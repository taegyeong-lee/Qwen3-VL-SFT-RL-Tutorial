"""
모든 체크포인트를 한번에 평가 (배치 인퍼런스 지원).

Usage:
  python inference/evaluate_all.py --checkpoints-dir outputs/sft_lora
  python inference/evaluate_all.py --checkpoints-dir outputs/sft_lora --max-eval 50
  python inference/evaluate_all.py --checkpoints-dir outputs/sft_lora --batch-size 8
"""

import os
import sys
import json
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

from shared.dataset_utils import load_dataset_splits
from inference.predict import parse_output, DEFAULT_MODEL, SYSTEM_PROMPT, USER_PROMPT
from inference.metrics import compute_metrics, print_classification_report


def find_checkpoints(checkpoints_dir: str) -> list:
    """checkpoint-* 폴더와 final 폴더를 찾아서 정렬 반환."""
    candidates = []

    for name in os.listdir(checkpoints_dir):
        path = os.path.join(checkpoints_dir, name)
        if not os.path.isdir(path):
            continue
        if not os.path.exists(os.path.join(path, "adapter_config.json")):
            continue
        if name.startswith("checkpoint-"):
            step = int(name.split("-")[1])
            candidates.append((step, name, path))
        elif name == "final":
            candidates.append((999999, name, path))

    candidates.sort(key=lambda x: x[0])
    return candidates


def batch_predict(model, processor, images: list) -> list:
    """배치 인퍼런스. 여러 이미지를 한번에 처리."""
    texts = []
    pil_images = []

    for img in images:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        pil_images.append(img)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)

    inputs = processor(text=texts, images=pil_images, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    # 각 샘플에서 입력 토큰 이후만 디코딩
    outputs = []
    for i in range(len(images)):
        input_len = inputs["input_ids"][i].ne(processor.tokenizer.pad_token_id).sum().item()
        trimmed = generated_ids[i][input_len:]
        decoded = processor.decode(trimmed, skip_special_tokens=True)
        outputs.append(decoded)

    return outputs


def evaluate_checkpoint(base_model, processor, test_dataset, adapter_path: str,
                        max_eval: int = None, batch_size: int = 1) -> dict:
    """단일 체크포인트 평가."""
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    total = 0
    correct = 0
    parse_fail = 0
    results = []
    confusion = {"LONG": {"LONG": 0, "SHORT": 0, "NEUTRAL": 0},
                 "SHORT": {"LONG": 0, "SHORT": 0, "NEUTRAL": 0},
                 "NEUTRAL": {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}}

    n = len(test_dataset)
    if max_eval:
        n = min(n, max_eval)

    # 데이터 준비
    samples = []
    for i in range(n):
        sample = test_dataset[i]
        metadata = sample.get("metadata", {})
        actual_signal = metadata.get("actual_signal")
        if actual_signal:
            samples.append((i, sample["images"][0], actual_signal, metadata))

    # 배치 단위 처리
    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start:batch_start + batch_size]
        batch_images = [s[1] for s in batch]

        if batch_size > 1:
            try:
                raw_outputs = batch_predict(model, processor, batch_images)
            except Exception:
                # 배치 실패 시 하나씩 처리
                from inference.predict import predict
                raw_outputs = [predict(model, processor, img) for img in batch_images]
        else:
            from inference.predict import predict
            raw_outputs = [predict(model, processor, batch_images[0])]

        for j, (idx, image, actual_signal, metadata) in enumerate(batch):
            raw_output = raw_outputs[j]
            result = parse_output(raw_output)
            predicted = result.get("signal", "?")
            total += 1

            is_correct = False
            if predicted not in ("LONG", "SHORT", "NEUTRAL"):
                parse_fail += 1
                print(f"  [{total}/{len(samples)}] PARSE FAIL: {predicted}")
            else:
                is_correct = predicted == actual_signal
                if is_correct:
                    correct += 1
                confusion[actual_signal][predicted] += 1
                mark = "O" if is_correct else "X"
                print(f"  [{total}/{len(samples)}] pred={predicted} actual={actual_signal} {mark}")

            results.append({
                "index": idx,
                "actual_signal": actual_signal,
                "predicted_signal": predicted,
                "correct": is_correct,
                "response": result,
            })

    accuracy = correct / total * 100 if total > 0 else 0

    del model
    torch.cuda.empty_cache()

    return {
        "total": total,
        "correct": correct,
        "parse_fail": parse_fail,
        "accuracy": accuracy,
        "confusion": confusion,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints")
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "eval_results"))
    args = parser.parse_args()

    checkpoints_dir = os.path.join(PROJECT_ROOT, args.checkpoints_dir) if not os.path.isabs(args.checkpoints_dir) else args.checkpoints_dir
    checkpoints = find_checkpoints(checkpoints_dir)

    if not checkpoints:
        print(f"No checkpoints found in {checkpoints_dir}")
        return

    print("=" * 60)
    print(f"Evaluate All Checkpoints")
    print(f"Dir: {checkpoints_dir}")
    print(f"Found: {[name for _, name, _ in checkpoints]}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max eval: {args.max_eval or 'all'}")
    print("=" * 60)

    # base 모델 1번만 로드
    print(f"\nLoading base model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    _, _, test_dataset = load_dataset_splits()

    all_results = []
    for step, name, path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} ({path})")
        print(f"{'='*60}")

        metrics = evaluate_checkpoint(base_model, processor, test_dataset, path, args.max_eval, args.batch_size)
        metrics["name"] = name
        metrics["path"] = path
        all_results.append(metrics)

        print(f"\n  Accuracy: {metrics['accuracy']:.1f}% ({metrics['correct']}/{metrics['total']}) | Parse fail: {metrics['parse_fail']}")
        metrics["class_report"] = print_classification_report(metrics["confusion"], name)

    # 비교 테이블
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    print(f"{'Checkpoint':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'ParseFail':>10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['name']:<20} {r['accuracy']:>9.1f}% {r['correct']:>10} {r['total']:>8} {r['parse_fail']:>10}")

    best = max(all_results, key=lambda x: x["accuracy"])
    print(f"\nBest: {best['name']} ({best['accuracy']:.1f}%)")

    # 결과 저장
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, f"eval_all_{timestamp}.json")

    save_data = {
        "timestamp": timestamp,
        "checkpoints_dir": checkpoints_dir,
        "max_eval": args.max_eval,
        "batch_size": args.batch_size,
        "summary": [{
            "name": r["name"],
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "total": r["total"],
            "parse_fail": r["parse_fail"],
            "confusion": r["confusion"],
        } for r in all_results],
        "best": best["name"],
        "details": all_results,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved → {save_path}")


if __name__ == "__main__":
    main()
