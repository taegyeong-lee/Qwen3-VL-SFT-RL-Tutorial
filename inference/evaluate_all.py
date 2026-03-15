"""
모든 체크포인트를 한번에 평가 (1개씩 순차 인퍼런스).

Usage:
  python inference/evaluate_all.py --checkpoints-dir outputs/sft_lora
  python inference/evaluate_all.py --checkpoints-dir outputs/sft_lora --max-eval 50

  # 이미지 로딩 체크 (평가 없이 이미지가 제대로 들어가는지 확인)
  python inference/evaluate_all.py --checkpoints-dir outputs/sft_lora --check-image
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
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

from shared.dataset_utils import load_dataset_splits
from inference.predict import parse_output, predict, DEFAULT_MODEL
from inference.metrics import print_classification_report


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


def evaluate_checkpoint(base_model, processor, test_dataset, adapter_path: str,
                        max_eval: int = None) -> dict:
    """단일 체크포인트 평가. 1개씩 순차 처리."""
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

    samples = []
    for i in range(n):
        sample = test_dataset[i]
        metadata = sample.get("metadata", {})
        actual_signal = metadata.get("actual_signal")
        if actual_signal:
            samples.append((i, sample["images"][0], actual_signal, metadata))

    for idx, image, actual_signal, metadata in samples:
        raw_output = predict(model, processor, image)
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


def check_image(test_dataset, model, processor, adapter_path: str = None):
    """이미지 로딩 체크. 첫 3개 샘플의 이미지가 제대로 로드되고 모델에 들어가는지 확인."""
    print("=" * 60)
    print("Image Check Mode")
    print("=" * 60)

    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        check_model = PeftModel.from_pretrained(model, adapter_path)
        check_model.eval()
    else:
        check_model = model

    n = min(3, len(test_dataset))
    for i in range(n):
        sample = test_dataset[i]
        img = sample["images"][0]  # PIL.Image (lazy loaded)
        metadata = sample.get("metadata", {})
        actual = metadata.get("actual_signal", "?")

        print(f"\n--- Sample {i+1}/{n} ---")
        print(f"Size: {img.size}, Mode: {img.mode}")
        print(f"Actual signal: {actual}")

        # Describe this image
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this image in detail."},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(check_model.device)

        print(f"Input tokens: {inputs['input_ids'].shape[1]}")
        pv = inputs.get("pixel_values")
        print(f"Pixel values: shape={pv.shape}, dtype={pv.dtype}" if pv is not None else "Pixel values: None")

        with torch.no_grad():
            generated = check_model.generate(**inputs, max_new_tokens=256)

        trimmed = generated[0][inputs["input_ids"].shape[1]:]
        response = processor.decode(trimmed, skip_special_tokens=True)
        print(f"Response: {response[:300]}{'...' if len(response) > 300 else ''}")

    if adapter_path:
        del check_model
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("Image check passed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints")
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "eval_results"))
    parser.add_argument("--check-image", action="store_true",
                        help="평가 없이 이미지 로딩 + 모델 인식 체크만 수행")
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

    if args.check_image:
        adapter_path = checkpoints[0][2] if checkpoints else None
        check_image(test_dataset, base_model, processor, adapter_path)
        return

    all_results = []
    for _, name, path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} ({path})")
        print(f"{'='*60}")

        metrics = evaluate_checkpoint(base_model, processor, test_dataset, path, args.max_eval)
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

    print(f"\nResults saved -> {save_path}")


if __name__ == "__main__":
    main()
