"""
vLLM 배치 인퍼런스로 모든 체크포인트 평가 (매우 빠름).

사전 준비:
  pip install vllm

Usage:
  # 먼저 LoRA를 base 모델에 머지
  python inference/merge_lora.py --adapter outputs/sft_lora/checkpoint-100 --output outputs/merged/checkpoint-100
  python inference/merge_lora.py --adapter outputs/sft_lora/checkpoint-200 --output outputs/merged/checkpoint-200

  # 머지된 모델들로 평가
  python inference/evaluate_all_vllm.py --models-dir outputs/merged
  python inference/evaluate_all_vllm.py --models-dir outputs/merged --max-eval 50
"""

import os
import sys
import json
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
from shared.dataset_utils import load_dataset_splits
from inference.predict import parse_output, SYSTEM_PROMPT, USER_PROMPT
from inference.metrics import compute_metrics, print_classification_report


def build_prompt(image):
    """vLLM용 프롬프트 생성."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{USER_PROMPT}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def evaluate_model_vllm(model_path: str, test_dataset, max_eval: int = None) -> dict:
    """vLLM으로 한 모델 평가. 전체 테스트셋을 한번에 배치 처리."""
    from vllm import LLM, SamplingParams

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

    print(f"  Samples to evaluate: {len(samples)}")

    # vLLM 로드
    print(f"  Loading vLLM model: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(max_tokens=512, temperature=0)

    # 배치 요청 생성
    requests = []
    for idx, image, actual_signal, metadata in samples:
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image
        prompt = build_prompt(img)
        requests.append({
            "prompt": prompt,
            "multi_modal_data": {"image": img},
        })

    # 한번에 배치 인퍼런스
    print(f"  Running batch inference ({len(requests)} samples)...")
    outputs = llm.generate(requests, sampling_params=sampling_params)

    # 결과 처리
    total = 0
    correct = 0
    parse_fail = 0
    results = []
    confusion = {"LONG": {"LONG": 0, "SHORT": 0, "NEUTRAL": 0},
                 "SHORT": {"LONG": 0, "SHORT": 0, "NEUTRAL": 0},
                 "NEUTRAL": {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}}

    for j, (idx, image, actual_signal, metadata) in enumerate(samples):
        raw_output = outputs[j].outputs[0].text
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

    # vLLM 메모리 해제
    del llm
    import torch
    torch.cuda.empty_cache()

    return {
        "total": total,
        "correct": correct,
        "parse_fail": parse_fail,
        "accuracy": accuracy,
        "confusion": confusion,
        "results": results,
    }


def find_models(models_dir: str) -> list:
    """머지된 모델 폴더들을 찾아서 정렬 반환."""
    candidates = []
    for name in os.listdir(models_dir):
        path = os.path.join(models_dir, name)
        if not os.path.isdir(path):
            continue
        # config.json이 있으면 머지된 모델
        if os.path.exists(os.path.join(path, "config.json")):
            if "checkpoint" in name:
                step = int(name.split("-")[1]) if "-" in name else 0
            elif name == "final":
                step = 999999
            else:
                step = 0
            candidates.append((step, name, path))

    candidates.sort(key=lambda x: x[0])
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models with vLLM")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory with merged models")
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "eval_results"))
    args = parser.parse_args()

    models_dir = os.path.join(PROJECT_ROOT, args.models_dir) if not os.path.isabs(args.models_dir) else args.models_dir
    models = find_models(models_dir)

    if not models:
        print(f"No merged models found in {models_dir}")
        print("먼저 LoRA를 머지하세요:")
        print("  python inference/merge_lora.py --adapter outputs/sft_lora/checkpoint-100 --output outputs/merged/checkpoint-100")
        return

    print("=" * 60)
    print(f"Evaluate All Models (vLLM)")
    print(f"Dir: {models_dir}")
    print(f"Found: {[name for _, name, _ in models]}")
    print(f"Max eval: {args.max_eval or 'all'}")
    print("=" * 60)

    _, _, test_dataset = load_dataset_splits()

    all_results = []
    for step, name, path in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        metrics = evaluate_model_vllm(path, test_dataset, args.max_eval)
        metrics["name"] = name
        metrics["path"] = path
        all_results.append(metrics)

        print(f"\n  Accuracy: {metrics['accuracy']:.1f}% ({metrics['correct']}/{metrics['total']}) | Parse fail: {metrics['parse_fail']}")
        metrics["class_report"] = print_classification_report(metrics["confusion"], name)

    # 비교 테이블
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'ParseFail':>10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['name']:<20} {r['accuracy']:>9.1f}% {r['correct']:>10} {r['total']:>8} {r['parse_fail']:>10}")

    best = max(all_results, key=lambda x: x["accuracy"])
    print(f"\nBest: {best['name']} ({best['accuracy']:.1f}%)")

    # 결과 저장
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, f"eval_vllm_{timestamp}.json")

    save_data = {
        "timestamp": timestamp,
        "models_dir": models_dir,
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

    print(f"\nResults saved → {save_path}")


if __name__ == "__main__":
    main()
