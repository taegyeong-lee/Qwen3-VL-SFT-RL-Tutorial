"""
Test set evaluation for fine-tuned models.

Usage:
  # SFT 모델 평가
  python inference/evaluate.py --adapter outputs/sft_lora/final
  python inference/evaluate.py --adapter outputs/sft_lora/final --max-eval 50

  # DPO 모델 평가
  python inference/evaluate.py --adapter outputs/dpo_lora/final

  # GRPO 모델 평가
  python inference/evaluate.py --adapter outputs/grpo_lora/final

  # vLLM
  python inference/evaluate.py --backend vllm --model outputs/sft_merged
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.dataset_utils import load_dataset_splits
from inference.predict import (
    load_model_unsloth, predict_unsloth,
    load_model_vllm, predict_vllm,
    parse_output,
)


def evaluate_test_set(predict_fn, test_dataset, max_eval: int = None):
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    total = 0
    correct = 0
    parse_fail = 0
    confusion = {"LONG": {"LONG": 0, "SHORT": 0, "NEUTRAL": 0},
                 "SHORT": {"LONG": 0, "SHORT": 0, "NEUTRAL": 0},
                 "NEUTRAL": {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}}

    n = len(test_dataset)
    if max_eval:
        n = min(n, max_eval)

    for i in range(n):
        sample = test_dataset[i]
        image = sample["images"][0]
        metadata = sample.get("metadata", {})
        actual_signal = metadata.get("actual_signal")

        if not actual_signal:
            continue

        result = parse_output(predict_fn(image))
        predicted = result.get("signal", "?")
        total += 1

        if predicted not in ("LONG", "SHORT", "NEUTRAL"):
            parse_fail += 1
            print(f"  [{i+1}/{n}] PARSE FAIL: {predicted}")
            continue

        is_correct = predicted == actual_signal
        if is_correct:
            correct += 1
        confusion[actual_signal][predicted] += 1

        mark = "O" if is_correct else "X"
        print(f"  [{i+1}/{n}] pred={predicted} actual={actual_signal} {mark}")

    print(f"\n--- Results ---")
    print(f"Total: {total} | Correct: {correct} | Parse fail: {parse_fail}")
    if total > 0:
        print(f"Accuracy: {correct/total*100:.1f}%")

    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':>10} {'LONG':>8} {'SHORT':>8} {'NEUTRAL':>8}")
    for actual in ["LONG", "SHORT", "NEUTRAL"]:
        row = confusion[actual]
        print(f"{actual:>10} {row['LONG']:>8} {row['SHORT']:>8} {row['NEUTRAL']:>8}")


def main():
    parser = argparse.ArgumentParser(description="Test set evaluation")
    parser.add_argument("--backend", type=str, default="unsloth", choices=["unsloth", "vllm"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--no-4bit", dest="load_4bit", action="store_false", default=True)
    parser.add_argument("--max-eval", type=int, default=None)
    args = parser.parse_args()

    # Default model/adapter
    if args.model is None:
        if args.backend == "vllm":
            args.model = os.path.join(PROJECT_ROOT, "outputs", "sft_merged")
        else:
            args.model = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
    if args.adapter is None:
        args.adapter = os.path.join(PROJECT_ROOT, "outputs", "sft_lora", "final")

    # Load model
    if args.backend == "vllm":
        llm, sampling_params = load_model_vllm(args.model)
        predict_fn = lambda img: predict_vllm(llm, sampling_params, img)
    else:
        adapter = args.adapter if args.adapter and os.path.exists(args.adapter) else None
        model, tokenizer = load_model_unsloth(args.model, adapter, args.load_4bit)
        predict_fn = lambda img: predict_unsloth(model, tokenizer, img)

    _, _, test_dataset = load_dataset_splits()
    evaluate_test_set(predict_fn, test_dataset, args.max_eval)


if __name__ == "__main__":
    main()
