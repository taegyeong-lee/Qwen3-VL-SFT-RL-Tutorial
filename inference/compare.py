"""
Compare SFT vs DPO models on the same test set (transformers + PEFT).

Usage:
  python inference/compare.py
  python inference/compare.py --max-eval 50
  python inference/compare.py --adapters outputs/sft_lora/final outputs/dpo_lora/final
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

from shared.dataset_utils import load_dataset_splits
from inference.predict import predict, parse_output, DEFAULT_MODEL


def evaluate_adapter(base_model, processor, test_dataset, adapter_path: str, max_eval: int = None) -> dict:
    """Load adapter onto base model, evaluate, then unload."""
    name = os.path.basename(os.path.dirname(adapter_path))

    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
    else:
        print(f"  Adapter not found: {adapter_path}, using base model")
        model = base_model

    predict_fn = lambda img: predict(model, processor, img)

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
            continue

        if predicted == actual_signal:
            correct += 1
        confusion[actual_signal][predicted] += 1

    accuracy = (correct / total * 100) if total > 0 else 0

    # Unload adapter to reuse base model
    if isinstance(model, PeftModel):
        del model
        torch.cuda.empty_cache()

    return {
        "name": name,
        "adapter": adapter_path,
        "total": total,
        "correct": correct,
        "parse_fail": parse_fail,
        "accuracy": accuracy,
        "confusion": confusion,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare SFT vs DPO")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapters", nargs="+", default=None,
                        help="List of adapter paths to compare")
    parser.add_argument("--max-eval", type=int, default=None)
    args = parser.parse_args()

    # Default adapters
    if args.adapters is None:
        args.adapters = []
        for name in ["sft_lora", "dpo_lora"]:
            path = os.path.join(PROJECT_ROOT, "outputs", name, "final")
            if os.path.exists(path):
                args.adapters.append(path)

    if not args.adapters:
        print("No adapters found in outputs/. Specify with --adapters.")
        return

    print("=" * 60)
    print("Model Comparison")
    print(f"Adapters: {len(args.adapters)}")
    print("=" * 60)

    # Load base model once
    print(f"Loading base model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    _, _, test_dataset = load_dataset_splits()

    results = []
    for adapter_path in args.adapters:
        abs_path = os.path.join(PROJECT_ROOT, adapter_path) if not os.path.isabs(adapter_path) else adapter_path
        print(f"\n--- Evaluating: {abs_path} ---")
        r = evaluate_adapter(base_model, processor, test_dataset, abs_path, args.max_eval)
        results.append(r)

    # Summary table
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'ParseFail':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20} {r['accuracy']:>9.1f}% {r['correct']:>10} {r['total']:>8} {r['parse_fail']:>10}")

    # Best model
    best = max(results, key=lambda x: x["accuracy"])
    print(f"\nBest: {best['name']} ({best['accuracy']:.1f}%)")


if __name__ == "__main__":
    main()
