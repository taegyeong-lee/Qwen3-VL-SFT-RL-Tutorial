"""
Test set evaluation for fine-tuned models (transformers + PEFT).

Usage:
  python inference/evaluate.py --adapter outputs/sft_lora/final
  python inference/evaluate.py --adapter outputs/sft_lora/final --max-eval 50
  python inference/evaluate.py --adapter outputs/dpo_lora/final
"""

import os
import sys
import json
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.dataset_utils import load_dataset_splits
from inference.predict import load_model, predict, parse_output, DEFAULT_MODEL


def evaluate_test_set(predict_fn, test_dataset, max_eval: int = None, save_dir: str = None):
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

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

    for i in range(n):
        sample = test_dataset[i]
        image = sample["images"][0]
        metadata = sample.get("metadata", {})
        actual_signal = metadata.get("actual_signal")

        if not actual_signal:
            continue

        raw_output = predict_fn(image)
        result = parse_output(raw_output)
        predicted = result.get("signal", "?")
        total += 1

        is_correct = False
        if predicted not in ("LONG", "SHORT", "NEUTRAL"):
            parse_fail += 1
            print(f"  [{i+1}/{n}] PARSE FAIL: {predicted}")
        else:
            is_correct = predicted == actual_signal
            if is_correct:
                correct += 1
            confusion[actual_signal][predicted] += 1
            mark = "O" if is_correct else "X"
            print(f"  [{i+1}/{n}] pred={predicted} actual={actual_signal} {mark}")

        # 결과 저장
        results.append({
            "index": i,
            "image": getattr(image, "filename", str(image)),
            "actual_signal": actual_signal,
            "predicted_signal": predicted,
            "correct": is_correct,
            "response": result,
            "raw_output": raw_output,
            "metadata": metadata,
        })

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n--- Results ---")
    print(f"Total: {total} | Correct: {correct} | Parse fail: {parse_fail}")
    if total > 0:
        print(f"Accuracy: {accuracy:.1f}%")

    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':>10} {'LONG':>8} {'SHORT':>8} {'NEUTRAL':>8}")
    for actual in ["LONG", "SHORT", "NEUTRAL"]:
        row = confusion[actual]
        print(f"{actual:>10} {row['LONG']:>8} {row['SHORT']:>8} {row['NEUTRAL']:>8}")

    # 결과 파일 저장
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"eval_{timestamp}.json")

        save_data = {
            "timestamp": timestamp,
            "total": total,
            "correct": correct,
            "parse_fail": parse_fail,
            "accuracy": accuracy,
            "confusion": confusion,
            "results": results,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nResults saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Test set evaluation")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "eval_results"))
    args = parser.parse_args()

    if args.adapter is None:
        args.adapter = os.path.join(PROJECT_ROOT, "outputs", "sft_lora", "final")

    model, processor = load_model(args.model, args.adapter)
    predict_fn = lambda img: predict(model, processor, img)

    _, _, test_dataset = load_dataset_splits()
    evaluate_test_set(predict_fn, test_dataset, args.max_eval, args.save_dir)


if __name__ == "__main__":
    main()
