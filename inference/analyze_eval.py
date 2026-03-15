"""
Eval 결과 JSON 파일을 분석하여 시각화 및 통계를 출력한다.

Usage:
    python inference/analyze_eval.py --input eval_vllm_20260315_044221.json
    python inference/analyze_eval.py --input eval_vllm_20260315_044221.json --save-dir outputs/eval_analysis
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


SIGNALS = ["LONG", "SHORT", "NEUTRAL"]
COLORS = {"LONG": "#26a69a", "SHORT": "#ef5350", "NEUTRAL": "#78909c"}


def load_eval(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary(data: dict):
    print("=" * 60)
    print(f"Eval Results  |  Best: {data['best']}")
    print(f"Timestamp: {data['timestamp']}")
    print("=" * 60)

    header = f"{'Checkpoint':<20} {'Acc':>6} {'Correct':>8} {'Total':>6} {'Parse Fail':>11}"
    print(header)
    print("-" * 60)
    for s in data["summary"]:
        mark = " <-- best" if s["name"] == data["best"] else ""
        print(
            f"{s['name']:<20} {s['accuracy']:>5.1f}% {s['correct']:>8} {s['total']:>6} {s['parse_fail']:>11}{mark}"
        )
    print()


def print_confusion(detail: dict):
    print(f"\n--- {detail['name']} Confusion Matrix ---")
    cm = detail["confusion"]
    print(f"{'':>12} {'Pred LONG':>10} {'Pred SHORT':>11} {'Pred NEUTRAL':>13}")
    for actual in SIGNALS:
        row = cm.get(actual, {})
        vals = [row.get(pred, 0) for pred in SIGNALS]
        print(f"{'Act '+actual:>12} {vals[0]:>10} {vals[1]:>11} {vals[2]:>13}")
    print()


def print_class_report(detail: dict):
    cr = detail.get("class_report", {})
    if not cr:
        return
    print(f"--- {detail['name']} Classification Report ---")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("-" * 50)
    for sig in SIGNALS:
        m = cr.get(sig, {})
        print(
            f"{sig:<12} {m.get('precision',0):>9.1f}% {m.get('recall',0):>7.1f}% {m.get('f1',0):>7.1f}% {m.get('support',0):>9}"
        )
    macro = cr.get("macro_avg", {})
    print("-" * 50)
    print(
        f"{'Macro Avg':<12} {macro.get('precision',0):>9.1f}% {macro.get('recall',0):>7.1f}% {macro.get('f1',0):>7.1f}%"
    )
    print()


def plot_accuracy_curve(data: dict, save_dir: str | None = None):
    names = [s["name"] for s in data["summary"]]
    accs = [s["accuracy"] for s in data["summary"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, accs, color="#42a5f5", edgecolor="white", width=0.5)

    best_idx = names.index(data["best"])
    bars[best_idx].set_color("#26a69a")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, max(accs) * 1.3)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Checkpoint")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=150)
        print(f"Saved: {save_dir}/accuracy_curve.png")
    plt.show()


def plot_confusion_heatmap(detail: dict, save_dir: str | None = None):
    cm = detail["confusion"]
    matrix = np.array([[cm.get(a, {}).get(p, 0) for p in SIGNALS] for a in SIGNALS])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"Pred\n{s}" for s in SIGNALS])
    ax.set_yticklabels([f"Actual\n{s}" for s in SIGNALS])

    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=14, color=color)

    ax.set_title(f"Confusion Matrix — {detail['name']} (acc={detail['accuracy']:.1f}%)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"confusion_{detail['name']}.png"), dpi=150)
        print(f"Saved: {save_dir}/confusion_{detail['name']}.png")
    plt.show()


def plot_f1_comparison(data: dict, save_dir: str | None = None):
    details = data["details"]
    names = [d["name"] for d in details]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    width = 0.25

    for i, sig in enumerate(SIGNALS):
        f1s = []
        for d in details:
            cr = d.get("class_report", {})
            f1s.append(cr.get(sig, {}).get("f1", 0))
        ax.bar(x + i * width, f1s, width, label=sig, color=COLORS[sig])

    ax.set_xticks(x + width)
    ax.set_xticklabels(names)
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("F1 Score by Class & Checkpoint")
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "f1_comparison.png"), dpi=150)
        print(f"Saved: {save_dir}/f1_comparison.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze eval results JSON")
    parser.add_argument("--input", required=True, help="Path to eval JSON file")
    parser.add_argument("--save-dir", default=None, help="Directory to save plots (optional)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting, text only")
    args = parser.parse_args()

    data = load_eval(args.input)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Text summary
    print_summary(data)

    best_name = data["best"]
    for d in data["details"]:
        if d["name"] == best_name:
            print_confusion(d)
            print_class_report(d)
            break

    # All checkpoints class report
    print("\n=== All Checkpoints F1 ===")
    print(f"{'Checkpoint':<20} {'LONG F1':>8} {'SHORT F1':>9} {'NEUTRAL F1':>11} {'Macro F1':>9}")
    print("-" * 60)
    for d in data["details"]:
        cr = d.get("class_report", {})
        long_f1 = cr.get("LONG", {}).get("f1", 0)
        short_f1 = cr.get("SHORT", {}).get("f1", 0)
        neutral_f1 = cr.get("NEUTRAL", {}).get("f1", 0)
        macro_f1 = cr.get("macro_avg", {}).get("f1", 0)
        mark = " <--" if d["name"] == best_name else ""
        print(f"{d['name']:<20} {long_f1:>7.1f}% {short_f1:>8.1f}% {neutral_f1:>10.1f}% {macro_f1:>8.1f}%{mark}")

    # Plots
    if not args.no_plot:
        plot_accuracy_curve(data, args.save_dir)
        for d in data["details"]:
            if d["name"] == best_name:
                plot_confusion_heatmap(d, args.save_dir)
                break
        plot_f1_comparison(data, args.save_dir)


if __name__ == "__main__":
    main()
