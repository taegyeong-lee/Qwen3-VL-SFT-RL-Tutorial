"""
Teacher 데이터셋 통계 분석

Usage:
  python shared/analyze_dataset.py
  python shared/analyze_dataset.py --path data/teacher/dataset.jsonl
  python shared/analyze_dataset.py --path data/dataset.jsonl
"""

import os
import sys
import json
import argparse
from collections import Counter, defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(path: str) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def extract_fields(entry: dict) -> dict:
    """entry에서 분석에 필요한 필드 추출."""
    metadata = entry.get("metadata", {})

    # assistant content 파싱
    assistant_text = None
    if "messages" in entry:
        for msg in entry["messages"]:
            if msg["role"] == "assistant":
                assistant_text = msg["content"]
    elif "completion" in entry:
        for msg in entry["completion"]:
            if msg["role"] == "assistant":
                assistant_text = msg["content"]

    parsed = {}
    if assistant_text:
        text = assistant_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {"_parse_error": True}

    return {
        "signal": parsed.get("signal"),
        "confidence": parsed.get("confidence"),
        "stop_loss_pct": parsed.get("stop_loss_pct"),
        "take_profit_pct": parsed.get("take_profit_pct"),
        "risk_level": parsed.get("risk_level"),
        "has_reasoning": "reasoning" in parsed,
        "parse_error": parsed.get("_parse_error", False),
        "actual_signal": metadata.get("actual_signal"),
        "pct_change": metadata.get("pct_change"),
        "entry_price": metadata.get("entry_price"),
    }


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def analyze(entries: list[dict]):
    fields = [extract_fields(e) for e in entries]
    n = len(fields)

    # ── 1. 기본 통계 ──
    print_section("기본 통계")
    print(f"총 샘플 수:    {n:,}")

    parse_errors = sum(1 for f in fields if f["parse_error"])
    print(f"JSON 파싱 실패: {parse_errors} ({parse_errors/n*100:.1f}%)")

    has_metadata = sum(1 for f in fields if f["actual_signal"] is not None)
    print(f"메타데이터 있음: {has_metadata} ({has_metadata/n*100:.1f}%)")

    has_reasoning = sum(1 for f in fields if f["has_reasoning"])
    print(f"reasoning 포함: {has_reasoning} ({has_reasoning/n*100:.1f}%)")

    # ── 2. Signal 분포 ──
    print_section("Signal 분포 (GPT 예측)")
    signal_counts = Counter(f["signal"] for f in fields if f["signal"])
    total_signals = sum(signal_counts.values())
    for sig in ["LONG", "SHORT", "NEUTRAL"]:
        cnt = signal_counts.get(sig, 0)
        pct = cnt / total_signals * 100 if total_signals > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {sig:>8}: {cnt:>5} ({pct:>5.1f}%) {bar}")

    other = {k: v for k, v in signal_counts.items() if k not in ("LONG", "SHORT", "NEUTRAL")}
    if other:
        print(f"  기타: {other}")

    # ── 3. Actual Signal 분포 ──
    actual_counts = Counter(f["actual_signal"] for f in fields if f["actual_signal"])
    if actual_counts:
        print_section("Actual Signal 분포 (실제 시장 방향)")
        total_actual = sum(actual_counts.values())
        for sig in ["LONG", "SHORT", "NEUTRAL"]:
            cnt = actual_counts.get(sig, 0)
            pct = cnt / total_actual * 100 if total_actual > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {sig:>8}: {cnt:>5} ({pct:>5.1f}%) {bar}")

    # ── 4. 예측 정확도 ──
    matched = [(f["signal"], f["actual_signal"]) for f in fields
               if f["signal"] and f["actual_signal"]]
    if matched:
        print_section("예측 정확도 (GPT 예측 vs 실제)")
        correct = sum(1 for p, a in matched if p == a)
        total_m = len(matched)
        print(f"  전체 정확도: {correct}/{total_m} ({correct/total_m*100:.1f}%)")

        # 클래스별 정확도
        for sig in ["LONG", "SHORT", "NEUTRAL"]:
            cls_total = sum(1 for _, a in matched if a == sig)
            cls_correct = sum(1 for p, a in matched if a == sig and p == a)
            if cls_total > 0:
                print(f"  {sig:>8}: {cls_correct}/{cls_total} ({cls_correct/cls_total*100:.1f}%)")

        # Confusion matrix
        print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
        print(f"  {'':>10} {'LONG':>8} {'SHORT':>8} {'NEUTRAL':>8}")
        confusion = defaultdict(lambda: defaultdict(int))
        for p, a in matched:
            confusion[a][p] += 1
        for actual in ["LONG", "SHORT", "NEUTRAL"]:
            row = confusion[actual]
            print(f"  {actual:>10} {row['LONG']:>8} {row['SHORT']:>8} {row['NEUTRAL']:>8}")

    # ── 5. Confidence 분포 ──
    confidences = [f["confidence"] for f in fields if f["confidence"] is not None]
    if confidences:
        print_section("Confidence 분포")
        arr = np.array(confidences)
        print(f"  평균: {arr.mean():.1f}")
        print(f"  중간값: {np.median(arr):.1f}")
        print(f"  표준편차: {arr.std():.1f}")
        print(f"  범위: {arr.min():.0f} ~ {arr.max():.0f}")

        # 구간별 분포
        bins = [(0, 30), (30, 50), (50, 70), (70, 85), (85, 100)]
        print(f"\n  구간별:")
        for lo, hi in bins:
            cnt = int(np.sum((arr >= lo) & (arr < hi if hi < 100 else arr <= hi)))
            pct = cnt / len(arr) * 100
            bar = "█" * int(pct / 2)
            print(f"    {lo:>3}-{hi:<3}: {cnt:>5} ({pct:>5.1f}%) {bar}")

        # signal별 confidence
        print(f"\n  Signal별 평균 confidence:")
        for sig in ["LONG", "SHORT", "NEUTRAL"]:
            sig_confs = [f["confidence"] for f in fields
                         if f["signal"] == sig and f["confidence"] is not None]
            if sig_confs:
                print(f"    {sig:>8}: {np.mean(sig_confs):.1f} (n={len(sig_confs)})")

    # ── 6. Stop Loss / Take Profit ──
    sl_values = [f["stop_loss_pct"] for f in fields if f["stop_loss_pct"] is not None]
    tp_values = [f["take_profit_pct"] for f in fields if f["take_profit_pct"] is not None]
    if sl_values or tp_values:
        print_section("Stop Loss / Take Profit (%)")
        if sl_values:
            sl = np.array(sl_values)
            print(f"  Stop Loss:  평균 {sl.mean():.2f}%, 중간값 {np.median(sl):.2f}%, 범위 [{sl.min():.2f}, {sl.max():.2f}]")
        if tp_values:
            tp = np.array(tp_values)
            print(f"  Take Profit: 평균 {tp.mean():.2f}%, 중간값 {np.median(tp):.2f}%, 범위 [{tp.min():.2f}, {tp.max():.2f}]")

        # signal별
        if sl_values and tp_values:
            print(f"\n  Signal별 평균:")
            print(f"  {'':>10} {'SL (%)':>10} {'TP (%)':>10} {'RR':>8}")
            for sig in ["LONG", "SHORT", "NEUTRAL"]:
                sig_sl = [f["stop_loss_pct"] for f in fields
                          if f["signal"] == sig and f["stop_loss_pct"] is not None]
                sig_tp = [f["take_profit_pct"] for f in fields
                          if f["signal"] == sig and f["take_profit_pct"] is not None]
                if sig_sl and sig_tp:
                    avg_sl = np.mean(sig_sl)
                    avg_tp = np.mean(sig_tp)
                    rr = abs(avg_tp / avg_sl) if avg_sl != 0 else 0
                    print(f"  {sig:>10} {avg_sl:>10.2f} {avg_tp:>10.2f} {rr:>7.2f}x")

    # ── 7. Risk Level 분포 ──
    risk_counts = Counter(f["risk_level"] for f in fields if f["risk_level"])
    if risk_counts:
        print_section("Risk Level 분포")
        total_risk = sum(risk_counts.values())
        for level in ["LOW", "MEDIUM", "HIGH"]:
            cnt = risk_counts.get(level, 0)
            pct = cnt / total_risk * 100 if total_risk > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {level:>8}: {cnt:>5} ({pct:>5.1f}%) {bar}")

    # ── 8. pct_change 분포 (실제 수익률) ──
    pct_changes = [f["pct_change"] for f in fields if f["pct_change"] is not None]
    if pct_changes:
        print_section("실제 가격 변동률 (pct_change)")
        arr = np.array(pct_changes)
        print(f"  평균: {arr.mean():.4f}%")
        print(f"  중간값: {np.median(arr):.4f}%")
        print(f"  표준편차: {arr.std():.4f}%")
        print(f"  범위: {arr.min():.4f}% ~ {arr.max():.4f}%")

        # 구간별
        bins = [(-np.inf, -2), (-2, -1), (-1, -0.7), (-0.7, 0.7), (0.7, 1), (1, 2), (2, np.inf)]
        labels = ["< -2%", "-2~-1%", "-1~-0.7%", "-0.7~+0.7%", "+0.7~+1%", "+1~+2%", "> +2%"]
        print(f"\n  구간별 분포:")
        for (lo, hi), label in zip(bins, labels):
            cnt = int(np.sum((arr > lo) & (arr <= hi)))
            pct = cnt / len(arr) * 100
            bar = "█" * int(pct / 2)
            print(f"    {label:>12}: {cnt:>5} ({pct:>5.1f}%) {bar}")

    # ── 9. 가격대별 분포 ──
    entry_prices = [f["entry_price"] for f in fields if f["entry_price"] is not None]
    if entry_prices:
        print_section("Entry Price 분포")
        arr = np.array(entry_prices)
        print(f"  범위: ${arr.min():,.0f} ~ ${arr.max():,.0f}")
        print(f"  평균: ${arr.mean():,.0f}")

    print(f"\n{'=' * 60}")
    print(f"  분석 완료")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Teacher 데이터셋 통계 분석")
    parser.add_argument("--path", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "teacher", "dataset.jsonl"),
                        help="dataset.jsonl 경로")
    args = parser.parse_args()

    path = os.path.join(PROJECT_ROOT, args.path) if not os.path.isabs(args.path) else args.path

    print("=" * 60)
    print(f"  Dataset Analysis: {os.path.basename(path)}")
    print(f"  Path: {path}")
    print("=" * 60)

    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        sys.exit(1)

    entries = load_dataset(path)
    analyze(entries)


if __name__ == "__main__":
    main()
