"""
Step 0: CSV → 차트 이미지 + CSV 요약 + 메타데이터 생성 (오프라인, API 호출 없음)

- 슬라이딩 윈도우로 차트 PNG 이미지를 chart_images/에 저장
- 각 윈도우의 CSV 요약 텍스트 생성
- 미래 레이블(LONG/SHORT/NEUTRAL) 계산
- window_meta.json 에 모든 메타 정보 저장
"""

import os
import io
import json
import base64
import argparse
import random

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

# ── 설정 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "btc_trading_view.csv")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "chart_images")
META_PATH = os.path.join(PROJECT_ROOT, "data", "window_meta.json")

WINDOW_SIZE = 96        # 96 * 15min = 24시간
STEP_SIZE = 48          # 12시간 간격
FUTURE_CANDLES = 16     # 향후 4시간 (정답 레이블용)
LABEL_THRESHOLD = 0.7   # 기본 LONG/SHORT 판정 기준 (%)


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df.columns = [
        "time", "open", "high", "low", "close", "volume",
        "futures_oi", "oi_open", "oi_high", "oi_low", "oi_close", "funding_rate"
    ]
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    for col in ["open", "high", "low", "close", "volume",
                "oi_open", "oi_high", "oi_low", "oi_close", "funding_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


def render_chart(window: pd.DataFrame) -> bytes:
    """4패널 차트 이미지를 PNG bytes로 생성 (LLM 비전 모델 최적화)"""
    plt.style.use("dark_background")
    plt.rcParams.update({"font.size": 13})

    fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1, 2]})

    dates = mdates.date2num(window["datetime"])
    start = window["datetime"].iloc[0].strftime("%Y-%m-%d %H:%M")
    end = window["datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M")
    fig.suptitle(f"BTC/USDT 15m  |  {start} ~ {end}",
                 fontsize=18, fontweight="bold", y=0.995)

    if len(dates) > 1:
        candle_w = (dates[1] - dates[0]) * 0.7
    else:
        candle_w = 0.005

    # 1. 캔들차트
    ax1 = axes[0]
    ohlc = list(zip(dates, window["open"], window["high"],
                     window["low"], window["close"]))
    candlestick_ohlc(ax1, ohlc, width=candle_w,
                     colorup="#26a69a", colordown="#ef5350", alpha=0.9)
    ax1.set_ylabel("Price (USDT)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.25)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.tick_params(axis="y", labelsize=12)

    # 2. 거래량
    ax2 = axes[1]
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(window["close"], window["open"])]
    ax2.bar(dates, window["volume"], width=candle_w, color=colors, alpha=0.7)
    ax2.set_ylabel("Volume", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.25)
    ax2.tick_params(axis="y", labelsize=12)

    # 3. 펀딩레이트
    ax3 = axes[2]
    fr = window["funding_rate"].values
    pos = np.where(fr >= 0, fr, 0)
    neg = np.where(fr < 0, fr, 0)
    ax3.fill_between(dates, pos, 0, color="#00C853", alpha=0.6)
    ax3.fill_between(dates, neg, 0, color="#FF5252", alpha=0.6)
    ax3.axhline(y=0, color="white", linewidth=0.5, alpha=0.4)
    ax3.set_ylabel("Funding", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.25)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.3f}%"))
    ax3.tick_params(axis="y", labelsize=12)

    # 4. OI 캔들차트
    ax4 = axes[3]
    oi_valid = window.dropna(subset=["oi_open", "oi_high", "oi_low", "oi_close"])
    if len(oi_valid) > 0:
        oi_dates = mdates.date2num(oi_valid["datetime"])
        oi_ohlc = list(zip(oi_dates, oi_valid["oi_open"], oi_valid["oi_high"],
                           oi_valid["oi_low"], oi_valid["oi_close"]))
        candlestick_ohlc(ax4, oi_ohlc, width=candle_w,
                         colorup="#FFD54F", colordown="#FF8A65", alpha=0.9)
    ax4.set_ylabel("Open Interest", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.25)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.1f}K"))
    ax4.tick_params(axis="y", labelsize=12)

    ax4.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax4.tick_params(axis="x", labelsize=12, rotation=25)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.18)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_csv_summary(window: pd.DataFrame) -> str:
    """윈도우 데이터를 간결한 텍스트 테이블로 요약 (마지막 12개 캔들)"""
    tail = window.tail(12).copy()
    rows = ["time,close,vol_M,oi_K,oi_chg,fr_pct"]
    prev_oi = None
    for _, r in tail.iterrows():
        t = r["datetime"].strftime("%m-%d %H:%M")
        close = f"{r['close']:.0f}"
        vol = f"{r['volume'] / 1e6:.2f}" if pd.notna(r["volume"]) else "-"
        oi = r.get("oi_close", np.nan)
        oi_str = f"{oi / 1e3:.1f}" if pd.notna(oi) else "-"
        oi_chg = ""
        if pd.notna(oi) and prev_oi is not None and pd.notna(prev_oi):
            chg = oi - prev_oi
            oi_chg = f"{chg:+.0f}"
        prev_oi = oi
        fr = r.get("funding_rate", np.nan)
        fr_str = f"{fr * 100:.4f}" if pd.notna(fr) else "-"
        rows.append(f"{t},{close},{vol},{oi_str},{oi_chg},{fr_str}")
    return "\n".join(rows)


def compute_future_label(df: pd.DataFrame, window_end_idx: int, threshold: float = LABEL_THRESHOLD) -> dict | None:
    future_end = window_end_idx + FUTURE_CANDLES
    if future_end > len(df):
        return None

    entry_price = df.loc[window_end_idx - 1, "close"]
    future_close = df.loc[future_end - 1, "close"]
    pct_change = (future_close - entry_price) / entry_price * 100

    if pct_change > threshold:
        actual = "LONG"
    elif pct_change < -threshold:
        actual = "SHORT"
    else:
        actual = "NEUTRAL"

    return {
        "entry_price": round(entry_price, 1),
        "future_close": round(future_close, 1),
        "pct_change": round(pct_change, 4),
        "actual_signal": actual,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 0: 차트 이미지 + CSV 요약 + 메타 생성")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument("--step", type=int, default=STEP_SIZE)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--balanced", type=int, default=None,
                        help="LONG/SHORT/NEUTRAL 균등 샘플링 (총 개수, 예: 400)")
    parser.add_argument("--threshold", type=float, default=LABEL_THRESHOLD,
                        help="LONG/SHORT 판정 기준 %% (기본: 0.7)")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 0: Generate Chart Images + Meta")
    print("=" * 60)

    df = load_csv()
    print(f"Loaded {len(df):,} candles | {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    all_windows = []
    for start in range(0, len(df) - args.window - FUTURE_CANDLES + 1, args.step):
        all_windows.append((start, start + args.window))

    if args.balanced:
        # 미래 레이블 미리 계산해서 LONG/SHORT/NEUTRAL 분류
        buckets = {"LONG": [], "SHORT": [], "NEUTRAL": []}
        for start, end in all_windows:
            label = compute_future_label(df, end, args.threshold)
            if label:
                sig = label["actual_signal"]
                buckets[sig].append((start, end))

        per_class = args.balanced // 3
        print(f"\nBalanced sampling: {args.balanced} total ({per_class} per class)")
        print(f"  Available - LONG: {len(buckets['LONG'])}, SHORT: {len(buckets['SHORT'])}, NEUTRAL: {len(buckets['NEUTRAL'])}")

        windows = []
        for sig in ["LONG", "SHORT", "NEUTRAL"]:
            pool = buckets[sig]
            if len(pool) <= per_class:
                chosen = pool
            else:
                # 균등 간격으로 샘플링 (시간적으로 분산)
                step = len(pool) / per_class
                chosen = [pool[int(i * step)] for i in range(per_class)]
            windows.extend(chosen)
            print(f"  {sig}: picked {len(chosen)}")

        random.seed(42)
        random.shuffle(windows)
        # 시간순 정렬
        windows.sort(key=lambda x: x[0])
        print(f"  Total selected: {len(windows)}")
    else:
        windows = all_windows
        if args.max_samples:
            windows = windows[:args.max_samples]

    print(f"Total windows: {len(windows)}")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)

    meta_list = []
    for i, (start, end) in enumerate(windows):
        window = df.iloc[start:end].reset_index(drop=True)
        ts_start = window["datetime"].iloc[0].strftime("%Y%m%d_%H%M")
        ts_end = window["datetime"].iloc[-1].strftime("%Y%m%d_%H%M")
        custom_id = f"w{start}_{ts_start}_{ts_end}"
        img_filename = f"chart_{ts_start}_{ts_end}.png"

        # 차트 렌더링 + 저장
        img_bytes = render_chart(window)
        img_path = os.path.join(IMAGES_DIR, img_filename)
        with open(img_path, "wb") as img_f:
            img_f.write(img_bytes)

        # CSV 요약
        csv_data = build_csv_summary(window)

        # 미래 레이블
        future_label = compute_future_label(df, end, args.threshold)

        meta_entry = {
            "custom_id": custom_id,
            "window_start": start,
            "window_end": end,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "image_file": img_filename,
            "csv_summary": csv_data,
        }
        if future_label:
            meta_entry["future_label"] = future_label
        meta_list.append(meta_entry)

        if (i + 1) % 50 == 0 or i == len(windows) - 1:
            label = future_label["actual_signal"] if future_label else "N/A"
            print(f"  [{i+1}/{len(windows)}] {ts_start} ~ {ts_end}  actual={label}")

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False)

    total_img_mb = sum(
        os.path.getsize(os.path.join(IMAGES_DIR, m["image_file"]))
        for m in meta_list
    ) / (1024 * 1024)

    print(f"\nImages: {IMAGES_DIR}/ ({len(meta_list)} files, {total_img_mb:.1f} MB)")
    print(f"Meta:   {META_PATH}")
    print("Done! Next: python 1_prepare_batch.py")


if __name__ == "__main__":
    main()
