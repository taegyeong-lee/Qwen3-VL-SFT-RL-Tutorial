"""
Step 3: Batch 결과 + 메타 데이터 → 파인튜닝 JSONL 데이터셋 생성

- batch_results.jsonl에서 GPT 응답 파싱
- window_meta.json에서 이미지 + 미래 레이블 매칭
- TRL VLM 표준 포맷으로 dataset.jsonl 저장

출력 포맷:
  {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
      {"role": "assistant", "content": "..."}
    ],
    "images": ["data/chart_images/xxx.png"],
    "metadata": {"actual_signal": "LONG", ...}
  }
"""

import os
import json
import argparse

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "chart_images")
RESULT_JSONL = os.path.join(PROJECT_ROOT, "data", "teacher", "batch_results.jsonl")
META_PATH = os.path.join(PROJECT_ROOT, "data", "teacher", "window_meta.json")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "teacher", "dataset.jsonl")
PROMPTS_PATH = os.path.join(PROJECT_ROOT, "shared", "prompts.yaml")


def load_config(version: str = None) -> dict:
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ver = version or cfg["active"]
    prompts = cfg["versions"][ver]
    print(f"Prompt version: {ver} - {prompts['description']}")
    return prompts


def parse_gpt_response(raw: str) -> dict | None:
    """GPT 응답에서 JSON 파싱 (마크다운 코드블록 제거)"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Step 3: Batch 결과 -> 파인튜닝 데이터셋")
    parser.add_argument("--min-confidence", type=int, default=0,
                        help="최소 confidence 필터 (0=필터 없음)")
    parser.add_argument("--match-only", action="store_true",
                        help="GPT signal과 actual signal이 일치하는 것만 포함")
    parser.add_argument("--prompt-version", type=str, default=None,
                        help="프롬프트 버전 (기본: prompts.yaml active)")
    args = parser.parse_args()

    prompts = load_config(args.prompt_version)

    print("=" * 60)
    print("Step 3: Build Fine-tune Dataset")
    print("=" * 60)

    # 메타 로드
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta_list = json.load(f)
    meta_map = {m["custom_id"]: m for m in meta_list}
    print(f"Meta entries: {len(meta_map)}")

    # Batch 결과 로드
    results = {}
    with open(RESULT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            custom_id = entry["custom_id"]
            response = entry.get("response", {})
            if response.get("status_code") == 200:
                body = response["body"]
                raw = body["choices"][0]["message"]["content"]
                results[custom_id] = raw
    print(f"Batch results: {len(results)} (success)")

    # 데이터셋 생성
    stats = {"total": 0, "parsed": 0, "filtered": 0, "written": 0}
    signal_dist = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
    match_count = 0

    with open(DATASET_PATH, "w", encoding="utf-8") as out_f:
        for custom_id, raw_response in results.items():
            stats["total"] += 1

            meta = meta_map.get(custom_id)
            if not meta:
                continue

            gpt_result = parse_gpt_response(raw_response)
            if not gpt_result or "signal" not in gpt_result:
                stats["filtered"] += 1
                continue
            stats["parsed"] += 1

            # confidence 필터
            conf = gpt_result.get("confidence", 100)
            if conf < args.min_confidence:
                stats["filtered"] += 1
                continue

            # hindsight: actual label로 signal 강제 고정
            future_label = meta.get("future_label")
            is_hindsight = prompts.get("hindsight", False)

            if is_hindsight and future_label:
                gpt_result["signal"] = future_label["actual_signal"]

            # SL/TP 절대값 → 퍼센트 변환 (fallback)
            entry_price = future_label["entry_price"] if future_label else gpt_result.get("entry_price", 0)
            if entry_price and entry_price > 0:
                if "stop_loss" in gpt_result:
                    sl = gpt_result.pop("stop_loss")
                    gpt_result["stop_loss_pct"] = round((sl - entry_price) / entry_price * 100, 2)
                if "take_profit" in gpt_result:
                    tp = gpt_result.pop("take_profit")
                    gpt_result["take_profit_pct"] = round((tp - entry_price) / entry_price * 100, 2)
                gpt_result.pop("entry_price", None)

            if future_label:
                is_match = gpt_result["signal"] == future_label["actual_signal"]
                if is_match:
                    match_count += 1
                if args.match_only and not is_match:
                    stats["filtered"] += 1
                    continue

            signal = gpt_result["signal"]
            if signal in signal_dist:
                signal_dist[signal] += 1

            # 이미지 경로 확인
            img_path = os.path.join(IMAGES_DIR, meta["image_file"])
            if not os.path.exists(img_path):
                stats["filtered"] += 1
                continue

            # TRL VLM 표준 포맷으로 저장
            answer = json.dumps(gpt_result, ensure_ascii=False)
            entry = {
                "messages": [
                    {"role": "system", "content": prompts["finetune_system"].strip()},
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompts["finetune_user"].strip()},
                    ]},
                    {"role": "assistant", "content": answer},
                ],
                "images": [f"data/chart_images/{meta['image_file']}"],
            }

            if future_label:
                entry["metadata"] = future_label

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stats["written"] += 1

    # 통계 출력
    print(f"\n--- Stats ---")
    print(f"Total responses:  {stats['total']}")
    print(f"Parsed OK:        {stats['parsed']}")
    print(f"Filtered out:     {stats['filtered']}")
    print(f"Written:          {stats['written']}")
    print(f"\nSignal distribution: {signal_dist}")
    if stats['parsed'] > 0 and match_count > 0:
        print(f"GPT vs Actual match: {match_count}/{stats['parsed']} "
              f"({match_count/stats['parsed']*100:.1f}%)")
    print(f"\nDataset: {DATASET_PATH}")
    size_mb = os.path.getsize(DATASET_PATH) / (1024 * 1024)
    print(f"Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
