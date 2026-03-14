"""
Step 1: 저장된 이미지 + 메타 → OpenAI Batch API JSONL 생성

- 0_generate_charts.py에서 만든 chart_images/ + window_meta.json 읽기
- 이미지를 base64 인코딩해서 Batch API 요청 JSONL 생성
- 90MB 초과 시 자동으로 파일 분할 (OpenAI Batch API 파일 한도: 100MB)
"""

import os
import json
import base64
import argparse

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "chart_images")
META_PATH = os.path.join(PROJECT_ROOT, "data", "window_meta.json")
BATCH_DIR = os.path.join(PROJECT_ROOT, "data", "batch_parts")
PROMPTS_PATH = os.path.join(PROJECT_ROOT, "shared", "prompts.yaml")

MAX_FILE_SIZE = 90 * 1024 * 1024  # 90MB (100MB 한도에 여유)


def load_config(version: str = None) -> dict:
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ver = version or cfg["active"]
    prompts = cfg["versions"][ver]
    prompts["_model"] = cfg.get("model", "gpt-4.1-mini")
    prompts["_temperature"] = cfg.get("temperature", 0.3)
    prompts["_max_tokens"] = cfg.get("max_tokens", 800)
    print(f"Prompt version: {ver} - {prompts['description']}")
    print(f"Model: {prompts['_model']} | temp={prompts['_temperature']} | max_tokens={prompts['_max_tokens']}")
    return prompts


def make_batch_request(custom_id: str, image_b64: str, csv_data: str, prompts: dict,
                       future_label: dict = None) -> dict:
    fmt = {"csv_data": csv_data}
    if future_label:
        fmt.update(future_label)
    user_prompt = prompts["user"].strip().format(**fmt)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": prompts["_model"],
            "messages": [
                {"role": "system", "content": prompts["system"].strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            "max_tokens": prompts["_max_tokens"],
            "temperature": prompts["_temperature"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Step 1: 저장된 이미지 → Batch JSONL 생성 (자동 분할)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--prompt-version", type=str, default=None, help="프롬프트 버전 (기본: prompts.yaml active)")
    args = parser.parse_args()

    prompts = load_config(args.prompt_version)

    print("=" * 60)
    print("Step 1: Prepare Batch JSONL from saved images")
    print("=" * 60)

    if not os.path.exists(META_PATH):
        print(f"Error: {META_PATH} not found. Run 0_generate_charts.py first.")
        return

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta_list = json.load(f)
    if args.max_samples:
        meta_list = meta_list[:args.max_samples]
    print(f"Meta entries: {len(meta_list)}")

    os.makedirs(BATCH_DIR, exist_ok=True)

    part_idx = 0
    current_size = 0
    current_count = 0
    total_count = 0
    out_f = None
    files_created = []

    def open_new_part():
        nonlocal part_idx, current_size, current_count, out_f, files_created
        if out_f:
            out_f.close()
        part_path = os.path.join(BATCH_DIR, f"batch_part_{part_idx:03d}.jsonl")
        files_created.append(part_path)
        out_f = open(part_path, "w", encoding="utf-8")
        current_size = 0
        current_count = 0
        part_idx += 1

    open_new_part()

    for i, meta in enumerate(meta_list):
        img_path = os.path.join(IMAGES_DIR, meta["image_file"])
        if not os.path.exists(img_path):
            print(f"  SKIP: {meta['image_file']} not found")
            continue

        with open(img_path, "rb") as img_f:
            img_b64 = base64.b64encode(img_f.read()).decode("utf-8")

        csv_data = meta.get("csv_summary", "")
        future_label = meta.get("future_label")
        is_hindsight = prompts.get("hindsight", False)

        if is_hindsight and not future_label:
            print(f"  SKIP: no future_label for hindsight mode")
            continue

        req = make_batch_request(meta["custom_id"], img_b64, csv_data, prompts,
                                 future_label=future_label if is_hindsight else None)
        line = json.dumps(req, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        # 90MB 초과 시 새 파일
        if current_size + line_bytes > MAX_FILE_SIZE and current_count > 0:
            part_mb = current_size / (1024 * 1024)
            print(f"  Part {part_idx-1}: {current_count} requests ({part_mb:.1f} MB)")
            open_new_part()

        out_f.write(line)
        current_size += line_bytes
        current_count += 1
        total_count += 1

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(meta_list)}] part={part_idx}, size={current_size/(1024*1024):.1f}MB")

    if out_f:
        out_f.close()
        part_mb = current_size / (1024 * 1024)
        print(f"  Part {part_idx}: {current_count} requests ({part_mb:.1f} MB)")

    # 요약
    print(f"\n{'=' * 60}")
    print(f"Total requests: {total_count}")
    print(f"Batch files:    {len(files_created)}")
    for fp in files_created:
        sz = os.path.getsize(fp) / (1024 * 1024)
        print(f"  {os.path.basename(fp)} - {sz:.1f} MB")
    print(f"\nOutput dir: {BATCH_DIR}/")
    print("Done! Next: python 2_submit_batch.py")


if __name__ == "__main__":
    main()
