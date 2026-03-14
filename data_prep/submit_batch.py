"""
Step 2: OpenAI Batch API 제출 + 완료 대기 + 결과 다운로드

- batch_parts/*.jsonl 파일들을 순차적으로 제출
- 각 배치 완료 대기 후 결과 다운로드
- 모든 결과를 batch_results.jsonl 로 합침
"""

import os
import sys
import time
import json
import argparse
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from openai import OpenAI

BATCH_DIR = os.path.join(PROJECT_ROOT, "data", "batch_parts")
RESULT_JSONL = os.path.join(PROJECT_ROOT, "data", "batch_results.jsonl")
STATE_FILE = os.path.join(PROJECT_ROOT, "data", "batch_state.json")


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"batches": []}


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def submit_part(client: OpenAI, part_path: str) -> str:
    filename = os.path.basename(part_path)
    print(f"\nUploading {filename} ...")
    with open(part_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  File ID: {uploaded.id}")

    print("Creating batch ...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"BTC chart - {filename}"},
    )
    print(f"  Batch ID: {batch.id} | Status: {batch.status}")
    return batch.id


def poll(client: OpenAI, batch_id: str, interval: int = 30):
    print(f"Polling {batch_id} ...")
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else 0
        failed = batch.request_counts.failed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0

        print(f"  [{time.strftime('%H:%M:%S')}] {status} - {completed}/{total} done, {failed} failed")

        if status == "completed":
            return batch
        elif status in ("failed", "expired", "cancelled"):
            print(f"Batch {status}!")
            if batch.errors:
                for err in batch.errors.data:
                    print(f"  Error: {err.code} — {err.message}")
            return None

        time.sleep(interval)


def download_results(client: OpenAI, batch, out_f) -> int:
    output_file_id = batch.output_file_id
    if not output_file_id:
        print("  No output file.")
        return 0

    content = client.files.content(output_file_id)
    data = content.read()
    lines = data.decode("utf-8").strip().split("\n")
    for line in lines:
        out_f.write(line + "\n")
    out_f.flush()

    if batch.error_file_id:
        error_path = os.path.join(PROJECT_ROOT, f"batch_errors_{batch.id}.jsonl")
        err_content = client.files.content(batch.error_file_id)
        with open(error_path, "wb") as ef:
            ef.write(err_content.read())
        print(f"  Errors saved: {error_path}")

    return len(lines)


def main():
    parser = argparse.ArgumentParser(description="Step 2: Batch 파일 순차 제출 + 대기 + 다운로드")
    parser.add_argument("--interval", type=int, default=30, help="폴링 간격 (초)")
    parser.add_argument("--submit-only", action="store_true", help="제출만 (폴링 안함)")
    parser.add_argument("--resume", action="store_true", help="이전 상태에서 이어서")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: Submit Batch Parts → Poll → Download")
    print("=" * 60)

    # 배치 파일 목록
    part_files = sorted(glob.glob(os.path.join(BATCH_DIR, "batch_part_*.jsonl")))
    if not part_files:
        print(f"Error: No batch files in {BATCH_DIR}/. Run 1_prepare_batch.py first.")
        sys.exit(1)
    print(f"Found {len(part_files)} batch file(s)")

    client = OpenAI()
    state = load_state() if args.resume else {"batches": []}

    # 이미 제출된 파일 확인
    submitted_files = {b["file"] for b in state["batches"]}

    # 제출
    for part_path in part_files:
        filename = os.path.basename(part_path)
        if filename in submitted_files:
            print(f"\nSkip (already submitted): {filename}")
            continue

        batch_id = submit_part(client, part_path)
        state["batches"].append({
            "file": filename,
            "batch_id": batch_id,
            "downloaded": False,
        })
        save_state(state)

    if args.submit_only:
        print(f"\nAll parts submitted. State saved to {STATE_FILE}")
        print("Run again with --resume to poll and download.")
        return

    # 폴링 + 다운로드
    total_results = 0
    with open(RESULT_JSONL, "a" if args.resume else "w", encoding="utf-8") as out_f:
        for batch_info in state["batches"]:
            if batch_info.get("downloaded"):
                print(f"\nSkip (already downloaded): {batch_info['file']}")
                continue

            print(f"\n--- {batch_info['file']} ---")
            batch = poll(client, batch_info["batch_id"], interval=args.interval)

            if batch and batch.status == "completed":
                count = download_results(client, batch, out_f)
                total_results += count
                batch_info["downloaded"] = True
                batch_info["result_count"] = count
                save_state(state)
                print(f"  Downloaded {count} results")
            else:
                print(f"  Failed — skipping")

    print(f"\n{'=' * 60}")
    print(f"Total results: {total_results}")
    print(f"Merged to: {RESULT_JSONL}")
    print("Done! Next: python 3_build_dataset.py")


if __name__ == "__main__":
    main()
