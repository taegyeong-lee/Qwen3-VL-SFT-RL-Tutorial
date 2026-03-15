"""
테스트셋 이미지 + 정답 라벨을 폴더로 추출.

Usage:
  python inference/extract_testset.py
  python inference/extract_testset.py --output data/testset --max-samples 50
"""

import os
import sys
import json
import shutil
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "teacher", "dataset.jsonl")


def extract_testset(dataset_path: str, output_dir: str, max_samples: int = None,
                    train_ratio: float = 0.8, val_ratio: float = 0.1):
    # 전체 데이터 로드
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    # test split (마지막 10%)
    n = len(samples)
    test_start = int(n * (train_ratio + val_ratio))
    test_samples = samples[test_start:]

    if max_samples:
        test_samples = test_samples[:max_samples]

    print(f"Total: {n} | Test split: {len(test_samples)}")

    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # 정답 라벨 파일
    labels = []

    for i, sample in enumerate(test_samples):
        # 이미지 복사
        img_rel = sample["images"][0]
        img_src = os.path.join(PROJECT_ROOT, img_rel)
        img_name = os.path.basename(img_rel)
        img_dst = os.path.join(output_dir, img_name)

        if os.path.exists(img_src):
            shutil.copy2(img_src, img_dst)
        else:
            print(f"  [SKIP] Image not found: {img_src}")
            continue

        # 메타데이터에서 정답 추출
        metadata = sample.get("metadata", {})
        actual_signal = metadata.get("actual_signal", "UNKNOWN")
        pct_change = metadata.get("pct_change", None)

        # assistant 응답 (teacher label)
        teacher_response = ""
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                teacher_response = msg["content"]

        labels.append({
            "image": img_name,
            "actual_signal": actual_signal,
            "pct_change": pct_change,
            "teacher_response": teacher_response,
        })

        print(f"  [{i+1}/{len(test_samples)}] {img_name} → {actual_signal} ({pct_change:+.2f}%)" if pct_change else
              f"  [{i+1}/{len(test_samples)}] {img_name} → {actual_signal}")

    # labels.json 저장
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    # 통계
    signal_counts = {}
    for l in labels:
        s = l["actual_signal"]
        signal_counts[s] = signal_counts.get(s, 0) + 1

    print(f"\nExtracted {len(labels)} samples → {output_dir}")
    print(f"Labels saved → {labels_path}")
    print(f"Distribution: {signal_counts}")


def main():
    parser = argparse.ArgumentParser(description="Extract test set images + labels")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH)
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_ROOT, "data", "testset"))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    extract_testset(args.dataset, args.output, args.max_samples)


if __name__ == "__main__":
    main()
