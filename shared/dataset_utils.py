"""
Dataset loading utilities for Qwen VL fine-tuning / evaluation.

llava-instruct-mix 포맷 호환:
  - images: list[PIL.Image]  (lazy loaded)
  - prompt: list[dict]  (system + user messages)
  - completion: list[dict]  (assistant message)

Time-based split: train (80%) -> val (10%) -> test (10%)
Lazy loading: 이미지를 __getitem__ 시점에 로드하여 메모리 절약.
"""

import os
import json
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "teacher", "dataset.jsonl")


class VLMDataset(TorchDataset):
    """
    Dataset wrapper that returns llava-instruct-mix compatible dicts.
    이미지는 lazy loading으로 __getitem__ 시점에 로드.
    """

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        result = {
            "images": [img],
            "prompt": sample["prompt"],
            "completion": sample["completion"],
        }
        if "metadata" in sample:
            result["metadata"] = sample["metadata"]
        return result


def _parse_entry(entry: dict) -> dict | None:
    """
    Parse a dataset.jsonl entry.
    이미지 경로만 저장하고, 실제 로드는 VLMDataset.__getitem__에서 수행.
    """
    # --- New format: image_path + prompt + completion ---
    if "image_path" in entry:
        img_path = os.path.join(PROJECT_ROOT, entry["image_path"])
        if not os.path.exists(img_path):
            return None
        result = {
            "image_path": img_path,
            "prompt": entry["prompt"],
            "completion": entry["completion"],
        }
        if "metadata" in entry:
            result["metadata"] = entry["metadata"]
        return result

    # --- Old format: messages + images ---
    messages = entry.get("messages")
    if not messages:
        return None

    # 이미지 경로: entry["images"] 또는 messages 내 item["image"]
    image_path = None
    if "images" in entry and entry["images"]:
        image_path = os.path.join(PROJECT_ROOT, entry["images"][0])
    else:
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "image" and "image" in item:
                        image_path = os.path.join(PROJECT_ROOT, item["image"])

    if not image_path or not os.path.exists(image_path):
        return None

    prompt = []
    completion = []
    for msg in messages:
        if msg["role"] == "system":
            prompt.append({"role": "system", "content": msg["content"]})
        elif msg["role"] == "user":
            if isinstance(msg["content"], list):
                text_parts = [item["text"] for item in msg["content"] if item.get("type") == "text"]
                prompt.append({"role": "user", "content": "\n".join(text_parts)})
            else:
                prompt.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            completion.append({"role": "assistant", "content": msg["content"]})

    if not prompt or not completion:
        return None

    result = {
        "image_path": image_path,
        "prompt": prompt,
        "completion": completion,
    }
    if "metadata" in entry:
        result["metadata"] = entry["metadata"]
    return result


def load_dataset_splits(
    path: str = DATASET_PATH,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    max_samples: int = None,
) -> tuple[VLMDataset, VLMDataset, VLMDataset]:
    """
    Load dataset.jsonl and split by time into train/val/test.

    Returns datasets in llava-instruct-mix format (lazy loaded):
      {"images": [PIL.Image], "prompt": [...], "completion": [...]}

    Default: 80% train, 10% val, 10% test (time-ordered, no leakage).
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            parsed = _parse_entry(entry)
            if parsed:
                samples.append(parsed)

    if max_samples:
        samples = samples[:max_samples]

    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = samples[:train_end]
    val_data = samples[train_end:val_end]
    test_data = samples[val_end:]

    print(f"Dataset loaded: {n} total (lazy loading, llava-instruct-mix format)")
    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    return (
        VLMDataset(train_data),
        VLMDataset(val_data),
        VLMDataset(test_data),
    )
