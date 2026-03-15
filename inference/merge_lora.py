"""
LoRA adapter를 base 모델에 머지.

Usage:
  # 단일 체크포인트 머지
  python inference/merge_lora.py --adapter outputs/sft_lora/checkpoint-100 --output outputs/merged/checkpoint-100

  # 모든 체크포인트 한번에 머지
  python inference/merge_lora.py --checkpoints-dir outputs/sft_lora --output-dir outputs/merged
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

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def merge_single(base_model_name: str, adapter_path: str, output_path: str):
    """LoRA adapter를 base 모델에 머지하고 저장."""
    print(f"Loading base model: {base_model_name}")
    processor = AutoProcessor.from_pretrained(base_model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging...")
    model = model.merge_and_unload()

    print(f"Saving to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=str, default=None, help="Single adapter path")
    parser.add_argument("--output", type=str, default=None, help="Single output path")
    parser.add_argument("--checkpoints-dir", type=str, default=None, help="Merge all checkpoints in dir")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir for all merged models")
    args = parser.parse_args()

    if args.adapter and args.output:
        # 단일 머지
        adapter = os.path.join(PROJECT_ROOT, args.adapter) if not os.path.isabs(args.adapter) else args.adapter
        output = os.path.join(PROJECT_ROOT, args.output) if not os.path.isabs(args.output) else args.output
        merge_single(args.base_model, adapter, output)

    elif args.checkpoints_dir and args.output_dir:
        # 전체 머지
        ckpt_dir = os.path.join(PROJECT_ROOT, args.checkpoints_dir) if not os.path.isabs(args.checkpoints_dir) else args.checkpoints_dir
        out_dir = os.path.join(PROJECT_ROOT, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

        targets = []
        for name in os.listdir(ckpt_dir):
            path = os.path.join(ckpt_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
                targets.append((name, path))

        if not targets:
            print(f"No adapters found in {ckpt_dir}")
            return

        print(f"Found {len(targets)} adapters: {[t[0] for t in targets]}")

        for name, adapter_path in targets:
            output_path = os.path.join(out_dir, name)
            if os.path.exists(output_path):
                print(f"\nSkipping {name} (already exists)")
                continue
            print(f"\n{'='*60}")
            print(f"Merging: {name}")
            print(f"{'='*60}")
            merge_single(args.base_model, adapter_path, output_path)
    else:
        print("Usage:")
        print("  Single:  python merge_lora.py --adapter <path> --output <path>")
        print("  All:     python merge_lora.py --checkpoints-dir <dir> --output-dir <dir>")


if __name__ == "__main__":
    main()
