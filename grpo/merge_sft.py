"""
Merge SFT LoRA adapter into base model for GRPO training.

Usage:
  python grpo/merge_sft.py --config grpo/configs/a100.yaml
  python grpo/merge_sft.py --config grpo/configs/a100.yaml --sft-adapter outputs/sft_lora/final
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge SFT LoRA into base model")
    parser.add_argument("--config", type=str, required=True, help="GRPO YAML config path")
    parser.add_argument("--sft-adapter", type=str, default=None,
                        help="SFT LoRA adapter path (default: outputs/sft_lora/final)")
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]
    sft_adapter = args.sft_adapter or os.path.join(PROJECT_ROOT, cfg.get("sft_adapter_path", "outputs/sft_lora/final"))
    merged_path = os.path.join(PROJECT_ROOT, cfg.get("sft_merged_path", "outputs/sft_merged"))

    print("=" * 60)
    print(f"Merging SFT LoRA into base model")
    print(f"Base: {model_name}")
    print(f"Adapter: {sft_adapter}")
    print(f"Output: {merged_path}")
    print("=" * 60)

    if not os.path.exists(sft_adapter):
        print(f"ERROR: SFT adapter not found: {sft_adapter}")
        sys.exit(1)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoProcessor.from_pretrained(model_name)

    model = PeftModel.from_pretrained(model, sft_adapter)
    model = model.merge_and_unload()

    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"\nDone! Merged model saved to {merged_path}")


if __name__ == "__main__":
    main()
