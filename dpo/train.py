"""
DPO: Direct Preference Optimization Training

SFT 모델 위에 DPO로 chosen/rejected 쌍을 학습하여 정확도 향상.

Usage:
  # 1) 먼저 페어 생성
  python dpo/build_pairs.py --config dpo/configs/3060.yaml

  # 2) DPO 학습
  python dpo/train.py --config dpo/configs/3060.yaml

  A100:
    python dpo/train.py --config dpo/configs/a100.yaml
    accelerate launch --num_processes 2 dpo/train.py --config dpo/configs/a100.yaml
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from PIL import Image


def load_dpo_dataset(jsonl_path: str, max_samples: int = None) -> list[dict]:
    """Load DPO pairs into format expected by TRL DPOTrainer.

    Returns list of {"prompt": [...], "chosen": str, "rejected": str, "images": [PIL.Image]}
    """
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            parsed = _parse_dpo_entry(entry)
            if parsed:
                samples.append(parsed)

    if max_samples:
        samples = samples[:max_samples]

    print(f"DPO Dataset: {len(samples)} pairs")
    return samples


def _parse_dpo_entry(entry: dict) -> dict | None:
    """Convert DPO pair entry to TRL format."""
    image_path = entry.get("image_path")
    if not image_path:
        return None

    abs_path = os.path.join(PROJECT_ROOT, image_path)
    if not os.path.exists(abs_path):
        return None

    chosen = entry.get("chosen", "")
    rejected = entry.get("rejected", "")
    if not chosen or not rejected:
        return None

    img = Image.open(abs_path).convert("RGB")

    # Build prompt messages in VLM format
    prompt_msgs = []
    for msg in entry.get("prompt", []):
        if msg["role"] == "system":
            prompt_msgs.append({"role": "system", "content": [{"type": "text", "text": msg["content"]}]})
        elif msg["role"] == "user":
            prompt_msgs.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": msg["content"]},
                ],
            })

    return {
        "prompt": prompt_msgs,
        "chosen": [{"role": "assistant", "content": [{"type": "text", "text": chosen}]}],
        "rejected": [{"role": "assistant", "content": [{"type": "text", "text": rejected}]}],
        "images": [img],
    }


def main():
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--sft-adapter", type=str, default=None,
                        help="SFT LoRA adapter path (override yaml)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    import torch
    from unsloth import FastVisionModel
    from trl import DPOConfig, DPOTrainer

    model_name = cfg["model"]
    sft_adapter = args.sft_adapter or os.path.join(PROJECT_ROOT, cfg.get("sft_adapter_path", "outputs/sft_lora/final"))
    load_in_4bit = cfg.get("quantize_4bit", True)

    print("=" * 60)
    print(f"DPO Training: {model_name}")
    print(f"Config: {os.path.basename(config_path)}")
    print("=" * 60)

    # Load model
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    # Load SFT adapter first, then apply new LoRA for DPO
    if os.path.exists(sft_adapter):
        print(f"Loading SFT adapter: {sft_adapter}")
        model.load_adapter(sft_adapter)

    model = FastVisionModel.get_peft_model(
        model,
        r=cfg.get("lora_rank", 32),
        lora_alpha=cfg.get("lora_alpha", 64),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    # Dataset
    dataset_path = os.path.join(PROJECT_ROOT, cfg.get("dataset_path", "data/dpo_pairs.jsonl"))
    max_samples = args.max_samples or cfg.get("max_samples")
    train_data = load_dpo_dataset(dataset_path, max_samples)

    # Split train/eval
    n = len(train_data)
    split_idx = int(n * 0.9)
    eval_data = train_data[split_idx:] if split_idx < n else None
    train_data = train_data[:split_idx]
    print(f"Train: {len(train_data)} | Eval: {len(eval_data) if eval_data else 0}")

    output_dir = os.path.join(PROJECT_ROOT, cfg.get("output_dir", "outputs/dpo_lora"))

    dpo_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("lr", 5e-6),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        optim=cfg.get("optim", "adamw_8bit"),
        fp16=False,
        bf16=cfg.get("bf16", True),
        max_grad_norm=cfg.get("max_grad_norm", 0.1),
        beta=cfg.get("beta", 0.1),
        max_length=cfg.get("max_length", 2048),
        max_prompt_length=cfg.get("max_prompt_length", 1024),
        loss_type=cfg.get("loss_type", "sigmoid"),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        report_to=cfg.get("report_to", "tensorboard"),
        remove_unused_columns=False,
    )

    training_args = DPOConfig(**dpo_kwargs)

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    q = "QLoRA 4-bit" if load_in_4bit else "LoRA bf16"
    eff = cfg.get("per_device_train_batch_size", 1) * cfg.get("gradient_accumulation_steps", 8)
    print(f"\n{q} | rank={cfg.get('lora_rank', 32)}, alpha={cfg.get('lora_alpha', 64)}")
    print(f"Batch: {cfg.get('per_device_train_batch_size', 1)} x {cfg.get('gradient_accumulation_steps', 8)} = {eff} effective")
    print(f"Beta: {cfg.get('beta', 0.1)} | Loss: {cfg.get('loss_type', 'sigmoid')}")
    print(f"Output: {output_dir}\n")

    FastVisionModel.for_training(model)
    trainer.train(resume_from_checkpoint=args.resume)

    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"\nDone! DPO model saved to {output_dir}/final")


if __name__ == "__main__":
    main()
