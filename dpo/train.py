"""
DPO: Direct Preference Optimization Training (TRL + PEFT)

Merged SFT 모델을 base로 로드하고, peft_config로 새 LoRA를 전달하여 DPO 학습.
Reference model = merged SFT 모델 (adapter 비활성화 시 base = merged SFT).

Usage:
  python dpo/train.py --config dpo/configs/single.yaml
  python dpo/train.py --config dpo/configs/single.yaml --max-samples 100

  # Multi GPU (FSDP2)
  accelerate launch --config_file sft/configs/fsdp2.yaml dpo/train.py --config dpo/configs/multi.yaml
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import yaml
from PIL import Image
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import DPOConfig, DPOTrainer


def _parse_dpo_entry(entry: dict) -> dict | None:
    """DPO pair entry 파싱. 텍스트만 추출, 이미지는 lazy load."""
    images = entry.get("images", [])
    if not images:
        return None

    abs_path = os.path.join(PROJECT_ROOT, images[0])

    chosen = entry.get("chosen")
    rejected = entry.get("rejected")
    if not chosen or not rejected:
        return None

    prompt = entry.get("prompt", [])
    chosen_msgs = chosen if isinstance(chosen, list) else [{"role": "assistant", "content": chosen}]
    rejected_msgs = rejected if isinstance(rejected, list) else [{"role": "assistant", "content": rejected}]

    return {
        "image_path": abs_path,
        "prompt_data": json.dumps(prompt, ensure_ascii=False),
        "chosen_data": json.dumps(chosen_msgs, ensure_ascii=False),
        "rejected_data": json.dumps(rejected_msgs, ensure_ascii=False),
    }


def load_dpo_dataset(jsonl_path: str, max_samples: int = None):
    """DPO pairs를 로드하여 train/eval split 반환. with_transform으로 lazy image loading.

    주의: distributed 환경에서 모든 rank가 동일한 dataset을 보도록
    필터링은 JSONL 파싱 시점에만 하고, transform에서는 절대 skip하지 않음.
    """
    # 1) JSONL 파싱 + 필터링 (여기서만 drop)
    samples = []
    skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            parsed = _parse_dpo_entry(entry)
            if parsed:
                samples.append(parsed)
            else:
                skipped += 1

    if skipped > 0:
        print(f"[Dataset] Skipped {skipped} invalid entries")

    if max_samples:
        samples = samples[:max_samples]

    n = len(samples)
    split_idx = int(n * 0.9)
    train_data = samples[:split_idx]
    eval_data = samples[split_idx:] if split_idx < n else None

    # 2) Dataset 생성 — transform에서는 절대 skip 없이 모든 sample 처리
    def make_dataset(data):
        ds = Dataset.from_dict({
            "image_path": [s["image_path"] for s in data],
            "prompt_data": [s["prompt_data"] for s in data],
            "chosen_data": [s["chosen_data"] for s in data],
            "rejected_data": [s["rejected_data"] for s in data],
        })

        def transform(examples):
            batch = {"images": [], "prompt": [], "chosen": [], "rejected": []}
            for i in range(len(examples["image_path"])):
                img = Image.open(examples["image_path"][i]).convert("RGB")
                batch["images"].append([img])
                batch["prompt"].append(json.loads(examples["prompt_data"][i]))
                batch["chosen"].append(json.loads(examples["chosen_data"][i]))
                batch["rejected"].append(json.loads(examples["rejected_data"][i]))
            return batch

        return ds.with_transform(transform)

    train_ds = make_dataset(train_data)
    eval_ds = make_dataset(eval_data) if eval_data else None

    print(f"DPO Dataset: {n} pairs | Train: {len(train_data)} | Eval: {len(eval_data) if eval_data else 0}")
    return train_ds, eval_ds


def check_image(model_name: str, dataset_path: str):
    """DPO 학습 전 이미지 로딩 체크."""
    from qwen_vl_utils import process_vision_info

    print("=" * 60)
    print("DPO Image Check Mode")
    print("=" * 60)

    with open(dataset_path, "r", encoding="utf-8") as f:
        entry = json.loads(f.readline())

    img_rel = entry["images"][0]
    img_path = os.path.join(PROJECT_ROOT, img_rel)
    print(f"Image: {img_path}")
    print(f"Exists: {os.path.exists(img_path)}")

    print(f"\nLoading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    def generate_response(messages, max_tokens=256):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt",
        ).to(model.device)
        print(f"  Input tokens: {inputs['input_ids'].shape[1]}")
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_tokens)
        trimmed = generated[0][inputs["input_ids"].shape[1]:]
        return processor.decode(trimmed, skip_special_tokens=True)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": "Describe this image in detail."},
    ]}]
    response = generate_response(messages)
    print(f"\n{'─' * 60}")
    print(f"Prompt: Describe this image in detail.")
    print(f"{'─' * 60}")
    print(f"Response:\n{response}")
    print(f"{'─' * 60}")

    prompt = entry.get("prompt", [])
    user_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        user_text = item["text"]
            elif isinstance(msg["content"], str):
                user_text = msg["content"]

    if user_text:
        messages2 = [{"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": user_text},
        ]}]
        response2 = generate_response(messages2, max_tokens=512)
        print(f"\nPrompt: {user_text[:100]}...")
        print(f"{'─' * 60}")
        print(f"Response:\n{response2}")
        print(f"{'─' * 60}")

    print(f"\nChosen: {entry.get('chosen', [{}])[0].get('content', '')[:100]}...")
    print(f"Rejected: {entry.get('rejected', [{}])[0].get('content', '')[:100]}...")
    print("\nImage check passed!")


def main():
    parser = argparse.ArgumentParser(description="DPO Training (TRL + PEFT)")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--check-image", action="store_true",
                        help="학습 없이 이미지 로딩 + 모델 인식 체크만 수행")
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 머지 모델 경로
    sft_merged_path = os.path.join(PROJECT_ROOT, cfg.get("sft_merged_path", "outputs/sft_merged"))
    if not os.path.exists(sft_merged_path):
        print(f"ERROR: Merged SFT model not found: {sft_merged_path}")
        sys.exit(1)

    dataset_path = os.path.join(PROJECT_ROOT, cfg.get("dataset_path", "data/dpo_dataset/dpo_pairs.jsonl"))
    output_dir = os.path.join(PROJECT_ROOT, cfg.get("output_dir", "outputs/dpo_lora"))
    max_samples = args.max_samples or cfg.get("max_samples")

    # 이미지 체크 모드
    if args.check_image:
        check_image(sft_merged_path, dataset_path)
        return

    print("=" * 60)
    print(f"DPO Training — Merged SFT + New LoRA")
    print(f"  Model (merged SFT): {sft_merged_path}")
    print(f"  Config: {os.path.basename(config_path)}")
    print("=" * 60)

    # Dataset
    train_ds, eval_ds = load_dpo_dataset(dataset_path, max_samples)

    # 머지 모델 로드
    print(f"\nLoading merged SFT model: {sft_merged_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        sft_merged_path,

        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Processor
    processor_kwargs = {}
    if cfg.get("max_pixels"):
        processor_kwargs["max_pixels"] = cfg["max_pixels"]
    if cfg.get("min_pixels"):
        processor_kwargs["min_pixels"] = cfg["min_pixels"]
    processor = AutoProcessor.from_pretrained(sft_merged_path, trust_remote_code=True, **processor_kwargs)

    # 새 LoRA 수동 적용
    target_modules = cfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    peft_config = LoraConfig(
        r=cfg.get("lora_rank", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    # DPOConfig
    dpo_kwargs = dict(
        output_dir=output_dir,
        beta=cfg.get("beta", 0.1),
        max_length=cfg.get("max_length"),
        # Training
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(cfg.get("lr", 5e-6)),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        optim=cfg.get("optim", "adamw_torch"),
        # Precision
        bf16=cfg.get("bf16", True),
        tf32=cfg.get("tf32", True),
        # Gradient
        max_grad_norm=cfg.get("max_grad_norm", 0.1),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Logging & Save
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 5),
        report_to=cfg.get("report_to", "tensorboard"),
        # Data
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
        # DPO specific
        loss_type=cfg.get("loss_type", "sigmoid"),
        max_prompt_length=cfg.get("max_prompt_length"),
        precompute_ref_log_probs=cfg.get("precompute_ref_log_probs", False),
    )

    dpo_kwargs["eval_strategy"] = cfg.get("eval_strategy", "no")
    if dpo_kwargs["eval_strategy"] != "no" and eval_ds:
        dpo_kwargs["eval_steps"] = cfg.get("eval_steps", 100)

    training_args = DPOConfig(**dpo_kwargs)

    # DPOTrainer: peft_config 전달 → TRL이 LoRA 적용 + reference 관리
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        peft_config=peft_config,
    )

    # Summary
    eff = cfg.get("per_device_train_batch_size", 1) * cfg.get("gradient_accumulation_steps", 8)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"\nTrainable params: {trainable:,} / {total:,}")
    print(f"Data collator: {type(trainer.data_collator).__name__}")
    print(f"Distributed: {trainer.accelerator.distributed_type}")
    print(f"Batch: {cfg.get('per_device_train_batch_size', 1)} x {cfg.get('gradient_accumulation_steps', 8)} = {eff} effective")
    print(f"Beta: {cfg.get('beta', 0.1)} | Loss: {cfg.get('loss_type', 'sigmoid')}")
    print(f"Output: {output_dir}\n")

    # Train
    trainer.train(resume_from_checkpoint=args.resume)

    # Save
    trainer.save_model(os.path.join(output_dir, "final"))
    print(f"\nDone! DPO model saved to {output_dir}/final")


if __name__ == "__main__":
    main()
