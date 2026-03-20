"""
DPO: Direct Preference Optimization Training (TRL + PEFT)

SFT 머지 모델 위에 DPO LoRA를 학습하여 선호도 기반 정렬.

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

import yaml
from datasets import Dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer


def _parse_dpo_entry(entry: dict) -> dict | None:
    """DPO pair entry를 TRL VLM DPO 포맷으로 변환."""
    images = entry.get("images", [])
    if not images:
        return None

    abs_path = os.path.join(PROJECT_ROOT, images[0])
    if not os.path.exists(abs_path):
        return None

    chosen = entry.get("chosen")
    rejected = entry.get("rejected")
    if not chosen or not rejected:
        return None

    # prompt 구성
    prompt = entry.get("prompt", [])

    # chosen/rejected: TRL은 content가 string이어야 함
    # build_pairs.py 출력: [{"role": "assistant", "content": "text"}]
    chosen_msgs = chosen if isinstance(chosen, list) else [{"role": "assistant", "content": chosen}]
    rejected_msgs = rejected if isinstance(rejected, list) else [{"role": "assistant", "content": rejected}]

    return {
        "prompt": prompt,
        "chosen": chosen_msgs,
        "rejected": rejected_msgs,
        "images": [abs_path],
    }


def load_dpo_dataset(jsonl_path: str, max_samples: int = None):
    """DPO pairs를 로드하여 train/eval split 반환."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            parsed = _parse_dpo_entry(entry)
            if parsed:
                samples.append(parsed)

    if max_samples:
        samples = samples[:max_samples]

    # 90/10 split
    n = len(samples)
    split_idx = int(n * 0.9)
    train_data = samples[:split_idx]
    eval_data = samples[split_idx:] if split_idx < n else None

    def gen(data):
        for item in data:
            yield item

    train_ds = Dataset.from_generator(lambda: gen(train_data))
    eval_ds = Dataset.from_generator(lambda: gen(eval_data)) if eval_data else None

    print(f"DPO Dataset: {n} pairs | Train: {len(train_data)} | Eval: {len(eval_data) if eval_data else 0}")
    return train_ds, eval_ds


def check_image(model_name: str, dataset_path: str):
    """DPO 학습 전 이미지 로딩 체크. 모델이 이미지를 제대로 인식하는지 확인."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    import torch

    print("=" * 60)
    print("DPO Image Check Mode")
    print("=" * 60)

    # 데이터셋에서 첫 번째 이미지 로드
    with open(dataset_path, "r", encoding="utf-8") as f:
        entry = json.loads(f.readline())

    img_rel = entry["images"][0]
    img_path = os.path.join(PROJECT_ROOT, img_rel)
    print(f"Image: {img_path}")
    print(f"Exists: {os.path.exists(img_path)}")

    img = Image.open(img_path).convert("RGB")
    print(f"Size: {img.size}, Mode: {img.mode}")

    # 모델 로드
    print(f"\nLoading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # "Describe this image in detail" 로 테스트
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": "Describe this image in detail."},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

    print(f"\nInput tokens: {inputs['input_ids'].shape[1]}")
    print(f"Pixel values shape: {inputs.get('pixel_values', 'N/A')}")
    print(f"\nGenerating response...")

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=256)

    trimmed = generated[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(trimmed, skip_special_tokens=True)

    print(f"\n{'─' * 60}")
    print(f"Prompt: Describe this image in detail.")
    print(f"{'─' * 60}")
    print(f"Response:\n{response}")
    print(f"{'─' * 60}")

    # DPO prompt로도 테스트
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
        messages2 = [
            {"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": user_text},
            ]},
        ]
        text2 = processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        inputs2 = processor(text=[text2], images=[img], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated2 = model.generate(**inputs2, max_new_tokens=512)

        trimmed2 = generated2[0][inputs2["input_ids"].shape[1]:]
        response2 = processor.decode(trimmed2, skip_special_tokens=True)

        print(f"\nPrompt: {user_text[:100]}...")
        print(f"{'─' * 60}")
        print(f"Response:\n{response2}")
        print(f"{'─' * 60}")

    # chosen/rejected 확인
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

    # SFT 머지 모델을 base로 사용 (LoRA가 아닌 풀 모델)
    model_name = os.path.join(PROJECT_ROOT, cfg.get("sft_merged_path", "outputs/sft_merged"))
    if not os.path.exists(model_name):
        # fallback: base model
        model_name = cfg["model"]
        print(f"WARNING: sft_merged_path not found, using base model: {model_name}")

    dataset_path = os.path.join(PROJECT_ROOT, cfg.get("dataset_path", "data/dpo_pairs.jsonl"))
    output_dir = os.path.join(PROJECT_ROOT, cfg.get("output_dir", "outputs/dpo_lora"))
    max_samples = args.max_samples or cfg.get("max_samples")

    # 이미지 체크 모드
    if args.check_image:
        check_image(model_name, dataset_path)
        return

    print("=" * 60)
    print(f"DPO Training: {model_name}")
    print(f"Config: {os.path.basename(config_path)}")
    print("=" * 60)

    # Dataset
    train_ds, eval_ds = load_dpo_dataset(dataset_path, max_samples)

    # LoRA config
    target_modules = cfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    peft_config = LoraConfig(
        r=cfg.get("lora_rank", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    # DPOConfig
    dpo_kwargs = dict(
        output_dir=output_dir,
        model_init_kwargs={"torch_dtype": "bfloat16", "device_map": None},
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("lr", 5e-6),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        optim=cfg.get("optim", "adamw_torch"),
        fp16=False,
        bf16=cfg.get("bf16", True),
        max_grad_norm=cfg.get("max_grad_norm", 0.1),
        beta=cfg.get("beta", 0.1),
        loss_type=cfg.get("loss_type", "sigmoid"),
        # VLM: 이미지 토큰 잘림 방지
        max_length=None,
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 100),
        report_to=cfg.get("report_to", "tensorboard"),
        remove_unused_columns=False,
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        dataset_num_proc=cfg.get("dataset_num_proc", os.cpu_count()),
    )

    if eval_ds:
        dpo_kwargs["eval_strategy"] = "steps"
        dpo_kwargs["eval_steps"] = cfg.get("eval_steps", 100)

    training_args = DPOConfig(**dpo_kwargs)

    # DPOTrainer: model을 string으로 전달하면 from_pretrained 자동 호출
    trainer = DPOTrainer(
        model=model_name,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    # Summary
    eff = cfg.get("per_device_train_batch_size", 1) * cfg.get("gradient_accumulation_steps", 8)
    print(f"\nLoRA bf16 | rank={cfg.get('lora_rank', 16)}, alpha={cfg.get('lora_alpha', 32)}")
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
