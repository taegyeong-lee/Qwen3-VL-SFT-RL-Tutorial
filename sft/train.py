"""
SFT: Qwen3-VL 4B LoRA Fine-tuning (TRL + PEFT)

TRL SFTTrainer의 VLM 네이티브 지원을 사용.
- DataCollatorForVisionLanguageModeling 자동 적용
- 데이터셋: {"messages": [...], "images": [path]} 포맷

Usage:
  Single GPU:
    python sft/train.py --config sft/configs/single.yaml
    python sft/train.py --config sft/configs/single.yaml --max-samples 50

  Multi GPU:
    python sft/train.py --config sft/configs/multi.yaml
    accelerate launch --num_processes 2 sft/train.py --config sft/configs/multi.yaml

  이미지 로딩 체크 (학습 없이 모델이 이미지를 인식하는지 확인):
    python sft/train.py --config sft/configs/single.yaml --check-image
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
from trl import SFTConfig, SFTTrainer


def load_sft_dataset(
    jsonl_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    max_samples: int = None,
):
    """dataset.jsonl → TRL VLM 포맷 HuggingFace Dataset.

    TRL이 기대하는 VLM 데이터셋 포맷:
      - "messages": 대화 메시지 (image placeholder 포함)
      - "images": 이미지 경로 리스트 (별도 컬럼)

    Note: messages 내 content 타입이 혼합(str/list)되어 Dataset.from_list()의
    Arrow 변환이 실패하므로 from_generator()를 사용.

    ref: https://huggingface.co/docs/trl/sft_trainer#training-vision-language-models
    """
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
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

    print(f"Dataset: {n} total | Train: {len(train_data)} | Val: {len(val_data)}")

    # from_generator: Arrow 스키마 추론을 우회하여 혼합 타입(str/list) content 지원
    def make_generator(data):
        def gen():
            for item in data:
                yield item
        return gen

    train_ds = Dataset.from_generator(make_generator(train_data))
    val_ds = Dataset.from_generator(make_generator(val_data)) if val_data else None

    return train_ds, val_ds


def _parse_entry(entry: dict) -> dict | None:
    """dataset.jsonl entry 로드. 상대 경로 → 절대 경로 변환만 수행.

    dataset.jsonl은 이미 TRL VLM 표준 포맷:
      {"messages": [...], "images": ["data/chart_images/xxx.png"], "metadata": {...}}
    """
    if "messages" not in entry or "images" not in entry:
        return None

    # 이미지 상대 경로 → 절대 경로
    images = []
    for img_rel in entry["images"]:
        img_abs = os.path.join(PROJECT_ROOT, img_rel)
        if not os.path.exists(img_abs):
            return None
        images.append(img_abs)

    # TRL prepare_multimodal_messages는 모든 content가 list[dict] 형태여야 함
    # string content → [{"type": "text", "text": ...}] 로 통일
    messages = []
    for msg in entry["messages"]:
        content = msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        messages.append({"role": msg["role"], "content": content})

    return {
        "messages": messages,
        "images": images,
    }


def check_image(model_name: str, dataset_path: str):
    """학습 전 이미지 로딩 체크. 모델이 이미지를 제대로 인식하는지 확인."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    import torch

    print("=" * 60)
    print("Image Check Mode")
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

    # 데이터셋 원래 프롬프트로도 테스트
    original_messages = entry["messages"]
    # messages에서 user text 추출
    user_text = ""
    for msg in original_messages:
        if msg["role"] == "user":
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "text":
                        user_text = item["text"]

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

    print(f"\nPrompt: {user_text}")
    print(f"{'─' * 60}")
    print(f"Response (base model, no fine-tune):\n{response2}")
    print(f"{'─' * 60}")
    print("\nImage check passed!")


def main():
    parser = argparse.ArgumentParser(description="SFT: Qwen3-VL LoRA Fine-tuning (TRL + PEFT)")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--check-image", action="store_true",
                        help="학습 없이 이미지 로딩 + 모델 인식 체크만 수행")
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]
    dataset_path = os.path.join(PROJECT_ROOT, cfg.get("dataset_path", "data/teacher/dataset.jsonl"))

    # 이미지 체크 모드
    if args.check_image:
        check_image(model_name, dataset_path)
        return

    max_samples = args.max_samples or cfg.get("max_samples")

    print("=" * 60)
    print(f"SFT Training: {model_name}")
    print(f"Config: {os.path.basename(config_path)}")
    print("=" * 60)

    # Dataset
    train_ds, val_ds = load_sft_dataset(
        dataset_path,
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        max_samples=max_samples,
    )

    # LoRA config (SFTTrainer에 직접 전달)
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

    # SFTConfig
    output_dir = os.path.join(PROJECT_ROOT, cfg.get("output_dir", "outputs/sft_lora"))

    sft_kwargs = dict(
        output_dir=output_dir,
        # 모델 로드 시 device_map 명시 (meta device 문제 방지)
        model_init_kwargs={"torch_dtype": "bfloat16", "device_map": None},
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        fp16=False,
        bf16=cfg.get("bf16", True),
        optim=cfg.get("optim", "adamw_torch"),
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 2),
        report_to=cfg.get("report_to", "tensorboard"),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        # VLM: 이미지 토큰 잘림 방지
        max_length=None,
    )

    if val_ds:
        sft_kwargs["eval_strategy"] = "steps"
        sft_kwargs["eval_steps"] = cfg.get("eval_steps", 200)
        sft_kwargs["load_best_model_at_end"] = True
        sft_kwargs["metric_for_best_model"] = "eval_loss"

    training_args = SFTConfig(**sft_kwargs)

    # SFTTrainer: VLM 감지 시 DataCollatorForVisionLanguageModeling 자동 적용
    trainer = SFTTrainer(
        model=model_name,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
    )

    # Summary
    eff = cfg["batch_size"] * cfg["grad_accum"]
    print(f"\nLoRA bf16 | rank={cfg.get('lora_rank', 16)}, alpha={cfg.get('lora_alpha', 32)}")
    print(f"Batch: {cfg['batch_size']} x {cfg['grad_accum']} = {eff} effective")
    print(f"Epochs: {cfg['epochs']} | LR: {cfg['lr']}")
    print(f"Output: {output_dir}\n")

    # Train
    trainer.train(resume_from_checkpoint=args.resume)

    # Save
    trainer.save_model(os.path.join(output_dir, "final"))
    print(f"\nDone! Model saved to {output_dir}/final")


if __name__ == "__main__":
    main()
