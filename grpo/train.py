"""
GRPO: Reinforcement Learning on SFT-trained LoRA

SFT 학습 후 GRPO로 추가 학습하여 실제 시장 방향 예측 정확도를 높임.

Usage:
  RTX 3060 (no vLLM):
    python grpo/train.py --config grpo/configs/3060.yaml

  A100 x2 (vLLM server mode):
    # 1) SFT LoRA 머지 (최초 1회)
    python grpo/merge_sft.py --config grpo/configs/a100.yaml

    # 2) 터미널 1: vLLM 서버
    CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model outputs/sft_merged

    # 3) 터미널 2: GRPO 학습
    CUDA_VISIBLE_DEVICES=0 python grpo/train.py --config grpo/configs/a100.yaml
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
from grpo.rewards import format_reward_func, signal_reward_func, set_log_path


def load_grpo_dataset(jsonl_path: str, max_samples: int = None) -> list[dict]:
    """Load dataset for GRPO. Returns list with prompt, image, actual_signal."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            parsed = _parse_grpo_entry(entry)
            if parsed:
                samples.append(parsed)

    if max_samples:
        samples = samples[:max_samples]

    print(f"GRPO Dataset: {len(samples)} samples")
    return samples


def _parse_grpo_entry(entry: dict) -> dict | None:
    """Convert dataset entry to GRPO format."""
    metadata = entry.get("metadata", {})
    actual_signal = metadata.get("actual_signal")
    if not actual_signal:
        return None

    image_path = None
    system_text = ""
    user_text = ""

    if "messages" in entry:
        for msg in entry["messages"]:
            if msg["role"] == "system":
                system_text = msg["content"]
            elif msg["role"] == "user":
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            image_path = os.path.join(PROJECT_ROOT, item["image"])
                        elif item.get("type") == "text":
                            user_text = item["text"]
                else:
                    user_text = msg["content"]
    elif "image_path" in entry:
        image_path = os.path.join(PROJECT_ROOT, entry["image_path"])
        for msg in entry.get("prompt", []):
            if msg["role"] == "system":
                system_text = msg["content"]
            elif msg["role"] == "user":
                user_text = msg["content"]

    if not image_path or not os.path.exists(image_path):
        return None
    if not user_text:
        return None

    img = Image.open(image_path).convert("RGB")

    prompt = []
    if system_text:
        prompt.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
    prompt.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ],
    })

    return {
        "prompt": prompt,
        "image": img,
        "actual_signal": actual_signal,
    }


def main():
    parser = argparse.ArgumentParser(description="GRPO RL Training")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--sft-adapter", type=str, default=None,
                        help="SFT LoRA adapter path (override yaml)")
    parser.add_argument("--sft-merged", type=str, default=None,
                        help="Path to merged SFT model (override yaml)")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.max_samples:
        cfg["max_samples"] = args.max_samples

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model
    from trl import GRPOConfig, GRPOTrainer

    use_vllm = cfg.get("use_vllm", False)
    sft_adapter = args.sft_adapter or os.path.join(PROJECT_ROOT, cfg.get("sft_adapter_path", "outputs/sft_lora/final"))

    if use_vllm:
        merged_path = args.sft_merged or os.path.join(PROJECT_ROOT, cfg.get("sft_merged_path", "outputs/sft_merged"))
        if not os.path.exists(merged_path):
            print(f"ERROR: Merged model not found: {merged_path}")
            print(f"Run first: python grpo/merge_sft.py --config {args.config}")
            sys.exit(1)
        model_path = merged_path
    else:
        model_path = cfg["model"]

    print("=" * 60)
    print(f"GRPO Training {'(vLLM colocate)' if use_vllm else '(no vLLM)'}")
    print(f"Model: {model_path}")
    print(f"Config: {os.path.basename(config_path)}")
    print("=" * 60)

    model_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="cpu" if use_vllm else None,
    )
    if cfg.get("quantize_4bit", False):
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    tokenizer = AutoProcessor.from_pretrained(model_path)

    if not use_vllm and os.path.exists(sft_adapter):
        from peft import PeftModel
        print(f"Loading SFT adapter: {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter)

    peft_config = LoraConfig(
        r=cfg.get("lora_rank", 64),
        lora_alpha=cfg.get("lora_alpha", 128),
        lora_dropout=cfg.get("lora_dropout", 0),
        target_modules=cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    # Dataset
    dataset_path = os.path.join(PROJECT_ROOT, cfg.get("dataset_path", "data/dataset.jsonl"))
    train_data = load_grpo_dataset(dataset_path, cfg.get("max_samples"))

    # Output
    output_dir = os.path.join(PROJECT_ROOT, cfg.get("output_dir", "outputs/grpo_lora"))

    # Set log path for rewards
    set_log_path(os.path.join(output_dir, "generations.jsonl"))

    # Build GRPOConfig from yaml
    grpo_kwargs = dict(
        learning_rate=cfg.get("lr", 5e-6),
        adam_beta1=cfg.get("adam_beta1", 0.9),
        adam_beta2=cfg.get("adam_beta2", 0.99),
        weight_decay=cfg.get("weight_decay", 0.1),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        optim=cfg.get("optim", "adamw_8bit"),
        logging_steps=cfg.get("logging_steps", 1),
        log_completions=False,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        num_generations=cfg.get("num_generations", 8),
        max_completion_length=cfg.get("max_completion_length", 1024),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        save_steps=cfg.get("save_steps", 100),
        max_grad_norm=cfg.get("max_grad_norm", 0.1),
        report_to=cfg.get("report_to", "tensorboard"),
        output_dir=output_dir,
        fp16=False,
        bf16=cfg.get("bf16", True),
        remove_unused_columns=False,
        loss_type=cfg.get("loss_type", "dr_grpo"),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
    )

    if cfg.get("generation_batch_size"):
        grpo_kwargs["generation_batch_size"] = cfg["generation_batch_size"]

    # vLLM specific
    if use_vllm:
        vllm_mode = cfg.get("vllm_mode", "colocate")
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_mode"] = vllm_mode
        if vllm_mode == "colocate":
            grpo_kwargs["vllm_gpu_memory_utilization"] = cfg.get("vllm_gpu_memory_utilization", 0.4)
            grpo_kwargs["vllm_enable_sleep_mode"] = cfg.get("vllm_enable_sleep_mode", True)

    if cfg.get("temperature"):
        grpo_kwargs["temperature"] = cfg["temperature"]

    training_args = GRPOConfig(**grpo_kwargs)

    # vLLM colocate: 학습 모델을 CPU로 옮겨 GPU 비운 후 vLLM 초기화
    if use_vllm and cfg.get("vllm_mode") == "colocate":
        print("Moving training model to CPU for vLLM initialization...")
        model = model.to("cpu")
        torch.cuda.empty_cache()

    # TRL GRPOTrainer가 model.warnings_issued에 접근하는데 PeftModel에는 없음
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            signal_reward_func,
        ],
        train_dataset=train_data,
    )

    print(f"\nGRPO | rank={cfg.get('lora_rank', 64)} | generations={cfg.get('num_generations', 8)}")
    print(f"Output: {output_dir}\n")

    trainer.train()

    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"\nDone! GRPO model saved to {output_dir}/final")


if __name__ == "__main__":
    main()
