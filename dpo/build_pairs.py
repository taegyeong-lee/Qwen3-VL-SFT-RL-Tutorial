"""
DPO: Build chosen/rejected pairs via vLLM batch sampling.

SFT LoRA를 base에 머지한 모델을 vLLM으로 로드하고,
같은 차트에 대해 N번 생성 → actual_signal과 비교하여
chosen (정답) / rejected (오답) 쌍을 구성.

사전 준비:
  1) SFT LoRA → base 모델 머지
     python grpo/merge_sft.py --config dpo/configs/single.yaml

Usage:
  python dpo/build_pairs.py --config dpo/configs/single.yaml
  python dpo/build_pairs.py --config dpo/configs/single.yaml --max-samples 100
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
from vllm import LLM, SamplingParams


def parse_model_output(text: str) -> dict | None:
    """Parse JSON from model output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_source_dataset(jsonl_path: str, max_samples: int = None) -> list[dict]:
    """actual_signal이 있는 dataset entries 로드.

    TRL VLM 포맷: {"messages": [...], "images": [...], "metadata": {...}}
    """
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            metadata = entry.get("metadata", {})
            if not metadata.get("actual_signal"):
                continue
            if "messages" not in entry or "images" not in entry:
                continue

            # 이미지 경로 확인
            image_path = os.path.join(PROJECT_ROOT, entry["images"][0])
            if not os.path.exists(image_path):
                continue

            # messages에서 system/user 텍스트 추출
            system_text = ""
            user_text = ""
            for msg in entry["messages"]:
                if msg["role"] == "system":
                    system_text = msg["content"]
                elif msg["role"] == "user":
                    if isinstance(msg["content"], list):
                        for item in msg["content"]:
                            if item.get("type") == "text":
                                user_text = item["text"]
                    else:
                        user_text = msg["content"]

            if not user_text:
                continue

            samples.append({
                "image_path": image_path,
                "system_text": system_text,
                "user_text": user_text,
                "actual_signal": metadata["actual_signal"],
            })

    if max_samples:
        samples = samples[:max_samples]

    print(f"Source dataset: {len(samples)} samples with actual_signal")
    return samples


def generate_pairs(samples: list[dict], cfg: dict) -> list[dict]:
    """vLLM으로 SFT 모델 N번 샘플링 → chosen/rejected 쌍 생성."""
    model_path = os.path.join(PROJECT_ROOT, cfg.get("sft_merged_path", "outputs/sft_merged"))
    pair_cfg = cfg.get("pair_generation", {})
    num_samples = pair_cfg.get("num_samples_per_image", 8)
    temperature = pair_cfg.get("temperature", 1.0)
    max_new_tokens = pair_cfg.get("max_new_tokens", 512)

    if not os.path.exists(model_path):
        print(f"ERROR: Merged model not found: {model_path}")
        print(f"Run first: python grpo/merge_sft.py --config <your_config>")
        sys.exit(1)

    print(f"Loading vLLM model: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        n=num_samples,  # N번 샘플링을 한 번의 요청으로
    )

    # vLLM 입력 구성
    print(f"Building {len(samples)} prompts (n={num_samples} per image)...")
    prompts = []
    for sample in samples:
        img = Image.open(sample["image_path"]).convert("RGB")

        prompt = f"<|im_start|>system\n{sample['system_text']}<|im_end|>\n" if sample["system_text"] else ""
        prompt += (
            f"<|im_start|>user\n"
            f"<|vision_start|><|image_pad|><|vision_end|>"
            f"{sample['user_text']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompts.append({"prompt": prompt, "multi_modal_data": {"image": img}})

    # vLLM batch generate
    print(f"Generating {len(prompts)} x {num_samples} = {len(prompts) * num_samples} outputs...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    # chosen/rejected 분류
    pairs = []
    for i, (sample, output) in enumerate(zip(samples, outputs)):
        chosen_outputs = []
        rejected_outputs = []

        for completion in output.outputs:
            output_text = completion.text

            parsed = parse_model_output(output_text)
            if not parsed or "signal" not in parsed:
                rejected_outputs.append(output_text)
                continue

            if parsed["signal"] == sample["actual_signal"]:
                chosen_outputs.append(output_text)
            else:
                rejected_outputs.append(output_text)

        # chosen × rejected 조합으로 쌍 구성
        for chosen in chosen_outputs:
            for rejected in rejected_outputs:
                pairs.append({
                    "image_path": os.path.relpath(sample["image_path"], PROJECT_ROOT),
                    "prompt": [
                        {"role": "system", "content": sample["system_text"]},
                        {"role": "user", "content": sample["user_text"]},
                    ] if sample["system_text"] else [
                        {"role": "user", "content": sample["user_text"]},
                    ],
                    "chosen": chosen,
                    "rejected": rejected,
                    "actual_signal": sample["actual_signal"],
                })

        if (i + 1) % 100 == 0 or i == len(samples) - 1:
            print(f"  [{i+1}/{len(samples)}] chosen={len(chosen_outputs)} rejected={len(rejected_outputs)} total_pairs={len(pairs)}")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="DPO: Build chosen/rejected pairs (vLLM)")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--source-dataset", type=str, default=None,
                        help="Source dataset.jsonl path (default: data/teacher/dataset.jsonl)")
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    source_path = args.source_dataset or os.path.join(PROJECT_ROOT, "data", "teacher", "dataset.jsonl")
    output_path = os.path.join(PROJECT_ROOT, cfg.get("dataset_path", "data/dpo_pairs.jsonl"))

    print("=" * 60)
    print("DPO: Build Chosen/Rejected Pairs (vLLM)")
    print("=" * 60)

    samples = load_source_dataset(source_path, args.max_samples)
    pairs = generate_pairs(samples, cfg)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
