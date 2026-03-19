"""
DPO: Build chosen/rejected pairs via vLLM batch sampling + BGE-M3 scoring.

SFT LoRA를 base에 머지한 모델을 vLLM으로 로드하고,
같은 차트에 대해 N번 생성 → 복합 스코어링으로
chosen (best) / rejected (worst) 쌍을 구성.

스코어링 기준:
  - signal_match: actual_signal 일치 여부 (가중치 10)
  - reasoning_sim: teacher reasoning과 BGE-M3 cosine similarity (가중치 5)
  - confidence_cal: confidence calibration (가중치 1)

사전 준비:
  1) SFT LoRA → base 모델 머지
     python inference/merge_lora.py --adapter outputs/sft_lora/checkpoint-200 --output outputs/sft_merged

Usage:
  python dpo/build_pairs.py --config dpo/configs/single.yaml
  python dpo/build_pairs.py --config dpo/configs/single.yaml --max-samples 100
  python dpo/build_pairs.py --config dpo/configs/single.yaml --no-embedding  # embedding 없이
"""

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import sys
import json
import argparse
from collections import defaultdict
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from PIL import Image
from transformers import AutoProcessor
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


def load_embedding_model(model_name: str = "BAAI/bge-m3"):
    """BGE-M3 임베딩 모델 로드."""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name} (CPU)")
        model = SentenceTransformer(model_name, device="cpu")
        return model
    except ImportError:
        print("WARNING: sentence-transformers not installed. pip install sentence-transformers")
        return None


def get_reasoning_text(parsed: dict) -> str:
    """reasoning dict를 필드별 텍스트로 변환."""
    reasoning = parsed.get("reasoning", {})
    if isinstance(reasoning, dict):
        parts = []
        for field in ["market_context", "price_action", "volume_oi", "risk_assessment"]:
            if field in reasoning:
                parts.append(f"{field}: {reasoning[field]}")
        return " ".join(parts)
    elif isinstance(reasoning, str):
        return reasoning
    return ""


def compute_reasoning_similarity(embed_model, teacher_text: str, student_text: str) -> float:
    """BGE-M3로 teacher/student reasoning cosine similarity 계산."""
    if not embed_model or not teacher_text or not student_text:
        return 0.0
    embeddings = embed_model.encode([teacher_text, student_text], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def score_output(parsed: dict, actual_signal: str, teacher_reasoning: str,
                 embed_model=None) -> float:
    """복합 스코어링.

    - signal_match: 10점
    - reasoning_sim: 0~5점 (BGE-M3 cosine sim * 5)
    - confidence_cal: 0~1점
    """
    score = 0.0

    # 1) Signal match (가중치 10)
    signal_match = parsed.get("signal") == actual_signal
    if signal_match:
        score += 10.0

    # 2) Reasoning similarity (가중치 5)
    student_reasoning = get_reasoning_text(parsed)
    if embed_model and teacher_reasoning and student_reasoning:
        sim = compute_reasoning_similarity(embed_model, teacher_reasoning, student_reasoning)
        score += sim * 5.0

    # 3) Confidence calibration (가중치 1)
    confidence = parsed.get("confidence", 50)
    if isinstance(confidence, (int, float)):
        conf_norm = confidence / 100.0
        if signal_match:
            score += conf_norm  # 맞았을 때 confidence 높으면 +
        else:
            score += (1.0 - conf_norm)  # 틀렸을 때 confidence 낮으면 +

    return score


def load_source_dataset(jsonl_path: str, max_samples: int = None) -> list[dict]:
    """actual_signal + teacher reasoning이 있는 dataset entries 로드."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            metadata = entry.get("metadata", {})
            if not metadata.get("actual_signal"):
                continue
            if "messages" not in entry or "images" not in entry:
                continue

            image_path = os.path.join(PROJECT_ROOT, entry["images"][0])
            if not os.path.exists(image_path):
                continue

            # messages에서 system/user/assistant 텍스트 추출
            system_text = ""
            user_text = ""
            teacher_response = ""
            for msg in entry["messages"]:
                if msg["role"] == "system":
                    content = msg["content"]
                    system_text = content[0]["text"] if isinstance(content, list) else content
                elif msg["role"] == "user":
                    if isinstance(msg["content"], list):
                        for item in msg["content"]:
                            if item.get("type") == "text":
                                user_text = item["text"]
                    else:
                        user_text = msg["content"]
                elif msg["role"] == "assistant":
                    content = msg["content"]
                    teacher_response = content[0]["text"] if isinstance(content, list) else content

            if not user_text:
                continue

            # Teacher reasoning 추출
            teacher_reasoning = ""
            if teacher_response:
                try:
                    teacher_parsed = json.loads(teacher_response)
                    teacher_reasoning = get_reasoning_text(teacher_parsed)
                except json.JSONDecodeError:
                    pass

            samples.append({
                "image_path": image_path,
                "system_text": system_text,
                "user_text": user_text,
                "actual_signal": metadata["actual_signal"],
                "teacher_reasoning": teacher_reasoning,
            })

    if max_samples:
        samples = samples[:max_samples]

    print(f"Source dataset: {len(samples)} samples with actual_signal")
    return samples


def generate_pairs(samples: list[dict], cfg: dict, use_embedding: bool = True,
                    output_path: str = None) -> list[dict]:
    """vLLM으로 SFT 모델 N번 샘플링 → 복합 스코어링 → chosen/rejected 쌍 생성."""
    model_path = os.path.join(PROJECT_ROOT, cfg.get("sft_merged_path", "outputs/sft_merged"))
    pair_cfg = cfg.get("pair_generation", {})
    num_samples = pair_cfg.get("num_samples_per_image", 8)
    temperature = pair_cfg.get("temperature", 1.0)
    max_new_tokens = pair_cfg.get("max_new_tokens", 768)

    if not os.path.exists(model_path):
        print(f"ERROR: Merged model not found: {model_path}")
        print(f"Run first: python inference/merge_lora.py --adapter <adapter_path> --output {model_path}")
        sys.exit(1)

    # BGE-M3 로드
    embed_model = None
    if use_embedding:
        embed_model = load_embedding_model()
        if not embed_model:
            print("Falling back to signal-only scoring")

    # Processor 로드 (apply_chat_template용)
    base_model = cfg.get("model", "Qwen/Qwen3-VL-4B-Instruct")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    # vLLM 로드
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
        n=num_samples,
    )

    # 청크 단위 처리 + 중간 저장
    chunk_size = pair_cfg.get("chunk_size", 500)
    pairs = []
    stats = {"total": 0, "both_correct": 0, "both_wrong": 0, "mixed": 0, "parse_fail": 0}

    for chunk_start in range(0, len(samples), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(samples))
        chunk_samples = samples[chunk_start:chunk_end]
        print(f"\n--- Chunk {chunk_start//chunk_size + 1} [{chunk_start+1}~{chunk_end}/{len(samples)}] ---")

        # vLLM 입력 구성 (processor.apply_chat_template 방식)
        prompts = []
        for sample in chunk_samples:
            img = Image.open(sample["image_path"]).convert("RGB")

            messages = []
            if sample["system_text"]:
                messages.append({"role": "system", "content": [
                    {"type": "text", "text": sample["system_text"]},
                ]})
            messages.append({"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": sample["user_text"]},
            ]})

            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append({"prompt": prompt_text, "multi_modal_data": {"image": img}})

        # vLLM batch generate
        print(f"  Generating {len(prompts)} x {num_samples} = {len(prompts) * num_samples} outputs...")
        outputs = llm.generate(prompts, sampling_params=sampling_params)

        # 파싱 + 임베딩 텍스트 수집
        parsed_results = []  # [(sample_idx, output_text, parsed, student_reasoning)]
        for i, (sample, output) in enumerate(zip(chunk_samples, outputs)):
            for completion in output.outputs:
                output_text = completion.text
                parsed = parse_model_output(output_text)
                if not parsed or "signal" not in parsed:
                    stats["parse_fail"] += 1
                    continue
                student_reasoning = get_reasoning_text(parsed)
                parsed_results.append((i, output_text, parsed, student_reasoning))

        # 배치 임베딩 (한 번에 encode)
        sim_scores = {}
        if embed_model and parsed_results:
            teacher_texts = []
            student_texts = []
            embed_indices = []
            for idx, (sample_idx, _, _, student_reasoning) in enumerate(parsed_results):
                teacher_reasoning = chunk_samples[sample_idx]["teacher_reasoning"]
                if teacher_reasoning and student_reasoning:
                    teacher_texts.append(teacher_reasoning)
                    student_texts.append(student_reasoning)
                    embed_indices.append(idx)

            if teacher_texts:
                print(f"  Computing {len(teacher_texts)} embeddings in batch...")
                all_texts = teacher_texts + student_texts
                all_embeddings = embed_model.encode(all_texts, normalize_embeddings=True,
                                                     batch_size=256, show_progress_bar=False)
                n = len(teacher_texts)
                for j, idx in enumerate(embed_indices):
                    sim = float(np.dot(all_embeddings[j], all_embeddings[n + j]))
                    sim_scores[idx] = sim

        # 결과를 sample별로 그룹핑
        sample_outputs = defaultdict(list)
        for idx, (sample_idx, output_text, parsed, _) in enumerate(parsed_results):
            sample = chunk_samples[sample_idx]
            signal_match = parsed.get("signal") == sample["actual_signal"]

            # 스코어 계산 (임베딩은 미리 계산된 값 사용)
            score = 0.0
            if signal_match:
                score += 10.0
            if idx in sim_scores:
                score += sim_scores[idx] * 5.0
            confidence = parsed.get("confidence", 50)
            if isinstance(confidence, (int, float)):
                conf_norm = confidence / 100.0
                score += conf_norm if signal_match else (1.0 - conf_norm)

            sample_outputs[sample_idx].append({
                "text": output_text,
                "score": score,
                "signal_match": signal_match,
            })

        # chosen/rejected 쌍 구성
        for i, sample in enumerate(chunk_samples):
            scored_outputs = sample_outputs.get(i, [])

            if len(scored_outputs) < 2:
                continue

            stats["total"] += 1

            # best와 worst 선택
            scored_outputs.sort(key=lambda x: x["score"], reverse=True)
            best = scored_outputs[0]
            worst = scored_outputs[-1]

            all_correct = all(o["signal_match"] for o in scored_outputs)
            all_wrong = not any(o["signal_match"] for o in scored_outputs)

            if all_correct:
                stats["both_correct"] += 1
                if embed_model and (best["score"] - worst["score"]) > 1.0:
                    pass  # 아래에서 쌍 생성
                else:
                    continue
            elif all_wrong:
                stats["both_wrong"] += 1
                continue
            else:
                stats["mixed"] += 1

            pairs.append({
                "prompt": [
                    {"role": "system", "content": sample["system_text"]},
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": sample["user_text"]},
                    ]},
                ] if sample["system_text"] else [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": sample["user_text"]},
                    ]},
                ],
                "images": [os.path.relpath(sample["image_path"], PROJECT_ROOT)],
                "chosen": [{"role": "assistant", "content": best["text"]}],
                "rejected": [{"role": "assistant", "content": worst["text"]}],
                "actual_signal": sample["actual_signal"],
                "best_score": round(best["score"], 3),
                "worst_score": round(worst["score"], 3),
                "all_outputs": [
                    {
                        "text": o["text"],
                        "score": round(o["score"], 3),
                        "signal_match": o["signal_match"],
                    } for o in scored_outputs
                ],
            })

        print(f"  Chunk done. pairs={len(pairs)} "
              f"(mixed={stats['mixed']} both_correct={stats['both_correct']} "
              f"both_wrong={stats['both_wrong']})")

        # 중간 저장
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            print(f"  Saved {len(pairs)} pairs to {output_path}")

    print(f"\n--- Pair Generation Stats ---")
    print(f"Total processed:  {stats['total']}")
    print(f"Mixed (usable):   {stats['mixed']}")
    print(f"Both correct:     {stats['both_correct']} (used if score gap > 1.0)")
    print(f"Both wrong:       {stats['both_wrong']} (skipped)")
    print(f"Parse failures:   {stats['parse_fail']}")
    print(f"Final pairs:      {len(pairs)}")

    return pairs


def check_image(source_dataset: list[dict], cfg: dict, num_samples: int = 3):
    """vLLM으로 이미지가 제대로 인식되는지 'Describe this image in detail' 테스트."""
    import random

    model_path = os.path.join(PROJECT_ROOT, cfg.get("sft_merged_path", "outputs/sft_merged"))
    base_model = cfg.get("model", "Qwen/Qwen3-VL-4B-Instruct")

    if not os.path.exists(model_path):
        print(f"ERROR: Merged model not found: {model_path}")
        sys.exit(1)

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    print(f"Loading vLLM model for image check: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(max_tokens=256, temperature=0.0)

    test_samples = random.sample(source_dataset, min(num_samples, len(source_dataset)))

    for i, sample in enumerate(test_samples):
        img = Image.open(sample["image_path"]).convert("RGB")
        print(f"\n--- Sample {i+1}/{len(test_samples)} ---")
        print(f"Image: {os.path.basename(sample['image_path'])}")
        print(f"Size: {img.size}, Mode: {img.mode}")
        print(f"Actual signal: {sample['actual_signal']}")

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in detail."},
        ]}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        output = llm.generate(
            [{"prompt": prompt_text, "multi_modal_data": {"image": img}}],
            sampling_params=sampling_params,
        )
        response = output[0].outputs[0].text
        print(f"Model response:\n{response[:500]}")

    del llm
    print("\nImage check complete.")


def main():
    parser = argparse.ArgumentParser(description="DPO: Build chosen/rejected pairs (vLLM + BGE-M3)")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-embedding", action="store_true",
                        help="Skip BGE-M3 embedding, use signal-only scoring")
    parser.add_argument("--check-image", action="store_true",
                        help="Test image loading with 'Describe this image' before building pairs")
    parser.add_argument("--model", type=str, default=None,
                        help="Merged SFT model path (overrides config sft_merged_path)")
    parser.add_argument("--source-dataset", type=str, default=None,
                        help="Source dataset.jsonl path (default: data/teacher/dataset.jsonl)")
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["sft_merged_path"] = args.model

    source_path = args.source_dataset or os.path.join(PROJECT_ROOT, "data", "teacher", "dataset.jsonl")
    output_path = os.path.join(PROJECT_ROOT, cfg.get("dataset_path", "data/dpo_pairs.jsonl"))

    print("=" * 60)
    print("DPO: Build Chosen/Rejected Pairs (vLLM + BGE-M3)")
    print(f"  Embedding: {'OFF' if args.no_embedding else 'ON (BGE-M3)'}")
    print("=" * 60)

    samples = load_source_dataset(source_path, args.max_samples)

    if args.check_image:
        check_image(samples, cfg)
        return

    pairs = generate_pairs(samples, cfg, use_embedding=not args.no_embedding, output_path=output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Output: {output_path}")

    # Score distribution
    if pairs:
        scores = [(p["best_score"], p["worst_score"]) for p in pairs]
        avg_best = sum(s[0] for s in scores) / len(scores)
        avg_worst = sum(s[1] for s in scores) / len(scores)
        avg_gap = sum(s[0] - s[1] for s in scores) / len(scores)
        print(f"Avg best score:  {avg_best:.2f}")
        print(f"Avg worst score: {avg_worst:.2f}")
        print(f"Avg score gap:   {avg_gap:.2f}")


if __name__ == "__main__":
    main()
