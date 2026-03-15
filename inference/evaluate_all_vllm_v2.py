"""
vLLM 배치 인퍼런스로 모든 체크포인트 평가 (v2).

generate_dpo_data_vllm.py 방식 적용:
  - processor.apply_chat_template()로 프롬프트 생성 (수동 ChatML 제거)
  - 청크 단위 처리 (메모리 효율)
  - ThreadPoolExecutor로 이미지 I/O 병렬화
  - max_pixels 설정으로 SFT/DPO 학습과 동일한 이미지 해상도
  - --check-image로 이미지 로딩 + 모델 인식 확인
  - --resume로 중단된 평가 이어서 처리

Usage:
  python inference/evaluate_all_vllm_v2.py --models-dir outputs/merged
  python inference/evaluate_all_vllm_v2.py --models-dir outputs/merged --max-eval 50
  python inference/evaluate_all_vllm_v2.py --models-dir outputs/merged --check-image
  python inference/evaluate_all_vllm_v2.py --models-dir outputs/merged --chunk-size 200
"""

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import os
import sys
import json
import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
from shared.dataset_utils import load_dataset_splits
from inference.predict import parse_output, SYSTEM_PROMPT, USER_PROMPT
from inference.metrics import compute_metrics, print_classification_report


def find_models(models_dir: str) -> list:
    """머지된 모델 폴더들을 찾아서 정렬 반환."""
    candidates = []
    for name in os.listdir(models_dir):
        path = os.path.join(models_dir, name)
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "config.json")):
            if "checkpoint" in name:
                step = int(name.split("-")[1]) if "-" in name else 0
            elif name == "final":
                step = 999999
            else:
                step = 0
            candidates.append((step, name, path))

    candidates.sort(key=lambda x: x[0])
    return candidates


def build_prompt(processor, image_path: str) -> str:
    """processor.apply_chat_template()으로 프롬프트 생성."""
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": SYSTEM_PROMPT},
        ]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def check_image(test_dataset, model_path: str, max_pixels: int = 4194304):
    """이미지 로딩 + 모델 인식 체크."""
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    print("=" * 60)
    print("Image Check Mode (vLLM v2)")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(processor, 'image_processor'):
        processor.image_processor.max_pixels = max_pixels
        processor.image_processor.min_pixels = min(256 * 28 * 28, max_pixels)

    n = min(3, len(test_dataset))
    for i in range(n):
        sample = test_dataset[i]
        img = sample["images"][0]  # PIL.Image (lazy loaded)
        metadata = sample.get("metadata", {})
        actual = metadata.get("actual_signal", "?")

        print(f"\n--- Sample {i+1}/{n} ---")
        print(f"Size: {img.size}, Mode: {img.mode}")
        print(f"Actual signal: {actual}")

    # Describe image 테스트
    print(f"\nLoading vLLM model: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": max_pixels},
    )
    sampling_params = SamplingParams(max_tokens=256, temperature=0)

    describe_messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in detail."},
        ]},
    ]
    describe_prompt = processor.apply_chat_template(
        describe_messages, tokenize=False, add_generation_prompt=True
    )

    requests = []
    images_for_check = []
    for i in range(n):
        img = test_dataset[i]["images"][0]  # PIL.Image
        images_for_check.append(img)
        requests.append({
            "prompt": describe_prompt,
            "multi_modal_data": {"image": img},
        })

    outputs = llm.generate(requests, sampling_params=sampling_params)

    # 첫 샘플 프롬프트 토큰 수 출력
    if outputs:
        prompt_tokens = len(outputs[0].prompt_token_ids)
        print(f"\nPrompt tokens: {prompt_tokens} (max_pixels={max_pixels})")

    for i in range(n):
        actual = test_dataset[i].get("metadata", {}).get("actual_signal", "?")
        response = outputs[i].outputs[0].text
        print(f"\n--- Sample {i+1} (actual={actual}) ---")
        print(f"Response: {response[:300]}{'...' if len(response) > 300 else ''}")

    del llm
    import torch
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("Image check passed!")
    print("=" * 60)


def evaluate_model_vllm(model_path: str, test_dataset, max_eval: int = None,
                        chunk_size: int = 500, max_pixels: int = 4194304) -> dict:
    """vLLM으로 한 모델 평가. 청크 단위 배치 처리."""
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    n = len(test_dataset)
    if max_eval:
        n = min(n, max_eval)

    # 데이터 준비
    samples = []
    for i in range(n):
        sample = test_dataset[i]
        metadata = sample.get("metadata", {})
        actual_signal = metadata.get("actual_signal")
        if actual_signal:
            samples.append((i, sample["images"][0], actual_signal, metadata))

    print(f"  Samples to evaluate: {len(samples)}")

    # Processor 로드
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(processor, 'image_processor'):
        processor.image_processor.max_pixels = max_pixels
        processor.image_processor.min_pixels = min(256 * 28 * 28, max_pixels)

    prompt_text = build_prompt(processor, "")

    # vLLM 로드
    print(f"  Loading vLLM model: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": max_pixels},
    )
    sampling_params = SamplingParams(max_tokens=512, temperature=0)

    # 첫 샘플로 이미지 인식 체크 (describe)
    first_img = samples[0][1] if isinstance(samples[0][1], Image.Image) else Image.open(samples[0][1]).convert("RGB")
    describe_messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in detail."},
        ]},
    ]
    describe_prompt = processor.apply_chat_template(
        describe_messages, tokenize=False, add_generation_prompt=True
    )
    describe_out = llm.generate(
        [{"prompt": describe_prompt, "multi_modal_data": {"image": first_img}}],
        SamplingParams(max_tokens=256, temperature=0),
    )
    describe_response = describe_out[0].outputs[0].text
    prompt_tokens = len(describe_out[0].prompt_token_ids)
    print(f"  [Image Check] Prompt tokens: {prompt_tokens} (max_pixels={max_pixels})")
    print(f"  [Image Check] Describe: {describe_response[:200]}{'...' if len(describe_response) > 200 else ''}")
    del first_img, describe_out

    # 결과 초기화
    total = 0
    correct = 0
    parse_fail = 0
    results = []
    confusion = {"LONG": {"LONG": 0, "SHORT": 0, "NEUTRAL": 0},
                 "SHORT": {"LONG": 0, "SHORT": 0, "NEUTRAL": 0},
                 "NEUTRAL": {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}}

    num_chunks = (len(samples) + chunk_size - 1) // chunk_size
    print(f"  Processing in {num_chunks} chunks (chunk_size={chunk_size})")
    total_gen_time = 0.0

    def _load_image(img):
        """이미지 로드 (thread-safe). PIL 객체면 그대로 반환."""
        try:
            if isinstance(img, Image.Image):
                return img, None
            return Image.open(img).convert("RGB"), None
        except Exception as e:
            return None, e

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(samples))
        chunk = samples[chunk_start:chunk_end]

        # 이미지 병렬 로드
        with ThreadPoolExecutor(max_workers=8) as executor:
            loaded = list(executor.map(lambda s: _load_image(s[1]), chunk))

        vllm_inputs = []
        chunk_valid = []
        for sample, (img, err) in zip(chunk, loaded):
            if err is not None:
                print(f"  [ERR] image load: {sample[1]}: {err}")
                parse_fail += 1
                total += 1
                continue
            vllm_inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {"image": img},
            })
            chunk_valid.append(sample)

        del loaded

        if not chunk_valid:
            continue

        # vLLM 배치 생성
        gen_start = time.time()
        vllm_outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        gen_time = time.time() - gen_start
        total_gen_time += gen_time

        # 첫 청크에서 프롬프트 토큰 수 출력
        if chunk_idx == 0 and vllm_outputs:
            prompt_tokens = len(vllm_outputs[0].prompt_token_ids)
            print(f"  [Check] Prompt tokens: {prompt_tokens} (max_pixels={max_pixels})")

        print(f"  [Chunk {chunk_idx+1}/{num_chunks}] "
              f"{len(chunk_valid)} samples, {gen_time:.1f}s "
              f"({len(chunk_valid)/max(gen_time, 0.1):.1f} samples/s)")

        # 결과 처리
        for (idx, image_path, actual_signal, metadata), vllm_out in zip(chunk_valid, vllm_outputs):
            raw_output = vllm_out.outputs[0].text
            result = parse_output(raw_output)
            predicted = result.get("signal", "?")
            total += 1

            is_correct = False
            if predicted not in ("LONG", "SHORT", "NEUTRAL"):
                parse_fail += 1
            else:
                is_correct = predicted == actual_signal
                if is_correct:
                    correct += 1
                confusion[actual_signal][predicted] += 1

            results.append({
                "index": idx,
                "actual_signal": actual_signal,
                "predicted_signal": predicted,
                "correct": is_correct,
                "response": result,
            })

        del vllm_inputs, vllm_outputs

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"  Total generation time: {total_gen_time:.1f}s")

    del llm
    import torch
    torch.cuda.empty_cache()

    return {
        "total": total,
        "correct": correct,
        "parse_fail": parse_fail,
        "accuracy": accuracy,
        "confusion": confusion,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models with vLLM (v2)")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory with merged models")
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--max-pixels", type=int, default=4194304)
    parser.add_argument("--save-dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "eval_results"))
    parser.add_argument("--check-image", action="store_true",
                        help="평가 없이 이미지 로딩 + 모델 인식 체크만 수행")
    args = parser.parse_args()

    models_dir = os.path.join(PROJECT_ROOT, args.models_dir) if not os.path.isabs(args.models_dir) else args.models_dir
    models = find_models(models_dir)

    if not models:
        print(f"No merged models found in {models_dir}")
        return

    print("=" * 60)
    print(f"Evaluate All Models (vLLM v2)")
    print(f"Dir: {models_dir}")
    print(f"Found: {[name for _, name, _ in models]}")
    print(f"Max eval: {args.max_eval or 'all'}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Max pixels: {args.max_pixels}")
    print("=" * 60)

    _, _, test_dataset = load_dataset_splits()

    if args.check_image:
        check_image(test_dataset, models[0][2], args.max_pixels)
        return

    all_results = []
    for _, name, path in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        metrics = evaluate_model_vllm(
            path, test_dataset, args.max_eval,
            args.chunk_size, args.max_pixels,
        )
        metrics["name"] = name
        metrics["path"] = path
        all_results.append(metrics)

        print(f"\n  Accuracy: {metrics['accuracy']:.1f}% "
              f"({metrics['correct']}/{metrics['total']}) | "
              f"Parse fail: {metrics['parse_fail']}")
        metrics["class_report"] = print_classification_report(metrics["confusion"], name)

    # 비교 테이블
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'ParseFail':>10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['name']:<20} {r['accuracy']:>9.1f}% {r['correct']:>10} {r['total']:>8} {r['parse_fail']:>10}")

    best = max(all_results, key=lambda x: x["accuracy"])
    print(f"\nBest: {best['name']} ({best['accuracy']:.1f}%)")

    # 결과 저장
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, f"eval_vllm_{timestamp}.json")

    save_data = {
        "timestamp": timestamp,
        "models_dir": models_dir,
        "max_eval": args.max_eval,
        "max_pixels": args.max_pixels,
        "chunk_size": args.chunk_size,
        "summary": [{
            "name": r["name"],
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "total": r["total"],
            "parse_fail": r["parse_fail"],
            "confusion": r["confusion"],
        } for r in all_results],
        "best": best["name"],
        "details": all_results,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved -> {save_path}")


if __name__ == "__main__":
    main()
