"""
Single image prediction with fine-tuned Qwen VL.

Usage:
  # Unsloth (RTX 3060)
  python inference/predict.py --image chart_images/sample.png
  python inference/predict.py --adapter outputs/grpo_lora/final --image chart_images/sample.png

  # vLLM (A100)
  python inference/predict.py --backend vllm --model outputs/sft_merged --image chart_images/sample.png
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image

SYSTEM_PROMPT = (
    "You are a professional Bitcoin futures trader. "
    "Analyze 15-minute candlestick charts to predict the direction over the next 4 hours."
)
USER_PROMPT = (
    "BTCUSDT 15m chart. Predict the direction for the next 4 hours (16 candles).\n"
    "Respond in JSON."
)


def load_model_unsloth(base_model: str, adapter_path: str, load_in_4bit: bool = True):
    from unsloth import FastVisionModel

    print(f"[Unsloth] Loading model: {base_model}")
    model, tokenizer = FastVisionModel.from_pretrained(
        base_model, load_in_4bit=load_in_4bit,
    )
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter: {adapter_path}")
        model.load_adapter(adapter_path)
        if not load_in_4bit:
            model = model.to("cuda")
    FastVisionModel.for_inference(model)
    return model, tokenizer


def predict_unsloth(model, tokenizer, image) -> str:
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(images=[image], text=text, return_tensors="pt")
    inputs = inputs.to(model.device)

    import torch
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def load_model_vllm(model_path: str):
    from vllm import LLM, SamplingParams

    print(f"[vLLM] Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(max_tokens=512, temperature=0)
    return llm, sampling_params


def predict_vllm(llm, sampling_params, image) -> str:
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{USER_PROMPT}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    outputs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        sampling_params=sampling_params,
    )
    return outputs[0].outputs[0].text


def parse_output(output_text: str) -> dict:
    text = output_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": output_text}


def main():
    parser = argparse.ArgumentParser(description="Single image prediction")
    parser.add_argument("--backend", type=str, default="unsloth", choices=["unsloth", "vllm"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--no-4bit", dest="load_4bit", action="store_false", default=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    # Default model/adapter
    if args.model is None:
        if args.backend == "vllm":
            args.model = os.path.join(PROJECT_ROOT, "outputs", "sft_merged")
        else:
            args.model = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
    if args.adapter is None:
        args.adapter = os.path.join(PROJECT_ROOT, "outputs", "sft_lora", "final")

    # Load model
    if args.backend == "vllm":
        llm, sampling_params = load_model_vllm(args.model)
        predict_fn = lambda img: predict_vllm(llm, sampling_params, img)
    else:
        adapter = args.adapter if args.adapter and os.path.exists(args.adapter) else None
        model, tokenizer = load_model_unsloth(args.model, adapter, args.load_4bit)
        predict_fn = lambda img: predict_unsloth(model, tokenizer, img)

    image_path = os.path.join(PROJECT_ROOT, args.image) if not os.path.isabs(args.image) else args.image
    result = parse_output(predict_fn(image_path))

    print("\n" + "=" * 60)
    print("Prediction Result")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    signal = result.get("signal", "?")
    conf = result.get("confidence", "?")
    risk = result.get("risk_level", "?")
    print(f"\nSignal: {signal} | Confidence: {conf} | Risk: {risk}")


if __name__ == "__main__":
    main()
