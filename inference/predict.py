"""
Single image prediction with fine-tuned Qwen VL (transformers + PEFT).

Usage:
  python inference/predict.py --image data/chart_images/sample.png
  python inference/predict.py --adapter outputs/sft_lora/final --image data/chart_images/sample.png
  python inference/predict.py --adapter outputs/dpo_lora/final --image data/chart_images/sample.png
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are a professional Bitcoin futures trader. "
    "Analyze 15-minute candlestick charts to predict the direction over the next 4 hours."
)
USER_PROMPT = (
    "BTCUSDT 15m chart. Predict the direction for the next 4 hours (16 candles).\n"
    "Respond in JSON."
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def load_model(base_model: str, adapter_path: str = None):
    """Load base model + optional LoRA adapter."""
    print(f"Loading model: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, processor


def predict(model, processor, image) -> str:
    """Run inference on a single image."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    trimmed = generated_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True)


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
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    if args.adapter is None:
        args.adapter = os.path.join(PROJECT_ROOT, "outputs", "sft_lora", "final")

    model, processor = load_model(args.model, args.adapter)

    image_path = os.path.join(PROJECT_ROOT, args.image) if not os.path.isabs(args.image) else args.image
    result = parse_output(predict(model, processor, image_path))

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
