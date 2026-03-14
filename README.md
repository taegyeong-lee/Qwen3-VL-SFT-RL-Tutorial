# Qwen3-VL 4B SFT + DPO Tutorial

Qwen3-VL-4B 비전 언어 모델을 SFT → DPO 순서로 파인튜닝하여 BTC 15분봉 차트 이미지를 분석하고 매매 신호(LONG/SHORT/NEUTRAL)를 예측하는 튜토리얼.

## Pipeline Overview

```
btc_trading_view.csv
    │
    ▼
[0] 차트 이미지 생성 ──→ data/chart_images/*.png + window_meta.json
    │
    ▼
[1~2] GPT-4.1-mini Batch API ──→ batch_results.jsonl
    │
    ▼
[3] 데이터셋 빌드 ──→ data/teacher/dataset.jsonl
    │
    ├──→ [SFT]  sft/train.py ──→ outputs/sft_lora/final/
    │      │
    │      └──→ [DPO]  grpo/merge_sft.py (LoRA 머지)
    │                  dpo/build_pairs.py (vLLM 샘플링 → chosen/rejected)
    │                  dpo/train.py ──→ outputs/dpo_lora/final/
    │
    ▼
[Eval] inference/evaluate.py ──→ Accuracy, Confusion Matrix
[Cmp]  inference/compare.py  ──→ SFT vs DPO 비교
```

## Project Structure

```
btc_training/
├── data_prep/                    # 데이터 준비 파이프라인
│   ├── generate_charts.py        # CSV → 4패널 차트 PNG + 메타데이터
│   ├── prepare_batch.py          # 이미지 base64 → OpenAI Batch JSONL
│   ├── submit_batch.py           # Batch API 제출 + 폴링 + 다운로드
│   └── build_dataset.py          # GPT 응답 → 파인튜닝 JSONL 데이터셋
│
├── sft/                          # Supervised Fine-Tuning
│   ├── train.py                  # TRL SFTTrainer + PEFT LoRA
│   └── configs/
│       ├── single.yaml           # Single GPU (bf16 LoRA)
│       └── multi.yaml            # Multi GPU (bf16 LoRA)
│
├── dpo/                          # Direct Preference Optimization
│   ├── build_pairs.py            # vLLM 배치 샘플링 → chosen/rejected 쌍
│   ├── train.py                  # TRL DPOTrainer
│   └── configs/
│       ├── single.yaml
│       └── multi.yaml
│
├── grpo/                         # Group Relative Policy Optimization
│   ├── train.py                  # TRL GRPOTrainer
│   ├── rewards.py                # 보상 함수 (format + signal)
│   ├── merge_sft.py              # SFT LoRA → base 모델 머지
│   └── configs/
│       ├── single.yaml
│       └── multi.yaml
│
├── inference/                    # 추론 & 평가
│   ├── predict.py                # 단일 이미지 예측
│   ├── evaluate.py               # 테스트셋 평가
│   └── compare.py                # SFT vs DPO 비교
│
├── shared/                       # 공통 모듈
│   ├── dataset_utils.py          # 데이터셋 로더 (VLMDataset, split)
│   ├── analyze_dataset.py        # 데이터셋 통계 분석
│   └── prompts.yaml              # 프롬프트 설정 (v3: hindsight mode)
│
├── data/
│   ├── btc_trading_view.csv      # 원본 15분봉 (OHLCV + OI + Funding Rate)
│   ├── teacher/                  # Teacher 데이터
│   │   ├── dataset.jsonl         # GPT-4.1-mini 생성 데이터셋
│   │   ├── window_meta.json      # 윈도우 메타데이터
│   │   ├── batch_parts/          # Batch API 요청 파일
│   │   └── batch_results.jsonl   # GPT 배치 결과
│   ├── dpo_pairs.jsonl           # DPO chosen/rejected 쌍
│   └── chart_images/             # 차트 PNG
│
├── outputs/                      # 학습 결과 (.gitignore)
│   ├── sft_lora/final/           # SFT LoRA adapter
│   ├── sft_merged/               # SFT LoRA 머지 모델 (DPO 쌍 생성용)
│   └── dpo_lora/final/           # DPO LoRA adapter
│
├── .env                          # OpenAI API key
└── requirements.txt              # 의존성
```

---

## Setup

### 1. 환경 생성

```bash
conda create -n btc python=3.11 -y
conda activate btc
```

### 2. PyTorch 설치 (CUDA 버전에 맞게)

```bash
# CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. GPU 확인

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 5. .env (데이터 생성 시 필요)

```
OPENAI_API_KEY=sk-...
```

---

## Step 0~3: 데이터 준비

```bash
# 0) 차트 이미지 생성 (균형 샘플링)
python data_prep/generate_charts.py --balanced 6000

# 1) Batch API 요청 생성
python data_prep/prepare_batch.py

# 2) Batch API 제출 + 결과 수집
python data_prep/submit_batch.py

# 3) 데이터셋 빌드
python data_prep/build_dataset.py
```

### 데이터셋 통계 확인

```bash
python shared/analyze_dataset.py
python shared/analyze_dataset.py --path data/teacher/dataset.jsonl
```

---

## SFT Training

```bash
# Single GPU
python sft/train.py --config sft/configs/single.yaml

# Multi GPU
accelerate launch --num_processes 2 sft/train.py --config sft/configs/multi.yaml

# 소량 테스트
python sft/train.py --config sft/configs/single.yaml --max-samples 50
```

### SFT 하이퍼파라미터 (yaml로 조정)

| | Single GPU | Multi GPU |
|---|----------|-------------|
| Model | Qwen3-VL-4B-Instruct | Qwen3-VL-4B-Instruct |
| LoRA rank / alpha | 16 / 32 | 16 / 32 |
| Batch x Grad Accum | 1 x 16 = **16** | 4 x 4 = **16** (x2 GPU = **32**) |
| Learning Rate | 1e-4 | 5e-5 |
| Epochs | 2 | 2 |

---

## DPO Training

```bash
# 1) SFT LoRA → base 모델 머지 (vLLM용)
python grpo/merge_sft.py --config dpo/configs/single.yaml

# 2) vLLM으로 chosen/rejected 쌍 생성
python dpo/build_pairs.py --config dpo/configs/single.yaml

# 3) DPO 학습
python dpo/train.py --config dpo/configs/single.yaml

# Multi GPU
accelerate launch --num_processes 2 dpo/train.py --config dpo/configs/multi.yaml
```

### DPO 하이퍼파라미터 (yaml로 조정)

| 파라미터 | 설명 |
|---------|------|
| `beta` | KL penalty coefficient (기본 0.1) |
| `loss_type` | sigmoid / hinge / ipo |
| `pair_generation.num_samples_per_image` | vLLM에서 이미지당 N번 샘플링 |
| `pair_generation.temperature` | 샘플링 다양성 (기본 1.0) |

---

## Inference & Evaluation

```bash
# 단일 이미지 예측
python inference/predict.py --image data/chart_images/sample.png
python inference/predict.py --adapter outputs/dpo_lora/final --image data/chart_images/sample.png

# 테스트셋 평가
python inference/evaluate.py --adapter outputs/sft_lora/final
python inference/evaluate.py --adapter outputs/dpo_lora/final

# SFT vs DPO 비교
python inference/compare.py
python inference/compare.py --max-eval 50
```

---

## Monitoring

```bash
# SFT
tensorboard --logdir outputs/sft_lora --port 6006

# DPO
tensorboard --logdir outputs/dpo_lora --port 6007
```

---

## Tech Stack

| 구분 | SFT | DPO |
|------|-----|-----|
| 모델 | Qwen3-VL-4B-Instruct | Qwen3-VL-4B-Instruct |
| 프레임워크 | TRL + PEFT | TRL + PEFT |
| Trainer | SFTTrainer | DPOTrainer |
| 데이터 | teacher/dataset.jsonl | dpo_pairs.jsonl |
| 쌍 생성 | - | vLLM batch sampling |
| 파인튜닝 | LoRA (bf16) | SFT 위에 LoRA (bf16) |
| Loss | Cross-entropy | DPO sigmoid |
