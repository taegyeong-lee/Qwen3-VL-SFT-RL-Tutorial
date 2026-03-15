# Qwen3-VL 4B SFT + DPO Tutorial

Qwen3-VL-4B 비전 언어 모델을 SFT → DPO 순서로 파인튜닝하여 BTC 15분봉 차트 이미지를 분석하고 매매 신호(LONG/SHORT/NEUTRAL)를 예측하는 튜토리얼.

- 현재 데이터셋 구축까지 업데이트 완료 [2025.03.14]


## Setup

### 1. 환경 생성

```bash
conda create -n btc python=3.11 -y
conda activate btc
```

### 2. PyTorch 설치 (CUDA 버전에 맞게)

```bash
# CUDA 12.x (FSDP2 requires PyTorch >= 2.6.0)
pip install torch>=2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8
pip install torch>=2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. Locale 설정 (Linux 서버)

학습 중 체크포인트 저장 시 `UnicodeDecodeError`가 발생할 수 있다. `~/.bashrc`에 추가:

```bash
export LANG=en_US.UTF-8
export PYTHONIOENCODING=utf-8
```

### 5. GPU 확인

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 6. .env (데이터 생성 시 필요)

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
accelerate launch --config_file sft/configs/fsdp2.yaml sft/train.py --config sft/configs/multi.yaml

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

# 원격 서버에서 실행 시 (외부 접근 허용)
tensorboard --logdir outputs/sft_lora --port 6006 --bind_all
# 브라우저에서 http://<서버IP>:6006 접속
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
