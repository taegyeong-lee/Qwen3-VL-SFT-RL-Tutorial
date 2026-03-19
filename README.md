# Qwen3-VL 4B SFT + DPO Tutorial

Qwen3-VL-4B 비전 언어 모델을 SFT → DPO 순서로 파인튜닝하여 BTC 15분봉 차트 이미지를 분석하고 매매 신호(LONG/SHORT/NEUTRAL)를 예측하는 튜토리얼.

- [Qwen3-VL 4B 비트코인 차트 해석 모델 만들기 — 1편: 데이터셋 구축 (LoRA SFT/Distillation, DPO)](https://velog.io/@seawhale/Qwen3-VL-4B%EB%A1%9C-%EB%B9%84%ED%8A%B8%EC%BD%94%EC%9D%B8-%EC%B0%A8%ED%8A%B8-%ED%95%B4%EC%84%9D-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0-LoRA-SFT-Distillation) [2025.03.14]

- [Qwen3-VL 4B 비트코인 차트 해석 모델 만들기 — 2편: 허깅페이스 TRL로 LoRA 파인튜닝](https://velog.io/@seawhale/Qwen3-VL-4B-%EB%B9%84%ED%8A%B8%EC%BD%94%EC%9D%B8-%EC%B0%A8%ED%8A%B8-%ED%95%B4%EC%84%9D-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0-2%ED%8E%B8-%ED%97%88%EA%B9%85%ED%8E%98%EC%9D%B4%EC%8A%A4-TRL%EB%A1%9C-LoRA-%ED%8C%8C%EC%9D%B8%ED%8A%9C%EB%8B%9D)


## Setup

### 1. 환경 생성

```bash
# conda
conda create -n btc python=3.11 -y
conda activate btc

# Docker 등 conda가 없는 환경
python3.11 -m venv ~/btc_env
source ~/btc_env/bin/activate
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

### 6. SFT 체크포인트 다운로드 (LoRA 머지됨)

학습 없이 바로 추론/DPO를 진행하려면 SFT 머지 모델을 다운로드:

- [Google Drive: sft_merged/checkpoint-200](https://drive.google.com/file/d/1GlPNi0Ukr6GoxOoMDcon7qDPbsK7mLeX/view?usp=drive_link)

```bash
pip install gdown
gdown 1GlPNi0Ukr6GoxOoMDcon7qDPbsK7mLeX -O sft_merged_checkpoint-200.zip
unzip sft_merged_checkpoint-200.zip -d outputs/sft_merged/
# outputs/sft_merged/checkpoint-200/ 에 config.json, model*.safetensors 등이 위치해야 함
```

### 7. .env (데이터 생성 시 필요)

```
OPENAI_API_KEY=sk-...
```

---

## Chart Examples

| LONG | SHORT | NEUTRAL |
|------|-------|---------|
| ![LONG](assets/example_long.png) | ![SHORT](assets/example_short.png) | ![NEUTRAL](assets/example_neutral.png) |

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

### SFT Evaluation Results (checkpoint-200, Best)

테스트셋 600개 (LONG 214 / SHORT 259 / NEUTRAL 127)에 대한 vLLM 평가 결과.

**Accuracy: 42.7% | Macro F1: 42.4%**

#### Accuracy by Checkpoint

![Accuracy Curve](assets/accuracy_curve.png)

#### Confusion Matrix (checkpoint-200)

![Confusion Matrix](assets/confusion_matrix.png)

| | Pred LONG | Pred SHORT | Pred NEUTRAL | Recall |
|---|---|---|---|---|
| **LONG** | 71 | 60 | 83 | 33.2% |
| **SHORT** | 55 | 102 | 102 | 39.4% |
| **NEUTRAL** | 16 | 28 | 83 | 65.4% |

#### Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| LONG | 50.0% | 33.2% | 39.9% | 214 |
| SHORT | 53.7% | 39.4% | 45.4% | 259 |
| NEUTRAL | 31.0% | 65.4% | 42.0% | 127 |
| **Macro Avg** | 44.9% | 46.0% | 42.4% | |

#### F1 Score by Class & Checkpoint

![F1 Comparison](assets/f1_comparison.png)

| Checkpoint | Accuracy | LONG F1 | SHORT F1 | NEUTRAL F1 | Macro F1 |
|---|---|---|---|---|---|
| **checkpoint-200** | **42.7%** | **39.9%** | 45.4% | **42.0%** | **42.4%** |
| checkpoint-300 | 39.8% | 15.2% | **51.6%** | 39.6% | 35.5% |
| final | 41.7% | 19.3% | 52.6% | 41.6% | 37.8% |

> checkpoint-200 이후 LONG recall이 급락하며 SHORT 편향이 심화됨. epoch 1.33 시점이 최적.

---

## DPO Training

### 개요

DPO(Direct Preference Optimization)는 SFT 모델의 출력 중 **좋은 응답(chosen)과 나쁜 응답(rejected)** 쌍으로 선호도 학습을 수행한다.

```
SFT 모델 → 같은 차트에 N번 생성 → 복합 스코어링 → best(chosen) / worst(rejected) 쌍 구성 → DPO 학습
```

### Step 1: SFT LoRA 머지

vLLM은 LoRA를 직접 로드할 수 없으므로 base + adapter를 먼저 머지한다.

```bash
python inference/merge_lora.py --adapter outputs/sft_lora/checkpoint-200 --output outputs/sft_merged

# 또는 Google Drive에서 머지된 모델 다운로드 (Setup 6번 참고)
```

### Step 2: 이미지 로딩 체크

DPO 쌍 생성 전에 이미지가 제대로 인식되는지 확인한다. "Describe this image in detail"로 테스트.

```bash
python dpo/build_pairs.py --config dpo/configs/single.yaml --check-image
```

### Step 3: DPO Chosen/Rejected 쌍 생성

SFT 머지 모델로 각 차트에 대해 N번(기본 8번) 생성 후, **복합 스코어링**으로 best/worst를 선정한다.

```bash
# 전체 (6000개 차트 × 8번 생성 = 48000 outputs)
python dpo/build_pairs.py --config dpo/configs/single.yaml

# 테스트 (100개만)
python dpo/build_pairs.py --config dpo/configs/single.yaml --max-samples 100

# BGE-M3 임베딩 없이 (signal 일치 여부만으로 스코어링)
python dpo/build_pairs.py --config dpo/configs/single.yaml --no-embedding
```

#### 복합 스코어링 기준

각 생성 응답에 점수를 매겨 best(chosen) vs worst(rejected)를 선정:

| 기준 | 가중치 | 설명 |
|------|--------|------|
| Signal 일치 | **10** | actual_signal과 맞으면 +10 |
| Reasoning 유사도 | **5** | Teacher reasoning과 BGE-M3 cosine similarity × 5 |
| Confidence 보정 | **1** | 맞았을 때 confidence 높으면 +, 틀렸을 때 낮으면 + |

```
예시 (actual_signal = LONG):
  생성1: LONG  ✓ + sim=0.82 + conf=80 → score=14.9  ← chosen
  생성2: SHORT ✗ + sim=0.30 + conf=85 → score=1.75  ← rejected
```

#### 쌍 선택 로직

| 상황 | 처리 |
|------|------|
| Mixed (맞은 것 + 틀린 것 혼재) | best vs worst 쌍 생성 |
| All correct (전부 맞음) | score 차이 > 1.0이면 쌍 생성 (reasoning 품질 차이) |
| All wrong (전부 틀림) | 스킵 |

출력 파일(`data/dpo_pairs.jsonl`)에는 chosen/rejected 외에 **모든 생성 결과(`all_outputs`)**도 저장된다.

### Step 4: DPO 학습

```bash
# Single GPU
python dpo/train.py --config dpo/configs/single.yaml

# Multi GPU
accelerate launch --config_file sft/configs/fsdp2.yaml dpo/train.py --config dpo/configs/multi.yaml
```

### Step 5: DPO 모델 평가

```bash
# DPO LoRA 머지
python inference/merge_lora.py --adapter outputs/dpo_lora/checkpoint-100 --output outputs/dpo_merged/checkpoint-100

# 평가
python inference/evaluate_all_vllm_v2.py --models-dir outputs/dpo_merged
```

### DPO 하이퍼파라미터 (yaml로 조정)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `beta` | 0.1 | KL penalty coefficient |
| `loss_type` | sigmoid | sigmoid / hinge / ipo |
| `lr` | 5e-6 | SFT보다 낮게 |
| `lora_rank` / `lora_alpha` | 16 / 32 | SFT와 동일 |
| `pair_generation.num_samples_per_image` | 8 | 이미지당 생성 횟수 |
| `pair_generation.temperature` | 1.0 | 샘플링 다양성 |

---

## Inference & Evaluation

```bash
# 테스트셋 이미지 + 정답 라벨 추출
python inference/extract_testset.py
python inference/extract_testset.py --max-samples 50

# 단일 이미지 예측
python inference/predict.py --image data/testset/chart_xxx.png
python inference/predict.py --adapter outputs/sft_lora/final --image data/testset/chart_xxx.png

# 테스트셋 일괄 평가 (결과는 outputs/eval_results/에 JSON 저장)
python inference/evaluate.py --adapter outputs/sft_lora/final
python inference/evaluate.py --adapter outputs/sft_lora/final --max-eval 50

# 이미지 로딩 체크 (학습/평가 전 이미지가 제대로 들어가는지 확인)
python inference/evaluate_all.py --checkpoints-dir outputs/sft_lora --check-image
python inference/evaluate_all_vllm_v2.py --models-dir outputs/merged --check-image

# 모든 체크포인트 한번에 평가 (precision/recall/F1 포함)
python inference/evaluate_all.py --checkpoints-dir outputs/sft_lora
python inference/evaluate_all.py --checkpoints-dir outputs/sft_lora --max-eval 50

# vLLM으로 빠르게 평가 (LoRA 머지 필요)
python inference/merge_lora.py --checkpoints-dir outputs/sft_lora --output-dir outputs/merged
python inference/evaluate_all_vllm_v2.py --models-dir outputs/merged

# SFT vs DPO 비교
python inference/compare.py
python inference/compare.py --max-eval 50

# Eval 결과 분석 (accuracy curve, confusion matrix, F1 비교)
python inference/analyze_eval.py --input eval_vllm_20260315_044221.json
python inference/analyze_eval.py --input eval_vllm_20260315_044221.json --save-dir outputs/eval_analysis
python inference/analyze_eval.py --input eval_vllm_20260315_044221.json --no-plot  # 텍스트만
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
