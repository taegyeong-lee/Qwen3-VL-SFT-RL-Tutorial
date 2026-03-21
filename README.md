# Qwen3-VL 4B SFT + DPO Tutorial

Qwen3-VL-4B 비전 언어 모델을 SFT → DPO 순서로 파인튜닝하여 BTC 15분봉 차트 이미지를 분석하고 매매 신호(LONG/SHORT/NEUTRAL)를 예측하는 튜토리얼.

- [Qwen3-VL 4B 비트코인 차트 해석 모델 만들기 — 1편: 데이터셋 구축 (LoRA SFT/Distillation, DPO)](https://velog.io/@seawhale/Qwen3-VL-4B%EB%A1%9C-%EB%B9%84%ED%8A%B8%EC%BD%94%EC%9D%B8-%EC%B0%A8%ED%8A%B8-%ED%95%B4%EC%84%9D-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0-LoRA-SFT-Distillation) [2025.03.14]

- [Qwen3-VL 4B 비트코인 차트 해석 모델 만들기 — 2편: 허깅페이스 TRL로 LoRA 파인튜닝](https://velog.io/@seawhale/Qwen3-VL-4B-%EB%B9%84%ED%8A%B8%EC%BD%94%EC%9D%B8-%EC%B0%A8%ED%8A%B8-%ED%95%B4%EC%84%9D-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0-2%ED%8E%B8-%ED%97%88%EA%B9%85%ED%8E%98%EC%9D%B4%EC%8A%A4-TRL%EB%A1%9C-LoRA-%ED%8C%8C%EC%9D%B8%ED%8A%9C%EB%8B%9D)

- Qwen3-VL 4B 비트코인 차트 해석 모델 만들기 — 3편: DPO로 응답 품질 정렬하기 (작성 중)

## Chart Examples

| LONG | SHORT | NEUTRAL |
|------|-------|---------|
| ![LONG](assets/example_long.png) | ![SHORT](assets/example_short.png) | ![NEUTRAL](assets/example_neutral.png) |

---

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

### 2. 의존성 설치

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
```

> **Note:** RunPod 등 CUDA torch가 이미 설치된 환경에서는 `--index-url` 없이 `pip install -r requirements.txt`만 실행해도 됩니다.

### 3. Locale 설정 (Linux 서버)

학습 중 체크포인트 저장 시 `UnicodeDecodeError`가 발생할 수 있다. `~/.bashrc`에 추가:

```bash
export LANG=en_US.UTF-8
export PYTHONIOENCODING=utf-8
```

### 4. GPU 확인

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 5. SFT 체크포인트 다운로드 (LoRA 머지됨)

학습 없이 바로 추론/DPO를 진행하려면 SFT 머지 모델을 다운로드:

- [Google Drive: sft_merged/checkpoint-200](https://drive.google.com/file/d/1GlPNi0Ukr6GoxOoMDcon7qDPbsK7mLeX/view?usp=drive_link)

```bash
pip install gdown
gdown 1GlPNi0Ukr6GoxOoMDcon7qDPbsK7mLeX -O sft_merged_checkpoint-200.zip
unzip sft_merged_checkpoint-200.zip -d outputs/sft_merged/
# 또는 Python으로: python -c "import zipfile; zipfile.ZipFile('sft_merged_checkpoint-200.zip').extractall('outputs/sft_merged/')"
# outputs/sft_merged/checkpoint-200/ 에 config.json, model*.safetensors 등이 위치해야 함
```

### 6. .env (데이터 생성 시 필요)

```
OPENAI_API_KEY=sk-...
```

---

## 데이터 준비

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

## Training

| 단계 | 설명 | 상세 |
|------|------|------|
| **SFT** | Teacher distillation으로 LoRA 파인튜닝 | [sft/README.md](sft/README.md) |
| **DPO** | SFT 모델 위에 선호도 학습 | [dpo/README.md](dpo/README.md) |

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

## Tested Environment

| 항목 | 버전 |
|------|------|
| Python | 3.11 |
| PyTorch | 2.10.0 (CUDA 12.x) |
| transformers | 4.57.6 |
| trl | 0.29.1 |
| peft | 0.18.1 |
| accelerate | 1.13.0 |
| vllm | 0.17.1 |

> 전체 패키지 버전은 [requirements.txt](requirements.txt) 참고
