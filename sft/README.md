# SFT Training

Teacher distillation 데이터로 Qwen3-VL-4B를 LoRA 파인튜닝한다.

## 학습

```bash
# Single GPU
python sft/train.py --config sft/configs/single.yaml

# Multi GPU
accelerate launch --config_file sft/configs/fsdp2.yaml sft/train.py --config sft/configs/multi.yaml

# 소량 테스트
python sft/train.py --config sft/configs/single.yaml --max-samples 50
```

## 하이퍼파라미터 (yaml로 조정)

| | Single GPU | Multi GPU |
|---|----------|-------------|
| Model | Qwen3-VL-4B-Instruct | Qwen3-VL-4B-Instruct |
| LoRA rank / alpha | 16 / 32 | 16 / 32 |
| Batch x Grad Accum | 1 x 16 = **16** | 4 x 4 = **16** (x2 GPU = **32**) |
| Learning Rate | 1e-4 | 5e-5 |
| Epochs | 2 | 2 |

## Evaluation Results (checkpoint-200, Best)

테스트셋 600개 (LONG 214 / SHORT 259 / NEUTRAL 127)에 대한 vLLM 평가 결과.

**Accuracy: 42.7% | Macro F1: 42.4%**

### Accuracy by Checkpoint

![Accuracy Curve](../assets/accuracy_curve.png)

### Confusion Matrix (checkpoint-200)

![Confusion Matrix](../assets/confusion_matrix.png)

| | Pred LONG | Pred SHORT | Pred NEUTRAL | Recall |
|---|---|---|---|---|
| **LONG** | 71 | 60 | 83 | 33.2% |
| **SHORT** | 55 | 102 | 102 | 39.4% |
| **NEUTRAL** | 16 | 28 | 83 | 65.4% |

### Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| LONG | 50.0% | 33.2% | 39.9% | 214 |
| SHORT | 53.7% | 39.4% | 45.4% | 259 |
| NEUTRAL | 31.0% | 65.4% | 42.0% | 127 |
| **Macro Avg** | 44.9% | 46.0% | 42.4% | |

### F1 Score by Class & Checkpoint

![F1 Comparison](../assets/f1_comparison.png)

| Checkpoint | Accuracy | LONG F1 | SHORT F1 | NEUTRAL F1 | Macro F1 |
|---|---|---|---|---|---|
| **checkpoint-200** | **42.7%** | **39.9%** | 45.4% | **42.0%** | **42.4%** |
| checkpoint-300 | 39.8% | 15.2% | **51.6%** | 39.6% | 35.5% |
| final | 41.7% | 19.3% | 52.6% | 41.6% | 37.8% |

> checkpoint-200 이후 LONG recall이 급락하며 SHORT 편향이 심화됨. epoch 1.33 시점이 최적.
