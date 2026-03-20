# DPO Training

DPO(Direct Preference Optimization)는 SFT 모델의 출력 중 **좋은 응답(chosen)과 나쁜 응답(rejected)** 쌍으로 선호도 학습을 수행한다.

```
SFT 모델 → 같은 차트에 N번 생성 → 복합 스코어링 → best(chosen) / worst(rejected) 쌍 구성 → DPO 학습
```

## Step 1: SFT LoRA 머지

vLLM은 LoRA를 직접 로드할 수 없으므로 base + adapter를 먼저 머지한다.

```bash
python inference/merge_lora.py --adapter outputs/sft_lora/checkpoint-200 --output outputs/sft_merged

# 또는 Google Drive에서 머지된 모델 다운로드 (메인 README Setup 6번 참고)
```

## Step 2: 이미지 로딩 체크

DPO 쌍 생성 전에 이미지가 제대로 인식되는지 확인한다. "Describe this image in detail"로 테스트.

```bash
python dpo/build_pairs.py --config dpo/configs/single.yaml --check-image

# 모델 경로 직접 지정
python dpo/build_pairs.py --config dpo/configs/single.yaml --model outputs/sft_merged/checkpoint-200 --check-image
```

## Step 3: DPO Chosen/Rejected 쌍 생성

SFT 머지 모델로 각 차트에 대해 N번(기본 8번) 생성 후, **복합 스코어링**으로 best/worst를 선정한다.

### 2-Phase 파이프라인

vLLM과 BGE-M3를 동시에 GPU에 올리면 VRAM이 부족하므로, 2단계로 분리하여 GPU를 시분할한다.

```
Phase 1: vLLM Generation (GPU)
  ├─ 청크 500개씩 생성 (매 청크마다 .raw.jsonl 중간 저장)
  └─ 전부 끝나면 vLLM 해제 → GPU 메모리 반환

Phase 2: BGE-M3 Scoring (GPU)
  ├─ 전체 임베딩 한 번에 배치 처리 (GPU, batch_size=512)
  └─ 스코어링 + chosen/rejected 쌍 구성 → dpo_pairs.jsonl 저장
```

### 실행

```bash
# 전체 (6000개 차트 × 8번 생성 = 48000 outputs)
python dpo/build_pairs.py --config dpo/configs/single.yaml

# 모델 경로 직접 지정 (--model이 config의 sft_merged_path를 덮어씀)
python dpo/build_pairs.py --config dpo/configs/single.yaml --model outputs/sft_merged/checkpoint-200

# 테스트 (100개만)
python dpo/build_pairs.py --config dpo/configs/single.yaml --max-samples 100

# BGE-M3 임베딩 없이 (signal 일치 여부만으로 스코어링)
python dpo/build_pairs.py --config dpo/configs/single.yaml --no-embedding
```

### 복합 스코어링 기준

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

### 쌍 선택 로직

| 상황 | 처리 |
|------|------|
| Mixed (맞은 것 + 틀린 것 혼재) | best vs worst 쌍 생성 |
| All correct (전부 맞음) | score 차이 > 1.0이면 쌍 생성 (reasoning 품질 차이) |
| All wrong (전부 틀림) | 스킵 |

출력 파일(`data/dpo_pairs.jsonl`)에는 chosen/rejected 외에 **모든 생성 결과(`all_outputs`)**도 저장된다.

## Step 4: DPO 학습

```bash
# Single GPU
python dpo/train.py --config dpo/configs/single.yaml

# Multi GPU (FSDP2)
accelerate launch --config_file dpo/configs/fsdp2.yaml dpo/train.py --config dpo/configs/multi.yaml
```

## Step 5: DPO 모델 평가

```bash
# DPO LoRA 머지
python inference/merge_lora.py --adapter outputs/dpo_lora/checkpoint-100 --output outputs/dpo_merged/checkpoint-100

# 평가
python inference/evaluate_all_vllm_v2.py --models-dir outputs/dpo_merged
```

## 하이퍼파라미터 (yaml로 조정)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `beta` | 0.1 | KL penalty coefficient |
| `loss_type` | sigmoid | sigmoid / hinge / ipo |
| `lr` | 5e-6 | SFT보다 낮게 |
| `lora_rank` / `lora_alpha` | 16 / 32 | SFT와 동일 |
| `pair_generation.num_samples_per_image` | 8 | 이미지당 생성 횟수 |
| `pair_generation.temperature` | 1.0 | 샘플링 다양성 |
