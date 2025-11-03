# InfoMAE Usage Guide

## 📖 목차

1. [빠른 시작](#빠른-시작)
2. [단계별 실험](#단계별-실험)
3. [평가 및 시각화](#평가-및-시각화)
4. [하이퍼파라미터 튜닝](#하이퍼파라미터-튜닝)
5. [문제 해결](#문제-해결)

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성
conda create -n infomae python=3.9
conda activate infomae

# 의존성 설치
pip install -r requirements.txt
```

### 2. 간단한 테스트 (CIFAR-100)

```bash
# CIFAR-100으로 빠른 테스트 (자동 다운로드)
bash scripts/quick_start.sh
```

이 명령은:
- CIFAR-100 데이터셋 자동 다운로드
- Stage 2 (Adaptive Masking) 10 에포크 학습
- `outputs/quick_test/`에 결과 저장

### 3. 전체 실험 실행

```bash
# ImageNet-100으로 전체 4단계 실험
bash run_experiments.sh
```

---

## 🔬 단계별 실험

### Stage 0: Baseline MAE

**목적**: MAE 기본 성능 측정

```bash
python main.py \
    --stage stage0 \
    --mode train \
    --dataset imagenet100 \
    --data_dir ./data/imagenet \
    --epochs 200 \
    --batch_size 256
```

**특징**:
- ✅ Random masking (75%)
- ❌ Surprisal-weighted attention
- ❌ Adaptive masking
- ❌ Information bottleneck

---

### Stage 1: Surprisal-Weighted Attention (SWA)

**목적**: Attention bias 효과 측정

```bash
python main.py \
    --stage stage1 \
    --mode train \
    --dataset imagenet100 \
    --epochs 200
```

**특징**:
- ✅ Random masking
- ✅ **Surprisal-weighted attention** (λ: 0→1.5)
- ❌ Adaptive masking
- ❌ Information bottleneck

**주요 파라미터**:
- `lambda_start`: 0.0 (초기 surprisal 가중치)
- `lambda_end`: 1.5 (최종 surprisal 가중치)
- `lambda_warmup_epochs`: 20 (warm-up 기간)

---

### Stage 2: Adaptive Masking + IB

**목적**: 정보량 기반 동적 마스킹 효과

```bash
python main.py \
    --stage stage2 \
    --mode train \
    --dataset imagenet100 \
    --epochs 200
```

**특징**:
- ✅ **Adaptive masking** (정보량 기반)
- ✅ Surprisal-weighted attention
- ✅ **Information bottleneck regularization**
- ❌ Encoder fine-tuning

**주요 파라미터**:
- `masking_alpha`: 3.0 (masking 기준점)
- `masking_gamma`: 0.5→1.5 (sensitivity)
- `beta_ib`: 0.02 (IB regularization weight)

**Adaptive Masking 원리**:
```
p_mask(i) = sigmoid(α - γ·S_i)

S_i ↑ (높은 surprisal) → p_mask ↓ → 더 자주 학습
S_i ↓ (낮은 surprisal) → p_mask ↑ → 자주 가림
```

---

### Stage 3: Partial Encoder Fine-tuning

**목적**: Encoder 최적화로 최종 성능 향상

```bash
python main.py \
    --stage stage3 \
    --mode train \
    --dataset imagenet100 \
    --epochs 200
```

**특징**:
- ✅ Adaptive masking
- ✅ Surprisal-weighted attention
- ✅ Information bottleneck
- ✅ **마지막 2개 encoder block fine-tuning**

**주요 파라미터**:
- `unfreeze_last_n_blocks`: 2
- `encoder_lr`: 1e-5 (encoder 학습률)

---

## 📊 평가 및 시각화

### Linear Probe 평가

모델의 representation 품질 측정:

```bash
python main.py \
    --stage stage3 \
    --mode probe \
    --dataset imagenet100 \
    --data_dir ./data/imagenet
```

**결과**:
- Top-1 accuracy
- 학습 곡선
- Representation 품질

---

### 종합 평가

전체 메트릭 및 시각화 생성:

```bash
python evaluate.py --stage stage3
```

**생성되는 항목**:
1. **Reconstruction visualization**: 원본, 마스킹, 복원 비교
2. **Attention maps**: 레이어별 attention 패턴
3. **Surprisal heatmaps**: 정보량 분포
4. **Masking strategy**: Random vs Adaptive 비교

---

### 단계별 비교

모든 stage 성능 비교:

```bash
python evaluate.py --compare --output_dir ./outputs
```

**출력**:
- Stage별 reconstruction loss
- Stage별 surprisal 통계
- 비교 그래프 (`stage_comparison.png`)

---

## ⚙️ 하이퍼파라미터 튜닝

### 주요 하이퍼파라미터

#### 1. Surprisal Attention (λ)

```bash
# λ를 더 강하게
python main.py --stage stage1 --mode train \
    --lambda_end 2.0  # default: 1.5

# λ warm-up 기간 조정
python main.py --stage stage1 --mode train \
    --lambda_warmup_epochs 30  # default: 20
```

**가이드라인**:
- λ ↑ → attention이 high-surprisal 영역에 더 집중
- λ ↓ → 더 균등한 attention 분포
- 추천 범위: 1.0~2.0

---

#### 2. Adaptive Masking (α, γ)

```python
# config.py에서 수정
config.model.masking_alpha = 3.5  # default: 3.0
config.model.masking_gamma_start = 0.3  # default: 0.5
config.model.masking_gamma_end = 2.0  # default: 1.5
```

**가이드라인**:
- α ↑ → 전체적으로 masking 확률 증가
- γ ↑ → surprisal에 더 민감하게 반응
- 추천: α=3.0~4.0, γ=0.5~2.0

---

#### 3. Information Bottleneck (β)

```bash
python main.py --stage stage2 --mode train \
    # config.py에서 training.beta_ib 수정
```

**가이드라인**:
- β ↑ → surprisal 정보가 representation에 더 반영
- β ↓ → reconstruction에 더 집중
- 추천 범위: 0.01~0.05

---

### 학습 안정성 개선

#### Gradient Clipping

```python
# config.py
config.training.grad_clip = 1.0  # default
# 학습 불안정시 0.5로 감소
```

#### Learning Rate

```bash
# 작은 learning rate (더 안정적)
python main.py --stage stage3 --lr 5e-5

# 큰 learning rate (더 빠른 수렴)
python main.py --stage stage3 --lr 2e-4
```

---

## 🎯 데이터셋별 설정

### ImageNet-100

```bash
python main.py \
    --stage stage3 \
    --dataset imagenet100 \
    --data_dir ./data/imagenet \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-4
```

### CIFAR-100

```bash
python main.py \
    --stage stage3 \
    --dataset cifar100 \
    --data_dir ./data \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-4
```

### STL-10

```bash
python main.py \
    --stage stage3 \
    --dataset stl10 \
    --data_dir ./data \
    --epochs 100 \
    --batch_size 128 \
    --lr 5e-5
```

---

## 🐛 문제 해결

### GPU 메모리 부족

```bash
# Batch size 감소
python main.py --batch_size 128  # default: 256

# Gradient accumulation (향후 구현 예정)
```

### 학습 불안정

```bash
# Learning rate 감소 + gradient clipping
python main.py --lr 5e-5

# config.py에서:
config.training.grad_clip = 0.5
```

### 수렴 속도 느림

```bash
# Warm-up 기간 조정
# config.py에서:
config.training.warmup_epochs = 20  # default: 10

# Learning rate scheduler 조정
config.training.scheduler = 'cosine'  # 또는 'step'
```

### Checkpoint 로드 실패

```bash
# Resume 사용
python main.py --resume ./outputs/stage3_finetune/checkpoints/checkpoint_epoch_100.pth

# Pretrained weights 로드
python main.py --pretrained ./pretrained/mae_pretrain_vit_base.pth
```

---

## 📈 성능 최적화 팁

### 1. Progressive Training

Stage를 순차적으로 실행하며 체크포인트 활용:

```bash
# Stage 1 학습
python main.py --stage stage1 --epochs 100

# Stage 1 결과를 Stage 2의 초기값으로 사용
python main.py --stage stage2 \
    --resume ./outputs/stage1_swa/checkpoints/checkpoint_epoch_best.pth
```

### 2. Hyperparameter Search

Grid search 예시:

```bash
for lambda in 1.0 1.5 2.0; do
    for beta in 0.01 0.02 0.05; do
        python main.py --stage stage2 \
            --exp_name "lambda${lambda}_beta${beta}"
        # config.py에서 파라미터 수정 필요
    done
done
```

### 3. 데이터 증강

```python
# data/datasets.py에서 증강 강도 조정
transforms.RandomApply([
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
], p=0.8)  # p를 0.9로 증가
```

---

## 🔍 결과 분석

### Wandb 로깅

```bash
# Wandb 활성화 (기본값)
python main.py --stage stage3

# Wandb 비활성화
python main.py --stage stage3 --no_wandb
```

### TensorBoard 로깅

```bash
# 향후 구현 예정
tensorboard --logdir ./outputs/stage3_finetune/logs
```

---

## 📝 체크리스트

실험 전 확인사항:

- [ ] 데이터셋 다운로드 및 경로 확인
- [ ] GPU 메모리 충분 (최소 16GB 권장)
- [ ] 의존성 패키지 설치 완료
- [ ] `config.py`에서 설정 확인
- [ ] Output directory 쓰기 권한 확인

실험 후 확인사항:

- [ ] Checkpoint 저장 확인
- [ ] 시각화 결과 생성 확인
- [ ] Linear probe 결과 확인
- [ ] Wandb 로그 확인 (활성화시)

---

**문의사항이나 버그 발견시 GitHub Issues에 올려주세요!**

