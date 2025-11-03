# 🧪 InfoMAE 실험 진행 가이드

## 📋 실험 진행 단계

### 1️⃣ 사전 준비

#### 1.1 환경 설정
```bash
# 가상환경 활성화
source venv/bin/activate  # 또는 conda activate infomae

# 의존성 확인
python test_installation.py
```

#### 1.2 Pretrained 모델 다운로드
```bash
# MAE pretrained weights 다운로드 (필수!)
bash scripts/download_pretrained.sh
```

#### 1.3 데이터셋 준비

**CIFAR-100 (빠른 테스트용)**
```bash
# 자동으로 다운로드됨 (처음 실행 시)
# ./data/cifar-100-python/ 경로에 저장
```

**ImageNet-100 (본 실험용)**
```bash
# ImageNet-1K 다운로드 후 100개 클래스 선택
python scripts/prepare_imagenet100.py \
    --source_dir /path/to/imagenet/train \
    --target_dir ./data/imagenet100
```

---

## 2️⃣ 실험 실행 방법

### 방법 1: 전체 실험 자동 실행 (권장)

```bash
# 모든 Stage (0-3) 자동 실행
bash run_experiments.sh
```

**설정 변경:**
`run_experiments.sh` 파일에서 다음을 수정:
```bash
DATASET="cifar100"     # 또는 "imagenet100"
EPOCHS=100             # CIFAR-100: 100, ImageNet-100: 200
BATCH_SIZE=256
LR=1e-4
```

**예상 소요 시간:**
- CIFAR-100 (100 epochs): ~6-8시간 (GPU)
- ImageNet-100 (200 epochs): ~2-3일 (GPU)

---

### 방법 2: Stage별 개별 실행

#### Stage 0: Baseline MAE
```bash
python main.py \
    --stage stage0 \
    --mode train \
    --dataset cifar100 \
    --data_dir ./data \
    --pretrained ./pretrained/mae_pretrain_vit_base.pth \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-4 \
    --output_dir ./outputs
```

#### Stage 1: + Surprisal-Weighted Attention
```bash
python main.py \
    --stage stage1 \
    --mode train \
    --dataset cifar100 \
    --data_dir ./data \
    --pretrained ./pretrained/mae_pretrain_vit_base.pth \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-4 \
    --output_dir ./outputs
```

#### Stage 2: + Adaptive Masking + IB
```bash
python main.py \
    --stage stage2 \
    --mode train \
    --dataset cifar100 \
    --data_dir ./data \
    --pretrained ./pretrained/mae_pretrain_vit_base.pth \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-4 \
    --output_dir ./outputs
```

#### Stage 3: + Partial Encoder Fine-tuning
```bash
python main.py \
    --stage stage3 \
    --mode train \
    --dataset cifar100 \
    --data_dir ./data \
    --pretrained ./pretrained/mae_pretrain_vit_base.pth \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-4 \
    --output_dir ./outputs
```

---

### 방법 3: 빠른 테스트 (Quick Start)

```bash
# CIFAR-100, 10 epochs, Stage 2만 실행
bash scripts/quick_start.sh
```

---

## 3️⃣ 실험 모니터링

### 3.1 학습 진행 확인

**터미널 출력:**
```
Epoch 1/100: 100%|████████| 390/390 [01:39<00:00,  3.93it/s, loss=0.596]
Validation: loss=0.542, acc=0.623
```

**Wandb (웹 대시보드):**
```bash
# 자동으로 로그 업로드됨
# 브라우저에서 확인: https://wandb.ai/your-username/InfoMAE
```

**로컬 로그:**
```bash
# 학습 곡선 확인
ls outputs/stage2_adaptive/training_curves.png

# 체크포인트 확인
ls outputs/stage2_adaptive/checkpoints/
```

---

## 4️⃣ 평가 및 결과 확인

### 4.1 Linear Probe 평가

각 Stage 학습 후 자동으로 평가되거나 수동 실행:

```bash
# Stage별 평가
python main.py \
    --stage stage2 \
    --mode probe \
    --dataset cifar100 \
    --data_dir ./data
```

### 4.2 종합 평가

```bash
# 특정 Stage 상세 평가
python evaluate.py --stage stage2

# 모든 Stage 비교
python evaluate.py --compare
```

### 4.3 결과 확인

**출력 디렉토리 구조:**
```
outputs/
├── stage0_baseline/
│   ├── checkpoints/
│   │   ├── checkpoint_epoch_best.pth
│   │   └── checkpoint_epoch_100.pth
│   ├── training_curves.png
│   ├── logs.txt
│   └── visualizations/
│       ├── reconstruction_epoch_10.png
│       └── attention_epoch_10.png
├── stage1_swa/
├── stage2_adaptive/
└── stage3_finetune/
```

**주요 메트릭:**
- `top1_acc`: Top-1 정확도
- `loss`: Reconstruction loss
- `mi`: Mutual Information I(Z;S)
- `attention_entropy`: Attention entropy

---

## 5️⃣ 실험 설정 커스터마이징

### 5.1 Config 파일 수정

`config.py`에서 하이퍼파라미터 조정:

```python
# Stage별 설정
stage_configs = {
    'stage0': {
        'use_surprisal_attention': False,
        'adaptive_masking': False,
        # ...
    },
    'stage1': {
        'use_surprisal_attention': True,
        'lambda_end': 1.0,  # Surprisal attention weight
        # ...
    },
    'stage2': {
        'use_surprisal_attention': True,
        'adaptive_masking': True,
        'lambda_end': 1.0,
        'masking_gamma': 1.0,  # Adaptive masking sensitivity
        'beta_ib': 0.02,  # Information bottleneck weight
        # ...
    },
    # ...
}
```

### 5.2 명령줄 인자 사용

```bash
python main.py \
    --stage stage2 \
    --epochs 200 \
    --batch_size 128 \
    --lr 5e-5 \
    --lambda_end 1.5 \
    --gamma 1.5 \
    --beta 0.03
```

---

## 6️⃣ 예상 결과 해석

### CIFAR-100 (100 epochs)

| Stage | Top-1 Acc | 개선도 | 특징 |
|-------|-----------|--------|------|
| Stage 0 | ~65% | Baseline | 기본 MAE |
| Stage 1 | ~68% | +3%p | + Surprisal Attention |
| Stage 2 | ~71% | +3%p | + Adaptive Masking + IB |
| Stage 3 | ~74% | +3%p | + Encoder Fine-tuning |

**총 개선: +9%p**

### ImageNet-100 (200 epochs)

| Stage | Top-1 Acc | 개선도 |
|-------|-----------|--------|
| Stage 0 | ~67% | Baseline |
| Stage 1 | ~70% | +3%p |
| Stage 2 | ~73% | +3%p |
| Stage 3 | ~76% | +3%p |

**총 개선: +9%p**

---

## 7️⃣ 문제 해결

### GPU 메모리 부족
```bash
# Batch size 줄이기
python main.py --batch_size 128

# Gradient accumulation
python main.py --batch_size 64 --gradient_accumulation_steps 4
```

### 학습이 너무 느림
```bash
# GPU 확인
nvidia-smi

# Mixed precision 사용 (자동)
# config.py에서 확인
```

### 에러 발생
```bash
# 로그 확인
tail -f outputs/stage2_adaptive/logs.txt

# 재시작 (체크포인트에서)
python main.py \
    --stage stage2 \
    --resume outputs/stage2_adaptive/checkpoints/checkpoint_epoch_50.pth
```

---

## 8️⃣ 추천 워크플로우

### 빠른 프로토타이핑 (1-2시간)
```bash
# 1. Quick test
bash scripts/quick_start.sh

# 2. 결과 확인
python evaluate.py --stage stage2
```

### 본 실험 (1-3일)
```bash
# 1. 전체 실험 실행
bash run_experiments.sh

# 2. 결과 비교
python evaluate.py --compare

# 3. 시각화 생성
python evaluate.py --visualize_all
```

### 논문 제출용 (1주일+)
```bash
# 1. 여러 데이터셋에서 실험
# - CIFAR-100
bash run_experiments.sh  # DATASET="cifar100"

# - ImageNet-100
bash run_experiments.sh  # DATASET="imagenet100"

# - STL-10 (Transfer)
python main.py --stage stage3 --dataset stl10 --epochs 100

# 2. 여러 시드로 평균
for seed in 42 123 456; do
    python main.py --stage stage3 --seed $seed
done

# 3. 결과 분석
python evaluate.py --compare --statistics
```

---

## 9️⃣ 체크리스트

### 실험 시작 전
- [ ] 가상환경 활성화
- [ ] Pretrained weights 다운로드 완료
- [ ] 데이터셋 준비 완료
- [ ] GPU 메모리 확인
- [ ] Config 설정 확인
- [ ] Output directory 확인

### 실험 중
- [ ] 학습 진행 확인 (Wandb/터미널)
- [ ] 체크포인트 저장 확인
- [ ] 메모리 사용량 모니터링
- [ ] 에러 로그 확인

### 실험 후
- [ ] 모든 Stage 완료 확인
- [ ] 평가 실행
- [ ] 결과 비교 및 분석
- [ ] 시각화 생성
- [ ] 결과 백업

---

## 🔟 유용한 팁

1. **첫 실험은 CIFAR-100으로 시작**
   - 빠르게 (6-8시간) 전체 파이프라인 검증 가능

2. **중간 체크포인트 활용**
   - `checkpoints/checkpoint_epoch_best.pth`로 최고 성능 모델 저장
   - 재현성을 위해 시드 고정

3. **Wandb 활용**
   - 실시간 학습 곡선 확인
   - 하이퍼파라미터 추적

4. **병렬 실행**
   - 여러 GPU가 있으면 Stage별로 분산 실행 가능

---

**준비 완료! 실험 시작하세요! 🚀**

