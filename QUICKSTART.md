# ⚡ InfoMAE 빠른 시작 가이드

5분 안에 InfoMAE를 시작하세요!

---

## 🎯 초고속 시작 (3단계)

```bash
# 1. 환경 설정 (자동)
bash setup_venv.sh

# 2. 환경 활성화
source venv/bin/activate

# 3. 테스트 실행 (CIFAR-100, 10 epochs)
bash scripts/quick_start.sh
```

완료! 🎉

---

## 📝 명령어 치트시트

### 환경 관리

```bash
# 환경 활성화
source venv/bin/activate        # venv
conda activate infomae          # conda

# 환경 비활성화
deactivate                       # venv
conda deactivate                # conda

# 설치 확인
python test_installation.py
```

### 학습 실행

```bash
# Stage별 실행
python main.py --stage stage0   # Baseline MAE
python main.py --stage stage1   # + Surprisal Attention
python main.py --stage stage2   # + Adaptive Masking
python main.py --stage stage3   # + Encoder Fine-tune

# 데이터셋 지정
python main.py --dataset cifar100      # CIFAR-100
python main.py --dataset stl10         # STL-10
python main.py --dataset imagenet100   # ImageNet-100

# 파라미터 조정
python main.py --epochs 50 --batch_size 128 --lr 5e-5
```

### 평가

```bash
# Linear probe
python main.py --stage stage3 --mode probe

# 종합 평가
python evaluate.py --stage stage3

# Stage 비교
python evaluate.py --compare
```

### 유틸리티

```bash
# Pretrained 가중치 다운로드
bash scripts/download_pretrained.sh

# 전체 실험 실행 (Stage 0-3)
bash run_experiments.sh

# 빠른 테스트
bash scripts/quick_start.sh
```

---

## 📂 주요 파일

| 파일 | 설명 |
|------|------|
| `main.py` | 학습 메인 스크립트 |
| `config.py` | 설정 관리 |
| `engine.py` | 학습/평가 엔진 |
| `evaluate.py` | 평가 스크립트 |
| `models/infomae.py` | InfoMAE 모델 |
| `data/datasets.py` | 데이터 로더 |
| `utils/` | 유틸리티 함수들 |

---

## 🎨 주요 파라미터

### 모델 설정

```python
# config.py에서 수정
lambda_end = 1.5              # Surprisal attention weight
masking_gamma = 1.5           # Adaptive masking sensitivity
beta_ib = 0.02                # Information bottleneck weight
mask_ratio = 0.75             # Masking ratio
```

### 학습 설정

```python
epochs = 200                  # Training epochs
batch_size = 256              # Batch size
lr = 1e-4                     # Learning rate
weight_decay = 0.05           # Weight decay
```

---

## 🔍 결과 확인

### 출력 구조

```
outputs/
├── stage0_baseline/
│   ├── checkpoints/
│   │   ├── checkpoint_epoch_10.pth
│   │   └── checkpoint_epoch_best.pth
│   └── visualizations/
│       ├── reconstruction_epoch_10.png
│       └── attention_epoch_10.png
├── stage1_swa/
├── stage2_adaptive/
└── stage3_finetune/
```

### 로그 확인

```bash
# Wandb (웹 브라우저)
# https://wandb.ai/your-username/InfoMAE

# 로컬 로그
ls outputs/stage3_finetune/visualizations/
```

---

## ⚙️ 자주 사용하는 설정

### GPU 메모리 부족시

```bash
python main.py --batch_size 128     # Batch 줄이기
python main.py --device cpu         # CPU 사용
```

### 빠른 프로토타이핑

```bash
python main.py \
    --stage stage2 \
    --dataset cifar100 \
    --epochs 10 \
    --batch_size 128
```

### 최상의 결과를 위한 설정

```bash
python main.py \
    --stage stage3 \
    --dataset imagenet100 \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-4
```

---

## 🐛 빠른 문제 해결

| 문제 | 해결 |
|------|------|
| 환경 활성화 안됨 | `source venv/bin/activate` |
| CUDA 메모리 부족 | `--batch_size 128` |
| Import 오류 | `pip install -r requirements.txt` |
| 느린 학습 | GPU 확인: `nvidia-smi` |
| 권한 오류 | `chmod +x setup_venv.sh` |

---

## 📊 예상 성능 (ImageNet-100)

| Stage | Top-1 Acc | Time/Epoch (A100) |
|-------|-----------|-------------------|
| Stage 0 | ~65% | ~10 min |
| Stage 1 | ~68% | ~12 min |
| Stage 2 | ~71% | ~15 min |
| Stage 3 | ~74% | ~18 min |

---

## 📚 더 알아보기

- **README.md**: 전체 프로젝트 개요
- **USAGE.md**: 상세 사용법
- **SETUP_GUIDE.md**: 설치 가이드
- **EXPERIMENTS.md**: 실험 설계
- **IMPLEMENTATION_CHECK.md**: 구현 검증

---

## 🎯 일반적인 워크플로우

### 1. 초기 설정 (한 번만)

```bash
bash setup_venv.sh
source venv/bin/activate
python test_installation.py
```

### 2. 매일 사용

```bash
# 터미널 열고
cd /Users/ihagmyeong/Documents/VClab/2DInfoMAE
source venv/bin/activate

# 실험 실행
python main.py --stage stage2 --dataset cifar100

# 결과 확인
python evaluate.py --stage stage2

# 작업 종료
deactivate
```

### 3. 논문 실험

```bash
# 전체 ablation study
bash run_experiments.sh

# 결과 비교
python evaluate.py --compare

# 시각화 생성
python evaluate.py --stage stage3
```

---

## ✅ 체크리스트

시작 전:
- [ ] Python 3.8+ 설치
- [ ] GPU 사용시 CUDA 설치
- [ ] 가상환경 생성 및 활성화
- [ ] 의존성 설치 완료
- [ ] 테스트 통과

실험 전:
- [ ] 데이터셋 준비
- [ ] Config 설정 확인
- [ ] GPU 메모리 확인
- [ ] Output directory 설정

---

**준비 완료! 행운을 빕니다! 🚀**

