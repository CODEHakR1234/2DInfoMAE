# 📦 ImageNet-100 설정 가이드

## 🔍 ImageNet-100이란?

ImageNet-100은 ImageNet-1K (1,000개 클래스)에서 100개 클래스를 선택한 서브셋입니다.

## 📥 다운로드 방법

### 방법 1: ImageNet-1K에서 추출 (권장)

#### Step 1: ImageNet-1K 다운로드

ImageNet-1K는 **수동 다운로드**가 필요합니다:

1. **웹사이트 방문**
   ```
   https://www.image-net.org/download.php
   ```

2. **회원가입/로그인**
   - Academic 사용자: 무료 다운로드 가능
   - 비-Academic: 확인 필요

3. **다운로드**
   - `ILSVRC2012_img_train.tar` (138GB) - Train set
   - `ILSVRC2012_img_val.tar` (6.3GB) - Validation set

4. **압축 해제**
   ```bash
   # Train set 압축 해제
   mkdir -p data/imagenet_raw/train
   cd data/imagenet_raw/train
   tar -xf /path/to/ILSVRC2012_img_train.tar
   
   # 각 클래스별로 압축 해제 (1000개의 tar 파일)
   find . -name "*.tar" | while read NAME ; do
       mkdir -p "${NAME%.tar}"
       tar -xf "${NAME}" -C "${NAME%.tar}"
       rm -f "${NAME}"
   done
   
   # Validation set 압축 해제
   mkdir -p data/imagenet_raw/val
   cd data/imagenet_raw/val
   tar -xf /path/to/ILSVRC2012_img_val.tar
   ```

#### Step 2: ImageNet-100 추출

```bash
# prepare_imagenet100.py 사용
python scripts/prepare_imagenet100.py \
    --imagenet_root ./data/imagenet_raw \
    --output_dir ./data/imagenet \
    --mode symlink  # 또는 'copy' (symlink가 빠르고 공간 절약)
```

**결과:**
- `./data/imagenet/train/` - 100개 클래스의 train 이미지
- `./data/imagenet/val/` - 100개 클래스의 val 이미지

---

### 방법 2: 기존 ImageNet-1K 폴더가 있는 경우

이미 ImageNet-1K를 다른 프로젝트에서 사용 중이라면:

```bash
# 바로 ImageNet-100 추출
python scripts/prepare_imagenet100.py \
    --imagenet_root /path/to/imagenet/ILSVRC/Data/CLS-LOC \
    --output_dir ./data/imagenet \
    --mode symlink
```

**주의:** ImageNet-1K 구조가 다음과 같아야 합니다:
```
imagenet_root/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

---

### 방법 3: CIFAR-100 사용 (대안)

ImageNet-100이 준비되기 전까지 **CIFAR-100**을 사용할 수 있습니다:

```bash
# run_experiments.sh 수정:
DATASET="cifar100"
DATA_DIR="./data"

# 자동으로 다운로드됨!
bash run_experiments.sh
```

**장점:**
- ✅ 자동 다운로드 (약 169MB)
- ✅ 빠른 준비 (5분 내)
- ✅ 빠른 실험 (100 epochs: 6-8시간)

**단점:**
- ❌ 32×32 → 224×224 리사이즈 (아티팩트)
- ❌ 데이터가 적음 (60K vs 135K)
- ❌ Surprisal 평가에 덜 적합

---

## 🚀 빠른 시작 (요약)

### ImageNet-100이 없는 경우

```bash
# Option 1: CIFAR-100 사용 (즉시 시작 가능)
# run_experiments.sh에서:
DATASET="cifar100"
bash run_experiments.sh  # 자동 다운로드!

# Option 2: ImageNet-100 준비 (더 좋은 결과)
# 1. ImageNet-1K 다운로드 (수동)
# 2. 압축 해제
# 3. ImageNet-100 추출
python scripts/prepare_imagenet100.py \
    --imagenet_root ./data/imagenet_raw \
    --output_dir ./data/imagenet \
    --mode symlink

# 4. 실험 실행
bash run_experiments.sh  # ImageNet-100 사용
```

---

## 📊 데이터 크기 비교

| 데이터셋 | 다운로드 크기 | 준비 후 크기 | 준비 시간 |
|---------|-------------|------------|----------|
| CIFAR-100 | 169MB | ~200MB | 5분 (자동) |
| ImageNet-1K | 144GB | ~140GB | 수동 다운로드 |
| ImageNet-100 | (ImageNet-1K에서 추출) | ~14GB | 1-2시간 (추출) |

---

## ⚙️ prepare_imagenet100.py 옵션

```bash
python scripts/prepare_imagenet100.py \
    --imagenet_root /path/to/imagenet-1k \
    --output_dir ./data/imagenet \
    --mode symlink  # 또는 'copy'
```

**옵션 설명:**
- `--imagenet_root`: ImageNet-1K의 루트 디렉토리 (train/과 val/ 포함)
- `--output_dir`: ImageNet-100 저장 위치
- `--mode`:
  - `symlink`: 심볼릭 링크 생성 (빠름, 공간 절약) ⭐ 권장
  - `copy`: 실제 복사 (느림, 공간 필요)

---

## 🔍 확인 방법

```bash
# ImageNet-100 확인
ls ./data/imagenet/train/ | wc -l  # 100개 클래스
ls ./data/imagenet/val/ | wc -l    # 100개 클래스

# 이미지 개수 확인 (대략적)
find ./data/imagenet/train -name "*.JPEG" | wc -l  # ~130,000개
find ./data/imagenet/val -name "*.JPEG" | wc -l    # ~5,000개
```

---

## ❓ FAQ

**Q: ImageNet-1K를 다운받을 수 없어요**
→ CIFAR-100을 사용하세요 (자동 다운로드)

**Q: 다운로드는 어디서 하나요?**
→ https://www.image-net.org/download.php

**Q: 다운로드 시간이 얼마나 걸려요?**
→ 138GB이므로 인터넷 속도에 따라 다릅니다 (일반적으로 하루 이상)

**Q: symlink와 copy의 차이는?**
→ `symlink`: 빠르고 공간 절약, 원본 필요
→ `copy`: 느리지만 독립적, 공간 필요

---

## 🎯 추천 워크플로우

1. **빠른 테스트**: CIFAR-100 사용
   ```bash
   DATASET="cifar100" bash run_experiments.sh
   ```

2. **본 실험**: ImageNet-100 준비
   ```bash
   # ImageNet-1K 다운로드 (수동)
   # prepare_imagenet100.py 실행
   bash run_experiments.sh  # ImageNet-100 사용
   ```

