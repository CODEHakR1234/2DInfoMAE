# 🔍 Shared Memory (shm) 부족 문제 원인 분석

## 📋 문제 발생 이유

### 1. **DataLoader의 Multiprocessing 동작 방식**

PyTorch `DataLoader`는 `num_workers > 0`일 때 **멀티프로세싱**을 사용합니다:

```python
DataLoader(dataset, num_workers=8, batch_size=256)
```

**동작 과정:**

1. **Main Process**: 학습 루프 실행
2. **Worker Processes**: 데이터 로딩 전담 (num_workers개)
3. **Shared Memory**: Main ↔ Workers 간 데이터 전달 통로

```
┌─────────────┐
│ Main Process│ ← 학습/모델 실행
│  (GPU 사용) │
└──────┬──────┘
       │ IPC (Inter-Process Communication)
       │ Shared Memory 통해서 데이터 전달
       │
       ├─── Worker 1 ────┐
       ├─── Worker 2 ────┤
       ├─── Worker 3 ────┤  ← 각 Worker가 배치 준비
       ├─── Worker 4 ────┤     (데이터 로드, transform)
       ├─── Worker 5 ────┤
       ├─── Worker 6 ────┤
       ├─── Worker 7 ────┤
       └─── Worker 8 ────┘
```

### 2. **Shared Memory가 필요한 이유**

**문제:**
- Worker Process는 **별도의 프로세스** (독립된 메모리 공간)
- Main Process와 메모리를 공유할 수 없음
- 데이터 전달 방법 필요

**해결책: Shared Memory (shm)**
- Linux/Unix 시스템의 `/dev/shm` (RAM 기반 파일시스템)
- 프로세스 간 빠른 데이터 공유
- 디스크 I/O 없이 메모리 속도

**데이터 전달 과정:**

```python
# Worker Process (자식 프로세스)
batch = dataset[i]  # 데이터 로드
transformed = transform(batch)  # 변환
# → Shared Memory에 저장 (/dev/shm)

# Main Process (부모 프로세스)
batch = read_from_shm()  # Shared Memory에서 읽기
model(batch)  # 학습
```

### 3. **왜 메모리가 부족해지는가?**

#### A. **각 Worker가 사용하는 메모리**

```
메모리 사용량 = num_workers × (배치 크기 × 이미지 크기)

예시:
- num_workers = 8
- batch_size = 256
- 이미지 크기 = 224×224×3 = 150KB (float32)

Worker당 메모리:
  = 256 × 150KB × 2 (transform 전후) 
  ≈ 75MB per worker

총 필요 메모리:
  = 8 workers × 75MB
  ≈ 600MB
```

**실제로는 더 많이 사용:**
- Transform 중간 결과물 (augmentation)
- Queue에 쌓인 배치들 (prefetch)
- 메타데이터, 인덱스 등

**일반적으로:**
- Worker당 **50-200MB** 사용
- `num_workers=8`이면 **400MB-1.6GB** 필요

#### B. **시스템 Shared Memory 제한**

```bash
# Linux 기본 설정
df -h /dev/shm
# 일반적으로:
# - Docker: 64MB (매우 작음!)
# - Linux: RAM의 50% (예: 16GB RAM → 8GB shm)
# - 하지만 Docker 컨테이너는 기본 64MB로 제한됨
```

**문제:**
- Docker 환경에서 기본 shm 크기: **64MB**
- 8 workers × 75MB = **600MB 필요**
- **64MB < 600MB** → 부족! 💥

#### C. **메모리 사용 패턴**

```
Epoch 시작:
  Worker 1: 배치 0 준비 → shm에 저장 (75MB)
  Worker 2: 배치 1 준비 → shm에 저장 (75MB)
  Worker 3: 배치 2 준비 → shm에 저장 (75MB)
  ...
  Worker 8: 배치 7 준비 → shm에 저장 (75MB)
  
Main Process:
  배치 0 사용 → shm에서 삭제
  배치 1 사용 → shm에서 삭제
  ...

하지만:
- Queue에 미리 준비된 배치들이 쌓임
- Main이 처리하기 전까지 메모리에 유지
- 여러 Epoch를 넘나들며 메모리 누적 가능
```

### 4. **왜 "Bus Error"인가?**

**Bus Error (SIGBUS)** 발생 상황:

```
1. Worker가 shm에 데이터 쓰기 시도
2. shm 공간이 부족 (64MB 다 찼음)
3. OS가 "더 이상 쓸 수 없음" 신호
4. Worker Process가 강제 종료 (SIGBUS)
5. Main Process는 무한 대기 → 에러 발생
```

**에러 메시지:**
```
RuntimeError: DataLoader worker (pid 102099) is killed by signal: Bus error.
```

---

## 🔧 메모리 사용량 계산

### 공식

```
필요한 shm 크기 ≈ num_workers × batch_size × image_size × factor

factor = 2~4 (transform, queue, overhead)
```

### 실제 예시

#### 예시 1: CIFAR-100 (작은 이미지)
```
num_workers = 8
batch_size = 256
이미지 = 32×32×3 = 3KB

필요 메모리 = 8 × 256 × 3KB × 2 = ~12MB ✅ (64MB 안에 가능)
```

#### 예시 2: ImageNet-100 (큰 이미지) - **문제 발생!**
```
num_workers = 8
batch_size = 256
이미지 = 224×224×3 = 150KB

필요 메모리 = 8 × 256 × 150KB × 2 = ~600MB ❌ (64MB 초과!)
```

#### 예시 3: InfoMAE (우리 케이스)
```
num_workers = 8
batch_size = 256
이미지 = 224×224×3 = 150KB
IndexedDataset 오버헤드 포함

필요 메모리 ≈ 8 × 256 × 150KB × 3 = ~900MB ❌ (64MB 초과!)
```

---

## 📊 해결 방법 우선순위

### 방법 1: num_workers 줄이기 ⭐ (가장 간단)

```python
# Before (문제):
num_workers = 8  # 8 × 75MB = 600MB 필요
                  # 하지만 shm = 64MB → 부족!

# After (해결):
num_workers = 2  # 2 × 75MB = 150MB 필요
                  # 하지만 실제 사용은 50MB 정도
                  # 64MB 안에 가능 ✅
```

**메모리 계산:**
- `num_workers=2`: ~150MB (여유 있음)
- `num_workers=4`: ~300MB (여전히 부족 가능)
- `num_workers=0`: 0MB (가장 안전, 하지만 느림)

### 방법 2: Shared Memory 증가

#### Docker 환경:
```bash
docker run --shm-size=2g ...
# 또는
docker-compose.yml:
  shm_size: '2gb'
```

#### Linux 직접 실행:
```bash
# 임시 증가
sudo mount -o remount,size=2G /dev/shm

# 영구 설정
# /etc/fstab 수정:
tmpfs /dev/shm tmpfs defaults,size=2g 0 0
```

### 방법 3: Batch Size 줄이기

```python
# Before:
batch_size = 256  # 256 × 150KB = 38MB per worker

# After:
batch_size = 128  # 128 × 150KB = 19MB per worker
                  # 8 workers × 19MB = 152MB (여전히 부족하지만 완화)
```

---

## 🎯 권장 설정

### 시나리오별 최적 설정

#### 1. Docker 환경 (shm=64MB 제한)
```python
num_workers = 0  # Single-threaded, 가장 안전
# 또는
num_workers = 1-2  # 최소한의 멀티프로세싱
batch_size = 128-256
```

#### 2. Linux 직접 실행 (shm=RAM의 50%)
```python
# 8GB RAM → 4GB shm
num_workers = 4
batch_size = 256

# 16GB RAM → 8GB shm  
num_workers = 8
batch_size = 256
```

#### 3. 고성능 서버 (shm 증가 가능)
```python
num_workers = 8-16
batch_size = 256-512
```

---

## 🔍 디버깅: 메모리 사용량 확인

### Shared Memory 크기 확인
```bash
# 현재 shm 크기
df -h /dev/shm

# 사용량 모니터링
watch -n 1 'df -h /dev/shm'
```

### Worker 메모리 사용량 확인
```python
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Main process: {get_memory_usage():.1f} MB")
```

---

## 📝 요약

### 왜 발생하는가?

1. **PyTorch DataLoader는 멀티프로세싱 사용**
   - `num_workers` 개의 별도 프로세스 생성
   - 프로세스 간 데이터 전달 필요

2. **Shared Memory (`/dev/shm`) 사용**
   - Main ↔ Workers 간 통신 채널
   - Linux: RAM 기반, 빠름
   - Docker: 기본 64MB (작음!)

3. **메모리 사용량 = workers × batch_size × image_size**
   - `num_workers=8, batch_size=256, 224×224` → **~600MB 필요**
   - Docker shm=64MB → **부족!** 💥

4. **Bus Error 발생**
   - shm 공간 부족
   - Worker가 데이터 쓰기 실패
   - 프로세스 강제 종료

### 해결책

**즉시 해결:**
```bash
python main.py --num_workers 2  # workers 줄이기
```

**근본 해결:**
```bash
docker run --shm-size=2g ...  # Docker shm 증가
```

**최적 해결:**
```python
# 자동 조정 코드 (이미 구현됨)
if num_workers > 4 and batch_size > 128:
    num_workers = min(num_workers, 4)
```

---

## 🎓 참고 자료

- PyTorch DataLoader 문서: https://pytorch.org/docs/stable/data.html
- Linux Shared Memory: `/dev/shm` (tmpfs)
- Multiprocessing in Python: `multiprocessing` 모듈

**핵심:** `num_workers`는 성능 향상을 주지만, 시스템 제약(shm 크기)을 고려해야 합니다! 🚀

