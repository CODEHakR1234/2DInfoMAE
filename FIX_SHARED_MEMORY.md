# 🔧 Shared Memory (shm) 문제 해결 가이드

## 문제 증상

```
ERROR: Unexpected bus error encountered in worker. 
This might be caused by insufficient shared memory (shm).
RuntimeError: DataLoader worker (pid XXX) is killed by signal: Bus error.
```

## 해결 방법

### 방법 1: num_workers 줄이기 (가장 간단)

```bash
# 명령줄에서 num_workers 지정
python main.py --stage stage2 --num_workers 2

# 또는 0으로 설정 (single-threaded, 느리지만 안정적)
python main.py --stage stage2 --num_workers 0
```

### 방법 2: config.py 수정

`config.py`에서 `num_workers`를 줄입니다:

```python
@dataclass
class TrainingConfig:
    num_workers: int = 2  # 8에서 2로 변경
    # 또는
    num_workers: int = 0  # 완전히 비활성화 (느리지만 안정적)
```

### 방법 3: Shared Memory 증가 (Docker/Linux)

Docker를 사용하는 경우:

```bash
# Docker run 시
docker run --shm-size=8g ...

# 또는 docker-compose.yml
services:
  training:
    shm_size: '8gb'
```

Linux에서 직접 실행:

```bash
# 임시 증가 (현재 세션만)
sudo mount -o remount,size=8G /dev/shm

# 영구 설정 (주의: 시스템 재부팅 필요할 수 있음)
# /etc/fstab 수정
tmpfs /dev/shm tmpfs defaults,size=8g 0 0
```

### 방법 4: DataLoader 설정 자동 조정

코드가 이미 자동으로 조정합니다:
- `batch_size > 128`이고 `num_workers > 4`이면 자동으로 `num_workers`를 4로 줄임
- 추가로 `persistent_workers=True`로 worker 재사용하여 메모리 효율 향상

## 권장 설정

### 작은 메모리 시스템
```python
num_workers = 0  # Single-threaded
batch_size = 128
```

### 중간 메모리 시스템
```python
num_workers = 2
batch_size = 256
```

### 대용량 메모리 시스템
```python
num_workers = 4
batch_size = 256
```

## 추가 안내

- `num_workers=0`: 가장 안정적, CPU 1개만 사용, 느림
- `num_workers=2-4`: 균형잡힌 선택, 일반적으로 권장
- `num_workers=8+`: 빠르지만 메모리 많이 필요

## 참고

현재 코드는 자동으로 worker 수를 조정하지만, 여전히 문제가 발생하면 `num_workers=0`으로 설정하세요.

