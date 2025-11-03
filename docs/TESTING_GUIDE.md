# 🧪 InfoMAE 테스트 가이드

버그 수정 후 검증 방법입니다.

---

## ✅ 빠른 검증 (5분)

### 1. 설치 테스트
```bash
python test_installation.py
```

**기대 결과**: 6/6 테스트 통과
```
✅ Directory Structure: PASSED
✅ Basic Imports: PASSED
✅ PyTorch & CUDA: PASSED
✅ InfoMAE Modules: PASSED
✅ Model Instantiation: PASSED
✅ Data Loading: PASSED
```

---

## 🔍 상세 검증 (30분)

### 2. 다양한 배치 크기 테스트

```python
import torch
from models.infomae import InfoMAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InfoMAE(use_surprisal_attention=True, adaptive_masking=True).to(device)

# 다양한 배치 크기
batch_sizes = [1, 4, 16, 32, 64]

for bs in batch_sizes:
    print(f"\nTesting batch_size={bs}")
    x = torch.randn(bs, 3, 224, 224).to(device)
    
    try:
        loss, pred, mask, surprisal, latent = model(
            x, 
            mask_ratio=0.75,
            lambda_weight=1.0,
            alpha=3.0,
            gamma=1.0
        )
        print(f"  ✅ Forward pass OK")
        print(f"  - Loss shape: {loss.shape}")
        print(f"  - Pred shape: {pred.shape}")
        print(f"  - Latent shape: {latent.shape}")
        
        # Backward test
        loss.backward()
        print(f"  ✅ Backward pass OK")
        model.zero_grad()
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
```

---

### 3. 다양한 마스크 비율 테스트

```python
mask_ratios = [0.5, 0.75, 0.9, 0.95]

x = torch.randn(4, 3, 224, 224).to(device)

for ratio in mask_ratios:
    print(f"\nTesting mask_ratio={ratio}")
    try:
        loss, pred, mask, surprisal, latent = model(
            x,
            mask_ratio=ratio,
            lambda_weight=1.0
        )
        
        actual_ratio = mask.float().mean().item()
        print(f"  ✅ OK - Actual ratio: {actual_ratio:.3f}")
        print(f"  - Kept tokens: {latent.shape[1]}")
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
```

---

### 4. Stage별 테스트

```python
# Stage 0: Baseline (no surprisal, no adaptive)
print("\n=== Stage 0: Baseline ===")
model_s0 = InfoMAE(
    use_surprisal_attention=False,
    adaptive_masking=False
).to(device)

x = torch.randn(4, 3, 224, 224).to(device)
loss, _, _, _, _ = model_s0(x)
print(f"✅ Stage 0 OK - Loss: {loss.item():.4f}")

# Stage 1: + Surprisal attention
print("\n=== Stage 1: + SWA ===")
model_s1 = InfoMAE(
    use_surprisal_attention=True,
    adaptive_masking=False
).to(device)

loss, _, _, _, _ = model_s1(x, lambda_weight=1.0)
print(f"✅ Stage 1 OK - Loss: {loss.item():.4f}")

# Stage 2: + Adaptive masking
print("\n=== Stage 2: + Adaptive ===")
model_s2 = InfoMAE(
    use_surprisal_attention=True,
    adaptive_masking=True
).to(device)

loss, _, _, _, _ = model_s2(x, lambda_weight=1.0, alpha=3.0, gamma=1.0)
print(f"✅ Stage 2 OK - Loss: {loss.item():.4f}")
```

---

### 5. Surprisal 업데이트 검증

```python
model = InfoMAE(use_surprisal_attention=True).to(device).train()

print("\nTesting Epoch Cache Update")
model.initialize_epoch_cache(100, device='cpu')
x = torch.randn(16, 3, 224, 224).to(device)
image_ids = torch.arange(16)

# 여러 번 forward
for i in range(10):
    loss, _, _, surprisal, _ = model(x, mask_ratio=0.75, image_ids=image_ids)
    if i % 3 == 0:
        cached = model.surprisal_memory[image_ids.cpu()]
        print(f"Step {i}: cached surprisal mean = {cached.mean():.4f}")

print(f"\nFinal cached surprisal: {model.surprisal_memory[image_ids.cpu()].mean():.4f}")
print("✅ Surprisal is updating correctly")
```

---

### 6. 메모리 사용량 체크

```python
import torch.cuda as cuda

if torch.cuda.is_available():
    model = InfoMAE().cuda()
    
    # Before forward
    cuda.empty_cache()
    mem_before = cuda.memory_allocated() / 1024**2
    
    x = torch.randn(32, 3, 224, 224).cuda()
    loss, _, _, _, _ = model(x)
    
    mem_after = cuda.memory_allocated() / 1024**2
    mem_peak = cuda.max_memory_allocated() / 1024**2
    
    print(f"\nMemory Usage (batch_size=32):")
    print(f"  Before: {mem_before:.1f} MB")
    print(f"  After: {mem_after:.1f} MB")
    print(f"  Peak: {mem_peak:.1f} MB")
    print(f"  Used: {mem_after - mem_before:.1f} MB")
```

---

## 🚀 실전 테스트 (20-30분)

### 7. 빠른 학습 테스트

```bash
# CIFAR-100으로 빠른 학습 (5 epochs)
python main.py \
    --stage stage2 \
    --dataset cifar100 \
    --epochs 5 \
    --batch_size 128 \
    --lr 1e-4 \
    --no_wandb
```

**확인 사항:**
- [ ] 학습이 시작됨
- [ ] Loss가 감소함
- [ ] Surprisal이 업데이트됨
- [ ] 메모리 누수 없음
- [ ] Checkpoint 저장됨

---

### 8. 시각화 테스트

```python
from utils.visualization import (
    visualize_reconstruction,
    visualize_attention_maps,
    visualize_masking_strategy
)

model = InfoMAE().to(device).eval()
x = torch.randn(4, 3, 224, 224).to(device)

# Reconstruction
visualize_reconstruction(model, x, device, save_path='test_recon.png')
print("✅ Reconstruction visualization saved")

# Attention maps
visualize_attention_maps(model, x, device, save_path='test_attn.png')
print("✅ Attention visualization saved")

# Masking strategy
if model.adaptive_masking:
    visualize_masking_strategy(model, x, device, save_path='test_mask.png')
    print("✅ Masking visualization saved")
```

---

## 📊 성능 벤치마크

### 9. 속도 테스트

```python
import time

model = InfoMAE().to(device).train()
x = torch.randn(64, 3, 224, 224).to(device)

# Warmup
for _ in range(10):
    loss, _, _, _, _ = model(x)

# Benchmark
torch.cuda.synchronize() if torch.cuda.is_available() else None
start = time.time()

n_iters = 100
for _ in range(n_iters):
    loss, _, _, _, _ = model(x)
    loss.backward()
    model.zero_grad()

torch.cuda.synchronize() if torch.cuda.is_available() else None
end = time.time()

time_per_iter = (end - start) / n_iters
samples_per_sec = 64 / time_per_iter

print(f"\nPerformance (batch_size=64):")
print(f"  Time/iter: {time_per_iter*1000:.1f} ms")
print(f"  Samples/sec: {samples_per_sec:.1f}")
print(f"  GPU: {'Yes' if torch.cuda.is_available() else 'No'}")
```

---

## 🔍 버그 재현 테스트

### 10. 이전 버그 확인

```python
# 버그 1: 차원 불일치
print("\n=== Bug 1: Dimension Mismatch Test ===")
try:
    model = InfoMAE(use_surprisal_attention=True).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    loss, _, _, _, _ = model(x, lambda_weight=1.5, mask_ratio=0.75)
    print("✅ No dimension mismatch - BUG FIXED!")
except RuntimeError as e:
    if "size of tensor" in str(e):
        print(f"❌ BUG STILL EXISTS: {e}")
    else:
        raise

# 테스트 2: Epoch Cache 동작 확인
print("\n=== Test 2: Epoch Cache Update Test ===")
model = InfoMAE().to(device).train()

# Initialize cache
dataset_size = 100
model.initialize_epoch_cache(dataset_size, device='cpu')

image_ids = torch.tensor([0, 1, 2, 3])
x = torch.randn(4, 3, 224, 224).to(device)

# Forward pass
loss, _, _, surprisal, _ = model(x, image_ids=image_ids)

# Check if cache was updated
if model.surprisal_initialized[image_ids.cpu()].all():
    print("✅ Epoch cache is updating correctly!")
    print(f"   Cached surprisal shape: {model.surprisal_memory[image_ids.cpu()].shape}")
else:
    print("❌ Epoch cache not updating!")
```

---

## ✅ 최종 체크리스트

### 기본 기능:
- [ ] 모델 생성 성공
- [ ] Forward pass 성공
- [ ] Backward pass 성공
- [ ] Loss 계산 정상

### 차원 검증:
- [ ] 다양한 배치 크기 (1, 16, 64)
- [ ] 다양한 마스크 비율 (0.5, 0.75, 0.9)
- [ ] Surprisal bias 차원 일치

### Stage별 검증:
- [ ] Stage 0 (Baseline) 작동
- [ ] Stage 1 (+ SWA) 작동
- [ ] Stage 2 (+ Adaptive) 작동
- [ ] Stage 3 (+ Fine-tune) 작동

### 학습 검증:
- [ ] Loss 감소 확인
- [ ] Surprisal EMA 업데이트 확인
- [ ] Gradient flow 정상
- [ ] 메모리 누수 없음

### 시각화:
- [ ] Reconstruction 시각화
- [ ] Attention maps 생성
- [ ] Masking strategy 비교

---

## 🎯 성공 기준

**모든 테스트 통과 시:**
```
✅ All Tests Passed! (6/6)
✅ No dimension mismatches
✅ Surprisal EMA updating correctly
✅ All stages working
✅ Training stable
✅ Visualizations generated

🎉 InfoMAE is ready for production!
```

**문제 발견 시:**
1. 에러 메시지 기록
2. BUG_ANALYSIS.md 참조
3. GitHub Issue 생성
4. 추가 디버깅

---

**빠른 검증 스크립트:**

```bash
#!/bin/bash
# quick_test.sh

echo "Running InfoMAE Quick Tests..."

# 1. Installation test
python test_installation.py

# 2. Quick forward test
python -c "
import torch
from models.infomae import InfoMAE

model = InfoMAE(use_surprisal_attention=True, adaptive_masking=True)
x = torch.randn(4, 3, 224, 224)
loss, _, _, _, _ = model(x, lambda_weight=1.0)
print(f'✅ Quick test passed! Loss: {loss.item():.4f}')
"

echo "Quick tests complete!"
```

저장 후 실행:
```bash
chmod +x quick_test.sh
./quick_test.sh
```

