# 🔄 Epoch Cache Device Handling 설명

## 📋 핵심 개념

### Memory 저장 위치: **CPU** (권장)

```python
# main.py
model.initialize_epoch_cache(dataset_size, device='cpu')
# → CPU에 저장 (GPU 메모리 절약!)
```

**이유:**
- ImageNet-100: 102 MB
- GPU 메모리를 차지하지 않음 ✅
- Forward pass 중에만 GPU로 이동 (전송 비용 매우 작음)

---

## 🔄 Device Flow

### 1. **Training Loop 시작**

```python
# engine.py
images = images.to(self.device)  # GPU
image_ids = image_ids.to(self.device)  # GPU
```

### 2. **Forward Pass - Cache 가져오기**

```python
# models/infomae.py forward()
# Step 1: Indexing을 위해 CPU로 이동 (일시적!)
image_ids_cpu = image_ids.cpu()  # GPU → CPU

# Step 2: CPU에서 indexing
if self.surprisal_initialized[image_ids_cpu].all():
    surprisal_cached = self.surprisal_memory[image_ids_cpu]  # CPU tensor

# Step 3: 사용을 위해 GPU로 이동
surprisal_override = surprisal_cached.to(imgs.device)  # CPU → GPU
# 이제 GPU에서 사용!
```

### 3. **Forward Pass - Cache 저장**

```python
# Step 1: CPU로 이동
image_ids_cpu = image_ids.cpu()  # GPU → CPU
surprisal_cpu = surprisal.detach().to('cpu')  # GPU → CPU

# Step 2: CPU에 저장
self.surprisal_memory[image_ids_cpu] = surprisal_cpu  # CPU에 저장
```

---

## ⚠️ 왜 이렇게 하나?

### **PyTorch 제약사항:**

```python
# ❌ 불가능:
cpu_tensor = torch.zeros(100, device='cpu')
gpu_indices = torch.tensor([0, 1, 2], device='cuda')
result = cpu_tensor[gpu_indices]  # RuntimeError!

# ✅ 가능:
cpu_tensor = torch.zeros(100, device='cpu')
cpu_indices = torch.tensor([0, 1, 2], device='cpu')
result = cpu_tensor[cpu_indices]  # OK!

# 또는:
gpu_tensor = torch.zeros(100, device='cuda')
gpu_indices = torch.tensor([0, 1, 2], device='cuda')
result = gpu_tensor[gpu_indices]  # OK!
```

**규칙:** **Indexing 시 같은 device여야 함!**

---

## 💡 최적화 옵션

### Option 1: **CPU Memory (현재, 추천)** ✅

```python
# Memory: CPU
# Forward: GPU
# 전송: indexing 시 CPU로, 사용 시 GPU로
```

**장점:**
- GPU 메모리 절약 ✅
- 전송 비용 매우 작음 (batch × 196 × 4 bytes = ~200 KB)

**단점:**
- 약간의 CPU↔GPU 전송 (하지만 무시 가능)

---

### Option 2: **GPU Memory**

```python
# main.py 수정:
model.initialize_epoch_cache(dataset_size, device='cuda')
```

**장점:**
- 전송 없음 (빠름)

**단점:**
- GPU 메모리 102 MB 차지 ❌
- A100 (40GB): 0.27% ← 괜찮음
- RTX 4090 (24GB): 0.44% ← 괜찮음
- **하지만 불필요한 GPU 메모리 사용!**

---

## 🎯 결론

**현재 구현이 올바릅니다!** ✅

```python
# Memory: CPU (GPU 메모리 절약)
# Indexing: CPU에서 (PyTorch 요구사항)
# 사용: GPU에서 (계산)

# Flow:
GPU (image_ids) → CPU (indexing) → CPU (memory) → GPU (사용)
```

**만약 GPU memory를 사용하고 싶다면:**
- `device='cuda'`로 변경 가능
- 하지만 불필요 (CPU가 더 효율적)

---

## 📊 전송 비용

```python
# Batch size = 256
transfer_size = 256 × 4 bytes = 1 KB  # image_ids
surprisal_size = 256 × 196 × 4 bytes = 200 KB  # surprisal

# PCIe 4.0: ~32 GB/s
transfer_time = 201 KB / 32 GB/s ≈ 0.006 ms

# Forward pass: ~50 ms
overhead = 0.006 / 50 = 0.012%  # 무시 가능!
```

---

**결론: CPU memory 사용이 맞고, indexing을 위해 일시적으로 CPU로 이동하는 것이 정확합니다!** ✅

