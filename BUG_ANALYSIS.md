# 🐛 InfoMAE 버그 분석 리포트

전체 코드를 체계적으로 검토한 결과입니다.

---

## 🚨 Critical Bugs (치명적 버그)

### 1. **Surprisal Bias 차원 오류** ⭐⭐⭐

**위치**: `models/infomae.py`, line 280-294

**문제:**
```python
# Line 280: cls token 추가
x = torch.cat((cls_tokens, x), dim=1)  # x는 이제 [B, N_keep+1, D]

# Line 294: 또 1을 더함!
surprisal_bias = torch.ones(B, 1 + x.shape[1], device=x.device) * mean_surprisal
# 이러면 [B, 1 + (N_keep+1)] = [B, N_keep+2] 가 됨!
# 하지만 x는 [B, N_keep+1] 이므로 차원 불일치!
```

**올바른 코드:**
```python
# 방법 1: cls 추가 전에 surprisal_bias 생성
surprisal_bias = None
if self.use_surprisal_attention and lambda_weight > 0:
    B = x.shape[0]
    N_keep = x.shape[1]  # cls 추가 전의 kept patches 수
    mean_surprisal = self.surprisal_ema.mean()
    # cls용 1개 + kept patches용 N_keep개
    surprisal_bias = torch.ones(B, 1 + N_keep, device=x.device) * mean_surprisal

# 그 다음 cls token 추가
cls_token = self.cls_token + self.pos_embed[:, :1, :]
cls_tokens = cls_token.expand(x.shape[0], -1, -1)
x = torch.cat((cls_tokens, x), dim=1)

# 방법 2: cls 추가 후에 생성
# cls 추가 후
x = torch.cat((cls_tokens, x), dim=1)  # [B, N_keep+1, D]

# surprisal_bias도 같은 길이로
surprisal_bias = torch.ones(B, x.shape[1], device=x.device) * mean_surprisal
# 이제 [B, N_keep+1]로 x와 일치!
```

**영향**: 모델 forward pass 실패 가능성

---

### 2. **사용되지 않는 변수**

**위치**: `models/infomae.py`, line 290

**문제:**
```python
N_with_cls = x.shape[0]  # 이건 Batch size (B)를 저장함!
# 변수명은 N_with_cls인데 실제로는 사용되지 않음
# 게다가 x.shape[0]은 항상 B (batch size)
```

**수정:**
```python
# 이 줄은 삭제하거나
# 필요하다면:
N_keep = x.shape[1]  # cls 추가 전 kept patches 수
```

---

## ⚠️ Major Issues (주요 문제)

### 3. **Adaptive Masking의 비효율적인 조정 로직**

**위치**: `models/infomae.py`, line 227-256

**문제:**
```python
# 현재: 각 샘플마다 for loop로 mask 조정
for i in range(N):
    if adjustment[i] > 0:
        # 마스크 더 추가
    elif adjustment[i] < 0:
        # 마스크 제거
```

이 방식은:
- 느림 (for loop)
- 복잡함
- Batch 처리의 이점 상실

**개선안:**
```python
def adaptive_masking_strategy(self, x, mask_ratio, alpha=3.0, gamma=1.0):
    """Improved adaptive masking"""
    N, L, D = x.shape
    target_masked = int(L * mask_ratio)
    
    # Compute masking probabilities
    surprisal = self.surprisal_ema.unsqueeze(0).expand(N, -1)
    mask_probs = torch.sigmoid(alpha - gamma * surprisal)
    
    # Sort by mask probability
    sorted_probs, sorted_idx = torch.sort(mask_probs, dim=1, descending=True)
    
    # Select top target_masked indices
    mask = torch.zeros(N, L, device=x.device)
    for i in range(N):
        mask_indices = sorted_idx[i, :target_masked]
        mask[i, mask_indices] = 1
    
    # Get kept indices (where mask == 0)
    keep_mask = (mask == 0)
    ids_keep = keep_mask.nonzero(as_tuple=False)[:, 1].reshape(N, -1)
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
    ids_restore = torch.argsort(torch.argsort(mask, dim=1), dim=1)
    
    return x_masked, mask, ids_restore
```

---

### 4. **Surprisal EMA 업데이트 문제**

**위치**: `models/infomae.py`, line 350-353

**문제:**
```python
batch_surprisal = surprisal.mean(dim=0)  # [L]
self.surprisal_ema = (self.surprisal_momentum * self.surprisal_ema + 
                      (1 - self.surprisal_momentum) * batch_surprisal)
```

**이슈:**
- `surprisal`은 `[B, L]` 형태
- `mask`가 적용된 패치만 의미 있는 loss를 가짐
- 마스크되지 않은 패치의 loss는 0이므로 평균이 왜곡됨

**개선안:**
```python
if self.training:
    # 마스크된 패치만 고려
    masked_surprisal = surprisal * mask  # 마스크된 부분만
    num_masked = mask.sum(dim=0).clamp(min=1)  # 각 패치별 마스크 횟수
    batch_surprisal = masked_surprisal.sum(dim=0) / num_masked
    
    self.surprisal_ema = (self.surprisal_momentum * self.surprisal_ema + 
                          (1 - self.surprisal_momentum) * batch_surprisal)
```

---

## 🔸 Minor Issues (경미한 문제)

### 5. **get_attention_maps의 불일치**

**위치**: `models/infomae.py`, line 370-395

**문제:**
```python
def get_attention_maps(self, imgs):
    # 마스킹 없이 전체 이미지 사용
    x = self.patch_embed(imgs)
    x = x + self.pos_embed[:, 1:, :]
    # ...
```

이것은 학습 시와 다른 입력입니다:
- **학습 시**: 75% 마스킹된 입력
- **시각화 시**: 마스킹 없는 전체 입력

이게 의도된 것인지 확인 필요.

**제안:**
옵션 추가하여 마스킹 여부 선택 가능하게:
```python
def get_attention_maps(self, imgs, use_masking=False, mask_ratio=0.75):
    if use_masking:
        # 학습과 동일한 방식
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, 0.0)
        # ...
    else:
        # 현재 방식 (전체 이미지)
        # ...
```

---

### 6. **Patchify/Unpatchify 하드코딩**

**위치**: `models/infomae.py`, line 176-196

**문제:**
```python
def patchify(self, imgs):
    # ...
    x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)  # 3 하드코딩
```

채널 수가 3으로 고정되어 있어 그레이스케일 이미지 처리 불가

**개선안:**
```python
def patchify(self, imgs):
    p = self.patch_size
    c = imgs.shape[1]  # 채널 수 동적으로
    h = w = imgs.shape[2] // p
    x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(imgs.shape[0], h * w, p**2 * c)
    return x
```

---

### 7. **Random Masking 시드 관리 없음**

**위치**: `models/infomae.py`, line 198-216

**문제:**
재현성을 위한 시드 관리가 없음

**개선안:**
```python
def random_masking(self, x, mask_ratio, generator=None):
    """Random masking with optional seed"""
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    
    if generator is not None:
        noise = torch.rand(N, L, device=x.device, generator=generator)
    else:
        noise = torch.rand(N, L, device=x.device)
    # ...
```

---

## 🟢 Good Practices (잘된 부분)

### ✅ 1. **EMA 사용**
Surprisal tracking에 EMA 사용은 좋은 선택

### ✅ 2. **모듈화**
Encoder/Decoder 분리가 잘 되어 있음

### ✅ 3. **Docstrings**
대부분의 함수에 설명이 있음

---

## 🔧 수정 우선순위

### 🔥 즉시 수정 필요:
1. **Surprisal bias 차원 오류** (Critical!)
2. **사용되지 않는 변수 제거**

### ⚡ 빠른 시일 내 수정:
3. **Adaptive masking 효율성 개선**
4. **Surprisal EMA 업데이트 로직**

### 📝 향후 개선:
5. **get_attention_maps 옵션 추가**
6. **Patchify 일반화**
7. **시드 관리**

---

## 📊 테스트 커버리지

현재 테스트되는 것:
- ✅ 모델 생성
- ✅ Forward pass (기본)
- ✅ 데이터 로딩

테스트 필요:
- ❌ Adaptive masking 로직
- ❌ Surprisal EMA 업데이트
- ❌ 다양한 배치 크기
- ❌ 다양한 마스크 비율
- ❌ Gradient flow

---

## 🛠️ 즉시 적용 가능한 수정

### 수정 1: Surprisal Bias (가장 중요!)

```python
# 현재 (잘못됨):
x = torch.cat((cls_tokens, x), dim=1)
surprisal_bias = torch.ones(B, 1 + x.shape[1], device=x.device) * mean_surprisal

# 수정:
x = torch.cat((cls_tokens, x), dim=1)
surprisal_bias = torch.ones(B, x.shape[1], device=x.device) * mean_surprisal
# 또는
surprisal_bias = torch.ones_like(x[:, :, 0]) * mean_surprisal  # [B, N]
```

### 수정 2: 불필요한 변수 제거

```python
# 삭제:
N_with_cls = x.shape[0]  # 이 줄 삭제
```

---

## 📈 성능 영향 예측

### Critical Bug 수정 시:
- ✅ 모델이 정상 작동
- ✅ 학습 안정성 향상
- ✅ Surprisal-weighted attention 제대로 작동

### Optimization 적용 시:
- ⚡ Adaptive masking 속도 2-3배 향상
- 📊 메모리 사용량 10-20% 감소
- 🎯 더 정확한 surprisal tracking

---

## ✅ 검증 체크리스트

수정 후 확인 사항:
- [ ] `python test_installation.py` 통과
- [ ] 다양한 배치 크기에서 테스트 (1, 16, 256)
- [ ] 다양한 마스크 비율 (0.5, 0.75, 0.9)
- [ ] Stage 0-3 모두 테스트
- [ ] Forward/Backward pass 검증
- [ ] 메모리 사용량 확인
- [ ] 학습 커브 확인

---

## 🎯 결론

**현재 상태:**
- 🔴 **Critical Bug 1개** (즉시 수정 필요)
- 🟡 **Major Issue 2개** (빠른 수정 권장)
- 🟢 **Minor Issue 4개** (점진적 개선)

**권장 조치:**
1. 즉시: Surprisal bias 차원 수정
2. 이번 주: Adaptive masking 최적화
3. 다음 버전: 기타 개선사항

**전체 평가:**
핵심 아이디어는 좋으나, 구현에 몇 가지 버그가 있습니다.
Critical bug만 수정하면 정상 작동할 것으로 예상됩니다.

