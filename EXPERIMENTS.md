# InfoMAE 실험 가이드

## 🧪 실험 설계 및 예상 결과

본 문서는 InfoMAE 논문의 실험 설계와 예상 결과를 정리합니다.

---

## 📊 주요 실험

### Experiment 1: Ablation Study (Stage별 비교)

**목적**: 각 구성 요소의 효과 검증

| Stage | SWA | Adaptive Mask | IB | Encoder FT | Expected Top-1 Acc |
|-------|-----|---------------|----|-----------|--------------------|
| Stage 0 | ❌ | ❌ | ❌ | ❌ | ~65% |
| Stage 1 | ✅ | ❌ | ❌ | ❌ | ~68% (+3%) |
| Stage 2 | ✅ | ✅ | ✅ | ❌ | ~71% (+3%) |
| Stage 3 | ✅ | ✅ | ✅ | ✅ | ~74% (+3%) |

**실행 방법**:
```bash
bash run_experiments.sh
python evaluate.py --compare
```

**평가 지표**:
1. **Representation Quality**:
   - Top-1 accuracy (Linear probe)
   - KNN accuracy

2. **Attention Selectivity**:
   - Entropy ↓ (더 selective)
   - Gini coefficient ↑ (더 집중)
   - Top-10% concentration ↑

3. **Information Encoding**:
   - MI(Z; S) ↑ (surprisal 반영도)
   - Reconstruction loss ↓

---

### Experiment 2: Hyperparameter Sensitivity

#### 2.1 Lambda (λ) Sensitivity

**가설**: λ가 클수록 attention이 더 selective해짐

```bash
# λ = 0.5
python main.py --stage stage1 # config에서 lambda_end=0.5로 수정

# λ = 1.0
python main.py --stage stage1 # config에서 lambda_end=1.0로 수정

# λ = 1.5 (default)
python main.py --stage stage1

# λ = 2.0
python main.py --stage stage1 # config에서 lambda_end=2.0로 수정
```

**예상 결과**:

| λ | Attention Entropy | Top-1 Acc | MI(Z;S) |
|---|-------------------|-----------|---------|
| 0.5 | 3.0 | 66% | 0.20 |
| 1.0 | 2.7 | 68% | 0.28 |
| 1.5 | 2.5 | 68% | 0.30 |
| 2.0 | 2.3 | 67% | 0.32 |

**해석**: λ가 너무 크면 overfitting 가능

---

#### 2.2 Beta (β) Sensitivity

**가설**: β가 클수록 MI(Z;S)가 증가하지만 reconstruction 성능은 감소

```python
# config.py에서 수정
config.training.beta_ib = [0.0, 0.01, 0.02, 0.05, 0.1]
```

**예상 결과**:

| β | MI(Z;S) | Recon Loss | Top-1 Acc |
|---|---------|------------|-----------|
| 0.0 | 0.15 | 0.25 | 68% |
| 0.01 | 0.28 | 0.26 | 70% |
| 0.02 | 0.35 | 0.28 | 71% |
| 0.05 | 0.42 | 0.32 | 69% |
| 0.1 | 0.48 | 0.38 | 66% |

**해석**: β=0.02가 최적 trade-off

---

#### 2.3 Gamma (γ) Sensitivity

**가설**: γ가 클수록 adaptive masking이 더 공격적

```python
# config.py에서 수정
config.model.masking_gamma_end = [0.5, 1.0, 1.5, 2.0]
```

**예상 결과**:

| γ | Masking Diversity | Top-1 Acc |
|---|-------------------|-----------|
| 0.5 | Low (거의 uniform) | 69% |
| 1.0 | Medium | 70% |
| 1.5 | High (default) | 71% |
| 2.0 | Very High | 70% |

---

### Experiment 3: Transfer Learning

**목적**: 학습된 representation의 일반화 능력 검증

#### 3.1 CIFAR-100

```bash
python main.py \
    --stage stage3 \
    --dataset cifar100 \
    --epochs 100
```

**예상 결과**:
- Baseline MAE: ~75%
- InfoMAE: ~78-80%

#### 3.2 STL-10

```bash
python main.py \
    --stage stage3 \
    --dataset stl10 \
    --epochs 100
```

**예상 결과**:
- Baseline MAE: ~85%
- InfoMAE: ~87-89%

---

### Experiment 4: Attention Analysis

**목적**: InfoMAE의 attention이 인간의 주의와 유사한지 검증

#### 4.1 Attention Entropy Analysis

```python
from utils.visualization import compute_attention_entropy

# 각 stage의 attention entropy 비교
entropies = []
for stage in ['stage0', 'stage1', 'stage2', 'stage3']:
    model = load_model(stage)
    attn_maps = model.get_attention_maps(images)
    entropy = compute_attention_entropy(attn_maps)
    entropies.append(entropy)
```

**예상 결과**:

| Stage | Layer 3 | Layer 6 | Layer 9 | Layer 12 |
|-------|---------|---------|---------|----------|
| Stage 0 | 3.5 | 3.2 | 3.0 | 2.8 |
| Stage 1 | 3.2 | 2.8 | 2.6 | 2.3 |
| Stage 2 | 3.0 | 2.6 | 2.4 | 2.1 |
| Stage 3 | 2.8 | 2.4 | 2.2 | 1.9 |

**해석**: 깊은 layer일수록, stage가 진행될수록 더 selective

---

#### 4.2 Saliency Comparison (Optional)

MIT300 또는 OSIE 데이터셋으로 인간의 주의맵과 비교

```bash
# 데이터셋 다운로드 필요
python evaluate_saliency.py \
    --stage stage3 \
    --saliency_dataset mit300 \
    --saliency_path ./data/mit300
```

**예상 메트릭**:
- NSS (Normalized Scanpath Saliency): 1.5~2.0
- AUC-Judd: 0.75~0.85
- CC (Correlation Coefficient): 0.5~0.7

---

### Experiment 5: Surprisal Dynamics

**목적**: 학습 과정에서 surprisal이 어떻게 변화하는지 분석

```python
# Training 중 surprisal tracking
# engine.py에서 자동으로 수행됨

# Visualization
from utils.visualization import visualize_surprisal_evolution

surprisal_history = load_surprisal_history('outputs/stage3/surprisal_log.pkl')
visualize_surprisal_evolution(surprisal_history, save_path='surprisal_evolution.png')
```

**예상 패턴**:
1. **초기 (0-50 epoch)**: 균등한 surprisal 분포
2. **중기 (50-100 epoch)**: 객체 경계에서 높은 surprisal
3. **후기 (100+ epoch)**: Fine-grained details에 집중

---

### Experiment 6: Comparison with Baselines

**비교 대상**:
1. **MAE** (He et al., 2022)
2. **Self-Guided MAE** (Shin et al., 2024)
3. **Attention-Guided MAE** (Sick et al., 2024)
4. **InfoMAE** (Ours)

**평가 항목**:

| Method | Top-1 Acc | Attention Entropy | Training Time |
|--------|-----------|-------------------|---------------|
| MAE | 65% | 3.2 | 1.0x |
| Self-Guided MAE | 68% | 2.9 | 1.1x |
| Attention-Guided MAE | 70% | 2.7 | 1.2x |
| **InfoMAE** | **74%** | **2.3** | 1.15x |

---

## 📈 결과 분석 및 시각화

### 1. Training Curves

```python
from utils.visualization import plot_training_curves

plot_training_curves(
    train_losses,
    val_losses,
    save_path='training_curves.png'
)
```

### 2. Attention Maps

```python
from utils.visualization import visualize_attention_maps

visualize_attention_maps(
    model, 
    images, 
    device,
    layer_indices=[3, 6, 9, 11],
    save_path='attention_maps.png'
)
```

### 3. Reconstruction Quality

```python
from utils.visualization import visualize_reconstruction

visualize_reconstruction(
    model,
    images,
    device,
    save_path='reconstruction.png',
    num_samples=8
)
```

### 4. Masking Strategy

```python
from utils.visualization import visualize_masking_strategy

visualize_masking_strategy(
    model,
    images,
    device,
    save_path='masking_comparison.png'
)
```

---

## 🔬 추가 실험 아이디어

### 1. Architecture Variations

- ViT-Small vs ViT-Base vs ViT-Large
- Different mask ratios: 50%, 60%, 75%, 90%

### 2. Dataset Scale

- ImageNet-100 → ImageNet-1K
- Different data augmentation strategies

### 3. Multi-modal Extension

- Text + Image (CLIP-style)
- Audio + Visual

### 4. Attention Mechanism Variants

- Different surprisal computation methods
- Learnable λ instead of scheduled

---

## 📊 실험 체크리스트

### 필수 실험

- [ ] Stage 0-3 ablation study
- [ ] Lambda sensitivity analysis
- [ ] Beta sensitivity analysis
- [ ] Transfer learning (CIFAR-100, STL-10)
- [ ] Attention entropy analysis
- [ ] Baseline comparison

### 선택 실험

- [ ] Gamma sensitivity analysis
- [ ] Saliency comparison
- [ ] Surprisal dynamics
- [ ] Architecture variations
- [ ] Different datasets

### 결과 정리

- [ ] Training curves
- [ ] Attention visualizations
- [ ] Reconstruction examples
- [ ] Performance tables
- [ ] Statistical significance tests

---

## 📝 논문 작성 가이드

### Main Results Table

```
Table 1: Ablation Study on ImageNet-100

Method              | Top-1 | Top-5 | Entropy ↓ | MI(Z;S) ↑
--------------------|-------|-------|-----------|----------
MAE (baseline)      | 65.2  | 85.3  | 3.2       | 0.15
+ SWA (Stage 1)     | 68.1  | 87.8  | 2.8       | 0.25
+ Adaptive (Stage 2)| 71.3  | 89.5  | 2.5       | 0.35
+ Fine-tune (Stage 3)| 74.2  | 91.2  | 2.3       | 0.42
```

### Figure Suggestions

1. **Figure 1**: Architecture overview
2. **Figure 2**: Attention map comparison (MAE vs InfoMAE)
3. **Figure 3**: Surprisal evolution over training
4. **Figure 4**: Masking strategy visualization
5. **Figure 5**: Transfer learning results

---

## 💡 예상 인사이트

1. **Surprisal-weighted attention은 모델이 정보량 높은 영역에 집중하게 함**
   - Attention entropy 감소
   - 성능 향상

2. **Adaptive masking은 학습 효율을 높임**
   - 중요한 패치를 더 자주 학습
   - 수렴 속도 향상

3. **Information bottleneck은 representation quality를 개선**
   - MI(Z;S) 증가
   - Transfer learning 성능 향상

4. **인간의 visual attention과 유사한 패턴**
   - 객체 경계에 집중
   - 배경은 무시

---

**실험 진행시 결과를 기록하고 이 문서를 업데이트하세요!**

