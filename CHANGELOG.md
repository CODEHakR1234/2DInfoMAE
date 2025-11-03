# Changelog

All notable changes to InfoMAE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Distributed training support (Multi-GPU DDP)
- Mixed precision training (AMP)
- Complete saliency evaluation pipeline
- Jupyter notebook tutorials
- Docker container
- CI/CD pipeline

---

## [1.0.0] - 2025-11-03

### 🎉 Initial Release

This is the first public release of InfoMAE: Information-Driven Masked Autoencoding for Human-Like Visual Attention.

### ✨ Added

#### Core Features
- **Surprisal-Weighted Attention (SWA)**
  - Self-attention with surprisal-based bias
  - Dynamic λ warm-up schedule (0 → 1.5)
  - Configurable attention weighting

- **Adaptive Masking**
  - Information-driven masking strategy
  - Surprisal-based probability adjustment
  - Progressive γ scheduling (0.5 → 1.5)

- **Information Bottleneck Regularization**
  - Mutual information estimation
  - IB loss with configurable β weight
  - Representation quality optimization

- **Surprisal Tracking**
  - Exponential moving average (EMA)
  - Per-patch reconstruction error tracking
  - Automatic surprisal map updates

#### Model Architecture
- Vision Transformer (ViT) encoder
- Custom attention blocks with surprisal bias
- MAE-style decoder
- Flexible architecture configuration

#### Training Framework
- 4-stage progressive training (Stage 0-3)
- Automated checkpoint management
- Resume from checkpoint support
- Weights & Biases (wandb) integration
- Learning rate scheduling (cosine decay)
- Gradient clipping
- Mixed optimizer groups for encoder/decoder

#### Datasets
- ImageNet-100 support with automatic class selection
- CIFAR-100 (auto-download)
- STL-10 (auto-download)
- Saliency datasets (MIT300, OSIE)
- Custom dataset interface

#### Evaluation
- Linear probe evaluation
- Top-1 accuracy metrics
- Attention selectivity metrics (entropy, Gini, concentration)
- Mutual information I(Z;S)
- Reconstruction quality (MSE, PSNR)
- Saliency metrics (NSS, AUC-Judd, CC)

#### Visualization
- Reconstruction comparison (original, masked, reconstructed)
- Layer-wise attention maps
- Surprisal heatmaps
- Adaptive vs random masking comparison
- Training curves
- Masking strategy visualization

#### Documentation
- Comprehensive README with setup instructions
- Quick start guide (QUICKSTART.md)
- Detailed usage guide (USAGE.md)
- Installation guide (SETUP_GUIDE.md)
- Experiment design documentation (EXPERIMENTS.md)
- Implementation verification (IMPLEMENTATION_CHECK.md)
- Contributing guidelines (CONTRIBUTING.md)

#### Scripts & Tools
- Automated environment setup (venv & conda)
- Installation verification script
- Quick start test script (CIFAR-100, 10 epochs)
- Full experiment runner (all 4 stages)
- Pretrained MAE weights downloader
- Comprehensive evaluation script

#### Configuration
- YAML-style config management
- Stage-specific configurations
- Command-line argument override
- Environment variable support (.env)

### 🔧 Technical Details

**Model Sizes:**
- ViT-Base/16: ~86M parameters (encoder)
- Total: ~120M parameters (with decoder)

**Hyperparameters:**
- Learning rate: 1e-4 (decoder), 1e-5 (encoder)
- Batch size: 256 (default)
- Epochs: 200 (default)
- Mask ratio: 75%
- Optimizer: AdamW (β₁=0.9, β₂=0.95)
- Weight decay: 0.05

**Performance (Expected on ImageNet-100):**
- Stage 0 (Baseline): ~65% Top-1
- Stage 1 (+SWA): ~68% Top-1
- Stage 2 (+Adaptive): ~71% Top-1
- Stage 3 (+Fine-tune): ~74% Top-1

### 📦 Dependencies

**Core:**
- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- torchvision ≥ 0.15.0
- timm ≥ 0.9.0

**Full list:** See `requirements.txt`

### 🎓 Research

Based on:
- **MAE** (He et al., 2022): Masked Autoencoders Are Scalable Vision Learners
- **Predictive Coding** (Rao & Ballard, 1999)
- **Information Bottleneck** (Tishby et al., 2000)

### 📄 License

MIT License - See LICENSE file for details

---

## Release Notes Template for Future Versions

### [X.Y.Z] - YYYY-MM-DD

#### Added
- New features

#### Changed
- Changes in existing functionality

#### Deprecated
- Soon-to-be removed features

#### Removed
- Removed features

#### Fixed
- Bug fixes

#### Security
- Security improvements

---

**Legend:**
- 🎉 Major release
- ✨ New feature
- 🐛 Bug fix
- 📚 Documentation
- ⚡ Performance
- 🔧 Configuration
- 💥 Breaking change

