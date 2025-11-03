#!/bin/bash
# InfoMAE Conda Environment Setup Script
# Conda를 사용하는 경우 이 스크립트를 사용하세요

set -e  # Exit on error

echo "=================================================="
echo "InfoMAE Conda Environment Setup"
echo "=================================================="
echo ""

# 환경 이름
ENV_NAME="infomae"

# 1. Conda 설치 확인
echo "Step 1: Checking conda installation..."
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed"
    echo "Please install Anaconda or Miniconda from:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

CONDA_VERSION=$(conda --version)
echo "Found: $CONDA_VERSION"
echo "✓ Conda is installed"
echo ""

# 2. 기존 환경 확인
echo "Step 2: Checking for existing environment..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Warning: Environment '$ENV_NAME' already exists"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
        echo "✓ Existing environment removed"
    else
        echo "Keeping existing environment"
        echo "Activating $ENV_NAME..."
        eval "$(conda shell.bash hook)"
        conda activate $ENV_NAME
        echo "✓ Environment activated"
        echo ""
        echo "Updating packages..."
        pip install -r requirements.txt
        echo ""
        echo "=================================================="
        echo "Setup Complete!"
        echo "=================================================="
        echo ""
        echo "To activate this environment in the future, run:"
        echo "  conda activate $ENV_NAME"
        echo ""
        exit 0
    fi
fi
echo ""

# 3. Conda 환경 생성
echo "Step 3: Creating conda environment '$ENV_NAME' with Python 3.9..."
conda create -n $ENV_NAME python=3.9 -y
echo "✓ Conda environment created"
echo ""

# 4. 환경 활성화
echo "Step 4: Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME
echo "✓ Environment activated"
echo ""

# 5. pip 업그레이드
echo "Step 5: Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# 6. PyTorch 설치 (CUDA 지원)
echo "Step 6: Installing PyTorch with CUDA support..."
echo "Detecting CUDA availability..."

# CUDA 버전 감지
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "Detected CUDA Version: $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" =~ ^11\.[0-9]+$ ]]; then
        echo "Installing PyTorch for CUDA 11.8..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$CUDA_VERSION" =~ ^12\.[0-9]+$ ]]; then
        echo "Installing PyTorch for CUDA 12.1..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        echo "Installing PyTorch (CPU version)..."
        pip install torch torchvision
    fi
else
    echo "CUDA not detected, installing CPU version..."
    pip install torch torchvision
fi
echo "✓ PyTorch installed"
echo ""

# 7. 나머지 의존성 설치
echo "Step 7: Installing other dependencies..."
# requirements.txt에서 torch, torchvision 제외하고 설치
grep -v "^torch" requirements.txt > requirements_temp.txt
pip install -r requirements_temp.txt
rm requirements_temp.txt
echo "✓ All dependencies installed"
echo ""

# 8. 설치 확인
echo "Step 8: Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
    for i in $(seq 0 $(($(python -c "import torch; print(torch.cuda.device_count())")-1))); do
        python -c "import torch; print(f'GPU $i: {torch.cuda.get_device_name($i)}')"
    done
else
    echo "Note: Running in CPU mode"
fi
echo ""

# 9. 디렉토리 생성
echo "Step 9: Creating directories..."
mkdir -p data
mkdir -p outputs
mkdir -p pretrained
mkdir -p logs
echo "✓ Directories created"
echo ""

# 10. Jupyter 지원 (선택사항)
echo "Step 10: Setting up Jupyter support (optional)..."
read -p "Do you want to install Jupyter notebook support? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install jupyter ipykernel
    python -m ipykernel install --user --name=$ENV_NAME --display-name="InfoMAE"
    echo "✓ Jupyter support installed"
    echo "  You can now use 'InfoMAE' kernel in Jupyter"
else
    echo "Skipping Jupyter installation"
fi
echo ""

# 11. 환경 설정 파일
echo "Step 11: Creating configuration files..."

# .env 파일
cat > .env << 'EOF'
# InfoMAE Environment Variables
WANDB_PROJECT=InfoMAE
# WANDB_API_KEY=your_api_key_here
# WANDB_ENTITY=your_username_here

DATA_DIR=./data
OUTPUT_DIR=./outputs
PRETRAINED_DIR=./pretrained

# Hardware
CUDA_VISIBLE_DEVICES=0
EOF

# conda 환경 export
conda env export > environment.yml
echo "✓ Configuration files created"
echo "  - .env: Environment variables"
echo "  - environment.yml: Conda environment specification"
echo ""

# 12. 설치 테스트
echo "Step 12: Running comprehensive test..."
python << 'PYTHON_TEST'
import sys
print("Testing imports...")

try:
    import torch
    import torchvision
    import timm
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    import einops
    from PIL import Image
    import cv2
    
    print("✓ All packages imported successfully")
    print(f"✓ Python version: {sys.version.split()[0]}")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ GPU available: {torch.cuda.is_available()}")
    
    # Test model loading
    from models.infomae import InfoMAE
    print("✓ InfoMAE model can be imported")
    
    # Test config
    from config import get_config
    config = get_config('stage1')
    print("✓ Config loaded successfully")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    sys.exit(1)
PYTHON_TEST

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Setup Complete! 🎉"
    echo "=================================================="
    echo ""
    echo "Conda environment '$ENV_NAME' is ready!"
    echo ""
    echo "Environment info:"
    echo "  - Python version: $(python --version)"
    echo "  - Environment name: $ENV_NAME"
    echo "  - Location: $(conda info --base)/envs/$ENV_NAME"
    echo ""
    echo "Quick commands:"
    echo "  1. Activate environment:"
    echo "     conda activate $ENV_NAME"
    echo ""
    echo "  2. Quick test:"
    echo "     bash scripts/quick_start.sh"
    echo ""
    echo "  3. Download pretrained weights:"
    echo "     bash scripts/download_pretrained.sh"
    echo ""
    echo "  4. Run experiments:"
    echo "     bash run_experiments.sh"
    echo ""
    echo "  5. Deactivate environment:"
    echo "     conda deactivate"
    echo ""
    echo "Documentation:"
    echo "  - README.md: Overview"
    echo "  - USAGE.md: Usage guide"
    echo "  - EXPERIMENTS.md: Experiment designs"
    echo "  - IMPLEMENTATION_CHECK.md: Verification"
    echo ""
    echo "=================================================="
    echo "Happy researching! 🚀"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "Setup encountered errors"
    echo "=================================================="
    echo "Please check the error messages above"
    exit 1
fi

