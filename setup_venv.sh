#!/bin/bash
# InfoMAE Virtual Environment Setup Script
# 이 스크립트는 Python venv를 사용하여 개발 환경을 구성합니다

set -e  # Exit on error

echo "=================================================="
echo "InfoMAE Virtual Environment Setup"
echo "=================================================="
echo ""

# 1. Python 버전 확인
echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Found Python $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "Error: Python 3.8+ is required"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✓ Python version is compatible"
echo ""

# 2. venv 생성
echo "Step 2: Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Warning: venv directory already exists"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing venv..."
        rm -rf venv
    else
        echo "Keeping existing venv, skipping to activation..."
        source venv/bin/activate
        echo "✓ Virtual environment activated"
        echo ""
        # 의존성 업데이트
        echo "Step 3: Updating dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
        echo ""
        echo "=================================================="
        echo "Setup Complete!"
        echo "=================================================="
        echo ""
        echo "To activate the virtual environment in the future, run:"
        echo "  source venv/bin/activate"
        echo ""
        echo "To deactivate, run:"
        echo "  deactivate"
        echo ""
        exit 0
    fi
fi

python3 -m venv venv
echo "✓ Virtual environment created at ./venv"
echo ""

# 3. venv 활성화
echo "Step 3: Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# 4. pip 업그레이드
echo "Step 4: Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# 5. 의존성 설치
echo "Step 5: Installing dependencies from requirements.txt..."
echo "This may take a few minutes..."
echo ""

pip install -r requirements.txt

echo ""
echo "✓ All dependencies installed"
echo ""

# 6. PyTorch 설치 확인
echo "Step 6: Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'GPU devices: {torch.cuda.device_count()}')"
else
    echo "Warning: CUDA is not available. Training will run on CPU."
    echo "For GPU support, please install CUDA and reinstall PyTorch:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
fi
echo ""

# 7. 디렉토리 생성
echo "Step 7: Creating necessary directories..."
mkdir -p data
mkdir -p outputs
mkdir -p pretrained
echo "✓ Directories created"
echo ""

# 8. 환경 변수 설정 (선택사항)
echo "Step 8: Setting up environment variables..."
cat > .env << 'EOF'
# InfoMAE Environment Variables
# Wandb 설정 (선택사항)
# WANDB_API_KEY=your_api_key_here
# WANDB_ENTITY=your_username_here

# 데이터 경로
DATA_DIR=./data
OUTPUT_DIR=./outputs

# 하드웨어 설정
CUDA_VISIBLE_DEVICES=0
EOF
echo "✓ .env file created (you can edit it to customize settings)"
echo ""

# 9. Git 설정 확인
if [ -d ".git" ]; then
    echo "Step 9: Configuring git (adding .env to .gitignore)..."
    if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
        echo ".env" >> .gitignore
        echo "✓ Added .env to .gitignore"
    else
        echo "✓ .env already in .gitignore"
    fi
else
    echo "Step 9: Git not initialized (skipping)"
fi
echo ""

# 10. 설치 확인 테스트
echo "Step 10: Running installation test..."
python -c "
import torch
import torchvision
import timm
import numpy
import matplotlib
import seaborn
import tqdm
import einops
print('✓ All core packages imported successfully')
"
echo ""

# 완료 메시지
echo "=================================================="
echo "Setup Complete! 🎉"
echo "=================================================="
echo ""
echo "Virtual environment is activated and ready to use."
echo ""
echo "Quick start commands:"
echo "  1. Test with CIFAR-100:"
echo "     bash scripts/quick_start.sh"
echo ""
echo "  2. Download pretrained MAE weights:"
echo "     bash scripts/download_pretrained.sh"
echo ""
echo "  3. Run full experiments:"
echo "     bash run_experiments.sh"
echo ""
echo "  4. Run single stage:"
echo "     python main.py --stage stage2 --mode train --dataset cifar100"
echo ""
echo "=================================================="
echo ""
echo "To activate this environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "For more information, see:"
echo "  - README.md: Project overview"
echo "  - USAGE.md: Detailed usage guide"
echo "  - IMPLEMENTATION_CHECK.md: Implementation verification"
echo ""
echo "Happy researching! 🚀"
echo "=================================================="

