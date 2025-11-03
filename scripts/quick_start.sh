#!/bin/bash
# Quick start script for testing InfoMAE on a small scale

echo "InfoMAE Quick Start"
echo "=================="
echo "This will run a quick test on CIFAR-100 (auto-downloaded)"
echo ""

# Check if pretrained weights exist
PRETRAINED_PATH="./pretrained/mae_pretrain_vit_base.pth"

if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "⚠️  Warning: Pretrained MAE weights not found at $PRETRAINED_PATH"
    echo "Running with random initialization (not recommended for quick test)"
    echo ""
    echo "To download pretrained weights, run:"
    echo "  bash scripts/download_pretrained.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    PRETRAINED_ARG=""
else
    echo "✓ Using pretrained MAE weights from $PRETRAINED_PATH"
    echo ""
    PRETRAINED_ARG="--pretrained $PRETRAINED_PATH"
fi

# Stage 2 with reduced epochs for quick testing
python main.py \
    --stage stage2 \
    --mode train \
    --dataset cifar100 \
    --data_dir ./data \
    --output_dir ./outputs/quick_test \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-4 \
    --seed 42 \
    $PRETRAINED_ARG

echo ""
echo "Quick test complete!"
echo "Check outputs/quick_test/ for results"

