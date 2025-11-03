#!/bin/bash
# Quick start script for testing InfoMAE on a small scale

echo "InfoMAE Quick Start"
echo "=================="
echo "This will run a quick test on CIFAR-100 (auto-downloaded)"
echo ""

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
    --seed 42

echo ""
echo "Quick test complete!"
echo "Check outputs/quick_test/ for results"

