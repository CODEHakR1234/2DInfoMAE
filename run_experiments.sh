#!/bin/bash
# InfoMAE Experiment Scripts

# ========================================
# Stage 0: Baseline MAE Fine-tuning
# ========================================
echo "Stage 0: Baseline MAE Fine-tuning"
python main.py \
    --stage stage0 \
    --mode train \
    --dataset imagenet100 \
    --data_dir ./data/imagenet \
    --output_dir ./outputs \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --seed 42

# Linear probe evaluation
python main.py \
    --stage stage0 \
    --mode probe \
    --dataset imagenet100 \
    --data_dir ./data/imagenet

# ========================================
# Stage 1: Add Surprisal-Weighted Attention
# ========================================
echo "Stage 1: Surprisal-Weighted Attention"
python main.py \
    --stage stage1 \
    --mode train \
    --dataset imagenet100 \
    --data_dir ./data/imagenet \
    --output_dir ./outputs \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --seed 42

# Linear probe evaluation
python main.py \
    --stage stage1 \
    --mode probe \
    --dataset imagenet100 \
    --data_dir ./data/imagenet

# ========================================
# Stage 2: Add Adaptive Masking + IB
# ========================================
echo "Stage 2: Adaptive Masking + Information Bottleneck"
python main.py \
    --stage stage2 \
    --mode train \
    --dataset imagenet100 \
    --data_dir ./data/imagenet \
    --output_dir ./outputs \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --seed 42

# Linear probe evaluation
python main.py \
    --stage stage2 \
    --mode probe \
    --dataset imagenet100 \
    --data_dir ./data/imagenet

# ========================================
# Stage 3: Partial Encoder Fine-tuning
# ========================================
echo "Stage 3: Partial Encoder Fine-tuning"
python main.py \
    --stage stage3 \
    --mode train \
    --dataset imagenet100 \
    --data_dir ./data/imagenet \
    --output_dir ./outputs \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --seed 42

# Linear probe evaluation
python main.py \
    --stage stage3 \
    --mode probe \
    --dataset imagenet100 \
    --data_dir ./data/imagenet

# ========================================
# Transfer Learning: CIFAR-100
# ========================================
echo "Transfer Learning: CIFAR-100"
python main.py \
    --stage stage3 \
    --mode train \
    --dataset cifar100 \
    --data_dir ./data \
    --output_dir ./outputs/cifar100 \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-4 \
    --seed 42

# ========================================
# Transfer Learning: STL-10
# ========================================
echo "Transfer Learning: STL-10"
python main.py \
    --stage stage3 \
    --mode train \
    --dataset stl10 \
    --data_dir ./data \
    --output_dir ./outputs/stl10 \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-4 \
    --seed 42

echo "All experiments completed!"

