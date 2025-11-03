#!/bin/bash
# InfoMAE Experiment Scripts
# Full pipeline for reproducing paper results

set -e  # Exit on error

echo "=========================================="
echo "InfoMAE Full Experiment Pipeline"
echo "=========================================="
echo ""

# Check pretrained weights
PRETRAINED="./pretrained/mae_pretrain_vit_base.pth"
if [ ! -f "$PRETRAINED" ]; then
    echo "❌ Error: Pretrained MAE weights not found!"
    echo "Please download first: bash scripts/download_pretrained.sh"
    exit 1
fi
echo "✓ Pretrained weights found: $PRETRAINED"
echo ""

# Configuration (modify as needed)
# ⚠️ Note: CIFAR-100 (32×32) will be resized to 224×224
# For better results, use ImageNet-100 (native 224×224)
DATASET="cifar100"  # Options: "cifar100" (fast, resized), "imagenet100" (better, slower)
DATA_DIR="./data"
EPOCHS=100          # 100 for CIFAR-100, 200 for ImageNet-100
BATCH_SIZE=256
LR=1e-4
SEED=42

# ImageNet-100 설정 (권장)
# DATASET="imagenet100"
# DATA_DIR="./data/imagenet"
# EPOCHS=200

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Using pretrained: Yes"
echo ""

# ========================================
# Stage 0: Baseline MAE Fine-tuning
# ========================================
echo "=========================================="
echo "Stage 0: Baseline MAE Fine-tuning"
echo "=========================================="
python main.py \
    --stage stage0 \
    --mode train \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --output_dir ./outputs \
    --pretrained $PRETRAINED \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED

# Linear probe evaluation
echo "Evaluating Stage 0..."
python main.py \
    --stage stage0 \
    --mode probe \
    --dataset $DATASET \
    --data_dir $DATA_DIR

echo "✓ Stage 0 completed!"
echo ""

# ========================================
# Stage 1: Add Surprisal-Weighted Attention
# ========================================
echo "=========================================="
echo "Stage 1: + Surprisal-Weighted Attention"
echo "=========================================="
python main.py \
    --stage stage1 \
    --mode train \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --output_dir ./outputs \
    --pretrained $PRETRAINED \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED

# Linear probe evaluation
echo "Evaluating Stage 1..."
python main.py \
    --stage stage1 \
    --mode probe \
    --dataset $DATASET \
    --data_dir $DATA_DIR

echo "✓ Stage 1 completed!"
echo ""

# ========================================
# Stage 2: Add Adaptive Masking + IB
# ========================================
echo "=========================================="
echo "Stage 2: + Adaptive Masking + Info Bottleneck"
echo "=========================================="
python main.py \
    --stage stage2 \
    --mode train \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --output_dir ./outputs \
    --pretrained $PRETRAINED \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED

# Linear probe evaluation
echo "Evaluating Stage 2..."
python main.py \
    --stage stage2 \
    --mode probe \
    --dataset $DATASET \
    --data_dir $DATA_DIR

echo "✓ Stage 2 completed!"
echo ""

# ========================================
# Stage 3: Partial Encoder Fine-tuning
# ========================================
echo "=========================================="
echo "Stage 3: + Partial Encoder Fine-tuning"
echo "=========================================="
python main.py \
    --stage stage3 \
    --mode train \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --output_dir ./outputs \
    --pretrained $PRETRAINED \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED

# Linear probe evaluation
echo "Evaluating Stage 3..."
python main.py \
    --stage stage3 \
    --mode probe \
    --dataset $DATASET \
    --data_dir $DATA_DIR

echo "✓ Stage 3 completed!"
echo ""

# ========================================
# Optional: Transfer Learning Tests
# ========================================
# Uncomment to run transfer learning experiments
# echo "=========================================="
# echo "Transfer Learning: STL-10"
# echo "=========================================="
# python main.py \
#     --stage stage3 \
#     --mode train \
#     --dataset stl10 \
#     --data_dir ./data \
#     --output_dir ./outputs/stl10 \
#     --pretrained $PRETRAINED \
#     --epochs 100 \
#     --batch_size 256 \
#     --lr 1e-4 \
#     --seed 42

echo ""
echo "=========================================="
echo "🎉 All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  outputs/stage0_baseline/"
echo "  outputs/stage1_swa/"
echo "  outputs/stage2_adaptive/"
echo "  outputs/stage3_finetune/"
echo ""
echo "Next steps:"
echo "  1. Compare results: python evaluate.py --compare"
echo "  2. Check training curves in outputs/*/training_curves.png"
echo "  3. Review visualizations in outputs/*/visualizations/"
echo ""

