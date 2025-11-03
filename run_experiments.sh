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

# Check and auto-download dataset if needed
if [ "$DATASET" = "imagenet100" ]; then
    # ImageNet-100: Check if exists, otherwise try to prepare or fallback
    if [ ! -d "$DATA_DIR/train" ] || [ ! -d "$DATA_DIR/val" ]; then
        echo "⚠️  ImageNet-100 not found. Attempting automatic setup..."
        echo ""
        
        # Try auto-download script (may fallback to CIFAR-100)
        FALLBACK=$(bash scripts/auto_download_dataset.sh imagenet100 "$DATA_DIR" 2>&1 | grep -E "^cifar100$" || echo "")
        
        if [ -n "$FALLBACK" ]; then
            echo ""
            echo "🔄 Switching to CIFAR-100 (auto-downloads)..."
            DATASET="cifar100"
            DATA_DIR="./data"
            EPOCHS=100  # CIFAR-100 uses fewer epochs
            NUM_WORKERS=4  # CIFAR-100 can use more workers
        elif [ ! -d "$DATA_DIR/train" ] || [ ! -d "$DATA_DIR/val" ]; then
            echo ""
            echo "❌ Error: ImageNet-100 dataset not found and cannot be auto-downloaded"
            echo ""
            echo "ImageNet-100 requires ImageNet-1K to be prepared manually."
            echo "Please prepare it first:"
            echo "  bash scripts/download_imagenet.sh  # for instructions"
            echo "  python scripts/prepare_imagenet100.py \\"
            echo "    --imagenet_root /path/to/imagenet \\"
            echo "    --output_dir $DATA_DIR"
            echo ""
            echo "Or use CIFAR-100 (auto-downloads):"
            echo "  DATASET=\"cifar100\" bash run_experiments.sh"
            exit 1
        fi
    else
        echo "✓ ImageNet-100 dataset found"
    fi
elif [ "$DATASET" = "cifar100" ]; then
    echo "✓ CIFAR-100 will be auto-downloaded if needed"
fi
echo ""

# Configuration
# ✅ ImageNet-100: Better for surprisal evaluation (native 224×224, diverse content)
DATASET="imagenet100"
DATA_DIR="./data/imagenet"
EPOCHS=100          # Reduced from 200 for faster experimentation
BATCH_SIZE=256
LR=1e-4
SEED=42
NUM_WORKERS=2       # Reduce if shared memory issues occur

# Alternative configurations:
# - Full training (recommended for paper): EPOCHS=200
# - Fast test: EPOCHS=50
# - CIFAR-100 (faster): DATASET="cifar100", DATA_DIR="./data", EPOCHS=50, NUM_WORKERS=4

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Num workers: $NUM_WORKERS"
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
    --num_workers $NUM_WORKERS \
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
    --num_workers $NUM_WORKERS \
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
    --num_workers $NUM_WORKERS \
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
    --num_workers $NUM_WORKERS \
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

