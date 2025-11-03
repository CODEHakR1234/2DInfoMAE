#!/bin/bash
# Auto-download dataset based on name
# Falls back to CIFAR-100 if ImageNet-100 is not available

set -e

DATASET_NAME=$1
OUTPUT_DIR=${2:-"./data"}

if [ -z "$DATASET_NAME" ]; then
    echo "Usage: bash scripts/auto_download_dataset.sh <dataset_name> [output_dir]"
    echo "  dataset_name: cifar100, imagenet100"
    exit 1
fi

echo "=========================================="
echo "Auto-downloading dataset: $DATASET_NAME"
echo "=========================================="
echo ""

if [ "$DATASET_NAME" = "cifar100" ]; then
    # CIFAR-100: Auto-download via PyTorch
    echo "✓ CIFAR-100 will be auto-downloaded on first run"
    echo "  Location: $OUTPUT_DIR/cifar-100-python/"
    exit 0

elif [ "$DATASET_NAME" = "imagenet100" ]; then
    # ImageNet-100: Check if already exists
    IMAGENET_DIR="$OUTPUT_DIR/imagenet"
    if [ -d "$IMAGENET_DIR/train" ] && [ -d "$IMAGENET_DIR/val" ]; then
        echo "✓ ImageNet-100 already exists at $IMAGENET_DIR"
        exit 0
    fi
    
    # Try to use CIFAR-100 as fallback
    echo "⚠️  ImageNet-100 not found and cannot be auto-downloaded"
    echo "   (ImageNet requires manual download from image-net.org)"
    echo ""
    echo "Options:"
    echo "  1. Use CIFAR-100 instead (auto-downloads):"
    echo "     DATASET=\"cifar100\" bash run_experiments.sh"
    echo ""
    echo "  2. Prepare ImageNet-100 manually:"
    echo "     bash scripts/download_imagenet.sh"
    echo "     python scripts/prepare_imagenet100.py \\"
    echo "       --imagenet_root /path/to/imagenet \\"
    echo "       --output_dir $IMAGENET_DIR"
    echo ""
    
    # Ask user if they want to fallback to CIFAR-100
    read -p "Use CIFAR-100 instead? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Switching to CIFAR-100..."
        # Modify run_experiments.sh or return signal to caller
        echo "cifar100"
        exit 0
    else
        exit 1
    fi
else
    echo "❌ Unknown dataset: $DATASET_NAME"
    exit 1
fi

