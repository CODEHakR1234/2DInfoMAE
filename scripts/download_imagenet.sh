#!/bin/bash
# ImageNet-1K Download and Setup Guide
# This script provides instructions and links to automated scripts

set -e

echo "=========================================="
echo "ImageNet-1K Setup Guide"
echo "=========================================="
echo ""

IMAGENET_ROOT="./data/imagenet_raw"
IMAGENET_OUTPUT="./data/imagenet"

# Check if already prepared
if [ -d "$IMAGENET_OUTPUT/train" ] && [ -d "$IMAGENET_OUTPUT/val" ]; then
    echo "✓ ImageNet-100 already prepared at $IMAGENET_OUTPUT"
    exit 0
fi

echo "ImageNet-1K Setup Options:"
echo ""
echo "=========================================="
echo "Option 1: Automated Download + Extract (if URLs available)"
echo "=========================================="
echo "  export IMAGENET_TRAIN_URL='your_train_url'"
echo "  export IMAGENET_VAL_URL='your_val_url'"
echo "  bash scripts/download_and_extract_imagenet.sh"
echo ""
echo "  Then extract ImageNet-100:"
echo "    python scripts/prepare_imagenet100.py \\"
echo "      --imagenet_root $IMAGENET_ROOT \\"
echo "      --output_dir $IMAGENET_OUTPUT \\"
echo "      --mode symlink"
echo ""
echo "=========================================="
echo "Option 2: Manual Download"
echo "=========================================="
echo "  1. Visit: https://www.image-net.org/download.php"
echo "  2. Register/login"
echo "  3. Download:"
echo "     - ILSVRC2012_img_train.tar (138GB)"
echo "     - ILSVRC2012_img_val.tar (6.3GB)"
echo "  4. Extract using:"
echo "     bash scripts/extract_imagenet.sh"
echo "  5. Prepare ImageNet-100:"
echo "     python scripts/prepare_imagenet100.py \\"
echo "       --imagenet_root $IMAGENET_ROOT \\"
echo "       --output_dir $IMAGENET_OUTPUT \\"
echo "       --mode symlink"
echo ""
echo "=========================================="
echo "Option 3: Use CIFAR-100 (auto-downloads)"
echo "=========================================="
echo "  DATASET=\"cifar100\" bash run_experiments.sh"
echo ""

