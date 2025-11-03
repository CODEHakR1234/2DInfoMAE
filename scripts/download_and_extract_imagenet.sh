#!/bin/bash
# Download and extract ImageNet-1K dataset
# Note: This requires ImageNet download URLs (from image-net.org after registration)

set -e

echo "=========================================="
echo "ImageNet-1K Download and Extract Script"
echo "=========================================="
echo ""

IMAGENET_ROOT="./data/imagenet_raw"
TRAIN_TAR="$IMAGENET_ROOT/ILSVRC2012_img_train.tar"
VAL_TAR="$IMAGENET_ROOT/ILSVRC2012_img_val.tar"

# Create directory
mkdir -p "$IMAGENET_ROOT"

# Check if URLs are provided via environment variables
if [ -z "$IMAGENET_TRAIN_URL" ] || [ -z "$IMAGENET_VAL_URL" ]; then
    echo "⚠️  ImageNet-1K download URLs not provided"
    echo ""
    echo "To use this script, you need to:"
    echo "  1. Register at https://www.image-net.org/download.php"
    echo "  2. Get download URLs for:"
    echo "     - ILSVRC2012_img_train.tar (138GB)"
    echo "     - ILSVRC2012_img_val.tar (6.3GB)"
    echo "  3. Set environment variables:"
    echo "     export IMAGENET_TRAIN_URL='your_train_url'"
    echo "     export IMAGENET_VAL_URL='your_val_url'"
    echo "     bash scripts/download_and_extract_imagenet.sh"
    echo ""
    echo "Alternative: Manual download and extraction"
    echo "  See: docs/IMAGENET100_SETUP.md"
    exit 1
fi

# Download train set
if [ ! -f "$TRAIN_TAR" ]; then
    echo "Downloading train set (138GB - this will take a very long time)..."
    echo "URL: $IMAGENET_TRAIN_URL"
    wget -c "$IMAGENET_TRAIN_URL" -O "$TRAIN_TAR"
    echo "✓ Train set downloaded"
else
    echo "✓ Train set already exists: $TRAIN_TAR"
fi

# Download val set
if [ ! -f "$VAL_TAR" ]; then
    echo "Downloading validation set (6.3GB)..."
    echo "URL: $IMAGENET_VAL_URL"
    wget -c "$IMAGENET_VAL_URL" -O "$VAL_TAR"
    echo "✓ Validation set downloaded"
else
    echo "✓ Validation set already exists: $VAL_TAR"
fi

# Extract train set
TRAIN_DIR="$IMAGENET_ROOT/train"
if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A $TRAIN_DIR 2>/dev/null)" ]; then
    echo ""
    echo "Extracting train set (this will take a while)..."
    mkdir -p "$TRAIN_DIR"
    cd "$TRAIN_DIR"
    
    # Extract main tar
    tar -xf "$TRAIN_TAR"
    
    # Extract each class tar file
    echo "Extracting class directories..."
    find . -name "n*.tar" | while read TARFILE; do
        CLASS_DIR="${TARFILE%.tar}"
        mkdir -p "$CLASS_DIR"
        tar -xf "$TARFILE" -C "$CLASS_DIR"
        rm -f "$TARFILE"
        echo -n "."
    done
    echo ""
    echo "✓ Train set extracted"
    cd - > /dev/null
else
    echo "✓ Train set already extracted: $TRAIN_DIR"
fi

# Extract val set
VAL_DIR="$IMAGENET_ROOT/val"
if [ ! -d "$VAL_DIR" ] || [ -z "$(ls -A $VAL_DIR 2>/dev/null)" ]; then
    echo ""
    echo "Extracting validation set..."
    mkdir -p "$VAL_DIR"
    cd "$VAL_DIR"
    tar -xf "$VAL_TAR"
    
    # Organize val images by class (ImageNet val set needs reorganization)
    echo "Organizing validation images by class..."
    # Note: This requires val labels file or manual organization
    # For now, just extract - user can organize manually or use script
    echo "⚠️  Validation set extracted but may need class organization"
    echo "   See: https://github.com/pytorch/vision/issues/977"
    cd - > /dev/null
else
    echo "✓ Validation set already extracted: $VAL_DIR"
fi

echo ""
echo "=========================================="
echo "✓ ImageNet-1K download and extraction complete!"
echo "=========================================="
echo ""
echo "Location: $IMAGENET_ROOT"
echo "  Train: $TRAIN_DIR"
echo "  Val: $VAL_DIR"
echo ""
echo "Next step: Extract ImageNet-100"
echo "  python scripts/prepare_imagenet100.py \\"
echo "    --imagenet_root $IMAGENET_ROOT \\"
echo "    --output_dir ./data/imagenet \\"
echo "    --mode symlink"
echo ""

