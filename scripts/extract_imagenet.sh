#!/bin/bash
# Extract ImageNet-1K tar files (after manual download)

set -e

echo "=========================================="
echo "ImageNet-1K Extraction Script"
echo "=========================================="
echo ""

IMAGENET_ROOT="./data/imagenet_raw"
TRAIN_TAR="$IMAGENET_ROOT/ILSVRC2012_img_train.tar"
VAL_TAR="$IMAGENET_ROOT/ILSVRC2012_img_val.tar"

# Check if tar files exist
if [ ! -f "$TRAIN_TAR" ]; then
    echo "❌ Error: Train tar not found: $TRAIN_TAR"
    echo "Please download ILSVRC2012_img_train.tar first"
    exit 1
fi

if [ ! -f "$VAL_TAR" ]; then
    echo "❌ Error: Validation tar not found: $VAL_TAR"
    echo "Please download ILSVRC2012_img_val.tar first"
    exit 1
fi

# Extract train set
TRAIN_DIR="$IMAGENET_ROOT/train"
if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A $TRAIN_DIR 2>/dev/null)" ]; then
    echo "Extracting train set (this will take a while)..."
    mkdir -p "$TRAIN_DIR"
    cd "$TRAIN_DIR"
    
    # Extract main tar (contains 1000 class tar files)
    echo "  Step 1: Extracting main tar file..."
    tar -xf "$TRAIN_TAR"
    
    # Extract each class tar file
    echo "  Step 2: Extracting class directories (1000 classes)..."
    TOTAL=$(find . -maxdepth 1 -name "n*.tar" | wc -l)
    COUNT=0
    find . -maxdepth 1 -name "n*.tar" | sort | while read TARFILE; do
        COUNT=$((COUNT + 1))
        CLASS_DIR="${TARFILE%.tar}"
        mkdir -p "$CLASS_DIR"
        tar -xf "$TARFILE" -C "$CLASS_DIR" > /dev/null 2>&1
        rm -f "$TARFILE"
        if [ $((COUNT % 100)) -eq 0 ]; then
            echo "    Progress: $COUNT/$TOTAL classes"
        fi
    done
    echo "  ✓ Train set extracted: $TRAIN_DIR"
    cd - > /dev/null
else
    echo "✓ Train set already extracted: $TRAIN_DIR"
fi

# Extract val set
VAL_DIR="$IMAGENET_ROOT/val"
if [ ! -d "$VAL_DIR" ] || [ -z "$(ls -A $VAL_DIR 2>/dev/null)" ]; then
    echo "Extracting validation set..."
    mkdir -p "$VAL_DIR"
    cd "$VAL_DIR"
    tar -xf "$VAL_TAR"
    echo "  ✓ Validation set extracted: $VAL_DIR"
    echo "  ⚠️  Note: Validation images may need class organization"
    cd - > /dev/null
else
    echo "✓ Validation set already extracted: $VAL_DIR"
fi

echo ""
echo "=========================================="
echo "✓ ImageNet-1K extraction complete!"
echo "=========================================="
echo ""
echo "Location: $IMAGENET_ROOT"
echo "  Train: $TRAIN_DIR ($(ls -1 $TRAIN_DIR | wc -l) classes)"
echo "  Val: $VAL_DIR"
echo ""
echo "Next step: Extract ImageNet-100"
echo "  python scripts/prepare_imagenet100.py \\"
echo "    --imagenet_root $IMAGENET_ROOT \\"
echo "    --output_dir ./data/imagenet \\"
echo "    --mode symlink"
echo ""

