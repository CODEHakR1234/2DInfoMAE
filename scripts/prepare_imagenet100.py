#!/usr/bin/env python3
"""
Prepare ImageNet-100 subset from ImageNet-1K

Usage:
    python scripts/prepare_imagenet100.py \
        --imagenet_root /path/to/imagenet \
        --output_dir ./data/imagenet100
"""

import os
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import random

# ImageNet-100 classes (randomly selected for reproducibility)
# You can modify this list or use the standard ImageNet-100 classes
IMAGENET_100_CLASSES = [
    'n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475',
    'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878',
    'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544',
    'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084',
    'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779',
    'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777',
    'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541',
    'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366',
    'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811',
    'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457',
    'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322',
    'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381',
    'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939',
    'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244',
    'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797',
    'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675',
    'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143',
    'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313',
    'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805',
    'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672',
]


def prepare_imagenet100(imagenet_root: str, output_dir: str, copy_mode: str = 'symlink'):
    """
    Prepare ImageNet-100 from ImageNet-1K
    
    Args:
        imagenet_root: Path to ImageNet-1K (should contain train/ and val/)
        output_dir: Output directory for ImageNet-100
        copy_mode: 'copy' or 'symlink' (symlink is faster and saves space)
    """
    imagenet_root = Path(imagenet_root)
    output_dir = Path(output_dir)
    
    # Check if ImageNet exists
    train_dir = imagenet_root / 'train'
    val_dir = imagenet_root / 'val'
    
    if not train_dir.exists() or not val_dir.exists():
        raise ValueError(f"ImageNet directory not found or incomplete: {imagenet_root}")
    
    print(f"Source: {imagenet_root}")
    print(f"Output: {output_dir}")
    print(f"Mode: {copy_mode}")
    print(f"Classes: {len(IMAGENET_100_CLASSES)}")
    print("")
    
    # Create output directories
    output_train = output_dir / 'train'
    output_val = output_dir / 'val'
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    
    # Process train and val splits
    for split, src_dir, dst_dir in [
        ('train', train_dir, output_train),
        ('val', val_dir, output_val)
    ]:
        print(f"Processing {split} split...")
        
        for class_name in tqdm(IMAGENET_100_CLASSES, desc=f"  {split}"):
            src_class_dir = src_dir / class_name
            dst_class_dir = dst_dir / class_name
            
            if not src_class_dir.exists():
                print(f"  Warning: {src_class_dir} not found, skipping...")
                continue
            
            # Create destination class directory
            dst_class_dir.mkdir(exist_ok=True)
            
            # Copy or symlink images
            images = list(src_class_dir.glob('*.JPEG'))
            
            for img_path in images:
                dst_img_path = dst_class_dir / img_path.name
                
                if copy_mode == 'symlink':
                    if not dst_img_path.exists():
                        dst_img_path.symlink_to(img_path.absolute())
                else:  # copy
                    if not dst_img_path.exists():
                        shutil.copy2(img_path, dst_img_path)
                
                total_images += 1
        
        print(f"  {split}: {len(list(dst_dir.iterdir()))} classes")
    
    print(f"\n✓ ImageNet-100 prepared successfully!")
    print(f"  Total images: {total_images:,}")
    print(f"  Location: {output_dir}")
    print(f"\nYou can now use:")
    print(f"  DATASET='imagenet100'")
    print(f"  DATA_DIR='{output_dir}'")


def main():
    parser = argparse.ArgumentParser(description='Prepare ImageNet-100 subset')
    parser.add_argument('--imagenet_root', type=str, required=True,
                       help='Path to ImageNet-1K root directory')
    parser.add_argument('--output_dir', type=str, default='./data/imagenet100',
                       help='Output directory for ImageNet-100')
    parser.add_argument('--mode', type=str, default='symlink',
                       choices=['copy', 'symlink'],
                       help='Copy mode: symlink (fast, saves space) or copy')
    
    args = parser.parse_args()
    
    prepare_imagenet100(
        imagenet_root=args.imagenet_root,
        output_dir=args.output_dir,
        copy_mode=args.mode
    )


if __name__ == '__main__':
    main()

