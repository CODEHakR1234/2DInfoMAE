#!/bin/bash
# Download pretrained MAE weights

echo "Downloading pretrained MAE ViT-Base weights..."

mkdir -p pretrained

# MAE pretrained on ImageNet-1K
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth \
    -O pretrained/mae_pretrain_vit_base.pth

echo "Download complete! Pretrained weights saved to pretrained/"

