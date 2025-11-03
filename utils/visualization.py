"""
Visualization utilities for InfoMAE
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from typing import List, Tuple, Optional
import os


def visualize_reconstruction(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    save_path: Optional[str] = None,
    num_samples: int = 4,
):
    """
    Visualize original images, masked images, and reconstructions
    """
    model.eval()
    with torch.no_grad():
        images = images[:num_samples].to(device)
        
        # Forward pass
        loss, pred, mask, surprisal, latent = model(images, mask_ratio=model.mask_ratio)
        
        # Unpatchify
        pred_imgs = model.unpatchify(pred)
        
        # Create masked image
        mask_vis = mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])
        masked_patches = model.patchify(images) * (1 - mask_vis)
        masked_imgs = model.unpatchify(masked_patches)
        
        # Denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        images_vis = images * std + mean
        pred_imgs_vis = pred_imgs * std + mean
        masked_imgs_vis = masked_imgs * std + mean
        
        # Clamp to [0, 1]
        images_vis = torch.clamp(images_vis, 0, 1)
        pred_imgs_vis = torch.clamp(pred_imgs_vis, 0, 1)
        masked_imgs_vis = torch.clamp(masked_imgs_vis, 0, 1)
        
        # Plot
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes[np.newaxis, :]
        
        for i in range(num_samples):
            # Original
            axes[i, 0].imshow(images_vis[i].cpu().permute(1, 2, 0))
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            # Masked
            axes[i, 1].imshow(masked_imgs_vis[i].cpu().permute(1, 2, 0))
            axes[i, 1].set_title(f'Masked ({model.mask_ratio:.1%})')
            axes[i, 1].axis('off')
            
            # Reconstruction
            axes[i, 2].imshow(pred_imgs_vis[i].cpu().permute(1, 2, 0))
            axes[i, 2].set_title('Reconstruction')
            axes[i, 2].axis('off')
            
            # Surprisal map
            surprisal_map = surprisal[i].cpu().reshape(int(np.sqrt(surprisal.shape[1])), -1)
            im = axes[i, 3].imshow(surprisal_map, cmap='hot')
            axes[i, 3].set_title('Surprisal Map')
            axes[i, 3].axis('off')
            plt.colorbar(im, ax=axes[i, 3], fraction=0.046)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved reconstruction visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def visualize_attention_maps(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    save_path: Optional[str] = None,
    layer_indices: List[int] = [3, 6, 9, 11],
):
    """
    Visualize attention maps from different layers
    """
    model.eval()
    with torch.no_grad():
        images = images[:1].to(device)  # Single image
        
        # Get attention maps
        attention_maps = model.get_attention_maps(images)
        
        # Select specific layers
        selected_attns = [attention_maps[i] for i in layer_indices]
        
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        img_vis = images * std + mean
        img_vis = torch.clamp(img_vis, 0, 1)
        
        # Plot
        fig, axes = plt.subplots(1, len(layer_indices) + 1, figsize=(4 * (len(layer_indices) + 1), 4))
        
        # Original image
        axes[0].imshow(img_vis[0].cpu().permute(1, 2, 0))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Attention maps
        for idx, (layer_idx, attn_map) in enumerate(zip(layer_indices, selected_attns)):
            # Average attention from CLS token to all patches
            attn = attn_map[0, 0, 1:].cpu().numpy()  # [0, CLS, patches]
            
            # Reshape to spatial
            size = int(np.sqrt(len(attn)))
            attn = attn.reshape(size, size)
            
            # Resize to image size
            attn_resized = cv2.resize(attn, (224, 224))
            
            # Overlay on image
            img_np = img_vis[0].cpu().permute(1, 2, 0).numpy()
            heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            overlay = 0.6 * img_np + 0.4 * heatmap
            
            axes[idx + 1].imshow(overlay)
            axes[idx + 1].set_title(f'Layer {layer_idx}')
            axes[idx + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def visualize_surprisal_evolution(
    surprisal_history: List[np.ndarray],
    save_path: Optional[str] = None,
):
    """
    Visualize how surprisal map evolves during training
    """
    num_epochs = len(surprisal_history)
    selected_epochs = np.linspace(0, num_epochs - 1, min(8, num_epochs)).astype(int)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, epoch_idx in enumerate(selected_epochs):
        surprisal = surprisal_history[epoch_idx]
        size = int(np.sqrt(len(surprisal)))
        surprisal_map = surprisal.reshape(size, size)
        
        im = axes[idx].imshow(surprisal_map, cmap='hot', vmin=0, vmax=surprisal_map.max())
        axes[idx].set_title(f'Epoch {epoch_idx}')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved surprisal evolution to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_attention_entropy(attention_maps: List[torch.Tensor]) -> List[float]:
    """
    Compute entropy of attention distributions
    Lower entropy = more selective attention
    """
    entropies = []
    
    for attn_map in attention_maps:
        # attn_map: [B, num_heads, N, N]
        # Focus on attention from CLS token
        attn = attn_map[:, :, 0, 1:]  # [B, num_heads, num_patches]
        
        # Compute entropy
        attn = attn + 1e-8  # Numerical stability
        entropy = -(attn * torch.log(attn)).sum(dim=-1).mean()
        entropies.append(entropy.item())
    
    return entropies


def visualize_masking_strategy(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    save_path: Optional[str] = None,
):
    """
    Visualize adaptive masking vs random masking
    """
    model.eval()
    images = images[:2].to(device)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    for i in range(2):
        img = images[i:i+1]
        img_vis = img * std + mean
        img_vis = torch.clamp(img_vis, 0, 1)
        
        # Original
        axes[i, 0].imshow(img_vis[0].cpu().permute(1, 2, 0))
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Random masking
        model.adaptive_masking = False
        with torch.no_grad():
            x = model.patch_embed(img)
            x = x + model.pos_embed[:, 1:, :]
            _, mask_random, _ = model.random_masking(x, model.mask_ratio)
        
        mask_map = mask_random[0].cpu().reshape(int(np.sqrt(mask_random.shape[1])), -1)
        axes[i, 1].imshow(mask_map, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Random Masking')
        axes[i, 1].axis('off')
        
        # Adaptive masking
        model.adaptive_masking = True
        with torch.no_grad():
            x = model.patch_embed(img)
            x = x + model.pos_embed[:, 1:, :]
            _, mask_adaptive, _ = model.adaptive_masking_strategy(x, model.mask_ratio)
        
        mask_map = mask_adaptive[0].cpu().reshape(int(np.sqrt(mask_adaptive.shape[1])), -1)
        axes[i, 2].imshow(mask_map, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Adaptive Masking')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved masking strategy comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
):
    """Plot training and validation loss curves"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()

