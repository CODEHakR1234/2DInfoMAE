"""
Loss functions for InfoMAE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_mutual_information(z: torch.Tensor, s: torch.Tensor, bins: int = 50) -> torch.Tensor:
    """
    Estimate mutual information I(Z; S) using histogram-based approach
    
    Args:
        z: latent representations [B, N, D]
        s: surprisal values [B, N]
        bins: number of bins for histogram
        
    Returns:
        Mutual information estimate (scalar)
    """
    B, N, D = z.shape
    
    # Flatten
    z_flat = z.reshape(-1, D)  # [B*N, D]
    s_flat = s.reshape(-1)     # [B*N]
    
    # Use first principal component of z for simplicity
    # In practice, could use more sophisticated methods
    z_mean = z_flat.mean(dim=0, keepdim=True)
    z_centered = z_flat - z_mean
    
    # Simple approximation: use mean of latent as univariate representation
    z_uni = z_centered.mean(dim=1)  # [B*N]
    
    # Normalize to [0, 1]
    z_uni = (z_uni - z_uni.min()) / (z_uni.max() - z_uni.min() + 1e-8)
    s_norm = (s_flat - s_flat.min()) / (s_flat.max() - s_flat.min() + 1e-8)
    
    # Create 2D histogram
    z_indices = (z_uni * (bins - 1)).long().clamp(0, bins - 1)
    s_indices = (s_norm * (bins - 1)).long().clamp(0, bins - 1)
    
    # Joint histogram
    joint_hist = torch.zeros(bins, bins, device=z.device)
    for i in range(len(z_indices)):
        joint_hist[z_indices[i], s_indices[i]] += 1
    
    joint_hist = joint_hist / joint_hist.sum()  # Normalize
    
    # Marginal histograms
    p_z = joint_hist.sum(dim=1)  # P(Z)
    p_s = joint_hist.sum(dim=0)  # P(S)
    
    # Mutual information: I(Z;S) = sum P(z,s) log(P(z,s) / (P(z)P(s)))
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint_hist[i, j] > 0 and p_z[i] > 0 and p_s[j] > 0:
                mi += joint_hist[i, j] * torch.log(joint_hist[i, j] / (p_z[i] * p_s[j] + 1e-10) + 1e-10)
    
    return mi


def compute_mutual_information_kl(z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Alternative MI estimation using correlation-based approximation
    Faster but less accurate
    
    MI(Z;S) ≈ -0.5 * log(1 - ρ²) where ρ is correlation
    """
    B, N = s.shape
    D = z.shape[-1]
    
    # Pool latent to match surprisal dimension
    z_pooled = z.mean(dim=1)  # [B, D]
    
    # Average over batch dimension for simplicity
    # This gives us a per-sample correlation estimate
    s_flat = s.reshape(-1)  # [B*N]
    
    # Repeat z_pooled for each patch to match surprisal dimension
    z_flat = z_pooled.unsqueeze(1).expand(-1, N, -1).reshape(-1, D)  # [B*N, D]
    z_flat_mean = z_flat.mean(dim=-1)  # [B*N] - average across feature dimension
    
    # Compute correlation between surprisal and latent mean
    # Normalize
    s_norm = (s_flat - s_flat.mean()) / (s_flat.std() + 1e-8)
    z_norm = (z_flat_mean - z_flat_mean.mean()) / (z_flat_mean.std() + 1e-8)
    
    # Pearson correlation
    correlation = (s_norm * z_norm).mean()
    
    # MI approximation: MI ≈ -0.5 * log(1 - ρ²)
    # Clamp correlation to avoid log(0)
    correlation = torch.clamp(correlation, -0.999, 0.999)
    mi = -0.5 * torch.log(1 - correlation ** 2 + 1e-8)
    
    # Ensure non-negative
    mi = torch.clamp(mi, min=0.0)
    
    return mi


class InfoMAELoss(nn.Module):
    """
    Complete loss for InfoMAE
    L = ||x - x̂||² - β * I(Z; S)
    """
    def __init__(self, beta: float = 0.02, mi_estimator: str = 'simple'):
        super().__init__()
        self.beta = beta
        self.mi_estimator = mi_estimator
        
    def forward(self, imgs, pred, mask, surprisal, latent):
        """
        Args:
            imgs: original images [B, 3, H, W]
            pred: predicted patches [B, L, p²*3]
            mask: binary mask [B, L]
            surprisal: reconstruction error per patch [B, L]
            latent: encoder output [B, N, D]
            
        Returns:
            total_loss, recon_loss, mi_loss
        """
        # Reconstruction loss (already computed in model)
        # We recalculate here for consistency
        B, L, _ = pred.shape
        
        # Patchify images
        patch_size = int(np.sqrt(pred.shape[-1] // 3))
        imgs_patches = self.patchify(imgs, patch_size)
        
        # MSE loss
        recon_loss = (pred - imgs_patches) ** 2
        recon_loss = recon_loss.mean(dim=-1)  # [B, L]
        recon_loss = (recon_loss * mask).sum() / (mask.sum() + 1e-8)
        
        # Information bottleneck term: -β * I(Z; S)
        # We want to maximize I(Z; S), so minimize -I(Z; S)
        if self.beta > 0:
            if self.mi_estimator == 'simple':
                mi = compute_mutual_information_kl(latent, surprisal)
            else:
                mi = compute_mutual_information(latent, surprisal)
            mi_loss = -self.beta * mi
        else:
            mi_loss = torch.tensor(0.0, device=pred.device)
            mi = torch.tensor(0.0, device=pred.device)
        
        total_loss = recon_loss + mi_loss
        
        return total_loss, recon_loss, mi_loss, mi
    
    @staticmethod
    def patchify(imgs, patch_size):
        """Convert images to patches"""
        p = patch_size
        B, C, H, W = imgs.shape
        h = w = H // p
        
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(B, h * w, p**2 * C)
        return x

