"""
Evaluation metrics for InfoMAE
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from typing import Dict, Tuple


def compute_reconstruction_metrics(
    images: torch.Tensor,
    reconstructions: torch.Tensor,
    mask: torch.Tensor = None,
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics
    
    Args:
        images: original images [B, C, H, W]
        reconstructions: reconstructed images [B, C, H, W]
        mask: optional mask [B, L]
    
    Returns:
        Dictionary of metrics
    """
    # MSE
    mse = ((images - reconstructions) ** 2).mean().item()
    
    # PSNR
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))
    
    # SSIM (simplified version)
    # For full SSIM, use skimage or pytorch-msssim
    
    metrics = {
        'mse': mse,
        'psnr': psnr,
    }
    
    return metrics


def compute_saliency_metrics(
    predicted_saliency: np.ndarray,
    ground_truth_saliency: np.ndarray,
    fixation_points: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute saliency prediction metrics
    
    Args:
        predicted_saliency: predicted saliency map [H, W]
        ground_truth_saliency: ground truth saliency map [H, W]
        fixation_points: binary fixation map [H, W] (optional)
    
    Returns:
        Dictionary of saliency metrics
    """
    # Normalize
    pred_norm = (predicted_saliency - predicted_saliency.min()) / \
                (predicted_saliency.max() - predicted_saliency.min() + 1e-8)
    gt_norm = (ground_truth_saliency - ground_truth_saliency.min()) / \
              (ground_truth_saliency.max() - ground_truth_saliency.min() + 1e-8)
    
    # Pearson correlation coefficient (CC)
    cc, _ = pearsonr(pred_norm.flatten(), gt_norm.flatten())
    
    # KL divergence
    pred_dist = pred_norm.flatten() + 1e-8
    gt_dist = gt_norm.flatten() + 1e-8
    pred_dist /= pred_dist.sum()
    gt_dist /= gt_dist.sum()
    kl_div = (gt_dist * np.log(gt_dist / pred_dist)).sum()
    
    metrics = {
        'cc': cc,
        'kl_div': kl_div,
    }
    
    # If fixation points are provided
    if fixation_points is not None:
        # NSS (Normalized Scanpath Saliency)
        pred_normalized = (predicted_saliency - predicted_saliency.mean()) / \
                         (predicted_saliency.std() + 1e-8)
        nss = pred_normalized[fixation_points > 0].mean()
        
        # AUC-Judd
        fixation_flat = fixation_points.flatten()
        pred_flat = predicted_saliency.flatten()
        
        if fixation_flat.sum() > 0:
            try:
                auc_judd = roc_auc_score(fixation_flat, pred_flat)
            except:
                auc_judd = 0.5
        else:
            auc_judd = 0.5
        
        metrics['nss'] = nss
        metrics['auc_judd'] = auc_judd
    
    return metrics


def compute_attention_selectivity(attention_maps: torch.Tensor) -> Dict[str, float]:
    """
    Compute attention selectivity metrics
    
    Args:
        attention_maps: attention weights [B, num_heads, N, N]
    
    Returns:
        Dictionary of selectivity metrics
    """
    # Focus on attention from CLS token
    cls_attention = attention_maps[:, :, 0, 1:]  # [B, num_heads, num_patches]
    
    # Entropy (lower = more selective)
    attn = cls_attention + 1e-8
    entropy = -(attn * torch.log(attn)).sum(dim=-1).mean().item()
    
    # Gini coefficient (higher = more selective)
    # Sort attention values
    attn_sorted, _ = torch.sort(cls_attention, dim=-1)
    n = attn_sorted.shape[-1]
    index = torch.arange(1, n + 1, device=attn_sorted.device).float()
    gini = ((2 * index - n - 1) * attn_sorted).sum(dim=-1) / (n * attn_sorted.sum(dim=-1) + 1e-8)
    gini = gini.mean().item()
    
    # Top-k concentration (what % of attention goes to top 10% of patches)
    k = int(0.1 * cls_attention.shape[-1])
    top_k_attn, _ = torch.topk(cls_attention, k, dim=-1)
    top_k_concentration = top_k_attn.sum(dim=-1).mean().item()
    
    metrics = {
        'entropy': entropy,
        'gini': gini,
        'top_k_concentration': top_k_concentration,
    }
    
    return metrics


def compute_representation_quality(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute representation quality metrics
    
    Args:
        features: extracted features [N, D]
        labels: class labels [N]
    
    Returns:
        Dictionary of quality metrics
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Convert to numpy
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Split into train/test
    n = len(features_np)
    train_idx = np.random.choice(n, int(0.8 * n), replace=False)
    test_idx = np.setdiff1d(np.arange(n), train_idx)
    
    X_train, X_test = features_np[train_idx], features_np[test_idx]
    y_train, y_test = labels_np[train_idx], labels_np[test_idx]
    
    # KNN accuracy
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train, y_train)
    knn_acc = knn.score(X_test, y_test)
    
    # Linear probe accuracy (quick version)
    lr = LogisticRegression(max_iter=100, random_state=42)
    lr.fit(X_train, y_train)
    linear_acc = lr.score(X_test, y_test)
    
    metrics = {
        'knn_acc': knn_acc,
        'linear_acc': linear_acc,
    }
    
    return metrics


def aggregate_metrics(metrics_list: list) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate metrics across multiple evaluations
    
    Returns:
        Dictionary with (mean, std) for each metric
    """
    if not metrics_list:
        return {}
    
    # Get all keys
    keys = metrics_list[0].keys()
    
    aggregated = {}
    for key in keys:
        values = [m[key] for m in metrics_list]
        aggregated[key] = (np.mean(values), np.std(values))
    
    return aggregated

