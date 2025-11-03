"""
Comprehensive evaluation script for InfoMAE
"""
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import get_config
from models.infomae import InfoMAE
from data.datasets import build_dataset, build_dataloader
from utils.visualization import (
    visualize_reconstruction,
    visualize_attention_maps,
    visualize_masking_strategy,
)
from utils.metrics import (
    compute_reconstruction_metrics,
    compute_attention_selectivity,
)


def evaluate_model(config, checkpoint_path):
    """Comprehensive model evaluation"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
    print("Building model...")
    model = InfoMAE(
        img_size=config.model.img_size,
        patch_size=config.model.patch_size,
        in_chans=config.model.in_chans,
        embed_dim=config.model.embed_dim,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        decoder_embed_dim=config.model.decoder_embed_dim,
        decoder_depth=config.model.decoder_depth,
        decoder_num_heads=config.model.decoder_num_heads,
        mlp_ratio=config.model.mlp_ratio,
        mask_ratio=config.model.mask_ratio,
        use_surprisal_attention=config.model.use_surprisal_attention,
        adaptive_masking=config.model.adaptive_masking,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Build dataset
    print("Loading dataset...")
    val_dataset = build_dataset(
        config.data.dataset,
        config.data.data_dir,
        split='val',
        img_size=config.model.img_size,
        augment=False,
    )
    
    val_loader = build_dataloader(
        val_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )
    
    # Evaluation metrics
    all_metrics = {
        'recon_loss': [],
        'surprisal_mean': [],
        'surprisal_std': [],
        'mi': [],
    }
    
    print("\nEvaluating...")
    with torch.no_grad():
        for images, _ in tqdm(val_loader):
            images = images.to(device)
            
            # Forward pass
            loss, pred, mask, surprisal, latent = model(
                images,
                mask_ratio=config.model.mask_ratio,
                lambda_weight=config.model.lambda_end,
                alpha=config.model.masking_alpha,
                gamma=config.model.masking_gamma_end,
            )
            
            # Reconstruction metrics
            recon = model.unpatchify(pred)
            recon_metrics = compute_reconstruction_metrics(images, recon)
            
            # Track metrics
            all_metrics['recon_loss'].append(loss.item())
            all_metrics['surprisal_mean'].append(surprisal.mean().item())
            all_metrics['surprisal_std'].append(surprisal.std().item())
    
    # Aggregate metrics
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    for key, values in all_metrics.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Attention analysis
    print("\nAnalyzing attention patterns...")
    sample_images, _ = next(iter(val_loader))
    sample_images = sample_images[:8].to(device)
    
    attention_maps = model.get_attention_maps(sample_images)
    if attention_maps:
        attn_metrics = compute_attention_selectivity(attention_maps[-1])  # Last layer
        print("\nAttention Metrics:")
        for key, value in attn_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    vis_dir = os.path.join(config.output_dir, 'eval_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Reconstruction
    visualize_reconstruction(
        model, sample_images, device,
        save_path=os.path.join(vis_dir, 'reconstruction.png'),
        num_samples=4
    )
    
    # Attention maps
    visualize_attention_maps(
        model, sample_images, device,
        save_path=os.path.join(vis_dir, 'attention_maps.png')
    )
    
    # Masking strategy
    if config.model.adaptive_masking:
        visualize_masking_strategy(
            model, sample_images, device,
            save_path=os.path.join(vis_dir, 'masking_strategy.png')
        )
    
    print(f"\nVisualizations saved to: {vis_dir}")
    print("="*60)
    
    return all_metrics


def compare_stages(output_dir='./outputs'):
    """Compare different training stages"""
    
    stages = ['stage0_baseline', 'stage1_swa', 'stage2_adaptive', 'stage3_finetune']
    results = {}
    
    for stage in stages:
        stage_dir = os.path.join(output_dir, stage)
        checkpoint_path = os.path.join(stage_dir, 'checkpoints', 'checkpoint_epoch_best.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found for {stage}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {stage}")
        print(f"{'='*60}")
        
        config = get_config(stage.split('_')[0])
        config.output_dir = stage_dir
        
        metrics = evaluate_model(config, checkpoint_path)
        results[stage] = metrics
    
    # Plot comparison
    if results:
        plot_stage_comparison(results, output_dir)


def plot_stage_comparison(results, output_dir):
    """Plot comparison across stages"""
    
    stages = list(results.keys())
    metrics_to_plot = ['recon_loss', 'surprisal_mean']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(12, 4))
    
    for idx, metric in enumerate(metrics_to_plot):
        values = [np.mean(results[stage][metric]) for stage in stages]
        errors = [np.std(results[stage][metric]) for stage in stages]
        
        axes[idx].bar(range(len(stages)), values, yerr=errors, capsize=5)
        axes[idx].set_xticks(range(len(stages)))
        axes[idx].set_xticklabels([s.replace('_', '\n') for s in stages], rotation=0)
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'stage_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate InfoMAE')
    parser.add_argument('--stage', type=str, default='stage3',
                       help='Training stage to evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (if not using default)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all stages')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_stages(args.output_dir)
    else:
        # Evaluate single stage
        config = get_config(args.stage)
        
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = os.path.join(
                args.output_dir,
                f"{args.stage}_*",  # Will need to glob this
                'checkpoints',
                'checkpoint_epoch_best.pth'
            )
            
            # Find checkpoint
            import glob
            matches = glob.glob(checkpoint_path)
            if matches:
                checkpoint_path = matches[0]
            else:
                print(f"Checkpoint not found at {checkpoint_path}")
                return
        
        evaluate_model(config, checkpoint_path)


if __name__ == '__main__':
    main()

