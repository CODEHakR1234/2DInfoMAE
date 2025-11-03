"""
Main training script for InfoMAE
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_config, Config
from models.infomae import InfoMAE
from data.datasets import build_dataset, build_dataloader
from engine import Trainer, LinearProbe
from utils.visualization import (
    visualize_reconstruction,
    visualize_attention_maps,
    plot_training_curves,
    visualize_masking_strategy,
)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(config: Config) -> InfoMAE:
    """Build InfoMAE model"""
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
    
    return model


def build_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    """Build optimizer with layer-wise learning rates"""
    
    # Separate encoder and decoder parameters
    if config.model.freeze_encoder:
        # Freeze encoder completely
        for param in model.patch_embed.parameters():
            param.requires_grad = False
        for param in model.blocks.parameters():
            param.requires_grad = False
        for param in model.norm.parameters():
            param.requires_grad = False
        
        # Only train decoder
        param_groups = [
            {'params': model.decoder_embed.parameters()},
            {'params': model.decoder_blocks.parameters()},
            {'params': model.decoder_norm.parameters()},
            {'params': model.decoder_pred.parameters()},
            {'params': [model.mask_token, model.decoder_pos_embed]},
        ]
    
    elif config.model.unfreeze_last_n_blocks > 0:
        # Freeze all encoder except last N blocks
        for param in model.patch_embed.parameters():
            param.requires_grad = False
        
        n_blocks = len(model.blocks)
        freeze_until = n_blocks - config.model.unfreeze_last_n_blocks
        
        for i, block in enumerate(model.blocks):
            if i < freeze_until:
                for param in block.parameters():
                    param.requires_grad = False
        
        # Create parameter groups with different learning rates
        frozen_blocks = list(model.blocks[:freeze_until].parameters())
        trainable_blocks = list(model.blocks[freeze_until:].parameters())
        
        param_groups = [
            {'params': trainable_blocks, 'lr': config.training.encoder_lr},
            {'params': model.norm.parameters(), 'lr': config.training.encoder_lr},
            {'params': model.decoder_embed.parameters()},
            {'params': model.decoder_blocks.parameters()},
            {'params': model.decoder_norm.parameters()},
            {'params': model.decoder_pred.parameters()},
        ]
    
    else:
        # Train everything
        param_groups = [
            {'params': model.parameters()},
        ]
    
    # Build optimizer
    if config.training.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.training.lr,
            betas=config.training.betas,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=config.training.lr,
            betas=config.training.betas,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")
    
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, config: Config):
    """Build learning rate scheduler"""
    if config.training.scheduler == 'cosine':
        # Ensure T_max is at least 1 to avoid division by zero
        T_max = max(1, config.training.epochs - config.training.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=config.training.min_lr,
        )
    elif config.training.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1,
        )
    else:
        scheduler = None
    
    return scheduler


def load_pretrained_mae(model: InfoMAE, pretrained_path: str):
    """Load pretrained MAE weights"""
    if not os.path.exists(pretrained_path):
        print(f"Warning: Pretrained weights not found at {pretrained_path}")
        print("Starting from random initialization...")
        return model
    
    print(f"Loading pretrained MAE from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Extract model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load with strict=False to allow for architecture differences
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights: {msg}")
    
    return model


def train(config: Config):
    """Main training function"""
    
    # Set seed
    set_seed(config.seed)
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build datasets
    print("Building datasets...")
    train_dataset = build_dataset(
        config.data.dataset,
        config.data.data_dir,
        split='train',
        img_size=config.model.img_size,
        augment=True,
    )
    
    val_dataset = build_dataset(
        config.data.dataset,
        config.data.data_dir,
        split='val',
        img_size=config.model.img_size,
        augment=False,
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Build dataloaders
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
        drop_last=True,
    )
    
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
        drop_last=False,
    )
    
    # Build model
    print("Building model...")
    model = build_model(config)
    
    # Load pretrained weights if specified
    if config.model.pretrained_path:
        model = load_pretrained_mae(model, config.model.pretrained_path)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    
    # Initialize wandb if enabled
    logger = None
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.exp_name,
                config=vars(config),
            )
            logger = wandb
            print("Initialized Weights & Biases logging")
        except ImportError:
            print("wandb not installed, skipping logging")
    
    # Build trainer
    trainer = Trainer(model, optimizer, scheduler, device, config)
    
    # Resume from checkpoint if specified
    if config.training.resume:
        trainer.load_checkpoint(config.training.resume)
    
    # Training loop
    print(f"\nStarting training for {config.training.epochs} epochs...")
    train_losses = []
    val_losses = []
    val_epochs = []  # Track which epochs we evaluated on
    
    for epoch in range(trainer.epoch + 1, config.training.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.training.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch, logger)
        train_losses.append(train_metrics['loss'])
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Recon: {train_metrics['recon_loss']:.4f}, "
              f"MI: {train_metrics['mi']:.4f}")
        
        # Evaluate
        if epoch % config.training.eval_freq == 0:
            val_metrics = trainer.evaluate(val_loader, logger)
            val_losses.append(val_metrics['loss'])
            val_epochs.append(epoch)  # Record the epoch number
            
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Recon: {val_metrics['recon_loss']:.4f}, "
                  f"MI: {val_metrics['mi']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < trainer.best_loss
            if is_best:
                trainer.best_loss = val_metrics['loss']
            
            if epoch % config.training.save_freq == 0 or is_best:
                checkpoint_path = os.path.join(
                    config.output_dir,
                    'checkpoints',
                    f'checkpoint_epoch_{epoch}.pth'
                )
                trainer.save_checkpoint(checkpoint_path, is_best)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        # Visualizations
        if config.eval.visualize_attention and epoch % (config.training.save_freq * 2) == 0:
            print("\nGenerating visualizations...")
            
            # Get a batch for visualization
            sample_images, _ = next(iter(val_loader))
            
            # Reconstruction visualization
            vis_path = os.path.join(
                config.output_dir,
                'visualizations',
                f'reconstruction_epoch_{epoch}.png'
            )
            visualize_reconstruction(model, sample_images, device, vis_path)
            
            # Attention maps
            attn_path = os.path.join(
                config.output_dir,
                'visualizations',
                f'attention_epoch_{epoch}.png'
            )
            visualize_attention_maps(model, sample_images, device, attn_path)
            
            # Masking strategy
            if config.model.adaptive_masking:
                mask_path = os.path.join(
                    config.output_dir,
                    'visualizations',
                    f'masking_epoch_{epoch}.png'
                )
                visualize_masking_strategy(model, sample_images, device, mask_path)
    
    # Plot training curves
    curve_path = os.path.join(config.output_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, curve_path, val_epochs)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {trainer.best_loss:.4f}")
    print(f"Results saved to: {config.output_dir}")
    print(f"{'='*60}")
    
    # Close wandb
    if logger:
        wandb.finish()


def evaluate_linear_probe(config: Config):
    """Evaluate with linear probe"""
    print("Running linear probe evaluation...")
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    # Build datasets
    train_dataset = build_dataset(
        config.data.dataset,
        config.data.data_dir,
        split='train',
        img_size=config.model.img_size,
        augment=False,
    )
    
    val_dataset = build_dataset(
        config.data.dataset,
        config.data.data_dir,
        split='val',
        img_size=config.model.img_size,
        augment=False,
    )
    
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.eval.probe_batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
    )
    
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.eval.probe_batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
    )
    
    # Load model
    model = build_model(config).to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.output_dir, 'checkpoints', 'checkpoint_epoch_best.pth')
    if not os.path.exists(checkpoint_path):
        # Try latest checkpoint
        checkpoints = [f for f in os.listdir(os.path.join(config.output_dir, 'checkpoints'))
                      if f.endswith('.pth')]
        if checkpoints:
            checkpoint_path = os.path.join(config.output_dir, 'checkpoints', sorted(checkpoints)[-1])
        else:
            print("No checkpoint found!")
            return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Determine number of classes
    num_classes = 100 if config.data.dataset in ['imagenet100', 'cifar100'] else 10
    
    # Build linear probe
    probe = LinearProbe(model, num_classes, device)
    probe_optimizer = torch.optim.Adam(probe.classifier.parameters(), lr=config.eval.probe_lr)
    
    # Train linear probe
    print(f"\nTraining linear probe for {config.eval.probe_epochs} epochs...")
    best_acc = 0.0
    
    for epoch in range(1, config.eval.probe_epochs + 1):
        train_loss, train_acc = probe.train_epoch(train_loader, probe_optimizer)
        
        if epoch % 10 == 0:
            val_loss, val_acc = probe.evaluate(val_loader)
            print(f"Epoch {epoch}/{config.eval.probe_epochs} - "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
    
    # Final evaluation
    val_loss, val_acc = probe.evaluate(val_loader)
    print(f"\n{'='*60}")
    print(f"Linear Probe Results:")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='InfoMAE Training')
    parser.add_argument('--stage', type=str, default='stage1',
                       choices=['stage0', 'stage1', 'stage2', 'stage3'],
                       help='Training stage')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'probe'],
                       help='Mode: train or evaluate')
    parser.add_argument('--dataset', type=str, default='imagenet100',
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained MAE weights')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Get config for stage
    config = get_config(args.stage)
    
    # Override with command line arguments
    if args.dataset:
        config.data.dataset = args.dataset
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.pretrained:
        config.model.pretrained_path = args.pretrained
    if args.resume:
        config.training.resume = args.resume
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.lr = args.lr
    if args.seed:
        config.seed = args.seed
    if args.no_wandb:
        config.use_wandb = False
    
    # Create output directory
    config.__post_init__()
    
    # Print config
    print("="*60)
    print(f"InfoMAE - {args.stage.upper()}")
    print("="*60)
    print(f"Dataset: {config.data.dataset}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.lr}")
    print(f"Surprisal Attention: {config.model.use_surprisal_attention}")
    print(f"Adaptive Masking: {config.model.adaptive_masking}")
    print("="*60)
    
    # Run
    if args.mode == 'train':
        train(config)
    elif args.mode == 'probe':
        evaluate_linear_probe(config)
    elif args.mode == 'eval':
        print("Evaluation mode not fully implemented yet")
        evaluate_linear_probe(config)


if __name__ == '__main__':
    main()

