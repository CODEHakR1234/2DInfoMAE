"""
Training and evaluation engine for InfoMAE
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import time
import datetime
from tqdm import tqdm
import numpy as np

from utils.losses import InfoMAELoss


class Trainer:
    """Training engine for InfoMAE"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        config,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = InfoMAELoss(beta=config.training.beta_ib)
        
        # Tracking
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
    def train_epoch(self, train_loader: DataLoader, epoch: int, logger=None) -> Dict:
        """Train for one epoch"""
        self.model.train()
        self.epoch = epoch
        
        # Dynamic parameters
        lambda_weight = self.get_lambda_weight(epoch)
        gamma = self.get_gamma(epoch)
        alpha = self.config.model.masking_alpha
        
        metrics = {
            'loss': 0.0,
            'recon_loss': 0.0,
            'mi_loss': 0.0,
            'mi': 0.0,
            'lambda': lambda_weight,
            'gamma': gamma,
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.training.epochs}")
        
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch (with or without image_ids)
            if len(batch_data) == 3:
                # (image_ids, images, labels) - from IndexedDataset
                image_ids, images, _ = batch_data
                image_ids = image_ids.to(self.device)
            else:
                # (images, labels) - regular dataset
                images, _ = batch_data
                image_ids = None
            
            images = images.to(self.device)
            
            # Forward pass (with epoch cache if image_ids provided)
            loss, pred, mask, surprisal, latent = self.model(
                images,
                mask_ratio=self.config.model.mask_ratio,
                lambda_weight=lambda_weight,
                alpha=alpha,
                gamma=gamma,
                image_ids=image_ids,  # ✅ Pass image_ids for epoch caching
            )
            
            # Compute full loss with IB term
            total_loss, recon_loss, mi_loss, mi = self.criterion(
                images, pred, mask, surprisal, latent
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip
                )
            
            self.optimizer.step()
            
            # Update metrics
            metrics['loss'] += total_loss.item()
            metrics['recon_loss'] += recon_loss.item()
            metrics['mi_loss'] += mi_loss.item()
            metrics['mi'] += mi.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'recon': recon_loss.item(),
                'mi': mi.item(),
                'λ': lambda_weight,
            })
            
            self.global_step += 1
            
            # Logging
            if logger and batch_idx % self.config.training.log_freq == 0:
                logger.log({
                    'train/loss': total_loss.item(),
                    'train/recon_loss': recon_loss.item(),
                    'train/mi_loss': mi_loss.item(),
                    'train/mi': mi.item(),
                    'train/lambda': lambda_weight,
                    'train/gamma': gamma,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch,
                    'train/step': self.global_step,
                })
        
        # Average metrics
        num_batches = len(train_loader)
        for key in metrics:
            if key not in ['lambda', 'gamma']:
                metrics[key] /= num_batches
        
        # Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, logger=None) -> Dict:
        """Evaluate on validation set"""
        self.model.eval()
        
        metrics = {
            'loss': 0.0,
            'recon_loss': 0.0,
            'mi_loss': 0.0,
            'mi': 0.0,
        }
        
        pbar = tqdm(val_loader, desc="Validation")
        
        for batch_data in pbar:
            # ✅ FIXED: Handle IndexedDataset (image_ids, images, labels)
            if len(batch_data) == 3:
                # (image_ids, images, labels) - from IndexedDataset
                image_ids, images, _ = batch_data
                image_ids = image_ids.to(self.device)
            else:
                # (images, labels) - from regular dataset
                images, _ = batch_data
                image_ids = None
            
            images = images.to(self.device)
            
            # Forward pass
            loss, pred, mask, surprisal, latent = self.model(
                images,
                image_ids=image_ids,  # Pass image_ids for epoch cache
                mask_ratio=self.config.model.mask_ratio,
                lambda_weight=self.config.model.lambda_end,  # Use final lambda for eval
                alpha=self.config.model.masking_alpha,
                gamma=self.config.model.masking_gamma_end,
            )
            
            # Compute full loss
            total_loss, recon_loss, mi_loss, mi = self.criterion(
                images, pred, mask, surprisal, latent
            )
            
            # Update metrics
            metrics['loss'] += total_loss.item()
            metrics['recon_loss'] += recon_loss.item()
            metrics['mi_loss'] += mi_loss.item()
            metrics['mi'] += mi.item()
            
            pbar.set_postfix({'loss': total_loss.item()})
        
        # Average metrics
        num_batches = len(val_loader)
        for key in metrics:
            metrics[key] /= num_batches
        
        # Logging
        if logger:
            logger.log({
                f'val/{key}': value for key, value in metrics.items()
            })
        
        return metrics
    
    def get_lambda_weight(self, epoch: int) -> float:
        """Get current lambda weight with warm-up"""
        warmup_epochs = self.config.model.lambda_warmup_epochs
        lambda_start = self.config.model.lambda_start
        lambda_end = self.config.model.lambda_end
        
        if epoch < warmup_epochs:
            return lambda_start + (lambda_end - lambda_start) * epoch / warmup_epochs
        else:
            return lambda_end
    
    def get_gamma(self, epoch: int) -> float:
        """Get current gamma for adaptive masking"""
        total_epochs = self.config.training.epochs
        gamma_start = self.config.model.masking_gamma_start
        gamma_end = self.config.model.masking_gamma_end
        
        # Linear schedule
        progress = epoch / total_epochs
        return gamma_start + (gamma_end - gamma_start) * progress
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'config': self.config,
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")


class LinearProbe:
    """Linear probe for evaluation"""
    
    def __init__(self, encoder: nn.Module, num_classes: int, device: torch.device):
        self.encoder = encoder
        self.device = device
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        # Linear classifier
        embed_dim = encoder.pos_embed.shape[-1]
        self.classifier = nn.Linear(embed_dim, num_classes).to(device)
        
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from encoder"""
        # Get encoder output
        x = self.encoder.patch_embed(images)
        x = x + self.encoder.pos_embed[:, 1:, :]
        
        # Add cls token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Forward through blocks
        for blk in self.encoder.blocks:
            if hasattr(blk, 'forward'):
                x = blk(x) if not self.encoder.use_surprisal_attention else blk(x, None, 0.0)
        
        x = self.encoder.norm(x)
        
        # Return cls token
        return x[:, 0]
    
    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Train linear probe for one epoch"""
        self.classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        for images, labels in tqdm(train_loader, desc="Linear probe training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Extract features
            features = self.extract_features(images)
            
            # Forward
            logits = self.classifier(features)
            loss = criterion(logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        return total_loss / len(train_loader), acc
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate linear probe"""
        self.classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        for images, labels in tqdm(val_loader, desc="Linear probe evaluation"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Extract features
            features = self.extract_features(images)
            
            # Forward
            logits = self.classifier(features)
            loss = criterion(logits, labels)
            
            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        return total_loss / len(val_loader), acc

