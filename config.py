"""
Configuration for InfoMAE experiments
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Base model
    base_model: str = "vit_base_patch16_224"  # timm model name
    pretrained: bool = True
    pretrained_path: Optional[str] = None  # path to MAE pretrained weights
    
    # Architecture
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    mlp_ratio: float = 4.0
    
    # Masking
    mask_ratio: float = 0.75
    adaptive_masking: bool = True
    masking_alpha: float = 3.0  # adaptive masking parameter
    masking_gamma_start: float = 0.5
    masking_gamma_end: float = 1.5
    
    # Surprisal-Weighted Attention
    use_surprisal_attention: bool = True
    lambda_start: float = 0.0
    lambda_end: float = 1.5
    lambda_warmup_epochs: int = 20
    
    # Encoder fine-tuning
    freeze_encoder: bool = False
    unfreeze_last_n_blocks: int = 2
    encoder_lr_scale: float = 0.1  # encoder lr = base_lr * scale


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic
    epochs: int = 200
    warmup_epochs: int = 10
    batch_size: int = 256
    num_workers: int = 8
    
    # Optimizer
    optimizer: str = "adamw"
    lr: float = 1e-4
    encoder_lr: float = 1e-5
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.95)
    
    # Scheduler
    scheduler: str = "cosine"
    min_lr: float = 1e-6
    
    # Loss weights
    beta_ib: float = 0.02  # information bottleneck weight
    
    # Regularization
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    
    # Checkpointing
    save_freq: int = 10
    eval_freq: int = 5
    log_freq: int = 50
    
    # Resume
    resume: Optional[str] = None
    auto_resume: bool = True


@dataclass
class DataConfig:
    """Dataset configuration"""
    # Dataset
    dataset: str = "imagenet100"  # imagenet100, cifar100, stl10
    data_dir: str = "./data"
    
    # ImageNet
    imagenet_path: str = "./data/imagenet"
    imagenet100_classes: Optional[List[int]] = None
    
    # Augmentation
    color_jitter: float = 0.4
    aa: str = "rand-m9-mstd0.5-inc1"  # AutoAugment
    reprob: float = 0.25  # Random Erasing
    remode: str = "pixel"
    recount: int = 1
    
    # Normalization (ImageNet stats)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    # Linear probe
    linear_probe: bool = True
    probe_epochs: int = 100
    probe_lr: float = 1e-3
    probe_batch_size: int = 256
    
    # Attention analysis
    visualize_attention: bool = True
    num_vis_samples: int = 50
    
    # Saliency evaluation (MIT300, OSIE)
    eval_saliency: bool = False
    saliency_dataset: str = "mit300"
    saliency_path: Optional[str] = None


@dataclass
class Config:
    """Main configuration"""
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Experiment
    exp_name: str = "infomae_default"
    output_dir: str = "./outputs"
    seed: int = 42
    
    # Hardware
    device: str = "cuda"
    distributed: bool = False
    world_size: int = 1
    local_rank: int = 0
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "InfoMAE"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        """Create output directory"""
        self.output_dir = os.path.join(self.output_dir, self.exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)


def get_config(stage: str = "stage1") -> Config:
    """Get configuration for different experiment stages"""
    config = Config()
    
    if stage == "stage0":
        # Baseline MAE fine-tuning
        config.exp_name = "stage0_baseline"
        config.model.use_surprisal_attention = False
        config.model.adaptive_masking = False
        config.model.freeze_encoder = True
        config.training.beta_ib = 0.0
        
    elif stage == "stage1":
        # Add Surprisal-Weighted Attention
        config.exp_name = "stage1_swa"
        config.model.use_surprisal_attention = True
        config.model.adaptive_masking = False
        config.model.freeze_encoder = True
        config.training.beta_ib = 0.0
        
    elif stage == "stage2":
        # Add Adaptive Masking
        config.exp_name = "stage2_adaptive"
        config.model.use_surprisal_attention = True
        config.model.adaptive_masking = True
        config.model.freeze_encoder = True
        config.training.beta_ib = 0.02
        
    elif stage == "stage3":
        # Unfreeze encoder partially
        config.exp_name = "stage3_finetune"
        config.model.use_surprisal_attention = True
        config.model.adaptive_masking = True
        config.model.freeze_encoder = False
        config.model.unfreeze_last_n_blocks = 2
        config.training.beta_ib = 0.02
        
    return config

