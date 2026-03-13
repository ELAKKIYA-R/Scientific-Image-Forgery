"""
Configuration for Scientific Image Forgery Detection
"""
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """Data-related configuration"""
    # Paths
    data_root: Path = Path("recodai-luc-scientific-image-forgery-detection")
    train_images_dir: Path = field(default_factory=lambda: Path("recodai-luc-scientific-image-forgery-detection/train_images"))
    train_masks_dir: Path = field(default_factory=lambda: Path("recodai-luc-scientific-image-forgery-detection/train_masks"))
    supplemental_images_dir: Path = field(default_factory=lambda: Path("recodai-luc-scientific-image-forgery-detection/supplemental_images"))
    supplemental_masks_dir: Path = field(default_factory=lambda: Path("recodai-luc-scientific-image-forgery-detection/supplemental_masks"))
    test_images_dir: Path = field(default_factory=lambda: Path("recodai-luc-scientific-image-forgery-detection/test_images"))
    
    # Split data (from clustering)
    split_data_dir: Path = Path("split_data")
    
    # Image settings
    image_size: Tuple[int, int] = (512, 512)  # (H, W)
    num_mask_channels: int = 1  # Single channel mask (max across all channels)
    combine_mask_channels: bool = True  # Always True - masks converted to single channel
    
    # Data split
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    
    # Extensions
    image_extensions: List[str] = field(default_factory=lambda: ['.png', '.jpg', '.jpeg'])


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Backbone
    encoder_name: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    encoder_pretrained: bool = True
    
    # Feature dimensions (SegFormer-B2)
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 320, 512])
    
    # Decoder
    decoder_channels: int = 256
    
    # Self-correlation module
    use_self_correlation: bool = True
    correlation_feature_scale: int = 2  # Which encoder stage to use (0-3)
    correlation_temperature: float = 0.07
    
    # Output heads
    num_mask_classes: int = 1  # Single channel mask (max across all channels)
    num_classification_classes: int = 2  # authentic vs forged
    
    # Dropout
    dropout_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Batch size (conservative for 16GB M2)
    batch_size: int = 4
    num_workers: int = 4
    
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Training duration
    num_epochs: int = 100
    early_stopping_patience: int = 15
    
    # Loss weights
    segmentation_loss_weight: float = 1.0
    classification_loss_weight: float = 0.5
    correlation_loss_weight: float = 0.1
    
    # Focal loss settings for classification
    # If None, uses balanced (0.5, 0.5)
    # If tuple (alpha_authentic, alpha_forged), uses per-class weighting
    # Example: (0.25, 0.75) gives more weight to forged class (if it's minority)
    focal_alpha: Optional[Tuple[float, float]] = None
    
    # Dice loss settings
    dice_smooth: float = 1.0
    
    # Gradient settings
    gradient_clip_val: float = 1.0
    use_gradient_checkpointing: bool = True
    
    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_top_k: int = 3
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    
    # Logging
    log_dir: Path = Path("logs")
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1
    
    # Reproducibility
    seed: int = 42


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    # Geometric
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotate_prob: float = 0.3
    rotate_limit: int = 45
    
    # Color
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    
    # Noise
    gaussian_noise_prob: float = 0.2
    gaussian_blur_prob: float = 0.1
    
    # Advanced (for copy-move robustness)
    jpeg_compression_prob: float = 0.3
    jpeg_quality_range: Tuple[int, int] = (70, 95)


@dataclass 
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    # Experiment name
    experiment_name: str = "cmfd_segformer_v1"
    
    def __post_init__(self):
        """Create necessary directories"""
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training.log_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get default configuration"""
    return Config()


# Quick access to default config
DEFAULT_CONFIG = Config()

