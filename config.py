"""
Configuration for HMER DenseNet + Transformer pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    """Data and preprocessing configuration."""
    raw_dir: str = "dataset/raw"
    processed_dir: str = "dataset/processed"
    vocab_path: str = "dataset/processed/vocab.json"

    # Image settings
    img_height: int = 128
    img_max_width: int = 512
    img_channels: int = 1  # grayscale

    # Sequence settings
    max_seq_len: int = 200

    # DataLoader
    batch_size: int = 48
    num_workers: int = 8
    pin_memory: bool = True

    # Augmentation
    augment: bool = True
    rotation_range: float = 5.0       # degrees
    scale_range: Tuple[float, float] = (0.9, 1.1)
    shear_range: float = 0.1
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    noise_std: float = 0.02
    elastic_alpha: float = 30.0
    elastic_sigma: float = 4.0
    erosion_dilation_prob: float = 0.3
    erosion_dilation_kernel: int = 2


@dataclass
class EncoderConfig:
    """DenseNet encoder configuration."""
    in_channels: int = 1
    growth_rate: int = 32
    block_config: Tuple[int, ...] = (6, 12, 24, 16)  # DenseNet-121 style
    num_init_features: int = 64
    bn_size: int = 4
    drop_rate: float = 0.2
    compression: float = 0.5


@dataclass
class DecoderConfig:
    """Transformer decoder configuration."""
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.3
    max_seq_len: int = 200


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 200
    lr: float = 2e-4
    min_lr: float = 1e-7
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    label_smoothing: float = 0.1

    # Gradient clipping
    grad_clip: float = 5.0

    # Mixed precision
    use_amp: bool = True

    # Early stopping
    patience: int = 20

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    # Logging
    log_interval: int = 50  # log every N batches

    # Output
    output_dir: str = "outputs"

    # Beam search
    beam_size: int = 5


@dataclass
class Config:
    """Full configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Device
    device: str = "cuda"

    # Random seed
    seed: int = 42

    def __post_init__(self):
        """Ensure directories exist."""
        os.makedirs(self.data.processed_dir, exist_ok=True)
        os.makedirs(self.train.checkpoint_dir, exist_ok=True)
        os.makedirs(self.train.output_dir, exist_ok=True)
