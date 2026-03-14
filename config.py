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
    # Reduced elastic deformation: large alpha distorts math symbols (∑→∫)
    elastic_alpha: float = 8.0
    elastic_sigma: float = 6.0        # Smoother deformation
    erosion_dilation_prob: float = 0.15  # Halved: thick strokes break thin symbols
    erosion_dilation_kernel: int = 2


@dataclass
class EncoderConfig:
    """DenseNet encoder configuration."""
    in_channels: int = 1
    # Reduced growth_rate: fewer channels per layer → spatial downsampling is less aggressive
    growth_rate: int = 24
    # Lighter block config: fewer layers in blocks 3 & 4 → preserves spatial resolution
    block_config: Tuple[int, ...] = (6, 12, 16, 8)
    num_init_features: int = 64
    bn_size: int = 4
    drop_rate: float = 0.2
    # Higher compression: keeps more features through transition layers (was 0.5)
    compression: float = 0.8
    # 2 transitions -> 256 encoder tokens (8x32) for 128x512 input.
    # 3 transitions (original) -> only 64 tokens (4x16), too sparse for long expressions.
    num_transitions: int = 2
    # GroupNorm groups (replaces BatchNorm to avoid curriculum augmentation distribution shift)
    num_groups: int = 32


@dataclass
class DecoderConfig:
    """CoMER Transformer decoder configuration."""
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.3
    max_seq_len: int = 200
    # ARM (Attention Refinement Module) — CoMER ECCV 2022
    arm_kernel_size: int = 5    # 2D conv kernel for coverage map (paper: 5)
    arm_d_coverage: int = 32    # intermediate channels in ARM (paper: 32)
    # Counting module dropout
    counting_dropout: float = 0.1
    # Multi-scale counting common channel dim (CAN, ECCV 2022)
    counting_common_channels: int = 128
    # Bidirectional training (BTTR, ICCV 2021)
    bidirectional: bool = True


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 200
    lr: float = 2e-4
    min_lr: float = 1e-7
    weight_decay: float = 1e-4
    # Reduced warmup: decay starts sooner so LR reaches min_lr before early stopping
    warmup_epochs: int = 5
    label_smoothing: float = 0.1

    # Cosine annealing cycle length in epochs (with restarts).
    # Shorter than total epochs so each cycle completes fully even with early stopping.
    lr_cycle_epochs: int = 50

    # Gradient clipping
    grad_clip: float = 5.0

    # Mixed precision
    use_amp: bool = True

    # Early stopping — increased to give time after LR restarts
    patience: int = 30

    # CTC auxiliary loss weight (0 to disable).
    ctc_weight: float = 0.1
    # Ramp CTC weight from 0 → ctc_weight over this many epochs.
    ctc_warmup_epochs: int = 25

    # Counting auxiliary loss weight (binary-CE for symbol presence)
    # Inspired by CAN (ECCV 2022): weakly supervised counting signal
    counting_weight: float = 0.05

    # Curriculum augmentation: ramp aug probability from 0→1 over this many epochs.
    aug_warmup_epochs: int = 30

    # Scale augmentation heights (Li et al. ICFHR 2020).
    # During training, each image is randomly resized to one of these heights
    # (maintaining aspect ratio) before padding. Helps with multi-scale symbols.
    # Set to empty list to disable (use fixed img_height only).
    scale_heights: Tuple[int, ...] = (96, 112, 128, 144)

    # LR restart decay: multiply lr_max by this factor on each cosine cycle restart.
    # Run 2: decay=0.5 → cycle-2 peak=1e-4. Still overshoots: ep56 exprate -5pp.
    # Run 3: decay=0.5 still overshoots at ep56 (23.3% from 28.6%).
    # 0.3 → cycle-2 peak=6e-5, safe for fine-tuning without losing progress.
    lr_restart_decay: float = 0.3

    # Mini warmup steps within each cosine restart (in epochs).
    # Without this, the LR jumps immediately to the cycle peak on the first
    # batch after a restart, causing a gradient spike that overshoots the
    # current minimum. A 2-epoch ramp smooths the re-entry.
    lr_restart_warmup_epochs: int = 2

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