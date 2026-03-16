"""
Configuration for CoMER HMER pipeline.
Based on: https://github.com/Green-Wood/CoMER
Adapted for single-GPU (RTX 5060 Ti 16GB).
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    """Data and preprocessing configuration."""
    # CoMER dataset directory
    comer_data_dir: str = "D:/Workplace/CoMER/data"

    # Legacy paths (for CSV format fallback)
    raw_dir: str = "dataset/raw"
    processed_dir: str = "dataset/processed"

    # Vocab
    vocab_path: str = "dataset/processed/vocab_comer.json"

    # Image settings — reduced for fast training on single GPU
    img_channels: int = 1       # grayscale
    h_lo: int = 16
    h_hi: int = 128
    w_lo: int = 16
    w_hi: int = 512

    # Sequence settings
    max_seq_len: int = 200

    # DataLoader — small images, large batches
    batch_size: int = 64
    max_batch_pixels: int = 4_000_000
    num_workers: int = 4        # pipeline CPU preprocessing with GPU compute
    pin_memory: bool = True

    # Augmentation
    augment: bool = True
    scale_aug: bool = True      # ScaleAugmentation(0.7, 1.4)
    scale_lo: float = 0.7
    scale_hi: float = 1.4


@dataclass
class ModelConfig:
    """CoMER model configuration (encoder + decoder)."""
    # Shared
    d_model: int = 256

    # Encoder (DenseNet)
    growth_rate: int = 24
    num_layers: int = 16        # dense blocks per stage

    # Decoder (Transformer)
    nhead: int = 8
    num_decoder_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.3

    # ARM (Attention Refinement Module)
    dc: int = 32                # intermediate channels
    cross_coverage: bool = True
    self_coverage: bool = True

    # Beam search
    beam_size: int = 10
    max_len: int = 200
    alpha: float = 1.0
    early_stopping: bool = False
    temperature: float = 1.0


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 300          # CoMER default

    # Adam (CoMER original uses SGD 0.08 with 4 GPUs DDP)
    # Adam 1e-4 is more stable for single GPU training
    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # LR scheduler: ReduceLROnPlateau on val_ExpRate
    patience: int = 10
    lr_factor: float = 0.5     # multiply LR by this on plateau

    # Validation frequency
    val_every_n_epoch: int = 1

    # Gradient clipping
    grad_clip: float = 5.0

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    # Logging
    log_interval: int = 50

    # Output
    output_dir: str = "outputs"


@dataclass
class Config:
    """Full configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Device
    device: str = "cuda"

    # Random seed
    seed: int = 7  # ICAL uses seed 7

    def __post_init__(self):
        """Ensure directories exist."""
        os.makedirs(self.data.processed_dir, exist_ok=True)
        os.makedirs(self.train.checkpoint_dir, exist_ok=True)
        os.makedirs(self.train.output_dir, exist_ok=True)
