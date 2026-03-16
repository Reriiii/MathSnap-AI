"""
Configuration for ICAL HMER pipeline.
Architecture copied from: https://github.com/qingzhenduyu/ICAL
Hyperparams adapted for single-GPU (RTX 5060 Ti 16GB).
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

    # Image settings — ICAL uses variable size with limits
    img_height: int = 128       # target resize height (used in dataset)
    img_max_width: int = 512    # safety cap
    img_channels: int = 1       # grayscale

    # ICAL ScaleToLimitRange bounds
    h_lo: int = 16
    h_hi: int = 256
    w_lo: int = 16
    w_hi: int = 1024

    # Sequence settings
    max_seq_len: int = 200

    # DataLoader — ICAL uses pre-grouped batching by image area
    batch_size: int = 8         # per-GPU batch (ICAL default)
    max_batch_pixels: int = 320000  # max_size in ICAL
    num_workers: int = 5
    pin_memory: bool = True

    # Augmentation — ICAL uses only scale augmentation
    augment: bool = True
    scale_aug: bool = True      # ScaleAugmentation(0.7, 1.4)
    scale_lo: float = 0.7
    scale_hi: float = 1.4


@dataclass
class ModelConfig:
    """ICAL model configuration (encoder + decoder)."""
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
    """Training configuration — matches ICAL."""
    epochs: int = 250

    # ICAL uses SGD(lr=0.08, momentum=0.9, weight_decay=1e-4) on 4 GPUs.
    # Single GPU: scale down lr by 4x (linear scaling rule).
    optimizer: str = "sgd"
    lr: float = 0.02            # 0.08 / 4 (1 GPU vs 4 GPUs)
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # LR scheduler: ReduceLROnPlateau on val_ExpRate
    patience: int = 20
    lr_factor: float = 0.25     # multiply LR by this on plateau

    # Validation frequency — ICAL uses check_val_every_n_epoch=2
    val_every_n_epoch: int = 2

    # Dynamic weight for implicit loss (ICAL default: True)
    dynamic_weight: bool = True

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
