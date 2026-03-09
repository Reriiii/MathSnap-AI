from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """NAMER training configuration."""

    # ── Paths ──────────────────────────────────────────────────────────────
    data_root:  str          = 'D://dataset/HME100K'
    label_file: str          = 'D://dataset/HME100K/train.txt'
    checkpoint_dir: str      = './checkpoints'
    vocab_path: str          = './vocab.json'
    resume_checkpoint: Optional[str] = None
    # Path to pretrained DWAP (used for NAMER training).
    # Set to None to use uniform-spread position estimate instead.
    dwap_checkpoint: Optional[str]  = None

    # ── Split ──────────────────────────────────────────────────────────────
    train_ratio: float = 0.80
    val_ratio:   float = 0.10

    # ── Training ──────────────────────────────────────────────────────────
    epochs:       int   = 40
    batch_size:   int   = 32
    lr:           float = 1e-4
    lambda_pgd:   float = 0.5
    # Connectivity loss weight — paper uses equal weight (1.0).
    # Was set to 0.01 due to right=500 bug, now fixed (ends count→index).
    w_conn:       float = 1.0
    augment:      bool  = False
    eval_every:   int   = 5
    log_interval: int   = 50
    # Epochs using GT tokens for PGD (curriculum learning).
    # After this, NAMER switches to VAT predictions as PGD input.
    pgd_teacher_epochs: int = 8

    # ── Model (paper defaults) ─────────────────────────────────────────────
    d_model:    int   = 256
    nhead:      int   = 8
    pgd_layers: int   = 3     # paper: "three layer Transformer in PGD"
    drop:       float = 0.3
    img_h:      int   = 128
    img_w:      int   = 512
    max_len:    int   = 200

    # ── Misc ───────────────────────────────────────────────────────────────
    num_workers: int = 2
    seed:        int = 42


@dataclass
class DWAPConfig:
    """DWAP pretraining configuration."""

    # ── Paths (same dataset as NAMER) ─────────────────────────────────────
    data_root:  str = 'D://dataset/HME100K'
    label_file: str = 'D://dataset/HME100K/train.txt'
    checkpoint_dir: str = './checkpoints'
    vocab_path: str = './vocab.json'
    resume_checkpoint: Optional[str] = None

    # ── Split ──────────────────────────────────────────────────────────────
    train_ratio: float = 0.80
    val_ratio:   float = 0.10

    # ── Training ──────────────────────────────────────────────────────────
    epochs:      int   = 15       # 10-15 epochs is sufficient for DWAP
    batch_size:  int   = 32
    lr:          float = 1e-3
    augment:     bool  = False
    eval_every:  int   = 5
    log_interval: int  = 50

    # ── Model ──────────────────────────────────────────────────────────────
    d:       int   = 256   # attention + GRU hidden dim
    emb_dim: int   = 256
    drop:    float = 0.3
    img_h:   int   = 128
    img_w:   int   = 512
    max_len: int   = 200

    # ── Misc ───────────────────────────────────────────────────────────────
    num_workers: int = 2
    seed:        int = 42
