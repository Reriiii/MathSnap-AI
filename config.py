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
    # Path to pretrained DWAP checkpoint (used for VAT bipartite matching positions).
    # Set to None to fall back to uniform-spread position estimate.
    dwap_checkpoint: Optional[str]  = None

    # ── Split ──────────────────────────────────────────────────────────────
    train_ratio: float = 0.80
    val_ratio:   float = 0.10

    # ── Training ──────────────────────────────────────────────────────────
    epochs:       int   = 40
    batch_size:   int   = 32
    lr:           float = 2e-4      # paper: 2e-4
    lambda_pgd:   float = 0.5       # paper: lambda=0.5
    w_conn:       float = 1.0       # paper: equal weight on left/right connectivity
    augment:      bool  = False
    eval_every:   int   = 5
    log_interval: int   = 50

    # ── VAT recall tuning ─────────────────────────────────────────────────
    # Weight applied to the background (∅) class in VAT CrossEntropyLoss.
    # Paper uses plain CE (none_weight=1.0).  In practice plain CE causes
    # VAT recall to plateau at ~61%, starving PGD of tokens.
    #
    # none_weight=0.50 reduces background penalty: the model predicts more
    # real tokens (higher recall) at the cost of slightly more false positives
    # (lower precision).  PGD can tolerate FP but NOT missing tokens.
    #
    # Tuning guide:
    #   none_weight=1.00  →  rec≈61%  prec≈64%  avg_det≈8
    #   none_weight=0.50  →  rec≈80%  prec≈58%  avg_det≈12  ← recommended
    #   none_weight=0.20  →  rec≈90%  prec≈42%  avg_det≈22  ← too many FP
    none_weight: float = 0.30

    # ── Model (paper defaults) ─────────────────────────────────────────────
    d_model:    int   = 256
    nhead:      int   = 8
    pgd_layers: int   = 3     # paper: three-layer Transformer in PGD
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
    epochs:      int   = 15
    batch_size:  int   = 32
    lr:          float = 1e-3
    augment:     bool  = False
    eval_every:  int   = 5
    log_interval: int  = 50

    # ── Model ──────────────────────────────────────────────────────────────
    d:       int   = 256
    emb_dim: int   = 256
    drop:    float = 0.3
    img_h:   int   = 128
    img_w:   int   = 512
    max_len: int   = 200

    # ── Misc ───────────────────────────────────────────────────────────────
    num_workers: int = 2
    seed:        int = 42