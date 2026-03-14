"""
Full HMER model: DenseNet Encoder + CoMER Decoder + Counting Module.

Architecture:
  DenseNet encoder  → [B, S, d_model] feature sequence + (feat_h, feat_w)
  CoMER decoder     → [B, T, vocab_size] logits  (ARM coverage attention)
  Counting module   → [B, vocab_size] per-class presence probability
                       (weakly supervised auxiliary, no location annotation needed)

References:
  CoMER (ECCV 2022): ARM coverage attention for Transformer decoders
  CAN   (ECCV 2022): counting-aware module for HMER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from models.encoder import DenseNetEncoder
from models.decoder import CoMERDecoder


# ---------------------------------------------------------------------------
# Counting module (inspired by CAN, ECCV 2022)
# ---------------------------------------------------------------------------

class CountingModule(nn.Module):
    """
    Weakly supervised counting module.

    Predicts the probability of each vocabulary token appearing at least once
    in the output sequence, using only the encoder features and label counts
    as supervision (no spatial annotations required).

    Architecture:
      1. Per-position linear projection: [B, L, d_model] → [B, L, vocab_size]
      2. Average-pool over spatial positions: [B, L, V] → [B, V]
      3. Sigmoid activation: outputs probability ∈ (0, 1) per class

    Loss: Binary Cross-Entropy against binary presence targets derived from
    the ground-truth token sequence (1 if token appears, 0 otherwise).

    This auxiliary signal encourages the encoder to build a global symbol
    inventory, complementing the sequence-level cross-entropy in the decoder.
    """

    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory: [B, L, d_model]  encoder output

        Returns:
            counts: [B, vocab_size]  probability each token is present
        """
        x = self.drop(memory)
        logits = self.linear(x)                     # [B, L, vocab_size]
        return torch.sigmoid(logits.mean(dim=1))    # [B, vocab_size]


# ---------------------------------------------------------------------------
# Full HMER model
# ---------------------------------------------------------------------------

class HMERModel(nn.Module):
    """
    Handwritten Mathematical Expression Recognition model.

    DenseNet encodes the image into a spatial feature sequence.
    CoMER decoder generates the LaTeX token sequence with ARM coverage.
    CountingModule provides an auxiliary presence-prediction signal.

    The CTC head on the encoder output (optional) provides an additional
    alignment-aware auxiliary loss during training.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        # Encoder
        enc_growth_rate: int = 24,
        enc_block_config: tuple = (6, 12, 16, 8),
        enc_num_init_features: int = 64,
        enc_bn_size: int = 4,
        enc_drop_rate: float = 0.2,
        enc_compression: float = 0.8,
        enc_num_transitions: int = 2,
        # Decoder
        dec_nhead: int = 8,
        dec_num_layers: int = 6,
        dec_dim_feedforward: int = 2048,
        dec_dropout: float = 0.3,
        max_seq_len: int = 200,
        pad_idx: int = 0,
        # ARM
        arm_kernel_size: int = 5,
        arm_d_coverage: int = 32,
        # Counting
        counting_dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = DenseNetEncoder(
            in_channels=1,
            growth_rate=enc_growth_rate,
            block_config=enc_block_config,
            num_init_features=enc_num_init_features,
            bn_size=enc_bn_size,
            drop_rate=enc_drop_rate,
            compression=enc_compression,
            d_model=d_model,
            num_transitions=enc_num_transitions,
        )

        self.decoder = CoMERDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=dec_nhead,
            num_layers=dec_num_layers,
            dim_feedforward=dec_dim_feedforward,
            dropout=dec_dropout,
            max_seq_len=max_seq_len,
            pad_idx=pad_idx,
            arm_kernel_size=arm_kernel_size,
            arm_d_coverage=arm_d_coverage,
        )

        # CTC auxiliary head (on encoder output)
        self.ctc_head = nn.Linear(d_model, vocab_size)

        # Counting auxiliary head
        self.counting_module = CountingModule(d_model, vocab_size, counting_dropout)

        self.pad_idx = pad_idx
        self.d_model = d_model

    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Run only the encoder. Returns (memory, feat_h, feat_w).
        Used by the CTC training branch.
        """
        return self.encoder(images)

    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forcing forward pass.

        Args:
            images:  [B, 1, H, W]
            targets: [B, T]  token sequence (SOS + content + EOS + padding)

        Returns:
            logits: [B, T-1, vocab_size]
            counts: [B, vocab_size]       counting module predictions
        """
        memory, feat_h, feat_w = self.encoder(images)   # [B, S, d_model]

        tgt_input = targets[:, :-1]   # remove last token (shift right)
        logits = self.decoder(tgt_input, memory, feat_h, feat_w)

        counts = self.counting_module(memory)

        return logits, counts

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 200,
        beam_size: int = 1,
    ) -> torch.Tensor:
        """
        Generate LaTeX sequences from images.

        Args:
            images:    [B, 1, H, W]
            beam_size: 1 = greedy, >1 = beam search (per-sample)

        Returns:
            predictions: [B, T] predicted token indices
        """
        self.eval()
        memory, feat_h, feat_w = self.encoder(images)

        if beam_size <= 1:
            return self.decoder.greedy_decode(
                memory, sos_idx, eos_idx, feat_h, feat_w, max_len
            )

        results = []
        for i in range(images.size(0)):
            mem_i = memory[i:i + 1]
            pred = self.decoder.beam_search(
                mem_i, sos_idx, eos_idx, feat_h, feat_w, beam_size, max_len
            )
            results.append(pred.squeeze(0))

        max_len_pred = max(r.size(0) for r in results)
        padded = torch.full(
            (len(results), max_len_pred), self.pad_idx,
            dtype=torch.long, device=images.device
        )
        for i, r in enumerate(results):
            padded[i, :r.size(0)] = r
        return padded

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(vocab_size: int, config=None) -> HMERModel:
    """Build HMERModel from config."""
    from config import Config
    if config is None:
        config = Config()

    return HMERModel(
        vocab_size=vocab_size,
        d_model=config.decoder.d_model,
        enc_growth_rate=config.encoder.growth_rate,
        enc_block_config=config.encoder.block_config,
        enc_num_init_features=config.encoder.num_init_features,
        enc_bn_size=config.encoder.bn_size,
        enc_drop_rate=config.encoder.drop_rate,
        enc_compression=config.encoder.compression,
        enc_num_transitions=config.encoder.num_transitions,
        dec_nhead=config.decoder.nhead,
        dec_num_layers=config.decoder.num_layers,
        dec_dim_feedforward=config.decoder.dim_feedforward,
        dec_dropout=config.decoder.dropout,
        max_seq_len=config.decoder.max_seq_len,
        pad_idx=0,
        arm_kernel_size=config.decoder.arm_kernel_size,
        arm_d_coverage=config.decoder.arm_d_coverage,
        counting_dropout=config.decoder.counting_dropout,
    )