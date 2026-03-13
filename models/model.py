"""
Full HMER model: DenseNet Encoder + Transformer Decoder.

Wraps the encoder and decoder into a single model with
forward pass and generation methods.
"""

import torch
import torch.nn as nn
from models.encoder import DenseNetEncoder
from models.decoder import TransformerDecoder


class HMERModel(nn.Module):
    """
    Handwritten Mathematical Expression Recognition model.

    DenseNet encodes the image into feature vectors,
    Transformer decoder generates LaTeX token sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        # Encoder params
        enc_growth_rate: int = 24,
        enc_block_config: tuple = (6, 12, 24, 16),
        enc_num_init_features: int = 64,
        enc_bn_size: int = 4,
        enc_drop_rate: float = 0.2,
        enc_compression: float = 0.5,
        # Decoder params
        dec_nhead: int = 8,
        dec_num_layers: int = 3,
        dec_dim_feedforward: int = 1024,
        dec_dropout: float = 0.3,
        max_seq_len: int = 200,
        pad_idx: int = 0,
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
        )

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=dec_nhead,
            num_layers=dec_num_layers,
            dim_feedforward=dec_dim_feedforward,
            dropout=dec_dropout,
            max_seq_len=max_seq_len,
            pad_idx=pad_idx,
        )

        self.pad_idx = pad_idx
        self.d_model = d_model

    def forward(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with teacher forcing.

        Args:
            images: [B, 1, H, W] grayscale images
            targets: [B, T] target sequences (SOS + tokens, shifted right)

        Returns:
            logits: [B, T, vocab_size]
        """
        # Encode
        memory = self.encoder(images)  # [B, S, d_model]

        # Decode (teacher forcing: input is target shifted right)
        # Target input: [SOS, t1, t2, ..., tn-1]
        # Target output: [t1, t2, ..., tn-1, EOS]
        tgt_input = targets[:, :-1]  # Remove last token
        logits = self.decoder(tgt_input, memory)

        return logits

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
            images: [B, 1, H, W] grayscale images
            sos_idx: start-of-sequence token index
            eos_idx: end-of-sequence token index
            max_len: maximum generation length
            beam_size: beam size (1 = greedy)

        Returns:
            predictions: [B, T] predicted token indices
        """
        self.eval()
        memory = self.encoder(images)

        if beam_size <= 1:
            return self.decoder.greedy_decode(memory, sos_idx, eos_idx, max_len)
        else:
            # Beam search (per-sample)
            results = []
            for i in range(images.size(0)):
                mem_i = memory[i:i+1]
                pred = self.decoder.beam_search(mem_i, sos_idx, eos_idx, beam_size, max_len)
                results.append(pred.squeeze(0))

            # Pad to same length
            max_pred_len = max(r.size(0) for r in results)
            padded = torch.full(
                (len(results), max_pred_len), self.pad_idx,
                dtype=torch.long, device=images.device
            )
            for i, r in enumerate(results):
                padded[i, :r.size(0)] = r

            return padded

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(vocab_size: int, config=None) -> HMERModel:
    """
    Build HMER model from config.

    Args:
        vocab_size: vocabulary size
        config: Config instance

    Returns:
        HMERModel instance
    """
    from config import Config

    if config is None:
        config = Config()

    model = HMERModel(
        vocab_size=vocab_size,
        d_model=config.decoder.d_model,
        enc_growth_rate=config.encoder.growth_rate,
        enc_block_config=config.encoder.block_config,
        enc_num_init_features=config.encoder.num_init_features,
        enc_bn_size=config.encoder.bn_size,
        enc_drop_rate=config.encoder.drop_rate,
        enc_compression=config.encoder.compression,
        dec_nhead=config.decoder.nhead,
        dec_num_layers=config.decoder.num_layers,
        dec_dim_feedforward=config.decoder.dim_feedforward,
        dec_dropout=config.decoder.dropout,
        max_seq_len=config.decoder.max_seq_len,
        pad_idx=0,
    )

    return model
