"""
CoMER decoder: Transformer with ARM (Attention Refinement Module).
Based on: https://github.com/Green-Wood/CoMER
Stripped from ICAL: removed SCCM, FusionModule, implicit/fusion heads.
"""
from typing import List

import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor

from .pos_enc import WordPosEnc
from .transformer.arm import AttentionRefinementModule
from .transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)


def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    dc: int,
    cross_coverage: bool,
    self_coverage: bool,
) -> TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    if cross_coverage or self_coverage:
        arm = AttentionRefinementModule(
            nhead, dc, cross_coverage, self_coverage)
    else:
        arm = None

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm)
    return decoder


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        vocab_size: int,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )

        self.pos_enc = WordPosEnc(d_model=d_model)

        self.norm = nn.LayerNorm(d_model)

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        # Single projection head (CoMER)
        self.proj = nn.Linear(d_model, vocab_size)

    def _build_attention_mask(self, length, device):
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=device
        )
        mask.triu_(1)
        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """
        Parameters
        ----------
        src : FloatTensor [b, h, w, d]
        src_mask: LongTensor [b, h, w]
        tgt : LongTensor [b, l]

        Returns
        -------
        FloatTensor [b, l, vocab_size]
        """
        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l, tgt.device)
        tgt_pad_mask = tgt == self.pad_idx

        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]
        tgt = self.norm(tgt)

        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        out = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )

        out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)

        return out

    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        """For beam search: run forward and return logits."""
        assert len(src) == 1 and len(src_mask) == 1
        return self(src[0], src_mask[0], input_ids)
