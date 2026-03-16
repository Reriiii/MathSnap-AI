"""
ICAL model: DenseNet Encoder + Transformer Decoder with SCCM + FusionModule.
Copied from: https://github.com/qingzhenduyu/ICAL
Adapted: removed PyTorch Lightning, added greedy decode.
"""
from typing import List

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

from .decoder import Decoder
from .encoder import Encoder


class ICAL(nn.Module):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
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

        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            vocab_size=vocab_size,
            pad_idx=pad_idx,
        )

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        Tuple of FloatTensor
            exp_out, imp_out, fusion_out: each [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, h, w, d]
        feature = torch.cat((feature, feature), dim=0)  # [2b, h, w, d]
        mask = torch.cat((mask, mask), dim=0)

        exp_out, imp_out, fusion_out = self.decoder(feature, mask, tgt)

        return exp_out, imp_out, fusion_out

    def encode(self, img: FloatTensor, img_mask: LongTensor):
        """Encode image only (for beam search)."""
        return self.encoder(img, img_mask)

    @torch.no_grad()
    def greedy_decode(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 200,
    ) -> List[List[int]]:
        """Simple greedy decode for fast validation (l2r only, fusion head).

        Parameters
        ----------
        img : FloatTensor [b, 1, h, w]
        img_mask : LongTensor [b, h, w]
        sos_idx : int
        eos_idx : int
        max_len : int

        Returns
        -------
        List[List[int]]: decoded token indices for each sample (no SOS/EOS)
        """
        self.eval()
        b = img.size(0)
        feature, mask = self.encoder(img, img_mask)

        # Start with SOS token
        input_ids = torch.full(
            (b, 1), fill_value=sos_idx, dtype=torch.long, device=img.device
        )

        finished = torch.zeros(b, dtype=torch.bool, device=img.device)

        for _ in range(max_len):
            # Get fusion logits (no bidirectional needed for greedy)
            _, _, fusion_out = self.decoder(feature, mask, input_ids)
            next_token = fusion_out[:, -1, :].argmax(dim=-1)  # [b]

            # Mark finished
            finished = finished | (next_token == eos_idx)
            next_token[finished] = eos_idx

            input_ids = torch.cat(
                [input_ids, next_token.unsqueeze(1)], dim=1
            )

            if finished.all():
                break

        # Extract sequences (remove SOS, stop at EOS)
        results = []
        for i in range(b):
            seq = input_ids[i, 1:].tolist()  # skip SOS
            # Truncate at first EOS
            if eos_idx in seq:
                seq = seq[:seq.index(eos_idx)]
            results.append(seq)

        return results


def build_model(config, vocab_size: int) -> ICAL:
    """Build ICAL model from config."""
    model = ICAL(
        d_model=config.model.d_model,
        growth_rate=config.model.growth_rate,
        num_layers=config.model.num_layers,
        nhead=config.model.nhead,
        num_decoder_layers=config.model.num_decoder_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        dc=config.model.dc,
        cross_coverage=config.model.cross_coverage,
        self_coverage=config.model.self_coverage,
        vocab_size=vocab_size,
        pad_idx=0,
    )
    return model
