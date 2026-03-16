"""
CoMER model: DenseNet Encoder + Transformer Decoder with ARM.
Based on: https://github.com/Green-Wood/CoMER
Stripped from ICAL: single CE loss, no SCCM/FusionModule.
"""
from typing import List

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

from .decoder import Decoder
from .encoder import Encoder


class CoMER(nn.Module):
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
        """
        Parameters
        ----------
        img : FloatTensor [b, 1, h, w]
        img_mask: LongTensor [b, h, w]
        tgt : LongTensor [b, l]

        Returns
        -------
        FloatTensor [b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, h, w, d]
        out = self.decoder(feature, mask, tgt)
        return out

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
        """Greedy decode for validation (l2r only, no bidirectional).

        Parameters
        ----------
        img : FloatTensor [b, 1, h, w]
        img_mask : LongTensor [b, h, w]

        Returns
        -------
        List[List[int]]: decoded token indices (no SOS/EOS)
        """
        self.eval()
        b = img.size(0)
        feature, mask = self.encoder(img, img_mask)

        input_ids = torch.full(
            (b, 1), fill_value=sos_idx, dtype=torch.long, device=img.device
        )

        finished = torch.zeros(b, dtype=torch.bool, device=img.device)

        for _ in range(max_len):
            out = self.decoder(feature, mask, input_ids)  # [b, l, vocab]
            next_token = out[:, -1, :].argmax(dim=-1)  # [b]

            finished = finished | (next_token == eos_idx)
            next_token[finished] = eos_idx

            input_ids = torch.cat(
                [input_ids, next_token.unsqueeze(1)], dim=1
            )

            if finished.all():
                break

        results = []
        for i in range(b):
            seq = input_ids[i, 1:].tolist()
            if eos_idx in seq:
                seq = seq[:seq.index(eos_idx)]
            results.append(seq)

        return results

    @torch.no_grad()
    def beam_search_decode(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        sos_idx: int,
        eos_idx: int,
        pad_idx: int = 0,
        beam_size: int = 10,
        max_len: int = 200,
        alpha: float = 1.0,
    ) -> List[List[int]]:
        """Beam search decode (l2r only, clean implementation).

        Processes one sample at a time to avoid complexity of batched beam search.
        """
        import torch.nn.functional as F

        self.eval()
        b = img.size(0)
        device = img.device
        feature, mask = self.encoder(img, img_mask)  # [b, h, w, d]

        results = []
        for i in range(b):
            feat_i = feature[i:i+1]  # [1, h, w, d]
            mask_i = mask[i:i+1]     # [1, h, w]

            # Expand for beams: [beam, h, w, d]
            feat_beam = feat_i.expand(beam_size, -1, -1, -1)
            mask_beam = mask_i.expand(beam_size, -1, -1)

            # Each beam starts with SOS
            sequences = torch.full((beam_size, 1), sos_idx, dtype=torch.long, device=device)
            scores = torch.zeros(beam_size, device=device)
            scores[1:] = -1e9  # only first beam active initially

            # Store completed hypotheses: (score, sequence)
            completed = []

            for step in range(max_len):
                out = self.decoder(feat_beam, mask_beam, sequences)  # [beam, l, vocab]
                logits = out[:, -1, :]  # [beam, vocab]
                log_probs = F.log_softmax(logits, dim=-1)

                # Candidate scores: [beam, vocab]
                candidate_scores = scores.unsqueeze(-1) + log_probs
                vocab_size = log_probs.size(-1)

                if step == 0:
                    # Only first beam is active
                    candidate_scores = candidate_scores[0:1].reshape(-1)
                else:
                    candidate_scores = candidate_scores.reshape(-1)

                # Top-k candidates
                topk_scores, topk_ids = candidate_scores.topk(beam_size, dim=-1)
                beam_ids = topk_ids // vocab_size
                token_ids = topk_ids % vocab_size

                # Build new sequences
                new_sequences = torch.cat([
                    sequences[beam_ids], token_ids.unsqueeze(-1)
                ], dim=-1)
                new_scores = topk_scores

                # Check for completed beams (EOS generated)
                active_mask = token_ids != eos_idx
                for j in range(beam_size):
                    if not active_mask[j]:
                        seq = new_sequences[j, 1:].tolist()  # remove SOS
                        if eos_idx in seq:
                            seq = seq[:seq.index(eos_idx)]
                        # Length-normalized score
                        length_penalty = ((5.0 + len(seq)) / 6.0) ** alpha
                        normalized_score = new_scores[j].item() / length_penalty
                        completed.append((normalized_score, seq))

                # Keep only active beams
                active_indices = active_mask.nonzero(as_tuple=True)[0]
                if len(active_indices) == 0:
                    break
                if len(completed) >= beam_size:
                    break

                # Pad back to beam_size if needed
                if len(active_indices) < beam_size:
                    # Fill remaining slots with top active beams
                    pad_count = beam_size - len(active_indices)
                    pad_indices = active_indices[:pad_count].repeat(
                        (pad_count + len(active_indices) - 1) // len(active_indices) + 1
                    )[:pad_count]
                    all_indices = torch.cat([active_indices, pad_indices])
                    new_scores_padded = new_scores[all_indices]
                    new_scores_padded[len(active_indices):] = -1e9
                    sequences = new_sequences[all_indices]
                    scores = new_scores_padded
                else:
                    sequences = new_sequences[active_indices[:beam_size]]
                    scores = new_scores[active_indices[:beam_size]]

            # If no completed, use best active beam
            if not completed:
                seq = sequences[0, 1:].tolist()
                if eos_idx in seq:
                    seq = seq[:seq.index(eos_idx)]
                completed.append((scores[0].item(), seq))

            # Pick best hypothesis
            completed.sort(key=lambda x: x[0], reverse=True)
            results.append(completed[0][1])

        return results


def build_model(config, vocab_size: int) -> CoMER:
    """Build CoMER model from config."""
    model = CoMER(
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
