"""
Transformer decoder for HMER.

Standard Transformer decoder with multi-head cross-attention to
encoder features. Supports greedy decoding and beam search.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding1D(nn.Module):
    """Sinusoidal positional encoding for 1D sequences."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [B, T, d_model]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for autoregressive LaTeX sequence generation.

    Uses cross-attention to attend to encoder features and
    causal self-attention for autoregressive generation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        max_seq_len: int = 200,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Positional encoding
        self.pos_encoding = PositionalEncoding1D(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for better training stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal (upper triangular) attention mask."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def _generate_padding_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Generate padding mask for target sequence."""
        return (tgt == self.pad_idx)  # [B, T], True where padded

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).

        Args:
            tgt: [B, T] target token indices (with SOS, without final token for shift)
            memory: [B, S, d_model] encoder output
            tgt_mask: optional causal mask
            tgt_key_padding_mask: optional padding mask

        Returns:
            logits: [B, T, vocab_size]
        """
        B, T = tgt.shape

        # Generate causal mask
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(T, tgt.device)

        # Generate padding mask
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._generate_padding_mask(tgt)

        # Embed tokens
        x = self.embedding(tgt) * math.sqrt(self.d_model)  # [B, T, d_model]
        x = self.pos_encoding(x)

        # Decode
        output = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Project to vocabulary
        logits = self.output_proj(output)  # [B, T, vocab_size]
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        memory: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 200,
    ) -> torch.Tensor:
        """
        Greedy decoding (for inference).

        Args:
            memory: [B, S, d_model] encoder output
            sos_idx: start-of-sequence token index
            eos_idx: end-of-sequence token index
            max_len: maximum decoding length

        Returns:
            decoded: [B, max_len] predicted token indices
        """
        B = memory.size(0)
        device = memory.device

        # Start with SOS
        decoded = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = self._generate_causal_mask(decoded.size(1), device)
            tgt_key_padding_mask = self._generate_padding_mask(decoded)

            x = self.embedding(decoded) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)

            output = self.transformer_decoder(
                tgt=x,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

            logits = self.output_proj(output[:, -1:, :])  # Last position
            next_token = logits.argmax(dim=-1)  # [B, 1]

            decoded = torch.cat([decoded, next_token], dim=1)

            # Check if all sequences have generated EOS
            if (next_token.squeeze(-1) == eos_idx).all():
                break

        return decoded

    @torch.no_grad()
    def beam_search(
        self,
        memory: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        beam_size: int = 5,
        max_len: int = 200,
    ) -> torch.Tensor:
        """
        Beam search decoding.

        Args:
            memory: [B, S, d_model] encoder output (B should be 1 for beam search)
            sos_idx: start-of-sequence token index
            eos_idx: end-of-sequence token index
            beam_size: number of beams
            max_len: maximum decoding length

        Returns:
            best_sequence: [1, T] best decoded sequence
        """
        B = memory.size(0)
        device = memory.device
        assert B == 1, "Beam search currently supports batch_size=1"

        # Expand memory for beam search
        memory = memory.expand(beam_size, -1, -1)  # [beam, S, d_model]

        # Initialize beams: (log_prob, sequence)
        sequences = torch.full((beam_size, 1), sos_idx, dtype=torch.long, device=device)
        scores = torch.zeros(beam_size, device=device)
        scores[1:] = -float('inf')  # Only first beam is active initially

        completed_beams = []

        for step in range(max_len - 1):
            tgt_mask = self._generate_causal_mask(sequences.size(1), device)
            tgt_key_padding_mask = self._generate_padding_mask(sequences)

            x = self.embedding(sequences) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)

            output = self.transformer_decoder(
                tgt=x,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

            logits = self.output_proj(output[:, -1, :])  # [beam, vocab]
            log_probs = F.log_softmax(logits, dim=-1)

            # Calculate new scores
            vocab_size = log_probs.size(-1)
            next_scores = scores.unsqueeze(-1) + log_probs  # [beam, vocab]
            next_scores = next_scores.view(-1)  # [beam * vocab]

            # Select top-k
            topk_scores, topk_indices = next_scores.topk(beam_size)
            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size

            # Update sequences
            sequences = torch.cat([
                sequences[beam_indices],
                token_indices.unsqueeze(-1)
            ], dim=1)
            scores = topk_scores

            # Check for completed beams
            eos_mask = token_indices == eos_idx
            if eos_mask.any():
                for i in range(beam_size):
                    if eos_mask[i]:
                        completed_beams.append((scores[i].item(), sequences[i].clone()))

                # Keep non-EOS beams
                non_eos = ~eos_mask
                if non_eos.sum() == 0:
                    break
                sequences = sequences[non_eos]
                scores = scores[non_eos]

                # Refill beams if needed
                if sequences.size(0) < beam_size:
                    pad_count = beam_size - sequences.size(0)
                    sequences = torch.cat([
                        sequences,
                        sequences[:pad_count].clone()
                    ], dim=0)
                    scores = torch.cat([
                        scores,
                        torch.full((pad_count,), -float('inf'), device=device)
                    ], dim=0)

                memory = memory[:sequences.size(0)]

        # Add remaining beams
        for i in range(sequences.size(0)):
            completed_beams.append((scores[i].item(), sequences[i].clone()))

        if not completed_beams:
            return sequences[:1]

        # Return best beam
        best_score, best_seq = max(completed_beams, key=lambda x: x[0])
        return best_seq.unsqueeze(0)
