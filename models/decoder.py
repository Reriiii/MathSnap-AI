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
        length_norm_alpha: float = 0.6,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len
        # Length normalization exponent for beam search scores.
        # Divides accumulated log-prob by length^alpha to prevent the beam
        # from always preferring shorter sequences.  0.6 is a standard default.
        self.length_norm_alpha = length_norm_alpha

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
        Beam search decoding with length normalization.

        Scores are normalized by sequence_length ^ alpha before selecting
        the best completed beam, preventing the decoder from systematically
        preferring shorter outputs.

        Args:
            memory: [1, S, d_model] encoder output (batch_size must be 1)
            sos_idx: start-of-sequence token index
            eos_idx: end-of-sequence token index
            beam_size: number of beams
            max_len: maximum decoding length

        Returns:
            best_sequence: [1, T] best decoded sequence
        """
        device = memory.device
        assert memory.size(0) == 1, "Beam search supports batch_size=1 only"

        # Expand memory once for all beams — kept fixed throughout decoding
        # (never sliced), avoiding the alignment bug when beams complete early.
        memory_expanded = memory.expand(beam_size, -1, -1)  # [beam, S, d_model]

        # sequences[i] holds the current token sequence for beam i
        sequences = torch.full((beam_size, 1), sos_idx, dtype=torch.long, device=device)
        # Raw accumulated log-probs (unnormalized)
        scores = torch.zeros(beam_size, device=device)
        scores[1:] = -float('inf')  # Only beam 0 is active at step 0

        # active_mask: which beam slots are still generating (not yet hit EOS)
        active_mask = torch.ones(beam_size, dtype=torch.bool, device=device)

        completed_beams: list[tuple[float, torch.Tensor]] = []

        for step in range(max_len - 1):
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            n_active = active_indices.size(0)
            if n_active == 0:
                break

            # Run decoder only on currently active beams
            seqs_active = sequences[active_indices]           # [n_active, t]
            mem_active = memory_expanded[:n_active]           # [n_active, S, d]

            t = seqs_active.size(1)
            tgt_mask = self._generate_causal_mask(t, device)
            tgt_key_padding_mask = self._generate_padding_mask(seqs_active)

            x = self.embedding(seqs_active) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)
            output = self.transformer_decoder(
                tgt=x,
                memory=mem_active,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            logits = self.output_proj(output[:, -1, :])       # [n_active, vocab]
            log_probs = F.log_softmax(logits, dim=-1)          # [n_active, vocab]

            vocab_size = log_probs.size(-1)

            # Expand current scores and add new log_probs
            cur_scores = scores[active_indices].unsqueeze(-1)  # [n_active, 1]
            next_scores = cur_scores + log_probs               # [n_active, vocab]

            # Flatten and pick top beam_size candidates across all active beams
            flat = next_scores.view(-1)                        # [n_active * vocab]
            topk_scores, topk_flat = flat.topk(min(beam_size, flat.size(0)))

            # Map flat indices back to (beam_within_active, token)
            beam_within_active = topk_flat // vocab_size
            token_ids = topk_flat % vocab_size

            # Reconstruct new sequences for all candidate beams
            new_sequences = torch.cat([
                seqs_active[beam_within_active],               # parent sequences
                token_ids.unsqueeze(-1)                        # new tokens
            ], dim=1)                                          # [beam_size, t+1]

            # Update global state: place new beams back into fixed-size buffers
            new_scores = topk_scores
            filled = new_sequences.size(0)
            sequences = torch.zeros(beam_size, new_sequences.size(1),
                                    dtype=torch.long, device=device)
            sequences[:filled] = new_sequences
            scores = torch.full((beam_size,), -float('inf'), device=device)
            scores[:filled] = new_scores
            active_mask = torch.zeros(beam_size, dtype=torch.bool, device=device)

            # Handle EOS tokens
            for i in range(filled):
                if token_ids[i].item() == eos_idx:
                    seq_len = new_sequences.size(1)
                    norm = (seq_len ** self.length_norm_alpha)
                    completed_beams.append((new_scores[i].item() / norm,
                                            new_sequences[i].clone()))
                else:
                    active_mask[i] = True

        # Add remaining active beams as completed (no EOS found within max_len)
        for i in range(beam_size):
            if active_mask[i]:
                seq_len = sequences[i].size(0)
                norm = (seq_len ** self.length_norm_alpha)
                completed_beams.append((scores[i].item() / norm, sequences[i].clone()))

        if not completed_beams:
            return sequences[:1]

        # Return the beam with the highest length-normalized score
        best_score, best_seq = max(completed_beams, key=lambda x: x[0])
        return best_seq.unsqueeze(0)