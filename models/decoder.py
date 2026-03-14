"""
CoMER-style Transformer decoder with Attention Refinement Module (ARM).

Key references:
- CoMER (ECCV 2022, Zhao & Gao): coverage via ARM with self/cross-coverage
- BTTR (ICCV 2021): Transformer baseline for HMER
- Li et al. (ICFHR 2020): scale augmentation, drop-attention

ARM summary:
  Coverage c_t = sum_{k<t} A_k  (past attention weights, exclusive cumsum)
  Refinement R = LayerNorm(ReLU(Conv2d(reshape(C))) @ W_proj)
  Modified cross-attn energy: E' = E - R  (penalises already-parsed regions)
  Final attention: A = softmax(E' / sqrt(d_k))

Two coverage streams (CoMER fusion-coverage):
  Self-coverage:  exclusive cumsum of current layer's own raw attention
  Cross-coverage: inclusive cumsum of previous decoder layer's attention

Training: all T positions processed in parallel via cumsum (no serial dep).
Inference: full forward pass re-run at each step (correct, O(T^2), simple).
"""

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionRefinementModule(nn.Module):
    """
    ARM from CoMER (ECCV 2022).

    Computes a refinement term R from accumulated coverage C and subtracts
    it from raw cross-attention energies BEFORE softmax.

    R = LayerNorm(ReLU(Conv2d(C_2d)) @ W_proj)
    E' = E - R

    The coverage is reshaped from flat [L] to 2D [ho, wo] so that Conv2d
    captures local spatial neighbourhood in the encoder feature map.
    """

    def __init__(self, nhead: int, kernel_size: int = 5, d_coverage: int = 32):
        super().__init__()
        self.nhead = nhead
        self.d_coverage = d_coverage

        self.conv = nn.Conv2d(
            in_channels=nhead,
            out_channels=d_coverage,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.proj = nn.Linear(d_coverage, nhead, bias=False)
        self.norm = nn.LayerNorm(nhead)

    def forward(self, coverage: torch.Tensor, ho: int, wo: int) -> torch.Tensor:
        """
        Args:
            coverage: [B, H, T, L]  cumulative past attention weights
            ho, wo:   spatial dims of encoder feature map (ho * wo == L)
        Returns:
            R: [B, H, T, L]  refinement to subtract from attention energies
        """
        B, H, T, L = coverage.shape

        # [B, H, T, L] -> [B*T, H, ho, wo]
        C = coverage.permute(0, 2, 1, 3).reshape(B * T, H, ho, wo)

        # Conv: [B*T, H, ho, wo] -> [B*T, d_cov, ho, wo]
        F_cov = torch.relu(self.conv(C))

        # [B*T, d_cov, ho, wo] -> [B*T, L, d_cov] -> [B*T, L, H]
        F_cov = F_cov.permute(0, 2, 3, 1).reshape(B * T, L, self.d_coverage)
        R = self.norm(self.proj(F_cov))

        # [B*T, L, H] -> [B, H, T, L]
        return R.reshape(B, T, L, H).permute(0, 3, 1, 2)


class CoverageMultiheadCrossAttention(nn.Module):
    """
    Multi-head cross-attention with ARM coverage refinement.

    Self-coverage is computed via exclusive cumsum of raw softmax weights
    (parallel, differentiable, no serial dependency during training).
    Cross-coverage from the previous layer is passed in.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float,
                 arm: AttentionRefinementModule):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scale = math.sqrt(self.d_k)
        self.arm = arm

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.reshape(B, S, self.nhead, self.d_k).permute(0, 2, 1, 3)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cross_coverage: Optional[torch.Tensor],
        ho: int,
        wo: int,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = query.shape

        Q = self._split_heads(self.q_proj(query))
        K = self._split_heads(self.k_proj(key))
        V = self._split_heads(self.v_proj(value))

        E = torch.matmul(Q, K.transpose(-2, -1)) / self.scale   # [B, H, T, L]

        # Self-coverage via exclusive cumsum of raw attention
        A_raw = torch.softmax(E, dim=-1)
        self_coverage = torch.cumsum(A_raw, dim=2) - A_raw      # [B, H, T, L]

        # Fusion-coverage: combine self + cross (CoMER §3.3)
        total_coverage = self_coverage
        if cross_coverage is not None:
            total_coverage = total_coverage + cross_coverage

        # Subtract ARM refinement from energies
        R = self.arm(total_coverage, ho, wo)
        E_refined = E - R

        if key_padding_mask is not None:
            E_refined = E_refined.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        A = torch.softmax(E_refined, dim=-1)
        out = torch.matmul(self.attn_drop(A), V)
        out = out.permute(0, 2, 1, 3).reshape(B, T, self.d_model)
        return self.out_proj(out), A.detach()


class CoMERDecoderLayer(nn.Module):
    """
    Pre-norm decoder layer with ARM coverage.

    1. Causal self-attention  (standard)
    2. Coverage cross-attention  (ARM-refined)
    3. FFN with GELU
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, arm: AttentionRefinementModule):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = CoverageMultiheadCrossAttention(d_model, nhead, dropout, arm)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        cross_coverage: Optional[torch.Tensor],
        ho: int,
        wo: int,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = tgt
        x2, _ = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(x2)

        x2, attn = self.cross_attn(
            query=self.norm2(x),
            key=memory,
            value=memory,
            cross_coverage=cross_coverage,
            ho=ho, wo=wo,
        )
        x = x + self.drop2(x2)
        x = x + self.drop3(self.ffn(self.norm3(x)))
        return x, attn


class CoMERDecoder(nn.Module):
    """
    Full CoMER decoder: N layers with self-coverage + cross-coverage ARM.

    Cross-coverage chaining:
      layer 0: cross_coverage = None
      layer i: cross_coverage = cumsum(attn_{i-1}, dim=2)

    Inference: re-runs full forward() at each step so coverage accumulates
    naturally via cumsum over the growing decoded sequence. Simple and correct.
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
        arm_kernel_size: int = 5,
        arm_d_coverage: int = 32,
        length_norm_alpha: float = 0.6,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len
        self.length_norm_alpha = length_norm_alpha

        self.embedding   = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding1D(d_model, max_len=max_seq_len, dropout=dropout)

        self.arms = nn.ModuleList([
            AttentionRefinementModule(nhead, arm_kernel_size, arm_d_coverage)
            for _ in range(num_layers)
        ])
        self.layers = nn.ModuleList([
            CoMERDecoderLayer(d_model, nhead, dim_feedforward, dropout, self.arms[i])
            for i in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def _pad_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        return tgt == self.pad_idx

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        feat_h: int,
        feat_w: int,
    ) -> torch.Tensor:
        B, T = tgt.shape
        device = tgt.device
        tgt_mask = self._causal_mask(T, device)
        tgt_kpm  = self._pad_mask(tgt)

        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        prev_attn: Optional[torch.Tensor] = None
        for layer in self.layers:
            x, attn = layer(
                tgt=x, memory=memory,
                cross_coverage=prev_attn,
                ho=feat_h, wo=feat_w,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_kpm,
            )
            prev_attn = torch.cumsum(attn, dim=2)

        return self.output_proj(x)

    @torch.no_grad()
    def greedy_decode(
        self,
        memory: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        feat_h: int,
        feat_w: int,
        max_len: int = 200,
    ) -> torch.Tensor:
        B = memory.size(0)
        device = memory.device
        decoded = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(decoded, memory, feat_h, feat_w)
            next_tok = logits[:, -1:, :].argmax(dim=-1)
            decoded = torch.cat([decoded, next_tok], dim=1)
            finished |= next_tok.squeeze(-1) == eos_idx
            if finished.all():
                break

        return decoded

    @torch.no_grad()
    def beam_search(
        self,
        memory: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        feat_h: int,
        feat_w: int,
        beam_size: int = 5,
        max_len: int = 200,
    ) -> torch.Tensor:
        """Beam search with length normalisation. memory: [1, S, d_model]."""
        device = memory.device
        assert memory.size(0) == 1

        mem = memory.expand(beam_size, -1, -1)
        sequences = torch.full((beam_size, 1), sos_idx, dtype=torch.long, device=device)
        scores    = torch.zeros(beam_size, device=device)
        scores[1:] = -float('inf')
        active    = torch.ones(beam_size, dtype=torch.bool, device=device)
        completed: List[Tuple[float, torch.Tensor]] = []

        for _ in range(max_len - 1):
            n_active = int(active.sum().item())
            if n_active == 0:
                break

            idx     = active.nonzero(as_tuple=True)[0]
            s_act   = sequences[idx]
            m_act   = mem[:n_active]

            logits    = self.forward(s_act, m_act, feat_h, feat_w)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

            V = log_probs.size(-1)
            next_scores = (scores[idx].unsqueeze(-1) + log_probs).view(-1)
            topk_scores, topk_flat = next_scores.topk(min(beam_size, next_scores.size(0)))
            beam_w  = topk_flat // V
            token_w = topk_flat % V

            new_seqs = torch.cat([s_act[beam_w], token_w.unsqueeze(-1)], dim=1)
            filled   = new_seqs.size(0)

            sequences = torch.zeros(beam_size, new_seqs.size(1), dtype=torch.long, device=device)
            sequences[:filled] = new_seqs
            scores = torch.full((beam_size,), -float('inf'), device=device)
            scores[:filled] = topk_scores
            active = torch.zeros(beam_size, dtype=torch.bool, device=device)

            for i in range(filled):
                if token_w[i].item() == eos_idx:
                    norm = new_seqs.size(1) ** self.length_norm_alpha
                    completed.append((topk_scores[i].item() / norm, new_seqs[i].clone()))
                else:
                    active[i] = True

        for i in range(beam_size):
            if active[i]:
                norm = sequences[i].size(0) ** self.length_norm_alpha
                completed.append((scores[i].item() / norm, sequences[i].clone()))

        if not completed:
            return sequences[:1]
        _, best = max(completed, key=lambda x: x[0])
        return best.unsqueeze(0)