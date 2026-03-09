"""
DWAP — Dynamic Weighted Attention Prediction (paper reference [47])

GRU decoder with coverage attention over the DenseNet feature map.
Used as a PRETRAINED model to provide accurate token position estimates T
for the VAT bipartite matching in NAMER training.

Architecture:
  Encoder: DenseNetEncoder (shared, stride-8 features F8x) → [B, C, H, W]
  Decoder: GRU + coverage attention
    h_0     = tanh(W_init * mean_pool(F8x))
    α_t     = softmax(v · tanh(W_a·proj(F8x) + U_a·h_{t-1} + V_a·coverage_t))
    c_t     = Σ α_t * F8x   (context = attended features)
    coverage_t = Σ_{k<t} α_k  (cumulative attention — avoids double-attend)
    h_t     = GRU([embed(y_{t-1}); c_t], h_{t-1})
    y_t     = softmax(W_out · h_t)

Training: teacher-forcing, CrossEntropyLoss on y_t.
Inference: greedy decoding (or beam-search), returns tokens + attention maps α_t.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWAP(nn.Module):
    def __init__(self, ch_8x: int, d: int, vocab_sz: int,
                 emb_dim: int = 256, drop: float = 0.3):
        """
        Args:
            ch_8x:    channel dim of F8x (from DenseNetEncoder.ch_8x)
            d:        internal dimension (attention + GRU hidden dim)
            vocab_sz: vocabulary size
            emb_dim:  token embedding dim
            drop:     dropout rate
        """
        super().__init__()

        # Project F8x to attention space once (cheaper per step)
        self.feat_proj = nn.Sequential(
            nn.Conv2d(ch_8x, d, 1, bias=False),
            nn.BatchNorm2d(d),
        )

        # Coverage projection (same d as feat_proj output)
        self.cov_proj  = nn.Linear(1, d, bias=False)   # coverage is [B,H*W,1]

        # Attention query from hidden state
        self.h_proj  = nn.Linear(d, d, bias=False)

        # Attention energy → scalar
        self.attn_v  = nn.Linear(d, 1, bias=False)

        # Context projection → input to GRU
        self.ctx_proj = nn.Linear(ch_8x, d, bias=True)

        # Token embedding
        self.embedding = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.embedding.weight.data[0].zero_()

        # GRU: input = [embed; context_d]
        self.gru = nn.GRUCell(emb_dim + d, d)

        # Initial hidden from mean-pooled features
        self.h_init = nn.Sequential(
            nn.Linear(ch_8x, d),
            nn.Tanh(),
        )

        # Output classifier
        self.out_fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(d, vocab_sz),
        )

        self.d        = d
        self.vocab_sz = vocab_sz
        self.drop     = nn.Dropout(drop)

    def _init_hidden(self, f8x):
        """h_0 = tanh(W * mean(F8x))"""
        # f8x: [B, C, H, W]
        ctx = f8x.mean(dim=[2, 3])   # [B, C]
        return self.h_init(ctx)      # [B, d]

    def _attend(self, f_proj_flat, h, coverage_flat):
        """
        Compute attention weights α.

        f_proj_flat: [B, H*W, d]   projected encoder features
        h:           [B, d]        current hidden state
        coverage_flat:[B, H*W, 1]  cumulative attention so far
        Returns:
            alpha:   [B, H*W]      attention weights (softmax)
        """
        # [B, H*W, d] + [B, 1, d] + [B, H*W, d]
        e = (f_proj_flat
             + self.h_proj(h).unsqueeze(1)
             + self.cov_proj(coverage_flat))       # [B, H*W, d]
        e = torch.tanh(e)
        e = self.attn_v(e).squeeze(-1)             # [B, H*W]
        return F.softmax(e, dim=-1)                # [B, H*W]

    def forward(self, f8x, token_ids, teacher_forcing: bool = True,
                 max_steps: int = 60):
        """
        Teacher-forcing forward pass (training).

        max_steps: cap sequence length to limit sequential GRU iterations.
                   Coverage attention is inherently O(L) — capping L is the
                   only way to speed up training without changing the architecture.
                   60 steps covers >95% of HME100K expressions.

        Returns:
            logits:  [B, min(L-1,max_steps), vocab_sz]
            alphas:  [B, min(L-1,max_steps), H*W]
        """
        B, C, H, W = f8x.shape
        # Cap L: reduces 200 sequential GRU steps → 60 (3.3× speedup)
        L      = min(token_ids.size(1), max_steps + 1)
        device = f8x.device

        f_proj      = self.feat_proj(f8x)             # [B, d, H, W]
        f_proj_flat = f_proj.flatten(2).transpose(1, 2)   # [B, H*W, d]
        f_flat      = f8x.flatten(2).transpose(1, 2)      # [B, H*W, C]

        h        = self._init_hidden(f8x)             # [B, d]
        coverage = torch.zeros(B, H * W, 1, device=device)

        logits_list = []
        alpha_list  = []

        for t in range(L - 1):
            alpha    = self._attend(f_proj_flat, h, coverage)   # [B, H*W]
            coverage = coverage + alpha.unsqueeze(-1)

            ctx_raw  = (alpha.unsqueeze(-1) * f_flat).sum(dim=1)  # [B, C]
            ctx      = self.ctx_proj(ctx_raw)                     # [B, d]
            ctx      = self.drop(ctx)

            emb      = self.embedding(token_ids[:, t])            # [B, emb]
            gru_in   = torch.cat([emb, ctx], dim=-1)
            h        = self.gru(gru_in, h)

            logits_list.append(self.out_fc(h))
            alpha_list.append(alpha)

        logits = torch.stack(logits_list, dim=1)    # [B, L-1, vocab_sz]
        alphas = torch.stack(alpha_list,  dim=1)    # [B, L-1, H*W]
        return logits, alphas

    @torch.no_grad()
    def decode(self, f8x, sos_idx: int, eos_idx: int, max_len: int = 150):
        """
        Greedy decoding for inference / DWAP position extraction.

        Returns:
            token_seqs: list of B lists (token ids, without SOS/EOS)
            attn_maps:  list of B tensors [n_t, H, W]  — attention at each step
                        Use argmax over [H, W] to get token position T[i].
        """
        B, C, H, W = f8x.shape
        device      = f8x.device

        f_proj      = self.feat_proj(f8x)
        f_proj_flat = f_proj.flatten(2).transpose(1, 2)
        f_flat      = f8x.flatten(2).transpose(1, 2)

        h           = self._init_hidden(f8x)
        coverage    = torch.zeros(B, H * W, 1, device=device)
        y           = torch.full((B,), sos_idx, dtype=torch.long, device=device)
        done        = torch.zeros(B, dtype=torch.bool, device=device)

        seqs        = [[] for _ in range(B)]
        attn_maps   = [[] for _ in range(B)]

        for _ in range(max_len):
            alpha   = self._attend(f_proj_flat, h, coverage)
            coverage = coverage + alpha.unsqueeze(-1)

            ctx_raw  = (alpha.unsqueeze(-1) * f_flat).sum(dim=1)
            ctx      = self.ctx_proj(ctx_raw)
            gru_in   = torch.cat([self.embedding(y), ctx], dim=-1)
            h        = self.gru(gru_in, h)
            y        = self.out_fc(h).argmax(dim=-1)   # [B]

            alpha_2d = alpha.view(B, H, W)              # [B, H, W]

            for b in range(B):
                if done[b]:
                    continue
                if y[b].item() == eos_idx:
                    done[b] = True
                else:
                    seqs[b].append(y[b].item())
                    attn_maps[b].append(alpha_2d[b].cpu())

            if done.all():
                break

        # Stack attention maps: list of [n_t, H, W] tensors
        attn_maps = [torch.stack(a) if a else torch.zeros(0, H, W)
                     for a in attn_maps]
        return seqs, attn_maps
