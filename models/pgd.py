import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PE2D


class _XAttnLayer(nn.Module):
    """Cross-attention (to visual features) + self-attention + FFN."""
    def __init__(self, d: int, heads: int, ff: int, drop: float = 0.1):
        super().__init__()
        self.xattn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.sattn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d, ff), nn.ReLU(inplace=False), nn.Dropout(drop), nn.Linear(ff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.dp    = nn.Dropout(drop)

    def forward(self, q, kv, key_padding_mask=None):
        q2, _ = self.xattn(q, kv, kv)
        q = self.n1(q + self.dp(q2))
        q2, _ = self.sattn(q, q, q, key_padding_mask=key_padding_mask)
        q = self.n2(q + self.dp(q2))
        return self.n3(q + self.dp(self.ffn(q)))


class _PGDHead(nn.Module):
    def __init__(self, d, heads, ff, n_layers, drop=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [_XAttnLayer(d, heads, ff, drop) for _ in range(n_layers)])

    def forward(self, q, kv, key_padding_mask=None):
        for layer in self.layers:
            q = layer(q, kv, key_padding_mask)
        return q


class PGD(nn.Module):
    """
    Parallel Graph Decoder — paper Section 3.3.

    Three independent NAR heads:
      SCH – Self-node Correction Head
      LCH – Left  Connectivity Head
      RCH – Right Connectivity Head

    Q0 = VisualFeature(sample F16x) + PE2D + WordEmbedding
    """
    def __init__(self, ch_16x: int, d: int, heads: int,
                 n_layers: int, num_cls: int, vocab_sz: int, drop: float = 0.1):
        super().__init__()
        self.proj_kv = nn.Sequential(
            nn.Conv2d(ch_16x, d, 1, bias=False),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
        )
        self.proj_q  = nn.Conv2d(ch_16x, d, 1, bias=True)
        self.vis_norm = nn.LayerNorm(d)
        self.pe       = PE2D(d)
        self.word_emb = nn.Embedding(vocab_sz, d, padding_idx=0)

        ff = d * 2
        self.sch = _PGDHead(d, heads, ff, n_layers, drop)
        self.lch = _PGDHead(d, heads, ff, n_layers, drop)
        self.rch = _PGDHead(d, heads, ff, n_layers, drop)
        self.cls_head = nn.Linear(d, num_cls)

        nn.init.normal_(self.word_emb.weight, std=0.02)
        self.word_emb.weight.data[0].zero_()   # padding idx = 0 → always zero

    def _sample_visual(self, f16x_proj, coords):
        """
        Sample visual features using grid_sample.
        coords: [B, N, 2] containing (row, col) indices into the f8x feature map.
        Grid sample expects x, y in [-1, 1].
        """
        B, d, H, W = f16x_proj.shape
        N = coords.size(1)
        if N == 0:
            return torch.zeros(B, 0, d, device=f16x_proj.device)
            
        # Coordinates come from f8x maps (stride 8). f16x is stride 16.
        # W_f8, H_f8 are inherently twice the size of f16x.
        W_f8 = W * 2
        H_f8 = H * 2
        
        # Convert coords to [-1, 1] range for grid_sample
        # coords are [row, col]. Grid sample needs [x(col), y(row)]
        x = coords[:, :, 1].float() / max(W_f8 - 1, 1) * 2.0 - 1.0
        y = coords[:, :, 0].float() / max(H_f8 - 1, 1) * 2.0 - 1.0
        
        # Grid shape needs to be [B, H_out, W_out, 2]. We want [B, 1, N, 2]
        grid = torch.stack([x, y], dim=-1).unsqueeze(1)
        
        # Sample: Output is [B, d, 1, N]
        sampled = F.grid_sample(f16x_proj, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return sampled.squeeze(2).transpose(1, 2)  # [B, N, d]

    def _build_q0(self, f16x, token_ids, coords):
        B, N = token_ids.shape
        safe_tids = token_ids.clamp(0, self.word_emb.num_embeddings - 1)
        word_feat = self.word_emb(safe_tids)
        
        # 2D Position Embedding at specific token coordinates
        # coords is [B, N, 2] (row, col) from f8x.
        H_f16, W_f16 = f16x.size(2), f16x.size(3)
        pe_full = self.pe(torch.zeros(1, self.pe.pe.size(1), H_f16, W_f16, device=f16x.device))
        pe_grid = pe_full.expand(B, -1, -1, -1)
        
        # Sample PE exact same way as visual features
        x = coords[:, :, 1].float() / max((W_f16 * 2) - 1, 1) * 2.0 - 1.0
        y = coords[:, :, 0].float() / max((H_f16 * 2) - 1, 1) * 2.0 - 1.0
        grid = torch.stack([x, y], dim=-1).unsqueeze(1)
        pos_feat = F.grid_sample(pe_grid, grid, mode='nearest', padding_mode='zeros', align_corners=True)
        pos_feat = pos_feat.squeeze(2).transpose(1, 2)
        
        f16x_q    = self.proj_q(f16x)
        vis_feat  = self.vis_norm(self._sample_visual(f16x_q, coords))
        return vis_feat + pos_feat + word_feat

    def _build_kv(self, f16x):
        f = self.pe(self.proj_kv(f16x))
        return f.flatten(2).transpose(1, 2)

    def forward(self, f16x, token_ids, coords):
        pad_mask = (token_ids == 0)   # True = padding position
        kv = self._build_kv(f16x)
        q0 = self._build_q0(f16x, token_ids, coords)
        qs = self.sch(q0, kv, pad_mask)
        ql = self.lch(q0, kv, pad_mask)
        qr = self.rch(q0, kv, pad_mask)
        return qs, ql, qr, q0

    def compute_scores(self, qs, ql, qr, q0, pad_mask=None):
        """
        Returns:
          cls_logits   [B, N, num_cls]
          edge_scores  [B, N, N]
          left_scores  [B, N, N]  softmax probs
          right_scores [B, N, N]  softmax probs
          left_logits  [B, N, N]  raw logits   (for CE loss)
          right_logits [B, N, N]  raw logits   (for CE loss)
        """
        cls_logits = self.cls_head(qs)
        scale = q0.size(-1) ** 0.5
        raw_left  = torch.bmm(q0, ql.transpose(1, 2)) / scale
        raw_right = torch.bmm(q0, qr.transpose(1, 2)) / scale

        if pad_mask is not None:
            raw_left  = raw_left.masked_fill(pad_mask.unsqueeze(1), float('-inf'))
            raw_right = raw_right.masked_fill(pad_mask.unsqueeze(1), float('-inf'))

        left_scores  = F.softmax(raw_left,  dim=-1)
        right_scores = F.softmax(raw_right, dim=-1)

        left_logits  = torch.nan_to_num(raw_left,  nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        right_logits = torch.nan_to_num(raw_right, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        left_scores  = torch.nan_to_num(left_scores,  nan=0.0)
        right_scores = torch.nan_to_num(right_scores, nan=0.0)

        edge_scores = right_scores + left_scores.transpose(1, 2)
        return cls_logits, edge_scores, left_scores, right_scores, left_logits, right_logits
