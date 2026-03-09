import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from .encoder import DenseNetEncoder
from .vat import VAT
from .pgd import PGD
from utils.metrics import _path_selection

# ∅ background token is always index 4 in vocab (_SPECIAL order)
_NONE_IDX = 4


class NAMER(nn.Module):
    """
    NAMER: Non-Autoregressive Modeling for HMER.
    Combines VAT (visual tokenizer) + PGD (parallel graph decoder).
    """
    def __init__(self, vocab_size: int, d: int = 256, heads: int = 8,
                 pgd_layers: int = 2, drop: float = 0.3):
        super().__init__()
        self.enc = DenseNetEncoder()
        ch8      = self.enc.ch_8x
        ch16     = self.enc.ch_16x
        self.vat = VAT(ch8, ch16, d, vocab_size)
        self.pgd = PGD(ch16, d, heads, pgd_layers, vocab_size, vocab_size, drop)

    def forward(self, images, token_ids=None):
        f8x, f16x         = self.enc(images)
        probs, vat_logits = self.vat(f8x, f16x)

        if token_ids is None:
            return self._infer(f16x, probs)

        qs, ql, qr, q0 = self.pgd(f16x, token_ids)
        cls_logits, edge_scores, left_scores, right_scores, left_logits, right_logits = \
            self.pgd.compute_scores(qs, ql, qr, q0)
        return {
            'vat_logits':     vat_logits,
            'pgd_cls_logits': cls_logits,
            'left_scores':    left_scores,
            'right_scores':   right_scores,
            'left_logits':    left_logits,
            'right_logits':   right_logits,
        }

    @torch.no_grad()
    def _infer(self, f16x, probs):
        """Greedy NAR inference: VAT → PGD → DAG path selection."""
        B, K1, Hm, Wm = probs.shape
        none_idx = _NONE_IDX
        vocab_sz = K1
        # SOS=1, EOS=2, PAD=0 — must match vocab._SPECIAL order
        sos_idx, eos_idx, pad_idx = 1, 2, 0
        results  = []

        for b in range(B):
            prob_b = probs[b]                             # [K+1, H, W]
            max_probs, pred = prob_b.max(dim=0)           # [H, W]
            
            # Require >0.5 probability (with sigmoid focal loss, anything above 0.5 is very strong)
            tok_mask = (pred != none_idx) & (max_probs > 0.5)
            pos  = tok_mask.nonzero(as_tuple=False)       # [n, 2] (row, col)
            if pos.size(0) == 0:
                results.append([])
                continue

            # ── Sort by column (left→right), matching training flow ──────
            col_order = pos[:, 1].argsort()
            pos = pos[col_order]
            detected_tids = pred[pos[:, 0], pos[:, 1]]  # [n]
            n = detected_tids.size(0)

            # ── Build PGD input: [SOS, tok1, tok2, ..., tokN, EOS] ───────
            # This MUST match how train_epoch builds pgd_input (VAT branch)
            pgd_len = n + 2
            pgd_input = torch.full((1, pgd_len), pad_idx,
                                    dtype=torch.long, device=f16x.device)
            coords = torch.zeros(1, pgd_len, 2, device=f16x.device)
            # Safe padding for SOS/EOS
            coords[:, :, 0] = f16x.size(2)  # H_f16 is roughly H_f8 // 2
            coords[:, :, 1] = 0
            
            pgd_input[0, 0]     = sos_idx
            pgd_input[0, 1:n+1] = detected_tids.clamp(0, vocab_sz - 1)
            coords[0, 1:n+1] = pos.float()  # Stacked [row, col] from non-zero
            pgd_input[0, n+1]   = eos_idx

            # ── PGD forward ──────────────────────────────────────────────
            pad_mask = (pgd_input == pad_idx)
            qs, ql, qr, q0 = self.pgd(f16x[b:b+1], pgd_input, coords)
            cls_logits, edge_scores, _, _, _, _ = \
                self.pgd.compute_scores(qs, ql, qr, q0, pad_mask=pad_mask)

            corrected = cls_logits[0].argmax(dim=-1).cpu().tolist()
            E         = edge_scores[0].cpu().numpy()
            results.append(_path_selection(corrected, E, none_idx))

        return results

