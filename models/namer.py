import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from .encoder import DenseNetEncoder
from .vat import VAT
from .pgd import PGD
from utils.metrics import _path_selection

# background token index — must match vocab._SPECIAL order
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

        # Training mode: caller accesses enc/vat/pgd directly; this branch is unused
        qs, ql, qr, q0 = self.pgd(f16x, token_ids)
        cls_logits, edge_scores, left_scores, right_scores, left_logits, right_logits = \
            self.pgd.compute_scores(qs, ql, qr, q0)
        return {
            'vat_logits':     vat_logits,
            'pgd_cls_logits': cls_logits,
            'left_logits':    left_logits,
            'right_logits':   right_logits,
        }

    @torch.no_grad()
    def _infer(self, f16x, probs):
        """
        Greedy NAR inference: VAT → filter → PGD → DAG path selection.

        Key design decisions:
        - NO probability threshold beyond argmax. If pred != none_idx, the
          model already assigned more mass to that token than to background.
          A secondary threshold 0.5 was silently dropping correct-but-uncertain
          symbols (visually similar pairs like alpha/a, etc.).
        - edge_scores are row-normalized before path_selection so the eps
          threshold is comparable across batches regardless of sequence length.
        """
        B, K1, Hm, Wm = probs.shape
        none_idx = _NONE_IDX
        vocab_sz = K1
        sos_idx, eos_idx, pad_idx = 1, 2, 0
        map_h, map_w = Hm, Wm
        results = []

        for b in range(B):
            prob_b = probs[b]                         # [K+1, H, W]
            pred   = prob_b.argmax(dim=0)             # [H, W]

            # Accept all positions where the model prefers a real token
            # over background — no secondary probability threshold.
            tok_mask = (pred != none_idx)
            pos = tok_mask.nonzero(as_tuple=False)    # [n, 2] (row, col)
            if pos.size(0) == 0:
                results.append([])
                continue

            # Sort by column (left → right), matches training VAT branch
            col_order     = pos[:, 1].argsort()
            pos           = pos[col_order]
            detected_tids = pred[pos[:, 0], pos[:, 1]]   # [n]
            n = detected_tids.size(0)

            # Build PGD input: [SOS, tok1, ..., tokN, EOS]
            pgd_len   = n + 2
            pgd_input = torch.full((1, pgd_len), pad_idx,
                                    dtype=torch.long, device=f16x.device)
            coords    = torch.zeros(1, pgd_len, 2, device=f16x.device)

            # SOS/EOS default coords: vertical center, left edge (matches training)
            coords[:, :, 0] = map_h // 2
            coords[:, :, 1] = 0

            pgd_input[0, 0]     = sos_idx
            pgd_input[0, 1:n+1] = detected_tids.clamp(0, vocab_sz - 1)
            coords[0, 1:n+1]    = pos.float()   # f8x coords [row, col]
            pgd_input[0, n+1]   = eos_idx

            # PGD forward
            pad_mask = (pgd_input == pad_idx)
            qs, ql, qr, q0 = self.pgd(f16x[b:b+1], pgd_input, coords)
            cls_logits, edge_scores, _, _, _, _ = \
                self.pgd.compute_scores(qs, ql, qr, q0, pad_mask=pad_mask)

            corrected = cls_logits[0].argmax(dim=-1).cpu().tolist()
            # Restore SOS/EOS (SCH ignores them in training, outputs garbage)
            corrected[0]   = sos_idx
            corrected[n+1] = eos_idx

            E = edge_scores[0].cpu().numpy()
            results.append(_path_selection(corrected, E, none_idx))

        return results