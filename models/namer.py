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
    def _infer(self, f16x, probs, vocab=None):
        """
        Greedy NAR inference: VAT → filter → insert imaginary } → PGD → DAG path.

        Imaginary tokens (Paper Sec 3.3):
          After each structural token (^, _, \\frac, \\sqrt, etc.), N imaginary '}'
          are appended to the PGD input sequence. PGD then learns to place them
          correctly in the graph, enabling proper structure reconstruction.

        Key design decisions:
        - NO probability threshold beyond argmax (see previous notes).
        - Imaginary } share coordinates with their parent structural token.
        - STRUCT_N_GROUPS (fixed dict) determines imaginary count per token type.
        - After path selection, '}' appear explicitly; '{' are re-inserted by
          a lightweight reconstruction pass using structural rules.
        """
        from train_namer import STRUCT_N_GROUPS   # avoid circular; lazy import

        B, K1, Hm, Wm = probs.shape
        none_idx = _NONE_IDX
        vocab_sz = K1
        sos_idx, eos_idx, pad_idx = 1, 2, 0
        map_h, map_w = Hm, Wm

        # close_idx: vocab index of '}', used for imaginary tokens
        close_idx = vocab.close_idx if vocab is not None else -1
        # struct_by_id: {token_id → n_imaginary_}} for quick lookup
        struct_by_id: dict[int, int] = {}
        if vocab is not None:
            for tok_str, n in STRUCT_N_GROUPS.items():
                tid = vocab.t2i.get(tok_str, -1)
                if tid >= 0:
                    struct_by_id[tid] = n

        results = []

        for b in range(B):
            prob_b = probs[b]                         # [K+1, H, W]
            pred   = prob_b.argmax(dim=0)             # [H, W]

            tok_mask = (pred != none_idx)
            pos = tok_mask.nonzero(as_tuple=False)    # [n_vis, 2] (row, col)
            if pos.size(0) == 0:
                results.append([])
                continue

            # Sort by column (left → right), consistent with training
            col_order     = pos[:, 1].argsort()
            pos           = pos[col_order]
            detected_tids = pred[pos[:, 0], pos[:, 1]]   # [n_vis]
            n_vis = detected_tids.size(0)

            # ── Insert imaginary } tokens after structural tokens ─────────────
            # This mirrors the training PGD input construction exactly.
            exp_tids   = []   # expanded token ids
            exp_coords = []   # expanded coordinates

            for k in range(n_vis):
                tid = int(detected_tids[k].item())
                y   = int(pos[k, 0].item())
                x   = int(pos[k, 1].item())

                exp_tids.append(tid)
                exp_coords.append((y, x))

                n_imag = struct_by_id.get(tid, 0)
                for _ in range(n_imag):
                    exp_tids.append(close_idx if close_idx >= 0 else tid)
                    exp_coords.append((y, x))   # same spatial position as parent

            n_expanded = len(exp_tids)

            # ── Build PGD input: [SOS, tok1..tokN, EOS] ───────────────────────
            pgd_len   = n_expanded + 2
            pgd_input = torch.full((1, pgd_len), pad_idx,
                                    dtype=torch.long, device=f16x.device)
            coords    = torch.zeros(1, pgd_len, 2, device=f16x.device)
            coords[:, :, 0] = map_h // 2
            coords[:, :, 1] = 0

            pgd_input[0, 0]              = sos_idx
            pgd_input[0, 1:n_expanded+1] = torch.tensor(
                exp_tids, dtype=torch.long, device=f16x.device)
            pgd_input[0, n_expanded+1]   = eos_idx

            if exp_coords:
                coords[0, 1:n_expanded+1] = torch.tensor(
                    exp_coords, dtype=torch.float, device=f16x.device)

            # PGD forward
            pad_mask = (pgd_input == pad_idx)
            qs, ql, qr, q0 = self.pgd(f16x[b:b+1], pgd_input, coords)
            cls_logits, edge_scores, _, _, _, _ = \
                self.pgd.compute_scores(qs, ql, qr, q0, pad_mask=pad_mask)

            corrected = cls_logits[0].argmax(dim=-1).cpu().tolist()
            corrected[0]            = sos_idx
            corrected[n_expanded+1] = eos_idx

            E = edge_scores[0].cpu().numpy()
            results.append(_path_selection(corrected, E, none_idx))

        return results