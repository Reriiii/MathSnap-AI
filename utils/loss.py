import torch
import torch.nn as nn
import torch.nn.functional as F


class NAMERLoss(nn.Module):
    """
    L_all = L_VAT + λ * L_PGD
    L_PGD = L_self + w_conn * (L_left + L_right)

    L_VAT = CrossEntropy(vat_logits, vat_tgt)   — paper Eq. (3)
    """
    def __init__(self, lam: float = 0.5, w_conn: float = 1.0, none_idx: int = 4):
        super().__init__()
        self.lam      = lam
        self.w_conn   = w_conn
        self.none_idx = none_idx
        # VAT loss: standard multi-class CE over all spatial positions.
        # Paper Eq.(3): L_VAT = CrossEntropy(P, P*)
        # No ignore_index needed — none_idx (∅) is a valid class to predict.
        self.vat_ce = nn.CrossEntropyLoss()
        self.ce     = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, out, vat_tgt, pgd_tgt, l_tgt, r_tgt, mask):
        B, L = pgd_tgt.shape

        # ── VAT Loss: CrossEntropy (paper Eq. 3) ──────────────────────────────
        # vat_logits: [B, vocab_size, H/8, W/8]   vat_tgt: [B, H/8, W/8]
        # CE treats this as multi-class classification at every spatial position.
        # The model is trained with softmax and inferred with argmax, which is
        # fully consistent with CrossEntropyLoss.
        L_vat = self.vat_ce(out['vat_logits'], vat_tgt)

        pgd_valid = (pgd_tgt.reshape(-1) != -100)
        if pgd_valid.sum() > 0:
            L_self = self.ce(
                out['pgd_cls_logits'].reshape(B * L, -1), pgd_tgt.reshape(-1))
        else:
            L_self = torch.tensor(0.0, device=vat_tgt.device)

        m      = mask.reshape(-1).bool()
        l_tgt_c = l_tgt.clamp(0, L - 1)
        r_tgt_c = r_tgt.clamp(0, L - 1)

        if m.sum() > 0:
            L_left  = self.ce(
                out['left_logits'].reshape(B * L, -1)[m],
                l_tgt_c.reshape(-1)[m])
            L_right = self.ce(
                out['right_logits'].reshape(B * L, -1)[m],
                r_tgt_c.reshape(-1)[m])
        else:
            L_left  = torch.tensor(0.0, device=vat_tgt.device)
            L_right = torch.tensor(0.0, device=vat_tgt.device)

        L_pgd = L_self + self.w_conn * (L_left + L_right)
        L_all = L_vat + self.lam * L_pgd
        return L_all, {
            'L_all':  L_all.item(),  'L_vat':   L_vat.item(),
            'L_pgd':  L_pgd.item(),  'L_self':  L_self.item(),
            'L_left': L_left.item(), 'L_right': L_right.item(),
        }


class DWAPLoss(nn.Module):
    """Standard CE loss for DWAP AR training."""
    def __init__(self, pad_idx: int = 0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, logits, token_ids):
        """
        logits:    [B, T, vocab_sz]   T = min(L-1, max_steps) (may be < L-1)
        token_ids: [B, L]             targets = token_ids[:, 1:T+1]
        """
        B, T, V = logits.shape
        # T may be capped (e.g. 60) — only supervise those T positions
        targets = token_ids[:, 1:T+1]    # shift right, take first T targets
        return self.ce(logits.reshape(B * T, V), targets.reshape(-1))