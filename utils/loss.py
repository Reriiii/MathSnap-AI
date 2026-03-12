import torch
import torch.nn as nn
import torch.nn.functional as F


class NAMERLoss(nn.Module):
    """
    L_all = L_VAT + lambda * L_PGD
    L_PGD = L_self + L_left + L_right   (equal weights, per paper Eq 8)
    L_VAT = CrossEntropy(P, P*)          (plain CE, per paper Eq 3, no background weight)
    """
    def __init__(self, lam: float = 0.5, w_conn: float = 1.0,
                 none_idx: int = 4, vocab_size: int = 191):
        super().__init__()
        self.lam      = lam
        self.w_conn   = w_conn
        self.none_idx = none_idx

        # Plain CE — no background weighting (paper Eq 3)
        self.vat_ce  = nn.CrossEntropyLoss()
        self.pgd_ce  = nn.CrossEntropyLoss(ignore_index=-100)
        self.conn_ce = nn.CrossEntropyLoss()

    def forward(self, out, vat_tgt, pgd_tgt, l_tgt, r_tgt, mask):
        B, L = pgd_tgt.shape

        # VAT loss (plain CE, paper Eq 3)
        L_vat = self.vat_ce(out['vat_logits'], vat_tgt)

        # PGD self-correction loss
        pgd_valid = (pgd_tgt.reshape(-1) != -100)
        if pgd_valid.sum() > 0:
            L_self = self.pgd_ce(
                out['pgd_cls_logits'].reshape(B * L, -1),
                pgd_tgt.reshape(-1))
        else:
            L_self = torch.tensor(0.0, device=vat_tgt.device)

        # Connectivity losses
        m       = mask.reshape(-1).bool()
        l_tgt_c = l_tgt.clamp(0, L - 1)
        r_tgt_c = r_tgt.clamp(0, L - 1)

        if m.sum() > 0:
            L_left  = self.conn_ce(
                out['left_logits'].reshape(B * L, -1)[m],
                l_tgt_c.reshape(-1)[m])
            L_right = self.conn_ce(
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
    def __init__(self, pad_idx: int = 0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, logits, token_ids):
        B, T, V = logits.shape
        targets = token_ids[:, 1:T+1]
        return self.ce(logits.reshape(B * T, V), targets.reshape(-1))