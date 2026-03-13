import torch
import torch.nn as nn
import torch.nn.functional as F


class NAMERLoss(nn.Module):
    """
    L_all = L_VAT + lambda * L_PGD
    L_PGD = L_self + w_conn * (L_left + L_right)

    L_VAT uses CrossEntropyLoss over K+1 classes (K vocab + 1 background).
    Paper Eq.(3): L_VAT = CE(P_vat, P*)

    none_weight: class weight for the background (none/∅) class in VAT loss.
      - 1.0  = plain CE, matches paper exactly
      - 0.50 = penalise background less → higher recall, lower FP rate
      Empirically plain CE gives rec≈61%; none_weight=0.50 targets rec≈80%.
    """
    def __init__(self, lam: float = 0.5, w_conn: float = 1.0,
                 none_idx: int = 4, vocab_size: int = 186,
                 none_weight: float = 1.0):
        super().__init__()
        self.lam         = lam
        self.w_conn      = w_conn
        self.none_idx    = none_idx
        self.none_weight = none_weight
        self.vocab_size  = vocab_size   # = K+1 total classes (background at none_idx)

        # VAT weight buffer — allocated lazily in forward() on the correct device.
        # none_weight < 1.0 penalises background less: model predicts more real tokens
        # (higher recall). none_weight = 1.0 skips weight tensor entirely (fast path).
        self._vat_weight = None

        # PGD self-correction: ignore SOS/EOS/PAD positions (marked -100 in targets)
        self.pgd_ce  = nn.CrossEntropyLoss(ignore_index=-100)
        # Connectivity heads
        self.conn_ce = nn.CrossEntropyLoss()

    def _vat_ce(self, device: torch.device) -> nn.CrossEntropyLoss:
        """Return VAT CrossEntropyLoss, rebuilding weight tensor if device changed."""
        if self.none_weight == 1.0:
            return nn.CrossEntropyLoss()
        if self._vat_weight is None or self._vat_weight.device != device:
            w = torch.ones(self.vocab_size, device=device)
            w[self.none_idx] = self.none_weight
            self._vat_weight = w
        return nn.CrossEntropyLoss(weight=self._vat_weight)

    def forward(self, out, vat_tgt, pgd_tgt, l_tgt, r_tgt, mask, conn_mask=None):
        """
        conn_mask: [B, L] float tensor — 1 where BOTH the token and its GT
                   left/right neighbor were detected by VAT (valid connectivity
                   target), 0 where the target would be a self-loop guess.
                   Falls back to `mask` if not provided (legacy behaviour).
        """
        B, L = pgd_tgt.shape
        device = vat_tgt.device

        # ── VAT Loss ──────────────────────────────────────────────────────────
        # vat_logits: [B, K+1, H, W]   vat_tgt: [B, H, W]
        L_vat = self._vat_ce(device)(out['vat_logits'], vat_tgt)

        # ── PGD Self-Correction Loss ──────────────────────────────────────────
        pgd_valid = (pgd_tgt.reshape(-1) != -100)
        if pgd_valid.sum() > 0:
            L_self = self.pgd_ce(
                out['pgd_cls_logits'].reshape(B * L, -1),
                pgd_tgt.reshape(-1))
        else:
            L_self = torch.tensor(0.0, device=device)

        # ── Connectivity Losses ───────────────────────────────────────────────
        # Use conn_mask (clean targets only) instead of the general padding mask.
        # This avoids self-loop targets for tokens whose GT neighbors are missing.
        cm = (conn_mask if conn_mask is not None else mask).reshape(-1).bool()
        l_tgt_c = l_tgt.clamp(0, L - 1)
        r_tgt_c = r_tgt.clamp(0, L - 1)

        if cm.sum() > 0:
            L_left  = self.conn_ce(
                out['left_logits'].reshape(B * L, -1)[cm],
                l_tgt_c.reshape(-1)[cm])
            L_right = self.conn_ce(
                out['right_logits'].reshape(B * L, -1)[cm],
                r_tgt_c.reshape(-1)[cm])
        else:
            L_left  = torch.tensor(0.0, device=device)
            L_right = torch.tensor(0.0, device=device)

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
        B, T, V = logits.shape
        targets = token_ids[:, 1:T+1]
        return self.ce(logits.reshape(B * T, V), targets.reshape(-1))