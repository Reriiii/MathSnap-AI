"""
Hungarian bipartite matching for VAT target generation.

Paper Section 3.2 + Listing 2.1:
  When using DWAP: T[i] = argmax(α_t_i) from DWAP attention map
  When no DWAP:    T[i] = uniform column spread (training from scratch)
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


def bipartite_match_vat_targets(
    token_ids, P_vat, map_h, map_w,
    none_idx, pad_idx, sos_idx, eos_idx,
    vocab_sz, km=5, device='cpu',
    T_positions=None,
):
    """
    Build VAT target map P* via Hungarian bipartite matching.

    Args:
        token_ids:   [B, L] GT token indices (with SOS/EOS/pad)
        P_vat:       [B, K+1, H, W] VAT predicted probabilities
        map_h, map_w: stride-8 feature map dimensions
        km:          matching window size (paper default 5)
        T_positions: optional [B, n, 2] DWAP-provided (row, col) per token.
                     If None → uniform column spread used instead.

    Returns:
        vat_tgt: [B, H, W] long tensor with token IDs or none_idx
    """
    B    = token_ids.size(0)
    L    = token_ids.size(1)
    vat_tgt = torch.full((B, map_h, map_w), none_idx,
                          dtype=torch.long, device=device)
    coords_tgt = torch.zeros(B, L, 2, dtype=torch.long, device=device)
    pad  = km // 2

    with torch.no_grad():
        for b in range(B):
            tids_b = token_ids[b]
            keep   = ((tids_b != pad_idx) & (tids_b != sos_idx)
                      & (tids_b != eos_idx))
            valid  = tids_b[keep].clamp(0, vocab_sz - 1)
            orig_idx = keep.nonzero(as_tuple=False).squeeze(-1)
            n      = valid.size(0)
            if n == 0:
                continue

            # ── Step 1: Estimate positions T ────────────────────────────────
            if T_positions is not None:
                # Use DWAP attention argmax positions
                T_b   = T_positions[b]                 # [n_dwap, 2]
                n_pos = min(n, T_b.size(0))
                rows_t = T_b[:n_pos, 0].long().to(device)
                cols_t = T_b[:n_pos, 1].long().to(device)
                # If DWAP gave fewer positions than tokens, pad with uniform
                if n_pos < n:
                    extra_cols = torch.linspace(0, map_w - 1, n - n_pos,
                                                device=device).long()
                    extra_rows = torch.full((n - n_pos,), map_h // 2,
                                             dtype=torch.long, device=device)
                    rows_t = torch.cat([rows_t, extra_rows])
                    cols_t = torch.cat([cols_t, extra_cols])
            else:
                # Uniform column spread (no DWAP)
                rows_t = torch.full((n,), map_h // 2, dtype=torch.long, device=device)
                cols_t = torch.linspace(0, map_w - 1, n, device=device).long()

            # Clamp to valid range
            rows_t = rows_t.clamp(0, map_h - 1)
            cols_t = cols_t.clamp(0, map_w  - 1)

            # ── Step 2: T → indicator matrix [n, H, W] ──────────────────────
            y_idx  = torch.arange(n, device=device)
            T_mat  = torch.zeros(n, map_h, map_w, device=device)
            T_mat[y_idx, rows_t, cols_t] = 1.0

            # ── Step 3: Max-pool km×km for local matching windows ────────────
            T_mat = F.max_pool2d(
                T_mat.unsqueeze(0),
                kernel_size=(km, km), stride=1, padding=pad
            )[0]

            # ── Step 4: Distance = |P_vat[y_label] - T_mat| ─────────────────
            P_b      = P_vat[b]
            dist_mat = (P_b[valid].float() - T_mat).abs()

            # ── Step 5: Outside window → large cost ──────────────────────────
            dist_mat = dist_mat * T_mat + (1.0 - T_mat) * 1e6

            # ── Step 6: Hungarian algorithm ───────────────────────────────────
            cost_np          = dist_mat.view(n, -1).cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            h_ind            = col_ind // map_w
            w_ind            = col_ind %  map_w

            # ── Step 7: Fill P* ──
            for ri, h_i, w_i in zip(row_ind, h_ind, w_ind):
                vat_tgt[b, h_i, w_i] = valid[ri].item()
                coords_tgt[b, orig_idx[ri], 0] = h_i
                coords_tgt[b, orig_idx[ri], 1] = w_i

    return vat_tgt, coords_tgt


def make_vat_targets(token_ids, P_vat, map_h, map_w,
                     vocab, device, T_positions=None, km=5):
    """Wrapper: build VAT targets and clamp to valid range."""
    vat_tgt, coords_tgt = bipartite_match_vat_targets(
        token_ids, P_vat, map_h, map_w,
        vocab.none_idx, vocab.pad_idx, vocab.sos_idx, vocab.eos_idx,
        len(vocab), km=km, device=device, T_positions=T_positions,
    )
    return vat_tgt.clamp(0, len(vocab) - 1), coords_tgt


def get_dwap_positions(dwap_model, encoder, images,
                       token_ids, sos_idx, eos_idx, map_h, map_w,
                       max_len=150, device='cpu'):
    """
    Run DWAP greedy decoding and return attention argmax positions.

    Used in NAMER train_epoch to get T_positions for bipartite matching.

    Returns:
        positions: list of B tensors [n_t, 2] (row, col) for each sample.
                   Length of each list may differ from GT token count.
    """
    with torch.no_grad():
        f8x, _ = encoder(images)
        _, attn_maps = dwap_model.decode(
            f8x, sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len)

    positions = []
    for b in range(len(attn_maps)):
        a = attn_maps[b]    # [n_t, H, W]
        if a.numel() == 0:
            positions.append(torch.zeros(0, 2, dtype=torch.long))
            continue
        flat_idx = a.view(a.size(0), -1).argmax(dim=-1)   # [n_t]
        rows = (flat_idx // map_w).unsqueeze(-1)
        cols = (flat_idx %  map_w).unsqueeze(-1)
        positions.append(torch.cat([rows, cols], dim=-1))  # [n_t, 2]

    return positions
