"""
Hungarian bipartite matching for VAT target generation.

Paper Section 3.2 + Listing 2.1:
  When using DWAP: T[i] = argmax(α_t_i) from DWAP attention map
  When no DWAP:    T[i] = uniform column spread (training from scratch)
"""
import torch
import torch.nn.functional as F
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import linear_sum_assignment

# One persistent pool — threads are reused across steps.
_MATCH_POOL = ThreadPoolExecutor(max_workers=4)


def _build_cost_matrix(b, token_ids, P_vat, map_h, map_w,
                        none_idx, pad_idx, sos_idx, eos_idx,
                        open_idx, close_idx,
                        vocab_sz, km, T_positions):
    """
    Build the cost matrix for one sample — runs in a thread.
    T_positions: list of B items, mỗi item là Tensor[n,2] hoặc None.
    Returns (b, valid_cpu, orig_idx_cpu, cost_np) or None if n==0.
    """
    pad    = km // 2
    device = P_vat.device

    tids_b   = token_ids[b]
    # Exclude: padding, SOS, EOS, open-brace {, close-brace }
    # { and } are structural/imaginary tokens — they have no spatial position
    keep = (
        (tids_b != pad_idx)  &
        (tids_b != sos_idx)  &
        (tids_b != eos_idx)  &
        (tids_b != open_idx) &
        (tids_b != close_idx)
    )
    valid    = tids_b[keep].clamp(0, vocab_sz - 1)
    orig_idx = keep.nonzero(as_tuple=False).squeeze(-1)
    n        = valid.size(0)
    if n == 0:
        return None

    # ── Estimate positions T ─────────────────────────────────────────────
    # T_b là None nếu ảnh không có trong cache → dùng uniform spread
    T_b = T_positions[b] if (T_positions is not None) else None

    if T_b is not None and T_b.size(0) > 0:
        n_pos  = min(n, T_b.size(0))
        rows_t = T_b[:n_pos, 0].long().to(device)
        cols_t = T_b[:n_pos, 1].long().to(device)
        if n_pos < n:
            extra_cols = torch.linspace(0, map_w - 1, n - n_pos, device=device).long()
            extra_rows = torch.full((n - n_pos,), map_h // 2, dtype=torch.long, device=device)
            rows_t = torch.cat([rows_t, extra_rows])
            cols_t = torch.cat([cols_t, extra_cols])
    else:
        # Uniform spread fallback
        rows_t = torch.full((n,), map_h // 2, dtype=torch.long, device=device)
        cols_t = torch.linspace(0, map_w - 1, n, device=device).long()

    rows_t = rows_t.clamp(0, map_h - 1)
    cols_t = cols_t.clamp(0, map_w  - 1)

    # ── T → indicator matrix [n, H, W], max-pool ────────────────────────
    y_idx = torch.arange(n, device=device)
    T_mat = torch.zeros(n, map_h, map_w, device=device)
    T_mat[y_idx, rows_t, cols_t] = 1.0
    T_mat = F.max_pool2d(T_mat.unsqueeze(0),
                         kernel_size=(km, km), stride=1, padding=pad)[0]

    # ── Distance matrix, cost outside window = 1e6 ──────────────────────
    P_b      = P_vat[b]
    dist_mat = (P_b[valid].float() - T_mat).abs()
    dist_mat = dist_mat * T_mat + (1.0 - T_mat) * 1e6

    cost_np = dist_mat.view(n, -1).cpu().numpy()
    return b, valid.cpu(), orig_idx.cpu(), cost_np


def _run_hungarian(args):
    """Pure numpy/scipy work — GIL released, safe to parallelise."""
    b, valid_cpu, orig_idx_cpu, cost_np, map_w = args
    row_ind, col_ind = linear_sum_assignment(cost_np)
    h_ind = col_ind // map_w
    w_ind = col_ind %  map_w
    return b, valid_cpu, orig_idx_cpu, row_ind, h_ind, w_ind


def bipartite_match_vat_targets(
    token_ids, P_vat, map_h, map_w,
    none_idx, pad_idx, sos_idx, eos_idx,
    open_idx, close_idx,
    vocab_sz, km=5, device='cpu',
    T_positions=None,
):
    """
    Build VAT target map P* via Hungarian bipartite matching.

    T_positions: None  → uniform spread cho cả batch
                 list  → mỗi phần tử là Tensor[n,2] hoặc None
                         (None entry = uniform spread cho sample đó)
    """
    B = token_ids.size(0)
    L = token_ids.size(1)

    vat_tgt     = torch.full((B, map_h, map_w), none_idx, dtype=torch.long, device=device)
    coords_tgt  = torch.zeros(B, L, 2,                    dtype=torch.long, device=device)
    raw_idx_tgt = torch.full((B, map_h, map_w), -1,       dtype=torch.long, device=device)

    with torch.no_grad():
        # Phase 1: build cost matrices on GPU (sequential)
        prepared = []
        for b in range(B):
            result = _build_cost_matrix(
                b, token_ids, P_vat, map_h, map_w,
                none_idx, pad_idx, sos_idx, eos_idx,
                open_idx, close_idx,
                vocab_sz, km, T_positions)
            if result is not None:
                prepared.append(result)

        if not prepared:
            return vat_tgt, coords_tgt, raw_idx_tgt

        # Phase 2: Hungarian in parallel on CPU
        hungarian_args = [(b, vc, oi, cn, map_w) for b, vc, oi, cn in prepared]
        results = list(_MATCH_POOL.map(_run_hungarian, hungarian_args))

        # Phase 3: fill output tensors (vectorised)
        for b, valid_cpu, orig_idx_cpu, row_ind, h_ind, w_ind in results:
            ri_t = torch.from_numpy(row_ind).long()
            hi_t = torch.from_numpy(h_ind).long()
            wi_t = torch.from_numpy(w_ind).long()
            oi_t = orig_idx_cpu[ri_t]

            vat_tgt[b, hi_t, wi_t]     = valid_cpu[ri_t].to(device)
            raw_idx_tgt[b, hi_t, wi_t] = oi_t.to(device)
            coords_tgt[b, oi_t, 0]     = hi_t.to(device)
            coords_tgt[b, oi_t, 1]     = wi_t.to(device)

    return vat_tgt, coords_tgt, raw_idx_tgt


def make_vat_targets(token_ids, P_vat, map_h, map_w,
                     vocab, device, T_positions=None, km=5):
    """Wrapper: build VAT targets and clamp to valid range.

    Excludes { (open_idx) and } (close_idx) from bipartite matching —
    these are structural/imaginary tokens with no spatial position.
    """
    vat_tgt, coords_tgt, raw_idx_tgt = bipartite_match_vat_targets(
        token_ids, P_vat, map_h, map_w,
        vocab.none_idx, vocab.pad_idx, vocab.sos_idx, vocab.eos_idx,
        vocab.open_idx, vocab.close_idx,
        len(vocab), km=km, device=device, T_positions=T_positions,
    )
    return vat_tgt.clamp(0, len(vocab) - 1), coords_tgt, raw_idx_tgt


def get_dwap_positions(dwap_model, encoder, images,
                       token_ids, sos_idx, eos_idx, map_h, map_w,
                       max_len=150, device='cpu'):
    """
    Greedy DWAP decode → attention argmax positions.
    Chỉ dùng khi build_dwap_cache không có sẵn.
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
        flat_idx = a.view(a.size(0), -1).argmax(dim=-1)
        rows = (flat_idx // map_w).unsqueeze(-1)
        cols = (flat_idx %  map_w).unsqueeze(-1)
        positions.append(torch.cat([rows, cols], dim=-1))

    return positions