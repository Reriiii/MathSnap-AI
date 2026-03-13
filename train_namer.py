"""
train_namer.py — Train NAMER on HME100K
=========================================
Optionally loads a pretrained DWAP checkpoint to provide accurate
token position estimates T for VAT bipartite matching.

Usage:
    python train_namer.py
    python train_namer.py --dwap checkpoints/dwap_best.pth
    python train_namer.py --resume checkpoints/ep025_11.04.pth
"""
import math, os, random, argparse
from collections import defaultdict
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from config import Config
from data import build_datasets, Vocabulary
from data.dataset import _collate
from models import NAMER, DenseNetEncoder, DWAP
from utils.loss import NAMERLoss
from utils.metrics import compute_exprate
from utils.matching import make_vat_targets, get_dwap_positions


# ── Imaginary token helpers ────────────────────────────────────────────────────
# Paper Sec 3.3: for each structural token, PGD receives N imaginary "}"
# appended immediately after it, representing its brace-group boundaries.
# This dict is used at INFERENCE (fixed rules).
# At TRAINING we compute the count dynamically from GT (handles x^2 vs x^{2}).
STRUCT_N_GROUPS: dict[str, int] = {
    '\\frac':      2,   # \frac{num}{den}
    '\\sqrt':      1,   # \sqrt{arg}
    '^':           1,   # x^{exp}
    '_':           1,   # x_{sub}
    '\\overset':   2,   # \overset{top}{base}
    '\\underset':  2,   # \underset{bot}{base}
    '\\binom':     2,   # \binom{n}{k}
    '\\stackrel':  2,   # \stackrel{top}{bot}
    '\\overline':  1,
    '\\underline': 1,
    '\\hat':       1,
    '\\tilde':     1,
    '\\dot':       1,
    '\\ddot':      1,
    '\\bar':       1,
    '\\vec':       1,
    '\\widetilde': 1,
    '\\widehat':   1,
    '\\mathbf':    1,
    '\\mathit':    1,
    '\\mathrm':    1,
    '\\limits':    1,
}


def _get_close_brace_parents(token_strings: list[str],
                              open_tok: str = '{',
                              close_tok: str = '}') -> list[tuple[int, int, int]]:
    """
    For each '}' in token_strings, find which structural token owns it
    and which group index (0-based) it closes.

    Algorithm: scan left-to-right. When a structural token is seen, push
    it onto a pending stack. When '{' is seen, the MOST RECENTLY pushed
    structural token claims it (LIFO). When '}' is seen, pop the current
    open-group owner.

    Returns:
        List of (close_pos, struct_pos, group_idx) tuples, one per '}'.
        Unmatched '}' (no owner structural token) are silently skipped.
    """
    # pending[i] = [struct_token_index, remaining_groups_count]
    pending: list[list[int]] = []
    stack:   list             = []   # currently open group owners
    result:  list             = []

    for i, tok in enumerate(token_strings):
        n_groups = STRUCT_N_GROUPS.get(tok, 0)
        if n_groups > 0:
            pending.append([i, n_groups])

        if tok == open_tok:
            owner = None
            # Most recently pushed structural token with remaining groups claims this {
            for k in range(len(pending) - 1, -1, -1):
                if pending[k][1] > 0:
                    s_idx  = pending[k][0]
                    n_orig = STRUCT_N_GROUPS[token_strings[s_idx]]
                    g_num  = n_orig - pending[k][1]   # 0-indexed group
                    pending[k][1] -= 1
                    if pending[k][1] == 0:
                        pending.pop(k)
                    owner = (s_idx, g_num)
                    break
            stack.append(owner)

        elif tok == close_tok:
            if stack:
                owner = stack.pop()
                if owner is not None:
                    result.append((i, owner[0], owner[1]))

    return result


def _set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn,
                 vocab: Vocabulary, device, checkpoint_dir: str,
                 log_interval: int = 50,
                 dwap_cache: dict = None):
        self.model        = model
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.loss_fn      = loss_fn
        self.vocab        = vocab
        self.device       = device
        self.ckpt_dir     = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.best_er      = 0.0
        self.history      = defaultdict(list)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Pre-computed DWAP positions: img_path -> Tensor[n, 2] (CPU)
        # None = không có DWAP, make_vat_targets dùng uniform spread thay thế
        self.dwap_cache   = dwap_cache

        # AMP dtype: bf16 trên Ampere+ (RTX 30xx/40xx/50xx), fp16 trên P100/V100/T4
        self.use_amp = (device.type == 'cuda')
        if self.use_amp:
            self.amp_dtype = (torch.bfloat16
                              if torch.cuda.is_bf16_supported()
                              else torch.float16)
        else:
            self.amp_dtype = torch.float32
        self.scaler = torch.amp.GradScaler(
            device='cuda',
            enabled=(self.use_amp and self.amp_dtype == torch.float16),
        )
        tqdm.write(f"AMP: {self.amp_dtype} | scaler: {self.scaler.is_enabled()}")

    # ------------------------------------------------------------------
    def train_epoch(self, loader, epoch: int, total_epochs: int,
                    map_h: int, map_w: int):
        """
        Train one epoch. PGD always receives VAT output (paper Sec 3.3) —
        no teacher forcing. Imaginary '}' tokens are inserted after structural
        tokens per the imaginary-token design (see STRUCT_N_GROUPS above).
        """
        self.model.train()
        run = defaultdict(float)

        pbar = tqdm(loader,
                    desc=f"Ep {epoch:>3}/{total_epochs} [train]",
                    leave=True, dynamic_ncols=True, unit='batch')

        for step, batch in enumerate(pbar):
            images       = batch['image'].to(self.device)
            token_ids_gt = batch['token_ids'].to(self.device)

            with torch.amp.autocast(device_type=self.device.type,
                                    dtype=self.amp_dtype, enabled=self.use_amp):
                f8x, f16x             = self.model.enc(images)
                vat_probs, vat_logits = self.model.vat(f8x, f16x)

            # Lấy DWAP positions từ cache (pre-computed), không chạy AR inference
            T_pos = None
            if self.dwap_cache is not None:
                img_paths = batch['img_path']
                T_pos = [self.dwap_cache.get(p) for p in img_paths]
                # None entry = ảnh không có cache → make_vat_targets tự dùng uniform

            vat_tgt, gt_coords, raw_idx_tgt = make_vat_targets(
                token_ids_gt, vat_probs.detach(), map_h, map_w,
                self.vocab, self.device, T_positions=T_pos
            )

            none_idx  = self.vocab.none_idx
            pad_idx   = self.vocab.pad_idx
            sos_idx   = self.vocab.sos_idx
            eos_idx   = self.vocab.eos_idx
            open_idx  = self.vocab.open_idx   # vocab index of '{'
            close_idx = self.vocab.close_idx  # vocab index of '}'
            B         = images.size(0)
            dev       = self.device
            MAX_PGD   = 128

            # ── Build PGD input with imaginary tokens ─────────────────────────
            # Paper Sec 3.3: PGD receives y'_vat = VAT predictions with
            # imaginary '}' appended after each structural token.
            # Each imaginary } maps to an actual } in the GT sequence (via
            # brace-ownership analysis), giving correct connectivity targets.
            #
            # Dynamic count: the number of imaginary } for each structural token
            # is determined by how many } it owns in the actual GT sequence.
            # This handles both x^{2} (1 group) and x^2 (0 groups) correctly.

            vat_pred  = vat_probs.detach().argmax(dim=1)        # [B, H, W]
            H, W      = vat_pred.shape[1], vat_pred.shape[2]
            pred_flat = vat_pred.view(B, -1)                    # [B, H*W]
            gt_flat   = vat_tgt.view(B, -1)                     # [B, H*W]
            col_idx   = torch.arange(W, device=dev).unsqueeze(0).expand(H, W).reshape(-1)
            y_idx_hw  = torch.arange(H, device=dev).unsqueeze(1).expand(H, W).reshape(-1)
            x_idx_hw  = torch.arange(W, device=dev).unsqueeze(0).expand(H, W).reshape(-1)

            is_tok   = (pred_flat != none_idx)
            # n_vis[b] = number of VAT-detected visible tokens (before imaginary expansion)
            n_vis    = is_tok.sum(dim=1).clamp(max=MAX_PGD - 2)   # [B]

            # ── Per-sample build (Python loop, then batch into tensors) ───────
            # We build each sample's sequence individually because imaginary token
            # counts vary per sample. Then we pad to the batch-maximum length.
            sample_seqs = []   # list of dicts, one per sample

            for b in range(B):
                n = int(n_vis[b].item())

                # ── 1. Parse GT for } ownership ───────────────────────────────
                # Work on the non-special portion of the GT sequence (excludes PAD/SOS/EOS)
                gt_b     = token_ids_gt[b]                  # [L] with pad
                gt_list  = gt_b.tolist()
                ns_toks  = [self.vocab.i2t.get(t, '') for t in gt_list
                            if t not in (pad_idx, sos_idx, eos_idx)]
                ns_orig  = [i for i, t in enumerate(gt_list)
                            if t not in (pad_idx, sos_idx, eos_idx)]

                # ownership[i] = (close_ns_pos, struct_ns_pos, group_num)
                ownership = _get_close_brace_parents(ns_toks)
                # Map: gt_original_index_of_close → (gt_original_index_of_struct, group)
                own_by_close: dict[int, tuple[int, int]] = {
                    ns_orig[c]: (ns_orig[s], g)
                    for c, s, g in ownership
                }

                # ── 2. Sorted visible VAT predictions ─────────────────────────
                tok_mask_b = is_tok[b]                      # [H*W] bool
                if n == 0:
                    sample_seqs.append({
                        'seq':    [sos_idx, eos_idx],
                        'tgt':    [-100, -100],
                        'coords': [(map_h // 2, 0), (map_h // 2, 0)],
                        'l':      [0, 1],
                        'r':      [1, 1],
                        'conn':   [0, 0],
                        'total':  2,
                    })
                    continue

                order     = col_idx[tok_mask_b].argsort()[:n]
                y_pos     = y_idx_hw[tok_mask_b][order]            # [n]
                x_pos     = x_idx_hw[tok_mask_b][order]            # [n]
                pred_toks = pred_flat[b][tok_mask_b][order]        # [n] predicted ids
                gt_toks_b = gt_flat[b][tok_mask_b][order]          # [n] VAT target ids
                gt_inds   = raw_idx_tgt[b].view(-1)[tok_mask_b][order]  # [n] GT seq indices

                # ── 3. Build expanded sequence with imaginary tokens ───────────
                pgd_seq   = [sos_idx]
                pgd_tgt   = [-100]         # SOS: always ignored by loss
                pgd_coords = [(map_h // 2, 0)]

                # gt_to_pgd: GT original sequence index → PGD position
                # Covers SOS, all visible tokens, all imaginary }, EOS
                gt_to_pgd: dict[int, int] = {}
                sos_gt_idx = gt_list.index(sos_idx)
                gt_to_pgd[sos_gt_idx] = 0

                for k in range(n):
                    tok_id = int(pred_toks[k].item())
                    gt_idx = int(gt_inds[k].item())     # GT seq index (-1 if unmatched)
                    y      = int(y_pos[k].item())
                    x      = int(x_pos[k].item())
                    gt_tgt = int(gt_toks_b[k].item())   # VAT target at this position

                    pgd_pos = len(pgd_seq)
                    pgd_seq.append(tok_id)
                    # SCH target: ignore if background-matched (false positive from VAT)
                    pgd_tgt.append(-100 if gt_tgt == none_idx else gt_tgt)
                    pgd_coords.append((y, x))
                    if gt_idx >= 0:
                        gt_to_pgd[gt_idx] = pgd_pos

                    # ── Insert imaginary } tokens ──────────────────────────────
                    # Count = number of } that this GT token owns in the GT sequence.
                    # Dynamically computed so x^2 (no {}) gets 0 and x^{2} gets 1.
                    tok_str    = self.vocab.i2t.get(tok_id, '')
                    n_possible = STRUCT_N_GROUPS.get(tok_str, 0)
                    if n_possible > 0 and gt_idx >= 0:
                        # Find the } tokens owned by this structural token in GT
                        for group_k in range(n_possible):
                            # Look for GT } that is group_k-th close of this structural
                            found_close_gt = -1
                            for c_orig, (s_orig, g) in own_by_close.items():
                                if s_orig == gt_idx and g == group_k:
                                    found_close_gt = c_orig
                                    break
                            if found_close_gt < 0:
                                break   # GT has fewer groups than expected → stop here

                            imag_pos = len(pgd_seq)
                            pgd_seq.append(close_idx)
                            pgd_tgt.append(close_idx)   # SCH always predicts } here
                            pgd_coords.append((y, x))   # same coords as parent structural
                            gt_to_pgd[found_close_gt] = imag_pos

                eos_pgd = len(pgd_seq)
                pgd_seq.append(eos_idx)
                pgd_tgt.append(-100)    # EOS: always ignored
                pgd_coords.append((map_h // 2, 0))

                # Map EOS GT index
                gt_len = int((gt_b != pad_idx).sum().item())
                gt_to_pgd[gt_len - 1] = eos_pgd   # last non-pad token is EOS

                total = len(pgd_seq)  # SOS + visible + imaginary + EOS

                # ── 4. Connectivity from GT-filtered sequence (GT without '{') ─
                # GT-filtered: all non-pad GT tokens except '{'.
                # Imaginary } tokens appear here too, with their own neighbors.
                #
                # CRITICAL: only supervise connectivity when BOTH the token AND
                # its GT neighbor were detected by VAT.  If either is missing we
                # have no valid target → mark conn_mask=0 so the loss skips it.
                # Old self-loop fallback was corrupting L_left: ~33% of targets
                # were garbage self-loops, flooding the gradient with wrong signal.
                gt_f_orig = [i for i, t in enumerate(gt_list)
                             if t != pad_idx and t != open_idx]
                gt_f_toks = [t for t in gt_list
                             if t != pad_idx and t != open_idx]

                l_b      = [0] * total
                r_b      = [0] * total
                conn_b   = [0] * total   # 1 = valid target, 0 = ignore

                for fi, (gt_orig, gt_tok) in enumerate(zip(gt_f_orig, gt_f_toks)):
                    pgd_pos = gt_to_pgd.get(gt_orig, -1)
                    if pgd_pos < 0:
                        continue   # this token not detected → skip entirely

                    l_fi   = fi - 1
                    r_fi   = fi + 1
                    l_orig = gt_f_orig[l_fi] if l_fi >= 0               else -1
                    r_orig = gt_f_orig[r_fi] if r_fi < len(gt_f_orig)   else -1

                    l_pgd = gt_to_pgd.get(l_orig, -1)
                    r_pgd = gt_to_pgd.get(r_orig, -1)

                    # Only set valid targets when BOTH neighbors are detected.
                    # SOS (pos 0) has no left, EOS has no right — handle edges:
                    #   SOS: l_orig=-1 → l_pgd=-1, but r must be detected
                    #   EOS: r_orig=-1 → r_pgd=-1, but l must be detected
                    l_valid = (l_orig == -1) or (l_pgd >= 0)  # -1 = sequence start
                    r_valid = (r_orig == -1) or (r_pgd >= 0)  # -1 = sequence end

                    if l_valid and r_valid:
                        l_b[pgd_pos]    = l_pgd if l_pgd >= 0 else pgd_pos
                        r_b[pgd_pos]    = r_pgd if r_pgd >= 0 else pgd_pos
                        conn_b[pgd_pos] = 1

                sample_seqs.append({
                    'seq':    pgd_seq,
                    'tgt':    pgd_tgt,
                    'coords': pgd_coords,
                    'l':      l_b,
                    'r':      r_b,
                    'conn':   conn_b,
                    'total':  total,
                })

            # ── Batch into tensors ─────────────────────────────────────────────
            max_l = min(max(d['total'] for d in sample_seqs), MAX_PGD)

            pgd_input = torch.full((B, max_l), pad_idx,  dtype=torch.long,  device=dev)
            pgd_tgt   = torch.full((B, max_l), -100,     dtype=torch.long,  device=dev)
            mask      = torch.zeros(B, max_l,                                device=dev)
            conn_mask = torch.zeros(B, max_l,                                device=dev)
            coords    = torch.zeros(B, max_l, 2,                             device=dev)
            coords[:, :, 0] = map_h // 2
            l_tgt2    = torch.zeros(B, max_l, dtype=torch.long,              device=dev)
            r_tgt2    = torch.zeros(B, max_l, dtype=torch.long,              device=dev)

            for b, d in enumerate(sample_seqs):
                L = min(d['total'], max_l)
                pgd_input[b, :L] = torch.tensor(d['seq'][:L],    dtype=torch.long,  device=dev)
                pgd_tgt[b,   :L] = torch.tensor(d['tgt'][:L],    dtype=torch.long,  device=dev)
                mask[b,      :L] = 1.0
                conn_mask[b, :L] = torch.tensor(d['conn'][:L],   dtype=torch.float, device=dev)
                coords[b,    :L] = torch.tensor(d['coords'][:L], dtype=torch.float, device=dev)
                l_tgt2[b,    :L] = torch.tensor(d['l'][:L],      dtype=torch.long,  device=dev)
                r_tgt2[b,    :L] = torch.tensor(d['r'][:L],      dtype=torch.long,  device=dev)
                # Clamp connectivity targets to valid range after truncation
                l_tgt2[b, :L].clamp_(0, L - 1)
                r_tgt2[b, :L].clamp_(0, L - 1)

            with torch.amp.autocast(device_type=self.device.type,
                                    dtype=self.amp_dtype, enabled=self.use_amp):
                pad_mask = (pgd_input == pad_idx)
                qs, ql, qr, q0 = self.model.pgd(f16x, pgd_input, coords)
                cls_logits, _, _, _, left_logits, right_logits = \
                    self.model.pgd.compute_scores(qs, ql, qr, q0, pad_mask=pad_mask)
                out = {
                    'vat_logits':     vat_logits,
                    'pgd_cls_logits': cls_logits,
                    'left_logits':    left_logits,
                    'right_logits':   right_logits,
                }
                loss, ld = self.loss_fn(out, vat_tgt, pgd_tgt, l_tgt2, r_tgt2, mask, conn_mask)

            if not torch.isfinite(loss):
                tqdm.write(
                    f"  [step {step+1}] skipped NaN —"
                    f" vat={ld['L_vat']:.3f} self={ld['L_self']:.3f}"
                    f" left={ld['L_left']:.3f} right={ld['L_right']:.3f}"
                )
                self.optimizer.zero_grad()
                if self.scheduler: self.scheduler.step()
                continue

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler: self.scheduler.step()

            for k, v in ld.items():
                run[k] += v

            # Periodic cache clear prevents VRAM fragmentation in pure-VAT mode.
            # Each batch creates variable-size tensors (max_l varies per batch),
            # causing allocator fragments that accumulate: ep26=29min->ep28=42min.
            if self.device.type == 'cuda' and (step + 1) % 500 == 0:
                torch.cuda.empty_cache()

            if (step + 1) % self.log_interval == 0 or (step + 1) == len(loader):
                n  = step + 1
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'L':   f"{run['L_all']/n:.4f}",
                    'vat': f"{run['L_vat']/n:.4f}",
                    'pgd': f"{run['L_pgd']/n:.4f}",
                    'lr':  f"{lr:.1e}",
                })

        pbar.close()
        n_steps = len(loader)
        avg = {k: v / n_steps for k, v in run.items()}
        tqdm.write(
            f"  ↳ Ep {epoch} train — "
            f"L={avg['L_all']:.4f}  vat={avg['L_vat']:.4f}  pgd={avg['L_pgd']:.4f}"
            f"  (self={avg['L_self']:.3f} left={avg['L_left']:.3f} right={avg['L_right']:.3f})"
        )
        self.history['train_loss'].append(avg['L_all'])
        return avg

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader, split: str = 'val'):
        """
        Evaluate ExpRate + VAT diagnostic.

        VAT diagnostic (logged when split=='val'):
          vat_prec  = TP / (TP + FP)   — of detected positions, how many are right token?
          vat_rec   = TP / total_GT     — of all GT tokens, how many did VAT find?
          vat_det   = avg detections per image

        These tell us if VAT or PGD is the bottleneck:
          - Low recall → VAT misses tokens → fix VAT training
          - Low precision → VAT has many false positives → PGD must correct
          - High prec+rec but low ExpRate → PGD or path_selection is the problem
        """
        self.model.eval()
        preds, gts = [], []
        # VAT diagnostic accumulators
        vat_tp = vat_fp = vat_fn = vat_det_total = 0
        n_samples = 0

        none_idx = self.vocab.none_idx
        pbar = tqdm(loader, desc=f"             [{split:>5}]",
                    leave=False, dynamic_ncols=True, unit='batch')
        for batch in pbar:
            images       = batch['image'].to(self.device)
            token_ids_gt = batch['token_ids'].to(self.device)

            # Full forward for VAT diagnostic
            f8x, f16x             = self.model.enc(images)
            vat_probs, _          = self.model.vat(f8x, f16x)
            pred_seqs             = self.model._infer(f16x, vat_probs, self.vocab)

            # VAT diagnostic: compare VAT predictions to GT token positions
            if split == 'val':
                vat_pred_map = vat_probs.argmax(dim=1)   # [B, H, W]
                B = images.size(0)
                for b in range(B):
                    pred_map = vat_pred_map[b]            # [H, W]
                    gt_tids  = token_ids_gt[b]

                    # GT foreground token set — exclude {, } (imaginary/structural, not visible)
                    gt_set = set(t.item() for t in gt_tids
                                 if t.item() not in (self.vocab.pad_idx,
                                                      self.vocab.sos_idx,
                                                      self.vocab.eos_idx,
                                                      self.vocab.open_idx,
                                                      self.vocab.close_idx,
                                                      none_idx))
                    n_gt = len(gt_set)

                    # Detected tokens (positions where pred != background)
                    det_mask  = (pred_map != none_idx)
                    det_tids  = pred_map[det_mask].cpu().tolist()
                    det_set   = set(det_tids)
                    n_det     = len(det_tids)

                    tp = len(det_set & gt_set)
                    fp = n_det - tp
                    fn = n_gt  - tp

                    vat_tp  += tp;  vat_fp += fp;  vat_fn += fn
                    vat_det_total += n_det
                    n_samples += 1

            for ids, gt_toks in zip(pred_seqs, batch['tokens']):
                preds.append(self.vocab.decode(ids))
                # Phase 1: strip both '{' and '}' from GT.
                # Model doesn't output { (not visual) or } (imaginary disabled).
                gts.append([t for t in gt_toks if t not in ('{', '}')])

        pbar.close()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        er, l1, l2 = compute_exprate(preds, gts)
        tqdm.write(f"  ↳ [{split:>5}] ExpRate={er*100:.2f}%  ≤1={l1*100:.2f}%  ≤2={l2*100:.2f}%")

        if split == 'val':
            prec = vat_tp / max(vat_tp + vat_fp, 1)
            rec  = vat_tp / max(vat_tp + vat_fn, 1)
            avg_det = vat_det_total / max(n_samples, 1)
            tqdm.write(
                f"         [VAT ] prec={prec*100:.1f}%  rec={rec*100:.1f}%"
                f"  avg_det={avg_det:.1f}"
            )
            self.history['val_exprate'].append(er)
            self.history['val_l1'].append(l1)
            self.history['val_l2'].append(l2)
            self.history['vat_prec'].append(prec)
            self.history['vat_rec'].append(rec)
        return er, l1, l2

    def save(self, epoch: int, er: float):
        ckpt = dict(epoch=epoch, exprate=er,
                    model=self.model.state_dict(),
                    optim=self.optimizer.state_dict(),
                    scheduler=self.scheduler.state_dict() if self.scheduler else None,
                    history=dict(self.history))
        fname = self.ckpt_dir / f'ep{epoch:03d}_{er*100:.2f}.pth'
        torch.save(ckpt, fname)
        if er > self.best_er:
            self.best_er = er
            torch.save(ckpt, self.ckpt_dir / 'best.pth')
            tqdm.write(f"  ✓ New best: {er*100:.2f}%  → best.pth")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optim'])
        if self.scheduler and ckpt.get('scheduler'):
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.best_er = ckpt.get('exprate', 0.0)
        if 'history' in ckpt:
            self.history = defaultdict(list, ckpt['history'])
        tqdm.write(f"Resumed <- {path}  epoch={ckpt['epoch']}  best_er={self.best_er*100:.2f}%")
        return ckpt['epoch']


# ─────────────────────────────────────────────────────────────────────────────
def build_dwap_cache(dwap_ckpt_path: str, train_ds, cfg, device,
                     cache_path: str = None) -> dict:
    """
    Chạy DWAP greedy decode một lần trên toàn bộ train set,
    lưu attention argmax positions vào dict {img_path: Tensor[n,2]}.

    - Chỉ chạy 1 lần duy nhất (~5-10 phút trên P100 cho 79K ảnh)
    - Kết quả cache vào RAM (~3MB) và tùy chọn lưu disk để resume nhanh
    - Sau khi xong, DWAP + encoder được xóa khỏi GPU (giải phóng ~1GB VRAM)
    """
    # Thử load cache từ disk trước
    if cache_path and os.path.exists(cache_path):
        tqdm.write(f"DWAP cache: loading from {cache_path}")
        cache = torch.load(cache_path, map_location='cpu', weights_only=False)
        tqdm.write(f"DWAP cache: loaded {len(cache):,} entries")
        return cache

    tqdm.write(f"DWAP cache: building from {dwap_ckpt_path} ...")

    ckpt_dwap    = torch.load(dwap_ckpt_path, map_location=device, weights_only=False)
    dwap_encoder = DenseNetEncoder().to(device)
    dwap_encoder.load_state_dict(ckpt_dwap['encoder'])
    dwap_model   = DWAP(
        ch_8x    = dwap_encoder.ch_8x,
        d        = 256,
        vocab_sz = ckpt_dwap.get('vocab_size', 186),
        emb_dim  = 256,
        drop     = 0.0,
    ).to(device)
    dwap_model.load_state_dict(ckpt_dwap['dwap'])
    dwap_model.eval(); dwap_encoder.eval()
    tqdm.write(f"  DWAP loaded (ExpRate={ckpt_dwap.get('exprate',0)*100:.2f}%)")

    map_h = cfg.img_h // 8
    map_w = cfg.img_w // 8

    vocab_pad = train_ds.vocab.pad_idx

    def _collate_with_path(batch):
        imgs = torch.stack([b['image'] for b in batch])
        return {'image': imgs, 'img_path': [b['img_path'] for b in batch]}

    # num_workers=0 để tránh pickling issue với dwap trên subprocess
    loader = DataLoader(
        train_ds, batch_size=64, shuffle=False,
        collate_fn=_collate_with_path,
        num_workers=0, pin_memory=True,
    )

    cache = {}
    sos_idx = train_ds.vocab.sos_idx
    eos_idx = train_ds.vocab.eos_idx

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Building DWAP cache", unit='batch', leave=False):
            imgs      = batch['image'].to(device)
            img_paths = batch['img_path']
            f8x, _    = dwap_encoder(imgs)
            _, attn_maps = dwap_model.decode(f8x, sos_idx=sos_idx,
                                              eos_idx=eos_idx, max_len=150)
            for b, path in enumerate(img_paths):
                a = attn_maps[b]                   # Tensor [n_t, H, W] (CPU)
                if a.size(0) == 0:
                    cache[path] = torch.zeros(0, 2, dtype=torch.long)
                    continue
                a_stack  = a                       # [n_t, H, W]
                flat_idx = a_stack.view(a_stack.size(0), -1).argmax(dim=-1)
                rows     = (flat_idx // map_w).unsqueeze(-1)
                cols     = (flat_idx %  map_w).unsqueeze(-1)
                cache[path] = torch.cat([rows, cols], dim=-1)  # [n_t, 2] CPU

    # Xóa DWAP khỏi GPU ngay sau khi xong
    del dwap_model, dwap_encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    tqdm.write(f"DWAP cache: {len(cache):,} entries built, DWAP removed from GPU")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        tqdm.write(f"DWAP cache: saved → {cache_path}")

    return cache


# ─────────────────────────────────────────────────────────────────────────────
def run_training(cfg: Config):
    _set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tqdm.write(f"Device: {device}")
    if device.type == 'cuda':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    train_ds, val_ds, test_ds, vocab = build_datasets(cfg)
    col = partial(_collate, pad_idx=vocab.pad_idx)
    dkw = dict(
        collate_fn=col,
        pin_memory=(device.type == 'cuda'),
        num_workers=cfg.num_workers,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=(2 if cfg.num_workers > 0 else None),
    )
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=True, **dkw)
    val_loader   = DataLoader(val_ds,   cfg.batch_size, shuffle=False, **dkw)
    test_loader  = DataLoader(test_ds,  cfg.batch_size, shuffle=False, **dkw)
    tqdm.write(f"Steps/epoch — train={len(train_loader):,} | val={len(val_loader):,} | test={len(test_loader):,}")

    model = NAMER(
        vocab_size=len(vocab), d=cfg.d_model, heads=cfg.nhead,
        pgd_layers=cfg.pgd_layers, drop=cfg.drop,
    ).to(device)
    tqdm.write(f"NAMER — vocab={len(vocab)} | params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # torch.compile: ~15-20% speedup sau lần compile đầu (~2-3 phút)
    # reduce-overhead tối ưu cho sequence length thay đổi theo batch
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            model = torch.compile(model, mode='reduce-overhead')
            tqdm.write("torch.compile: enabled (reduce-overhead)")
        except Exception as e:
            tqdm.write(f"torch.compile: skipped ({e})")

    map_h = cfg.img_h // 8
    map_w = cfg.img_w // 8

    # DWAP: pre-compute positions một lần, cache vào RAM + disk
    dwap_cache = None
    if cfg.dwap_checkpoint and os.path.exists(cfg.dwap_checkpoint):
        cache_path = str(Path(cfg.checkpoint_dir) / 'dwap_cache.pt')
        dwap_cache = build_dwap_cache(
            cfg.dwap_checkpoint, train_ds, cfg, device,
            cache_path=cache_path,
        )
    else:
        tqdm.write("No DWAP checkpoint — using uniform spread for VAT position estimate")

    # Single cosine LR schedule: warmup 1 epoch → lr_max, decay → lr_min
    optimizer    = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    total_steps  = cfg.epochs * len(train_loader)
    warmup_steps = len(train_loader)   # 1 epoch warmup

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    loss_fn   = NAMERLoss(lam=cfg.lambda_pgd, w_conn=cfg.w_conn,
                         vocab_size=len(vocab), none_weight=cfg.none_weight)
    tqdm.write(f"Loss: lambda_pgd={cfg.lambda_pgd}  w_conn={cfg.w_conn}  none_weight={cfg.none_weight}")
    tqdm.write(f"LR schedule: warmup 1ep -> {cfg.lr:.0e}, cosine -> {cfg.lr*1e-3:.0e}")

    trainer = Trainer(model, optimizer, scheduler, loss_fn,
                      vocab, device, cfg.checkpoint_dir, cfg.log_interval,
                      dwap_cache=dwap_cache)

    start_epoch = 0
    if cfg.resume_checkpoint:
        start_epoch = trainer.load(cfg.resume_checkpoint)

    epoch_bar = tqdm(range(start_epoch + 1, cfg.epochs + 1),
                     desc='Overall', position=0, leave=True,
                     dynamic_ncols=True, unit='epoch')

    for epoch in epoch_bar:
        epoch_bar.set_postfix({'best_val': f"{trainer.best_er*100:.2f}%"})
        avg = trainer.train_epoch(train_loader, epoch, cfg.epochs, map_h, map_w)
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            er, l1, l2 = trainer.evaluate(val_loader, split='val')
            trainer.save(epoch, er)

    epoch_bar.close()

    sep = '=' * 50
    tqdm.write(f"\n{sep}\n  Final Test Evaluation\n{sep}")
    best_path = Path(cfg.checkpoint_dir) / 'best.pth'
    if best_path.exists():
        trainer.load(str(best_path))
    er, l1, l2 = trainer.evaluate(test_loader, split='test')
    tqdm.write(f"  ExpRate   : {er*100:.2f}%")
    tqdm.write(f"  ExpRate≤1 : {l1*100:.2f}%")
    tqdm.write(f"  ExpRate≤2 : {l2*100:.2f}%")
    tqdm.write(sep)
    return er, l1, l2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dwap',   type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.dwap:
        cfg.dwap_checkpoint = args.dwap
    if args.resume:
        cfg.resume_checkpoint = args.resume

    run_training(cfg)