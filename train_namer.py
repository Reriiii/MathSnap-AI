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


def _set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn,
                 vocab: Vocabulary, device, checkpoint_dir: str,
                 log_interval: int = 50,
                 dwap=None, dwap_encoder=None):
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

        self.dwap         = dwap
        self.dwap_encoder = dwap_encoder
        if dwap is not None:
            self.dwap.eval()
            self.dwap_encoder.eval()

        # AMP: bf16 on RTX 5060 Ti ~2x faster than fp32
        self.use_amp   = (device.type == 'cuda')
        self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.scaler    = torch.amp.GradScaler(device='cuda', enabled=self.use_amp)

    # ------------------------------------------------------------------
    def train_epoch(self, loader, epoch: int, total_epochs: int,
                    map_h: int, map_w: int,
                    ss_start_epoch: int = 15,
                    ss_end_epoch:   int = 25):
        """
        Scheduled Sampling curriculum:
          ep 1..ss_start       : p_gt = 1.0  (pure teacher forcing)
          ep ss_start..ss_end  : p_gt 1.0->0.0 linear (per-batch coin flip)
          ep ss_end..total     : p_gt = 0.0  (pure VAT, matches inference)
        """
        self.model.train()
        run = defaultdict(float)

        if epoch <= ss_start_epoch:
            p_gt = 1.0
        elif epoch >= ss_end_epoch:
            p_gt = 0.0
        else:
            p_gt = 1.0 - (epoch - ss_start_epoch) / (ss_end_epoch - ss_start_epoch)

        pbar = tqdm(loader,
                    desc=f"Ep {epoch:>3}/{total_epochs} [train] p_gt={p_gt:.2f}",
                    leave=True, dynamic_ncols=True, unit='batch')

        for step, batch in enumerate(pbar):
            images       = batch['image'].to(self.device)
            token_ids_gt = batch['token_ids'].to(self.device)

            with torch.amp.autocast(device_type=self.device.type,
                                    dtype=self.amp_dtype, enabled=self.use_amp):
                f8x, f16x             = self.model.enc(images)
                vat_probs, vat_logits = self.model.vat(f8x, f16x)

            # VAT targets via Hungarian matching
            T_pos = None
            if self.dwap is not None:
                T_pos = get_dwap_positions(
                    self.dwap, self.dwap_encoder, images,
                    token_ids_gt,
                    sos_idx=self.vocab.sos_idx,
                    eos_idx=self.vocab.eos_idx,
                    map_h=map_h, map_w=map_w,
                    device=self.device,
                )

            vat_tgt, gt_coords, raw_idx_tgt = make_vat_targets(
                token_ids_gt, vat_probs.detach(), map_h, map_w,
                self.vocab, self.device, T_positions=T_pos
            )

            none_idx = self.vocab.none_idx
            pad_idx  = self.vocab.pad_idx
            sos_idx  = self.vocab.sos_idx
            eos_idx  = self.vocab.eos_idx
            B        = images.size(0)
            dev      = self.device
            MAX_PGD  = 128

            # Per-batch coin flip for smooth curriculum transition
            use_gt_pgd = (p_gt == 1.0) or (p_gt > 0.0 and torch.rand(1).item() < p_gt)

            if use_gt_pgd:
                raw   = token_ids_gt
                max_l = min(raw.size(1), MAX_PGD)

                vat_pred_full = vat_probs.detach().argmax(dim=1)
                pgd_input = torch.full((B, max_l), pad_idx, dtype=torch.long, device=dev)
                coords    = gt_coords[:, :max_l].contiguous().float()

                for b in range(B):
                    valid_mask = (raw[b, :max_l] != pad_idx)
                    num_valid  = valid_mask.sum().item()
                    rows = coords[b, :num_valid, 0].long().clamp(0, map_h - 1)
                    cols = coords[b, :num_valid, 1].long().clamp(0, map_w - 1)
                    pgd_input[b, :num_valid] = vat_pred_full[b, rows, cols]
                    sp_mask = ((raw[b, :num_valid] == sos_idx) |
                               (raw[b, :num_valid] == eos_idx))
                    pgd_input[b, :num_valid][sp_mask] = raw[b, :num_valid][sp_mask]

                pgd_tgt = raw[:, :max_l].contiguous().clone()
                pgd_tgt[pgd_tgt == pad_idx] = -100
                pgd_tgt[pgd_tgt == sos_idx] = -100
                pgd_tgt[pgd_tgt == eos_idx] = -100
                mask = (raw[:, :max_l] != pad_idx).float()

                sp_mask_coord = ((raw[:, :max_l] == pad_idx) |
                                 (raw[:, :max_l] == sos_idx) |
                                 (raw[:, :max_l] == eos_idx))
                coords[sp_mask_coord, 0] = map_h // 2
                coords[sp_mask_coord, 1] = 0

                pos    = torch.arange(max_l, device=dev)
                ends   = ((raw[:, :max_l] != pad_idx).sum(dim=1) - 1).clamp(0, max_l - 1)
                l_tgt2 = (pos - 1).clamp(min=0).unsqueeze(0).expand(B, -1).clone()
                r_tgt2 = (pos + 1).clamp(max=max_l - 1).unsqueeze(0).expand(B, -1).clone()
                for b in range(B):
                    r_tgt2[b] = r_tgt2[b].clamp(max=ends[b].item())

            else:
                vat_pred  = vat_probs.detach().argmax(dim=1)
                H, W      = vat_pred.shape[1], vat_pred.shape[2]
                pred_flat = vat_pred.view(B, -1)
                gt_flat   = vat_tgt.view(B, -1)
                col_idx   = torch.arange(W, device=dev).unsqueeze(0).expand(H, W).reshape(-1)
                is_tok    = (pred_flat != none_idx)
                lengths   = is_tok.sum(dim=1).clamp(max=MAX_PGD - 2)
                max_l     = int(lengths.max().item()) + 2

                pgd_input = torch.full((B, max_l), pad_idx, dtype=torch.long, device=dev)
                pgd_tgt   = torch.full((B, max_l), -100,    dtype=torch.long, device=dev)
                mask      = torch.zeros(B, max_l, device=dev)
                coords    = torch.zeros(B, max_l, 2, device=dev)
                coords[:, :, 0] = map_h // 2
                coords[:, :, 1] = 0
                pgd_input[:, 0] = sos_idx
                l_tgt2 = torch.zeros(B, max_l, dtype=torch.long, device=dev)
                r_tgt2 = torch.zeros(B, max_l, dtype=torch.long, device=dev)

                for b in range(B):
                    n = lengths[b].item()
                    if n == 0:
                        pgd_input[b, 1] = eos_idx; mask[b, :2] = 1.0; continue
                    tok_mask = is_tok[b]
                    order    = col_idx[tok_mask].argsort()[:n]

                    y_idx = torch.arange(H, device=dev).unsqueeze(1).expand(H, W).reshape(-1)
                    x_idx = torch.arange(W, device=dev).unsqueeze(0).expand(H, W).reshape(-1)
                    y_pos = y_idx[tok_mask][order]
                    x_pos = x_idx[tok_mask][order]
                    gt_indices = raw_idx_tgt[b].view(-1)[tok_mask][order]

                    pgd_input[b, 1:n+1] = pred_flat[b][tok_mask][order]
                    coords[b, 1:n+1]    = torch.stack([y_pos, x_pos], dim=-1).float()
                    pgd_input[b, n+1]   = eos_idx
                    pgd_tgt[b, 1:n+1]   = gt_flat[b][tok_mask][order]
                    mask[b, :n+2]       = 1.0

                    max_gt_len     = token_ids_gt.size(1)
                    raw_to_pgd_map = torch.zeros(max_gt_len + 2, dtype=torch.long, device=dev)
                    raw_to_pgd_map[0] = 0
                    for pgd_idx in range(1, n + 1):
                        r_i = gt_indices[pgd_idx - 1]
                        if r_i >= 0 and r_i < max_gt_len + 2:
                            raw_to_pgd_map[r_i] = pgd_idx

                    last_gt_token = ((token_ids_gt[b] != pad_idx).sum() - 2).clamp(min=1)
                    if last_gt_token + 1 < max_gt_len + 2:
                        raw_to_pgd_map[last_gt_token + 1] = n + 1

                    l_tgt2[b, 0] = 0
                    r_tgt2[b, 0] = raw_to_pgd_map[1] if n > 0 else 0
                    l_tgt2[b, n+1] = raw_to_pgd_map[last_gt_token] if n > 0 else 0
                    r_tgt2[b, n+1] = n + 1

                    for pgd_idx in range(1, n + 1):
                        r_i = gt_indices[pgd_idx - 1]
                        if r_i >= 1 and r_i <= last_gt_token:
                            l_tgt2[b, pgd_idx] = raw_to_pgd_map[r_i - 1]
                            r_tgt2[b, pgd_idx] = raw_to_pgd_map[r_i + 1]
                        else:
                            l_tgt2[b, pgd_idx] = pgd_idx
                            r_tgt2[b, pgd_idx] = pgd_idx

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
                loss, ld = self.loss_fn(out, vat_tgt, pgd_tgt, l_tgt2, r_tgt2, mask)

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
        self.model.eval()
        preds, gts = [], []
        pbar = tqdm(loader, desc=f"             [{split:>5}]",
                    leave=False, dynamic_ncols=True, unit='batch')
        for batch in pbar:
            pred_seqs = self.model(batch['image'].to(self.device), token_ids=None)
            for ids, gt_toks in zip(pred_seqs, batch['tokens']):
                preds.append(self.vocab.decode(ids))
                gts.append(gt_toks)
        pbar.close()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        er, l1, l2 = compute_exprate(preds, gts)
        tqdm.write(f"  ↳ [{split:>5}] ExpRate={er*100:.2f}%  ≤1={l1*100:.2f}%  ≤2={l2*100:.2f}%")
        if split == 'val':
            self.history['val_exprate'].append(er)
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

    map_h = cfg.img_h // 8
    map_w = cfg.img_w // 8

    dwap_model   = None
    dwap_encoder = None
    if cfg.dwap_checkpoint and os.path.exists(cfg.dwap_checkpoint):
        tqdm.write(f"Loading DWAP from {cfg.dwap_checkpoint}")
        ckpt_dwap    = torch.load(cfg.dwap_checkpoint, map_location=device, weights_only=False)
        dwap_encoder = DenseNetEncoder().to(device)
        dwap_encoder.load_state_dict(ckpt_dwap['encoder'])
        dwap_model   = DWAP(
            ch_8x   = dwap_encoder.ch_8x,
            d       = 256,
            vocab_sz= len(vocab),
            emb_dim = 256,
            drop    = 0.0,
        ).to(device)
        dwap_model.load_state_dict(ckpt_dwap['dwap'])
        dwap_model.eval(); dwap_encoder.eval()
        tqdm.write(f"DWAP loaded — val ExpRate was {ckpt_dwap.get('exprate',0)*100:.2f}%")
    else:
        tqdm.write("No DWAP checkpoint — using uniform spread for VAT position estimate")

    # Two-Phase LR: cosine phase1, then restart for pure-VAT phase
    optimizer    = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    phase1_steps = cfg.pgd_ss_end_epoch * len(train_loader)
    phase2_steps = (cfg.epochs - cfg.pgd_ss_end_epoch) * len(train_loader)
    warmup_steps = 2 * len(train_loader)
    lr_min_mult  = 1e-3
    vat_lr_mult  = cfg.vat_lr_restart / cfg.lr

    def _lr_lambda(step):
        if step <= phase1_steps:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            prog = (step - warmup_steps) / max(1, phase1_steps - warmup_steps)
            return max(lr_min_mult, 0.5 * (1.0 + math.cos(math.pi * prog)))
        step2 = step - phase1_steps
        prog2 = step2 / max(1, phase2_steps)
        return vat_lr_mult * max(lr_min_mult, 0.5 * (1.0 + math.cos(math.pi * prog2)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    loss_fn   = NAMERLoss(lam=cfg.lambda_pgd, w_conn=cfg.w_conn)
    tqdm.write(f"Loss: lambda_pgd={cfg.lambda_pgd}  w_conn={cfg.w_conn}")
    tqdm.write(f"LR schedule: phase1 cosine {cfg.lr:.0e} over ep1-{cfg.pgd_ss_end_epoch}, "
               f"restart {cfg.vat_lr_restart:.0e} over ep{cfg.pgd_ss_end_epoch}-{cfg.epochs}")

    trainer = Trainer(model, optimizer, scheduler, loss_fn,
                      vocab, device, cfg.checkpoint_dir, cfg.log_interval,
                      dwap=dwap_model, dwap_encoder=dwap_encoder)

    start_epoch = 0
    if cfg.resume_checkpoint:
        start_epoch = trainer.load(cfg.resume_checkpoint)

    epoch_bar = tqdm(range(start_epoch + 1, cfg.epochs + 1),
                     desc='Overall', position=0, leave=True,
                     dynamic_ncols=True, unit='epoch')

    for epoch in epoch_bar:
        epoch_bar.set_postfix({'best_val': f"{trainer.best_er*100:.2f}%"})
        avg = trainer.train_epoch(
            train_loader, epoch, cfg.epochs, map_h, map_w,
            ss_start_epoch=cfg.pgd_teacher_epochs,
            ss_end_epoch=cfg.pgd_ss_end_epoch,
        )
        in_pure_vat = (epoch > cfg.pgd_ss_end_epoch)
        if in_pure_vat or epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
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