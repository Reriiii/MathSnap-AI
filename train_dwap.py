"""
train_dwap.py — Pretrain DWAP on HME100K
=========================================
Run:
    python train_dwap.py
"""
import math, os, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import DWAPConfig
from data import build_datasets, _parse_label_file, _split3, _collate, Vocabulary
from models import DenseNetEncoder, DWAP
from utils.loss import DWAPLoss
from utils.metrics import compute_exprate
from functools import partial
from torch.utils.data import DataLoader


def _set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def run_dwap_training(cfg: DWAPConfig):
    _set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tqdm.write(f"Device: {device}")
    if device.type == 'cuda':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True   # auto-tune CUDA kernels → 20-30% faster

    # ── Data ────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, vocab = build_datasets(cfg)
    col = partial(_collate, pad_idx=vocab.pad_idx)
    # num_workers=0: Windows multiprocessing overhead > benefit for small workers
    dkw = dict(collate_fn=col, pin_memory=(device.type=='cuda'), num_workers=0)
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=True, **dkw)
    val_loader   = DataLoader(val_ds,   cfg.batch_size, shuffle=False, **dkw)
    tqdm.write(f"Steps/epoch — train={len(train_loader):,} | val={len(val_loader):,}")

    # ── Model ────────────────────────────────────────────────────────────────
    encoder = DenseNetEncoder().to(device)
    dwap    = DWAP(
        ch_8x   = encoder.ch_8x,
        d       = cfg.d,
        vocab_sz= len(vocab),
        emb_dim = cfg.emb_dim,
        drop    = cfg.drop,
    ).to(device)
    n_enc  = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_dwap = sum(p.numel() for p in dwap.parameters()    if p.requires_grad)
    tqdm.write(f"Encoder params: {n_enc:,}  |  DWAP decoder params: {n_dwap:,}")

    # ── Optimizer & LR ───────────────────────────────────────────────────────
    params        = list(encoder.parameters()) + list(dwap.parameters())
    optimizer     = torch.optim.Adam(params, lr=cfg.lr, weight_decay=1e-4)
    total_steps   = cfg.epochs * len(train_loader)
    warmup_steps  = len(train_loader)   # 1-epoch warmup

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    loss_fn   = DWAPLoss(pad_idx=vocab.pad_idx)

    ckpt_dir = Path(cfg.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_er  = 0.0
    start_ep = 0

    if cfg.resume_checkpoint:
        ckpt = torch.load(cfg.resume_checkpoint, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt['encoder'])
        dwap.load_state_dict(ckpt['dwap'])
        optimizer.load_state_dict(ckpt['optim'])
        best_er  = ckpt.get('exprate', 0.0)
        start_ep = ckpt.get('epoch', 0)
        tqdm.write(f"Resumed from {cfg.resume_checkpoint}  (ep={start_ep}, exprate={best_er*100:.2f}%)")

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(start_ep + 1, cfg.epochs + 1):
        encoder.train(); dwap.train()
        run = defaultdict(float)

        pbar = tqdm(train_loader,
                    desc=f"DWAP Ep {epoch:>3}/{cfg.epochs} [train]",
                    leave=True, dynamic_ncols=True, unit='batch')

        for step, batch in enumerate(pbar):
            images    = batch['image'].to(device)
            token_ids = batch['token_ids'].to(device)

            f8x, _ = encoder(images)
            logits, _ = dwap(f8x, token_ids, teacher_forcing=True)
            # logits: [B, min(L-1, 60), V]   token_ids: [B, L]
            # DWAPLoss uses logits length to determine how many targets to use
            loss = loss_fn(logits, token_ids)

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                scheduler.step()
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            scheduler.step()

            run['loss'] += loss.item()
            if (step + 1) % cfg.log_interval == 0:
                n  = step + 1
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({'L': f"{run['loss']/n:.4f}", 'lr': f"{lr:.1e}"})

        pbar.close()
        avg_loss = run['loss'] / len(train_loader)
        tqdm.write(f"  ↳ Ep {epoch} train — loss={avg_loss:.4f}")

        # ── Validation ─────────────────────────────────────────────────────
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            encoder.eval(); dwap.eval()
            preds, gts = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="[val]", leave=False):
                    imgs = batch['image'].to(device)
                    f8x, _ = encoder(imgs)
                    seqs, _ = dwap.decode(f8x,
                                          sos_idx=vocab.sos_idx,
                                          eos_idx=vocab.eos_idx)
                    for ids, gt_toks in zip(seqs, batch['tokens']):
                        preds.append(vocab.decode(ids))
                        gts.append(gt_toks)

            er, l1, l2 = compute_exprate(preds, gts)
            tqdm.write(f"  ↳ [val] ExpRate={er*100:.2f}%  ≤1={l1*100:.2f}%  ≤2={l2*100:.2f}%")

            # Save checkpoint
            ckpt = dict(epoch=epoch, exprate=er,
                        encoder=encoder.state_dict(),
                        dwap=dwap.state_dict(),
                        optim=optimizer.state_dict())
            save_path = ckpt_dir / f'dwap_ep{epoch:03d}_{er*100:.2f}.pth'
            torch.save(ckpt, save_path)
            if er > best_er:
                best_er = er
                torch.save(ckpt, ckpt_dir / 'dwap_best.pth')
                tqdm.write(f"  ✓ DWAP best: {er*100:.2f}%  → dwap_best.pth")

    tqdm.write(f"\nDWAP training done. Best val ExpRate: {best_er*100:.2f}%")
    tqdm.write(f"Checkpoint saved: {ckpt_dir}/dwap_best.pth")
    return best_er


if __name__ == '__main__':
    cfg = DWAPConfig()
    run_dwap_training(cfg)
