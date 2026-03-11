"""
train_namer.py — Train NAMER on HME100K
=========================================
Follows paper (arXiv:2407.11380) Sec 3.3 + 4.2:
  - PGD always receives VAT predictions as input (no teacher forcing)
  - LR: warmup 1 epoch -> lr_max, cosine decay -> lr_max * 1e-3
  - Recommended lr=2e-4 (paper value; default config uses 1e-4)

Usage:
    python train_namer.py
    python train_namer.py --dwap checkpoints/dwap_best.pth
    python train_namer.py --resume checkpoints/ep020_10.55.pth
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
from utils.matching import make_vat_targets


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
        self.dwap_cache   = dwap_cache

        # AMP: bf16 on Ampere+, fp16 on P100/V100/T4
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
        Paper Sec 3.3: PGD always receives VAT predictions as input.
        Q0 = VAT visual features + 2D position encoding + word embedding.
        No teacher forcing -- matches inference exactly from epoch 1.
        """
        self.model.train()
        run = defaultdict(float)

        pbar = tqdm(loader,
                    desc=f"Ep {epoch:>3}/{total_epochs} [train]",
                    leave=True, dynamic_ncols=True, unit='batch')

        for step, batch in enumerate(pbar):
            images       = batch['image'].to(self.device)
            token_ids_gt = batch['token_ids'].to(self.device)
            img_paths    = batch['img_path']

            with torch.amp.autocast(device_type=self.device.type,
                                    dtype=self.amp_dtype, enabled=self.use_amp):
                f8x, f16x             = self.model.enc(images)
                vat_probs, vat_logits = self.model.vat(f8x, f16x)

            # VAT bipartite matching -> targets + GT coordinates
            T_pos = None
            if self.dwap_cache is not None:
                T_pos = [self.dwap_cache.get(p) for p in img_paths]

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

            # Build PGD input from VAT predictions (paper Sec 3.3)
            vat_pred  = vat_probs.detach().argmax(dim=1)       # [B, H, W]
            H, W      = vat_pred.shape[1], vat_pred.shape[2]
            pred_flat = vat_pred.view(B, -1)                   # [B, H*W]
            gt_flat   = vat_tgt.view(B, -1)                    # [B, H*W]
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
            l_tgt = torch.zeros(B, max_l, dtype=torch.long, device=dev)
            r_tgt = torch.zeros(B, max_l, dtype=torch.long, device=dev)

            # Build index tensors once, reuse for all samples in batch
            y_idx      = torch.arange(H, device=dev).unsqueeze(1).expand(H, W).reshape(-1)
            x_idx      = torch.arange(W, device=dev).unsqueeze(0).expand(H, W).reshape(-1)
            max_gt_len = token_ids_gt.size(1)

            for b in range(B):
                n = lengths[b].item()
                if n == 0:
                    pgd_input[b, 1] = eos_idx
                    mask[b, :2] = 1.0
                    continue

                tok_mask   = is_tok[b]
                order      = col_idx[tok_mask].argsort()[:n]
                y_pos      = y_idx[tok_mask][order]
                x_pos      = x_idx[tok_mask][order]
                gt_indices = raw_idx_tgt[b].view(-1)[tok_mask][order]  # [n]

                pgd_input[b, 1:n+1] = pred_flat[b][tok_mask][order]
                coords[b, 1:n+1]    = torch.stack([y_pos, x_pos], dim=-1).float()
                pgd_input[b, n+1]   = eos_idx
                pgd_tgt[b, 1:n+1]   = gt_flat[b][tok_mask][order]
                mask[b, :n+2]       = 1.0

                # Build GT->PGD index map via scatter (no Python inner loop)
                pgd_positions  = torch.arange(1, n + 1, device=dev)
                valid_ri       = (gt_indices >= 0) & (gt_indices < max_gt_len + 2)
                raw_to_pgd_map = torch.zeros(max_gt_len + 2, dtype=torch.long, device=dev)
                raw_to_pgd_map.scatter_(0, gt_indices[valid_ri].long(),
                                        pgd_positions[valid_ri])

                last_gt_token = ((token_ids_gt[b] != pad_idx).sum() - 2).clamp(min=1)
                if last_gt_token + 1 < max_gt_len + 2:
                    raw_to_pgd_map[last_gt_token + 1] = n + 1

                # SOS/EOS connectivity
                l_tgt[b, 0]   = 0
                r_tgt[b, 0]   = raw_to_pgd_map[1] if n > 0 else 0
                l_tgt[b, n+1] = raw_to_pgd_map[last_gt_token] if n > 0 else 0
                r_tgt[b, n+1] = n + 1

                # Inner token connectivity (vectorized)
                r_i_all     = gt_indices                                  # [n]
                valid_conn  = (r_i_all >= 1) & (r_i_all <= last_gt_token)
                r_i_clamped = r_i_all.clamp(1, max_gt_len)
                l_vals = raw_to_pgd_map[r_i_clamped - 1]
                r_vals = raw_to_pgd_map[(r_i_clamped + 1).clamp(max=max_gt_len + 1)]
                # Invalid entries self-loop (pgd_idx -> itself)
                l_tgt[b, 1:n+1] = torch.where(valid_conn, l_vals, pgd_positions)
                r_tgt[b, 1:n+1] = torch.where(valid_conn, r_vals, pgd_positions)

            # PGD forward + loss
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
                loss, ld = self.loss_fn(out, vat_tgt, pgd_tgt, l_tgt, r_tgt, mask)

            if not torch.isfinite(loss):
                tqdm.write(
                    f"  [step {step+1}] skipped NaN --"
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

            # Periodic VRAM defrag (variable max_l causes fragmentation)
            if self.device.type == 'cuda' and (step + 1) % 200 == 0:
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
            f"  > Ep {epoch} train -- "
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
          prec = TP / (TP+FP)  -- correct among detections
          rec  = TP / total_GT -- GT tokens found by VAT
          avg_det = avg detections per image
        """
        self.model.eval()
        preds, gts = [], []
        vat_tp = vat_fp = vat_fn = vat_det_total = 0
        n_samples = 0

        none_idx = self.vocab.none_idx
        pbar = tqdm(loader, desc=f"             [{split:>5}]",
                    leave=False, dynamic_ncols=True, unit='batch')
        for batch in pbar:
            images       = batch['image'].to(self.device)
            token_ids_gt = batch['token_ids'].to(self.device)

            f8x, f16x        = self.model.enc(images)
            vat_probs, _     = self.model.vat(f8x, f16x)
            pred_seqs        = self.model._infer(f16x, vat_probs)

            if split == 'val':
                vat_pred_map = vat_probs.argmax(dim=1)
                B = images.size(0)
                for b in range(B):
                    pred_map = vat_pred_map[b]
                    gt_tids  = token_ids_gt[b]

                    gt_set = set(t.item() for t in gt_tids
                                 if t.item() not in (self.vocab.pad_idx,
                                                      self.vocab.sos_idx,
                                                      self.vocab.eos_idx,
                                                      none_idx))
                    n_gt = len(gt_set)

                    det_mask = (pred_map != none_idx)
                    det_tids = pred_map[det_mask].cpu().tolist()
                    det_set  = set(det_tids)
                    n_det    = len(det_tids)

                    tp = len(det_set & gt_set)
                    fp = n_det - tp
                    fn = n_gt  - tp

                    vat_tp += tp; vat_fp += fp; vat_fn += fn
                    vat_det_total += n_det
                    n_samples += 1

            for ids, gt_toks in zip(pred_seqs, batch['tokens']):
                preds.append(self.vocab.decode(ids))
                gts.append(gt_toks)

        pbar.close()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        er, l1, l2 = compute_exprate(preds, gts)
        tqdm.write(f"  > [{split:>5}] ExpRate={er*100:.2f}%  <=1={l1*100:.2f}%  <=2={l2*100:.2f}%")

        if split == 'val':
            prec    = vat_tp / max(vat_tp + vat_fp, 1)
            rec     = vat_tp / max(vat_tp + vat_fn, 1)
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
            tqdm.write(f"  + New best: {er*100:.2f}%  -> best.pth")

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


# -----------------------------------------------------------------------------
def build_dwap_cache(dwap_ckpt_path: str, train_ds, cfg, device,
                     cache_path: str = None) -> dict:
    """
    Run DWAP greedy decode once on entire train set.
    Cache attention argmax positions: {img_path: Tensor[n, 2]}.
    Saves to disk so resume skips rebuild.
    DWAP + encoder freed from GPU after completion.
    """
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
    tqdm.write(f"  DWAP loaded (ExpRate={ckpt_dwap.get('exprate', 0)*100:.2f}%)")

    map_h = cfg.img_h // 8
    map_w = cfg.img_w // 8

    def _collate_path(batch):
        imgs = torch.stack([b['image'] for b in batch])
        return {'image': imgs, 'img_path': [b['img_path'] for b in batch]}

    loader = DataLoader(
        train_ds, batch_size=64, shuffle=False,
        collate_fn=_collate_path, num_workers=0, pin_memory=True,
    )

    cache   = {}
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
                a = attn_maps[b]   # [n_t, H, W] CPU
                if a.size(0) == 0:
                    cache[path] = torch.zeros(0, 2, dtype=torch.long)
                    continue
                flat_idx = a.view(a.size(0), -1).argmax(dim=-1)
                rows = (flat_idx // map_w).unsqueeze(-1)
                cols = (flat_idx %  map_w).unsqueeze(-1)
                cache[path] = torch.cat([rows, cols], dim=-1)   # [n_t, 2]

    del dwap_model, dwap_encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    tqdm.write(f"DWAP cache: {len(cache):,} entries built, DWAP removed from GPU")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        tqdm.write(f"DWAP cache: saved -> {cache_path}")

    return cache


# -----------------------------------------------------------------------------
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
    tqdm.write(f"Steps/epoch -- train={len(train_loader):,} | val={len(val_loader):,} | test={len(test_loader):,}")

    model = NAMER(
        vocab_size=len(vocab), d=cfg.d_model, heads=cfg.nhead,
        pgd_layers=cfg.pgd_layers, drop=cfg.drop,
    ).to(device)
    tqdm.write(f"NAMER -- vocab={len(vocab)} | params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            model = torch.compile(model, mode='reduce-overhead')
            tqdm.write("torch.compile: enabled (reduce-overhead)")
        except Exception as e:
            tqdm.write(f"torch.compile: skipped ({e})")

    map_h = cfg.img_h // 8
    map_w = cfg.img_w // 8

    dwap_cache = None
    if cfg.dwap_checkpoint and os.path.exists(cfg.dwap_checkpoint):
        cache_path = str(Path(cfg.checkpoint_dir) / 'dwap_cache.pt')
        dwap_cache = build_dwap_cache(
            cfg.dwap_checkpoint, train_ds, cfg, device,
            cache_path=cache_path,
        )
    else:
        tqdm.write("No DWAP checkpoint -- using uniform spread for VAT position estimate")

    # LR schedule: paper Sec 4.2
    #   warmup 1 epoch -> lr_max, cosine -> lr_max * 1e-3
    #   Paper uses lr_max=2e-4; cfg.lr default is 1e-4
    optimizer    = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    total_steps  = cfg.epochs * len(train_loader)
    warmup_steps = 1 * len(train_loader)

    def _lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    loss_fn   = NAMERLoss(lam=cfg.lambda_pgd, w_conn=cfg.w_conn,
                          vocab_size=len(vocab)).to(device)
    tqdm.write(f"Loss: lambda_pgd={cfg.lambda_pgd}  w_conn={cfg.w_conn}")
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
    tqdm.write(f"  ExpRate<=1 : {l1*100:.2f}%")
    tqdm.write(f"  ExpRate<=2 : {l2*100:.2f}%")
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