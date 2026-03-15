"""
Training script for HMER: DenseNet Encoder + Transformer Decoder.

Features:
- AdamW optimizer with warmup + cosine annealing with restarts
- ReduceLROnPlateau secondary scheduler to escape plateaus
- CTC auxiliary loss on encoder output (weight configurable)
- Curriculum augmentation: ramp-up aug probability over first N epochs
- Mixed precision training (AMP)
- Validation with ExpRate and BLEU metrics
- Checkpointing (best model by ExpRate)
- Early stopping
- Overfitting test mode
- Post-training visualization
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
from pathlib import Path

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import Config
from data.vocab import Vocab
from data.dataset import get_dataloader
from data.symlg_parser import preprocess_dataset
from models.model import build_model
from utils.metrics import compute_exprate, compute_bleu, compute_token_accuracy
from utils.visualize import (
    plot_training_curves,
    plot_sample_predictions,
    save_history,
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_lr_scheduler(optimizer, config: Config, steps_per_epoch: int, constant_lr: bool = False):
    """
    Warmup + cosine annealing with decaying restarts and per-restart mini warmup.

    Structure per cycle after the initial warmup:
      - lr_restart_warmup_epochs: LR ramps 0 → cycle_peak (smooth re-entry)
      - remaining cycle steps:    LR cosines from cycle_peak → min_lr

    cycle_peak decays geometrically each restart:
      cycle 0: peak = lr_max         (2e-4)
      cycle 1: peak = lr_max × 0.3  (6e-5) — safe for fine-tuning
      cycle 2: peak = lr_max × 0.09 (1.8e-5)

    Run 3 post-mortem: decay=0.5 → cycle-2 peak=1e-4 still caused ep56 drop
    of 5pp. Decay=0.3 keeps the restart below the overshoot threshold while
    still providing enough gradient signal to escape shallow plateaus.
    """
    if constant_lr:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    warmup_steps         = config.train.warmup_epochs * steps_per_epoch
    T_cycle              = config.train.lr_cycle_epochs * steps_per_epoch
    restart_warmup_steps = config.train.lr_restart_warmup_epochs * steps_per_epoch
    decay                = config.train.lr_restart_decay
    min_ratio            = config.train.min_lr / config.train.lr

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)

        post_warmup  = step - warmup_steps
        cycle_idx    = post_warmup // T_cycle
        cycle_step   = post_warmup % T_cycle
        peak_ratio   = decay ** cycle_idx   # shrinks each restart

        # Mini warmup at the start of each restart cycle (cycle 1, 2, ...)
        # Prevents the LR from jumping immediately to the cycle peak.
        if cycle_idx > 0 and cycle_step < restart_warmup_steps:
            ramp = cycle_step / max(restart_warmup_steps, 1)
            return max(min_ratio, peak_ratio * ramp)

        # Cosine decay for the remainder of the cycle
        offset            = restart_warmup_steps if cycle_idx > 0 else 0
        effective_step    = cycle_step - offset
        effective_dur     = T_cycle    - offset
        progress          = effective_step / max(effective_dur, 1)
        cosine_val        = 0.5 * (1 + np.cos(np.pi * progress))
        return max(min_ratio, peak_ratio * cosine_val)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



@torch.no_grad()
def _calibrate_bn(model, dataloader, config, num_batches: int = 30):
    """Re-sync BatchNorm running stats before validation.

    Resets running stats, then accumulates fresh statistics using cumulative
    moving average (momentum=None) over num_batches forward passes. This
    produces exact batch statistics instead of the lagged exponential average
    that can diverge with low momentum and few training steps.
    """
    model.train()

    # Reset and switch to cumulative average for precise calibration
    for m in model.encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        images = batch['image'].to(config.device)
        model.encode(images)

    # Restore original momentum for training
    bn_momentum = 0.05
    for m in model.encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = bn_momentum

    model.eval()


def _reverse_targets(targets: torch.Tensor, pad_idx: int, sos_idx: int, eos_idx: int) -> torch.Tensor:
    """Reverse content tokens in target sequences, keeping SOS/EOS framing.

    [SOS, t1, t2, ..., tn, EOS, PAD, ...] -> [SOS, tn, ..., t2, t1, EOS, PAD, ...]
    Used for R2L decoder in bidirectional training (BTTR, ICCV 2021).
    """
    r2l = targets.clone()
    for i in range(targets.size(0)):
        content_mask = (targets[i] != pad_idx) & (targets[i] != sos_idx) & (targets[i] != eos_idx)
        content = targets[i, content_mask]
        r2l[i, content_mask] = content.flip(0)
    return r2l


def train_one_epoch(
    model, dataloader, optimizer, scheduler, criterion, ctc_criterion,
    scaler, config, epoch, pad_idx, vocab_size, sos_idx, eos_idx
):
    """
    Train for one epoch.

    Loss = CE_l2r + CE_r2l + ctc_w * CTC + counting_w * CountingBCE

    CTC weight is ramped from 0 -> ctc_weight over ctc_warmup_epochs.
    Counting loss is binary-CE: model predicts which vocab tokens are
    present in the target sequence (weakly supervised, no location needed).
    R2L branch (BTTR): trains a second decoder on reversed targets.
    """
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_ctc_loss = 0
    total_cnt_loss = 0
    num_batches = 0

    # Ramp CTC weight: 0 at epoch 1 -> full at epoch ctc_warmup_epochs+1
    if config.train.ctc_weight > 0 and config.train.ctc_warmup_epochs > 0:
        ctc_w = config.train.ctc_weight * min(1.0, epoch / config.train.ctc_warmup_epochs)
    else:
        ctc_w = config.train.ctc_weight

    counting_w = config.train.counting_weight
    use_bidi = model.bidirectional and model.decoder_r2l is not None

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
    use_ctc      = ctc_w > 0 and ctc_criterion is not None
    use_counting = counting_w > 0

    for batch_idx, batch in enumerate(pbar):
        images  = batch['image'].to(config.device)
        targets = batch['target'].to(config.device)
        B       = images.size(0)

        # Build R2L targets for bidirectional training
        targets_r2l = None
        if use_bidi:
            targets_r2l = _reverse_targets(targets, pad_idx, sos_idx, eos_idx)

        # Build counting targets: 1 if token appears in this sample's sequence
        if use_counting:
            count_targets = torch.zeros(B, vocab_size, device=images.device)
            for i in range(B):
                present = targets[i].unique()
                present = present[present != pad_idx]
                count_targets[i, present] = 1.0

        optimizer.zero_grad()

        def compute_losses():
            # Encode once, reuse for all heads
            memory, feat_h, feat_w, intermediates = model.encode(images)

            # L2R decoder (teacher forcing)
            tgt_input = targets[:, :-1]
            logits_l2r = model.decoder(tgt_input, memory, feat_h, feat_w)
            tgt_out = targets[:, 1:]
            logits_l2r = logits_l2r[:, :tgt_out.size(1), :]
            ce_l2r = criterion(logits_l2r.reshape(-1, logits_l2r.size(-1)), tgt_out.reshape(-1))

            # R2L decoder (bidirectional)
            ce_r2l = torch.zeros(1, device=images.device)
            if use_bidi and targets_r2l is not None:
                tgt_input_r2l = targets_r2l[:, :-1]
                logits_r2l = model.decoder_r2l(tgt_input_r2l, memory, feat_h, feat_w)
                tgt_out_r2l = targets_r2l[:, 1:]
                logits_r2l = logits_r2l[:, :tgt_out_r2l.size(1), :]
                ce_r2l = criterion(logits_r2l.reshape(-1, logits_r2l.size(-1)),
                                   tgt_out_r2l.reshape(-1))

            ce = ce_l2r + ce_r2l

            # Multi-scale counting loss
            cnt = torch.zeros(1, device=images.device)
            if use_counting:
                counts = model.counting_module(intermediates)
                cnt = F.binary_cross_entropy_with_logits(counts, count_targets)

            # CTC on encoder output (reuses memory)
            ctc = torch.zeros(1, device=images.device)
            if use_ctc:
                S = memory.size(1)
                ctc_log_probs = model.ctc_head(memory).log_softmax(-1).permute(1, 0, 2)
                in_lens  = torch.full((B,), S, dtype=torch.long, device=images.device)
                tgt_ctc  = targets[:, 1:]
                tgt_lens = (tgt_ctc != pad_idx).sum(dim=1).clamp(min=1)
                ctc = ctc_criterion(ctc_log_probs, tgt_ctc, in_lens, tgt_lens)

            total = ce + ctc_w * ctc + counting_w * cnt
            return total, ce, ctc, cnt

        if config.train.use_amp and config.device == 'cuda':
            with autocast('cuda'):
                loss, ce_loss, ctc_loss, cnt_loss = compute_losses()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, ce_loss, ctc_loss, cnt_loss = compute_losses()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()

        scheduler.step()

        total_loss     += loss.item()
        total_ce_loss  += ce_loss.item()
        total_ctc_loss += ctc_loss.item()
        total_cnt_loss += cnt_loss.item()
        num_batches    += 1

        if batch_idx % config.train.log_interval == 0:
            postfix = {'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'}
            if use_ctc:
                postfix['ctc'] = f'{ctc_loss.item():.3f}'
            if use_counting:
                postfix['cnt'] = f'{cnt_loss.item():.3f}'
            pbar.set_postfix(postfix)

    n = max(num_batches, 1)
    return total_loss / n, total_ce_loss / n, total_ctc_loss / n, total_cnt_loss / n


@torch.no_grad()
def validate(model, dataloader, vocab, criterion, config, run_generate: bool = False):
    """Validate and compute metrics.

    Args:
        run_generate: if True, run expensive autoregressive generation and
                      compute ExpRate/BLEU. If False, only compute val loss
                      (much faster — single parallel forward pass).
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    all_images = []
    all_image_paths = []
    max_gen_samples = config.train.val_generate_max_samples
    num_generated = 0

    desc = "Validating (full)" if run_generate else "Validating (loss)"
    for batch in tqdm(dataloader, desc=desc, leave=False):
        images = batch['image'].to(config.device)
        targets = batch['target'].to(config.device)

        # Compute validation loss (L2R CE only — parallel teacher-forcing, always fast)
        logits, _, _ = model(images, targets)
        tgt_out   = targets[:, 1:]
        logits    = logits[:, :tgt_out.size(1), :]
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total_loss += loss.item()
        num_batches += 1

        # Autoregressive generation — only on generate epochs, capped at max_gen_samples
        if run_generate and (max_gen_samples == 0 or num_generated < max_gen_samples):
            preds = model.generate(
                images,
                sos_idx=vocab.sos_idx,
                eos_idx=vocab.eos_idx,
                max_len=config.data.max_seq_len,
            )
            for i in range(preds.size(0)):
                pred_str = vocab.decode(preds[i].cpu().tolist())
                tgt_str = batch['latex'][i]
                all_predictions.append(pred_str)
                all_targets.append(tgt_str)
            num_generated += images.size(0)

            if len(all_images) < 16:
                all_images.append(images.cpu().numpy())
                all_image_paths.extend(batch['image_path'])

    metrics = {
        'val_loss': total_loss / max(num_batches, 1),
    }

    if run_generate and all_predictions:
        exprate_metrics = compute_exprate(all_predictions, all_targets)
        bleu_metrics = compute_bleu(all_predictions, all_targets)
        token_acc = compute_token_accuracy(all_predictions, all_targets)
        metrics.update(exprate_metrics)
        metrics.update(bleu_metrics)
        metrics['token_accuracy'] = token_acc

    vis_images = None
    if all_images:
        vis_images = np.concatenate(all_images, axis=0)[:16]

    return metrics, all_predictions[:16], all_targets[:16], vis_images, all_image_paths[:16]


def main():
    parser = argparse.ArgumentParser(description="Train HMER model")
    parser.add_argument('--overfit_test', action='store_true',
                        help='Run overfitting test on small subset')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples for overfitting test')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--preprocess', action='store_true',
                        help='Run data preprocessing before training')
    args = parser.parse_args()

    # Initialize config
    config = Config()

    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.train.lr = args.lr

    # Device
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.use_amp = False
        print("CUDA not available, using CPU")
    else:
        print(f"Using CUDA: {torch.cuda.get_device_name()}")

    set_seed(config.seed)

    # Step 1: Preprocess data if needed
    train_csv = Path(config.data.processed_dir) / "train.csv"
    if args.preprocess or not train_csv.exists():
        print("=" * 60)
        print("PREPROCESSING")
        print("=" * 60)
        preprocess_dataset(
            raw_dir=config.data.raw_dir,
            output_dir=config.data.processed_dir,
        )

    # Step 2: Build vocabulary
    vocab_path = Path(config.data.vocab_path)
    if not vocab_path.exists():
        print("\nBuilding vocabulary...")
        vocab = Vocab()
        vocab.build_from_csv(str(train_csv))
        vocab.save(str(vocab_path))
    else:
        vocab = Vocab.from_file(str(vocab_path))

    print(f"Vocabulary size: {len(vocab)}")

    # Step 3: Create data loaders
    print("\nLoading data...")

    if args.overfit_test:
        # For overfitting test: disable ALL regularization to allow memorization
        config.data.augment = False
        config.data.batch_size = min(args.num_samples, config.data.batch_size)
        config.data.num_workers = 0
        config.train.warmup_epochs = 0
        config.train.patience = config.train.epochs  # No early stopping
        config.train.label_smoothing = 0.0  # No label smoothing
        config.train.weight_decay = 0.0     # No weight decay
        config.train.lr = 5e-4              # Higher LR for fast convergence
        config.encoder.drop_rate = 0.0      # No dropout in encoder
        config.decoder.dropout = 0.0        # No dropout in decoder
        config.train.ctc_weight = 0.0       # No aux losses
        config.train.counting_weight = 0.0
        config.decoder.bidirectional = False  # Simpler for overfit test

        train_loader = get_dataloader('train', vocab, config, shuffle=False)
        # Limit to num_samples
        train_loader.dataset.entries = train_loader.dataset.entries[:args.num_samples]
        val_loader = train_loader  # Validate on same data
        print(f"\n[OVERFIT TEST] {args.num_samples} samples, {config.train.epochs} epochs")
        print(f"  Regularization disabled: label_smoothing=0, dropout=0, weight_decay=0")
    else:
        train_loader = get_dataloader('train', vocab, config)
        val_loader = get_dataloader('val', vocab, config)

    # Step 4: Build model
    model = build_model(vocab_size=len(vocab), config=config)
    model = model.to(config.device)
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Step 5: Setup training
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=config.train.label_smoothing,
    )

    # CTC auxiliary loss — zero_infinity=True avoids NaN on very long sequences
    ctc_criterion = nn.CTCLoss(blank=vocab.pad_idx, zero_infinity=True) \
        if config.train.ctc_weight > 0 else None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress lr_scheduler before optimizer.step() warning
        scheduler = get_lr_scheduler(
            optimizer, config, steps_per_epoch,
            constant_lr=args.overfit_test  # Constant LR for overfitting test
        )
        scheduler.step()  # initialize step counter so per-batch stepping works cleanly

    # Secondary scheduler: halves LR when ExpRate stops improving for patience//3
    # epochs. Works on top of the cosine restarts to escape deep plateaus.
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=config.train.patience // 3,
        min_lr=config.train.min_lr
    ) if not args.overfit_test else None

    scaler = GradScaler('cuda', enabled=config.train.use_amp and config.device == 'cuda')

    # Resume from checkpoint
    start_epoch = 1
    best_exprate = 0.0
    history = {
        'train_loss': [], 'train_ce_loss': [], 'train_ctc_loss': [], 'train_cnt_loss': [],
        'val_loss': [],
        'exprate': [], 'exprate_1': [], 'exprate_2': [],
        'bleu': [], 'bleu_1': [], 'bleu_4': [],
        'token_accuracy': [],
        'lr': [],
    }

    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_exprate = checkpoint.get('best_exprate', 0.0)
        history = checkpoint.get('history', history)
        print(f"Resumed from epoch {start_epoch - 1}, best ExpRate: {best_exprate:.2f}%")

    # Step 6: Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    epochs_without_improvement = 0
    # Track LR restart epochs to exempt them from early stopping.
    # A cosine restart causes a transient regression (run 3: ep56 dropped 5pp)
    # that is NOT a genuine plateau — it is a known side-effect of the LR jump.
    # Penalising these epochs causes premature stopping before the model can
    # recover from the restart and continue improving.
    prev_epoch_end_lr = None
    restart_grace_remaining = 0   # grace epochs left after detecting a restart

    for epoch in range(start_epoch, config.train.epochs + 1):
        epoch_start = time.time()

        # --- Curriculum augmentation ---
        # Ramp aug probability from 0 → 1 over aug_warmup_epochs.
        # This lets the model learn clean patterns first before seeing
        # heavily distorted training images.
        if not args.overfit_test and config.data.augment and config.train.aug_warmup_epochs > 0:
            aug_prob = min(1.0, epoch / config.train.aug_warmup_epochs)
            train_loader.dataset.aug_config['global_prob'] = aug_prob
        else:
            aug_prob = 1.0

        # Train
        train_loss, train_ce_loss, train_ctc_loss, train_cnt_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, ctc_criterion,
            scaler, config, epoch, vocab.pad_idx, len(vocab),
            vocab.sos_idx, vocab.eos_idx
        )

        # Calibrate BN before validation — resets running stats and recomputes
        # from training data so eval mode matches train mode exactly.
        _calibrate_bn(model, train_loader, config,
                       num_batches=config.train.bn_calibrate_batches)

        # Validate — run generate() only every N epochs (expensive O(T^2) decode)
        gen_every = config.train.val_generate_every
        is_generate_epoch = (
            epoch % gen_every == 0
            or epoch == config.train.epochs
            or epoch <= 1
            or args.overfit_test
        )
        val_metrics, val_preds, val_targets, val_images, val_paths = validate(
            model, val_loader, vocab, criterion, config,
            run_generate=is_generate_epoch,
        )

        epoch_time = time.time() - epoch_start

        # Step plateau scheduler based on val loss when ExpRate not available
        if plateau_scheduler is not None:
            if 'exprate' in val_metrics:
                plateau_scheduler.step(val_metrics['exprate'])
            else:
                # Use negative val_loss as proxy (lower loss = better)
                plateau_scheduler.step(-val_metrics['val_loss'])

        # Record history — fill missing metrics with last known value
        history['train_loss'].append(train_loss)
        history['train_ce_loss'].append(train_ce_loss)
        history['train_ctc_loss'].append(train_ctc_loss)
        history['train_cnt_loss'].append(train_cnt_loss)
        history['val_loss'].append(val_metrics['val_loss'])
        history['exprate'].append(val_metrics.get('exprate', history['exprate'][-1] if history['exprate'] else 0))
        history['exprate_1'].append(val_metrics.get('exprate_1', history['exprate_1'][-1] if history['exprate_1'] else 0))
        history['exprate_2'].append(val_metrics.get('exprate_2', history['exprate_2'][-1] if history['exprate_2'] else 0))
        history['bleu'].append(val_metrics.get('bleu', history['bleu'][-1] if history['bleu'] else 0))
        history['bleu_1'].append(val_metrics.get('bleu_1', history['bleu_1'][-1] if history['bleu_1'] else 0))
        history['bleu_4'].append(val_metrics.get('bleu_4', history['bleu_4'][-1] if history['bleu_4'] else 0))
        history['token_accuracy'].append(val_metrics.get('token_accuracy', history['token_accuracy'][-1] if history['token_accuracy'] else 0))
        history['lr'].append(scheduler.get_last_lr()[0])

        # Print epoch summary
        if config.train.ctc_weight > 0 and config.train.ctc_warmup_epochs > 0:
            ctc_w = config.train.ctc_weight * min(1.0, epoch / config.train.ctc_warmup_epochs)
        else:
            ctc_w = config.train.ctc_weight
        ctc_str = f"\n  CTC Loss:   {train_ctc_loss:.4f} (weight={ctc_w:.3f})" if config.train.ctc_weight > 0 else ""
        cnt_str = f"\n  Count Loss: {train_cnt_loss:.4f}" if config.train.counting_weight > 0 else ""
        aug_str = f"\n  Aug prob:   {aug_prob:.2f}" if config.data.augment else ""

        if is_generate_epoch:
            metric_str = (
                f"\n  ExpRate:    {val_metrics['exprate']:.2f}% "
                f"(+1: {val_metrics['exprate_1']:.2f}%, +2: {val_metrics['exprate_2']:.2f}%)"
                f"\n  BLEU-4:     {val_metrics['bleu']:.2f}%"
                f"\n  Token Acc:  {val_metrics['token_accuracy']:.2f}%"
            )
        else:
            metric_str = "\n  [generation skipped — loss only]"

        print(
            f"\nEpoch {epoch}/{config.train.epochs} ({epoch_time:.1f}s)"
            f"\n  Train Loss: {train_loss:.4f} (CE: {train_ce_loss:.4f}{ctc_str}{cnt_str})"
            f"\n  Val Loss:   {val_metrics['val_loss']:.4f}"
            f"{metric_str}"
            f"\n  LR:         {scheduler.get_last_lr()[0]:.2e}"
            f"{aug_str}"
        )

        # Print sample predictions (only on generate epochs)
        if is_generate_epoch and val_preds:
            n_show = min(3, len(val_preds))
            print(f"\n  Sample predictions:")
            for i in range(n_show):
                match = "OK" if val_preds[i].strip() == val_targets[i].strip() else "XX"
                print(f"    {match} GT:   {val_targets[i][:60]}")
                print(f"      Pred: {val_preds[i][:60]}")

        # --- Detect LR restart and apply grace period ---
        current_lr = scheduler.get_last_lr()[0]
        is_restart_epoch = (
            prev_epoch_end_lr is not None and current_lr > prev_epoch_end_lr * 1.5
        )
        if is_restart_epoch:
            restart_grace_remaining = config.train.lr_restart_warmup_epochs + 3
            print(f"  LR restart detected ({prev_epoch_end_lr:.2e} -> {current_lr:.2e}). "
                  f"Grace period: {restart_grace_remaining} epochs")
        prev_epoch_end_lr = current_lr
        if restart_grace_remaining > 0:
            restart_grace_remaining -= 1

        # Checkpointing — use ExpRate when available, skip on loss-only epochs
        is_best = False
        current_exprate = val_metrics.get('exprate', None)
        if current_exprate is not None:
            is_best = current_exprate > best_exprate
            if is_best:
                best_exprate = current_exprate
                epochs_without_improvement = 0
            elif restart_grace_remaining > 0:
                pass
            else:
                epochs_without_improvement += 1

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_exprate': best_exprate,
            'history': history,
            'vocab_size': len(vocab),
            'config': {
                'data': vars(config.data),
                'encoder': vars(config.encoder),
                'decoder': vars(config.decoder),
                'train': vars(config.train),
            }
        }

        # Save latest
        torch.save(checkpoint, os.path.join(config.train.checkpoint_dir, 'latest.pt'))

        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(config.train.checkpoint_dir, 'best.pt'))
            print(f"  * New best ExpRate: {best_exprate:.2f}%")

        # Save history and plots periodically
        save_history(history, config.train.output_dir)
        if epoch % 5 == 0 or epoch == config.train.epochs:
            plot_training_curves(history, config.train.output_dir)
            if val_images is not None:
                plot_sample_predictions(
                    val_images[:8],
                    val_preds[:8],
                    val_targets[:8],
                    config.train.output_dir,
                    image_paths=val_paths[:8],
                )

        # Early stopping
        if not args.overfit_test and epochs_without_improvement >= config.train.patience:
            print(f"\nEarly stopping after {epochs_without_improvement} epochs without improvement")
            break

    # Final visualization
    print("\n" + "=" * 60)
    print("GENERATING FINAL VISUALIZATIONS")
    print("=" * 60)

    plot_training_curves(history, config.train.output_dir)
    save_history(history, config.train.output_dir)

    if val_images is not None:
        plot_sample_predictions(
            val_images[:8],
            val_preds[:8],
            val_targets[:8],
            config.train.output_dir,
            filename="final_predictions.png",
            image_paths=val_paths[:8],
        )

    print(f"\nTraining complete!")
    print(f"Best ExpRate: {best_exprate:.2f}%")
    print(f"Outputs saved to: {config.train.output_dir}/")
    print(f"Checkpoints saved to: {config.train.checkpoint_dir}/")


if __name__ == "__main__":
    main()