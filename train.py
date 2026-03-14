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


def _calibrate_bn(model: nn.Module, dataloader, device: str, num_batches: int = 30):
    """
    Recalibrate BatchNorm running statistics before validation.

    During curriculum augmentation the training data distribution shifts
    gradually as aug_prob increases. BN running_mean/var track the augmented
    training distribution; when the model switches to eval() for validation
    (which uses clean, unaugmented images) the mismatch can cause a complete
    decoding failure for one epoch.

    This function forwards a small number of training batches through the
    encoder in train() mode (no gradient accumulation, no optimizer update)
    so that the running stats reflect the current epoch's distribution before
    validation begins.  30 batches is enough for momentum=0.1 to converge.
    """
    model.train()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            model.encoder(batch['image'].to(device))


def train_one_epoch(
    model, dataloader, optimizer, scheduler, criterion, ctc_criterion,
    scaler, config, epoch, pad_idx, vocab_size
):
    """
    Train for one epoch.

    Loss = CE  +  ctc_w * CTC  +  counting_w * CountingBCE

    CTC weight is ramped from 0 → ctc_weight over ctc_warmup_epochs.
    Counting loss is binary-CE: model predicts which vocab tokens are
    present in the target sequence (weakly supervised, no location needed).
    """
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_ctc_loss = 0
    total_cnt_loss = 0
    num_batches = 0

    # Ramp CTC weight: 0 at epoch 1 → full at epoch ctc_warmup_epochs+1
    if config.train.ctc_weight > 0 and config.train.ctc_warmup_epochs > 0:
        ctc_w = config.train.ctc_weight * min(1.0, epoch / config.train.ctc_warmup_epochs)
    else:
        ctc_w = config.train.ctc_weight

    counting_w = config.train.counting_weight

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
    use_ctc      = ctc_w > 0 and ctc_criterion is not None
    use_counting = counting_w > 0

    for batch_idx, batch in enumerate(pbar):
        images  = batch['image'].to(config.device)
        targets = batch['target'].to(config.device)
        B       = images.size(0)

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
            memory, feat_h, feat_w = model.encode(images)

            # Decoder forward (teacher forcing)
            tgt_input = targets[:, :-1]
            logits = model.decoder(tgt_input, memory, feat_h, feat_w)
            tgt_out = targets[:, 1:]
            logits  = logits[:, :tgt_out.size(1), :]
            ce = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            # Counting loss: binary cross-entropy over vocab presence
            cnt = torch.zeros(1, device=images.device)
            if use_counting:
                counts = model.counting_module(memory)
                cnt = F.binary_cross_entropy(counts, count_targets)

            # CTC on encoder output (reuses memory from above)
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
def validate(model, dataloader, vocab, criterion, config):
    """Validate and compute metrics."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    all_images = []
    all_image_paths = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch['image'].to(config.device)
        targets = batch['target'].to(config.device)

        # Compute validation loss (CE only — no aux losses during validation)
        logits, _ = model(images, targets)
        tgt_out   = targets[:, 1:]
        logits    = logits[:, :tgt_out.size(1), :]
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total_loss += loss.item()
        num_batches += 1

        # Generate predictions
        preds = model.generate(
            images,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            max_len=config.data.max_seq_len,
        )

        # Decode predictions and targets
        for i in range(preds.size(0)):
            pred_str = vocab.decode(preds[i].cpu().tolist())
            tgt_str = batch['latex'][i]
            all_predictions.append(pred_str)
            all_targets.append(tgt_str)

        # Save some images and paths for visualization
        if len(all_images) < 16:
            all_images.append(images.cpu().numpy())
            all_image_paths.extend(batch['image_path'])

    # Compute metrics
    exprate_metrics = compute_exprate(all_predictions, all_targets)
    bleu_metrics = compute_bleu(all_predictions, all_targets)
    token_acc = compute_token_accuracy(all_predictions, all_targets)

    metrics = {
        'val_loss': total_loss / max(num_batches, 1),
        **exprate_metrics,
        **bleu_metrics,
        'token_accuracy': token_acc,
    }

    # Collect images for visualization
    if all_images:
        vis_images = np.concatenate(all_images, axis=0)[:16]
    else:
        vis_images = None

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

        train_loader = get_dataloader('train', vocab, config, shuffle=False)
        # Limit to num_samples
        train_loader.dataset.entries = train_loader.dataset.entries[:args.num_samples]
        val_loader = train_loader  # Validate on same data
        print(f"\n⚡ Overfitting test: {args.num_samples} samples, {config.train.epochs} epochs")
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
    scheduler = get_lr_scheduler(
        optimizer, config, steps_per_epoch,
        constant_lr=args.overfit_test  # Constant LR for overfitting test
    )

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
            scaler, config, epoch, vocab.pad_idx, len(vocab)
        )

        # --- BN calibration before validation ---
        # During aug warmup, training data distribution shifts each epoch as
        # aug_prob increases. BN running stats track the augmented-train
        # distribution and diverge from clean validation images, causing
        # single-epoch exprate spikes (ep17: 17%→1.3%, ep21: 19%→2.6%).
        # Fix: forward unaugmented batches through encoder in train() mode
        # to re-sync running stats before eval.
        #
        # Run 3 post-mortem: spike at ep31 (+0.12 val_loss, -4.7pp exprate)
        # because aug_prob=1.0 exactly at ep30 → calibration stopped.
        # ep31 = first epoch with FULL augmentation but BN stats from ep30
        # partial aug. Extend window 2 epochs past ramp completion to cover
        # the transition from partial → full augmentation.
        calibrate_window = config.train.aug_warmup_epochs + 2
        if config.data.augment and epoch <= calibrate_window:
            _calibrate_bn(model, train_loader, config.device, num_batches=30)

        # Validate
        val_metrics, val_preds, val_targets, val_images, val_paths = validate(
            model, val_loader, vocab, criterion, config
        )

        epoch_time = time.time() - epoch_start

        # Step plateau scheduler based on val ExpRate
        if plateau_scheduler is not None:
            plateau_scheduler.step(val_metrics['exprate'])

        # Record history
        history['train_loss'].append(train_loss)
        history['train_ce_loss'].append(train_ce_loss)
        history['train_ctc_loss'].append(train_ctc_loss)
        history['train_cnt_loss'].append(train_cnt_loss)
        history['val_loss'].append(val_metrics['val_loss'])
        history['exprate'].append(val_metrics['exprate'])
        history['exprate_1'].append(val_metrics['exprate_1'])
        history['exprate_2'].append(val_metrics['exprate_2'])
        history['bleu'].append(val_metrics['bleu'])
        history['bleu_1'].append(val_metrics.get('bleu_1', 0))
        history['bleu_4'].append(val_metrics.get('bleu_4', 0))
        history['token_accuracy'].append(val_metrics['token_accuracy'])
        history['lr'].append(scheduler.get_last_lr()[0])

        # Print epoch summary
        # Recompute ctc_w here (mirrors the ramp logic inside train_one_epoch)
        # so the printed weight matches what was actually used this epoch.
        if config.train.ctc_weight > 0 and config.train.ctc_warmup_epochs > 0:
            ctc_w = config.train.ctc_weight * min(1.0, epoch / config.train.ctc_warmup_epochs)
        else:
            ctc_w = config.train.ctc_weight
        ctc_str = f"\n  CTC Loss:   {train_ctc_loss:.4f} (weight={ctc_w:.3f})" if config.train.ctc_weight > 0 else ""
        cnt_str = f"\n  Count Loss: {train_cnt_loss:.4f}" if config.train.counting_weight > 0 else ""
        aug_str = f"\n  Aug prob:   {aug_prob:.2f}" if config.data.augment else ""
        print(
            f"\nEpoch {epoch}/{config.train.epochs} ({epoch_time:.1f}s)"
            f"\n  Train Loss: {train_loss:.4f} (CE: {train_ce_loss:.4f}{ctc_str}{cnt_str})"
            f"\n  Val Loss:   {val_metrics['val_loss']:.4f}"
            f"\n  ExpRate:    {val_metrics['exprate']:.2f}% "
            f"(±1: {val_metrics['exprate_1']:.2f}%, ±2: {val_metrics['exprate_2']:.2f}%)"
            f"\n  BLEU-4:     {val_metrics['bleu']:.2f}%"
            f"\n  Token Acc:  {val_metrics['token_accuracy']:.2f}%"
            f"\n  LR:         {scheduler.get_last_lr()[0]:.2e}"
            f"{aug_str}"
        )

        # Print some sample predictions
        n_show = min(3, len(val_preds))
        print(f"\n  Sample predictions:")
        for i in range(n_show):
            match = "✓" if val_preds[i].strip() == val_targets[i].strip() else "✗"
            print(f"    {match} GT:   {val_targets[i][:60]}")
            print(f"      Pred: {val_preds[i][:60]}")

        # --- Detect LR restart and apply grace period ---
        # A cosine restart causes a transient regression before the model
        # recovers (run 3 ep56: -5.3pp). Penalising with the patience counter
        # causes premature early stopping before recovery is complete.
        # Detection: LR only increases between epochs at a cycle boundary.
        current_lr = scheduler.get_last_lr()[0]
        is_restart_epoch = (
            prev_epoch_end_lr is not None and current_lr > prev_epoch_end_lr * 1.5
        )
        if is_restart_epoch:
            restart_grace_remaining = config.train.lr_restart_warmup_epochs + 3
            print(f"  \u21ba LR restart detected ({prev_epoch_end_lr:.2e} \u2192 {current_lr:.2e}). "
                  f"Grace period: {restart_grace_remaining} epochs")
        prev_epoch_end_lr = current_lr
        if restart_grace_remaining > 0:
            restart_grace_remaining -= 1

        # Checkpointing
        is_best = val_metrics['exprate'] > best_exprate
        if is_best:
            best_exprate = val_metrics['exprate']
            epochs_without_improvement = 0
        elif restart_grace_remaining > 0:
            # Within grace period: transient restart regression, don't penalise
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
            print(f"  ★ New best ExpRate: {best_exprate:.2f}%")

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