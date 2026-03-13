"""
Training script for HMER: DenseNet Encoder + Transformer Decoder.

Features:
- AdamW optimizer with warmup + cosine annealing
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
    """Create warmup + cosine annealing learning rate scheduler."""
    if constant_lr:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    warmup_steps = config.train.warmup_epochs * steps_per_epoch
    total_steps = config.train.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(
                config.train.min_lr / config.train.lr,
                0.5 * (1 + np.cos(np.pi * progress))
            )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model, dataloader, optimizer, scheduler, criterion, scaler, config, epoch
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(config.device)
        targets = batch['target'].to(config.device)

        optimizer.zero_grad()

        if config.train.use_amp and config.device == 'cuda':
            with autocast('cuda'):
                logits = model(images, targets)
                # Target for loss: shift left (remove SOS, keep EOS)
                tgt_out = targets[:, 1:]  # [B, T-1]
                logits = logits[:, :tgt_out.size(1), :]  # Align lengths
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, targets)
            tgt_out = targets[:, 1:]
            logits = logits[:, :tgt_out.size(1), :]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % config.train.log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, vocab, criterion, config):
    """Validate and compute metrics."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    all_images = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch['image'].to(config.device)
        targets = batch['target'].to(config.device)

        # Compute loss
        logits = model(images, targets)
        tgt_out = targets[:, 1:]
        logits = logits[:, :tgt_out.size(1), :]
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1)
        )
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

        # Save some images for visualization
        if len(all_images) < 16:
            all_images.append(images.cpu().numpy())

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

    return metrics, all_predictions[:16], all_targets[:16], vis_images


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
    scaler = GradScaler('cuda', enabled=config.train.use_amp and config.device == 'cuda')

    # Resume from checkpoint
    start_epoch = 1
    best_exprate = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
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

    for epoch in range(start_epoch, config.train.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, config, epoch
        )

        # Validate
        val_metrics, val_preds, val_targets, val_images = validate(
            model, val_loader, vocab, criterion, config
        )

        epoch_time = time.time() - epoch_start

        # Record history
        history['train_loss'].append(train_loss)
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
        print(
            f"\nEpoch {epoch}/{config.train.epochs} ({epoch_time:.1f}s)"
            f"\n  Train Loss: {train_loss:.4f}"
            f"\n  Val Loss:   {val_metrics['val_loss']:.4f}"
            f"\n  ExpRate:    {val_metrics['exprate']:.2f}% "
            f"(±1: {val_metrics['exprate_1']:.2f}%, ±2: {val_metrics['exprate_2']:.2f}%)"
            f"\n  BLEU-4:     {val_metrics['bleu']:.2f}%"
            f"\n  Token Acc:  {val_metrics['token_accuracy']:.2f}%"
            f"\n  LR:         {scheduler.get_last_lr()[0]:.2e}"
        )

        # Print some sample predictions
        n_show = min(3, len(val_preds))
        print(f"\n  Sample predictions:")
        for i in range(n_show):
            match = "✓" if val_preds[i].strip() == val_targets[i].strip() else "✗"
            print(f"    {match} GT:   {val_targets[i][:60]}")
            print(f"      Pred: {val_preds[i][:60]}")

        # Checkpointing
        is_best = val_metrics['exprate'] > best_exprate
        if is_best:
            best_exprate = val_metrics['exprate']
            epochs_without_improvement = 0
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
            filename="final_predictions.png"
        )

    print(f"\nTraining complete!")
    print(f"Best ExpRate: {best_exprate:.2f}%")
    print(f"Outputs saved to: {config.train.output_dir}/")
    print(f"Checkpoints saved to: {config.train.checkpoint_dir}/")


if __name__ == "__main__":
    main()
