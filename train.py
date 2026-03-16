"""
ICAL HMER training script.

Architecture: DenseNet Encoder + Transformer Decoder with SCCM + FusionModule.
Loss: CE_exp + CE_imp(dynamic_weight) + CE_fusion (bidirectional l2r + r2l).
Optimizer: SGD + ReduceLROnPlateau (ICAL default).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
from data.dataset import get_dataloader, collate_fn
from data.vocab import Vocab
from models.model import build_model
from utils.ical_utils import ce_loss, plicit_tgt_out
from utils.metrics import compute_exprate, compute_bleu, compute_token_accuracy


def get_structural_indices(vocab: Vocab) -> set:
    """Get indices of structural tokens: {, }, ^, _"""
    structural_tokens = ['{', '}', '^', '_']
    indices = set()
    for tok in structural_tokens:
        if tok in vocab.token2idx:
            indices.add(vocab.token2idx[tok])
    return indices


def train_one_epoch(
    model, train_loader, optimizer, config, vocab, structural_indices, scaler, epoch
):
    """Train one epoch (ICAL-style: bidirectional + 3 CE losses)."""
    model.train()
    device = config.device

    total_loss = 0.0
    total_exp_loss = 0.0
    total_imp_loss = 0.0
    total_fusion_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"  Train E{epoch+1}", leave=True, ncols=120)
    for batch_idx, batch in enumerate(pbar):
        imgs = batch['image'].to(device)        # [b, 1, H, W]
        mask = batch['padding_mask'].to(device)  # [b, H, W]
        indices = batch['indices']               # List[List[int]]

        # Build bidirectional targets (ICAL-style)
        # fusion/exp targets: normal tokens
        fusion_tgt, fusion_out = plicit_tgt_out(
            indices, device,
            pad_idx=vocab.pad_idx, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
            space_idx=vocab.unk_idx, structural_indices=structural_indices,
            is_explicit=False, is_implicit=False,
        )
        # exp targets same as fusion
        exp_tgt, exp_out = fusion_tgt, fusion_out

        # implicit targets: non-structural tokens replaced with space/UNK
        _, implicit_out = plicit_tgt_out(
            indices, device,
            pad_idx=vocab.pad_idx, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
            space_idx=vocab.unk_idx, structural_indices=structural_indices,
            is_explicit=False, is_implicit=True,
        )

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=config.train.use_amp):
            # Forward pass: features duplicated inside model for bidirectional
            exp_out_hat, imp_out_hat, fusion_out_hat = model(imgs, mask, exp_tgt)

            # Compute losses
            exp_l = ce_loss(exp_out_hat, exp_out, ignore_idx=vocab.pad_idx)
            imp_l = ce_loss(
                imp_out_hat, implicit_out,
                ignore_idx=vocab.pad_idx,
                need_weight=config.train.dynamic_weight,
                class_of_interest=vocab.unk_idx,
                vocab_size=len(vocab),
            )
            fusion_l = ce_loss(fusion_out_hat, fusion_out, ignore_idx=vocab.pad_idx)

            loss = exp_l + imp_l + fusion_l

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_exp_loss += exp_l.item()
        total_imp_loss += imp_l.item()
        total_fusion_loss += fusion_l.item()
        num_batches += 1

        # Update tqdm bar
        pbar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'exp': f'{total_exp_loss/num_batches:.3f}',
            'imp': f'{total_imp_loss/num_batches:.3f}',
            'fus': f'{total_fusion_loss/num_batches:.3f}',
        })

    avg_loss = total_loss / max(num_batches, 1)
    avg_exp = total_exp_loss / max(num_batches, 1)
    avg_imp = total_imp_loss / max(num_batches, 1)
    avg_fus = total_fusion_loss / max(num_batches, 1)

    return avg_loss, avg_exp, avg_imp, avg_fus


@torch.no_grad()
def validate(model, val_loader, config, vocab, structural_indices):
    """Validate: compute loss + generate predictions for ExpRate."""
    model.eval()
    device = config.device

    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(val_loader, desc="  Val", leave=True, ncols=120)
    for batch in pbar:
        imgs = batch['image'].to(device)
        mask = batch['padding_mask'].to(device)
        indices = batch['indices']

        # Compute validation loss (teacher-forcing, bidirectional)
        fusion_tgt, fusion_out = plicit_tgt_out(
            indices, device,
            pad_idx=vocab.pad_idx, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
            space_idx=vocab.unk_idx, structural_indices=structural_indices,
            is_explicit=False, is_implicit=False,
        )
        _, implicit_out = plicit_tgt_out(
            indices, device,
            pad_idx=vocab.pad_idx, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
            space_idx=vocab.unk_idx, structural_indices=structural_indices,
            is_explicit=False, is_implicit=True,
        )

        with torch.amp.autocast('cuda', enabled=config.train.use_amp):
            exp_out_hat, imp_out_hat, fusion_out_hat = model(imgs, mask, fusion_tgt)
            exp_l = ce_loss(exp_out_hat, fusion_out, ignore_idx=vocab.pad_idx)
            imp_l = ce_loss(
                imp_out_hat, implicit_out,
                ignore_idx=vocab.pad_idx,
                need_weight=config.train.dynamic_weight,
                class_of_interest=vocab.unk_idx,
                vocab_size=len(vocab),
            )
            fusion_l = ce_loss(fusion_out_hat, fusion_out, ignore_idx=vocab.pad_idx)
            loss = exp_l + imp_l + fusion_l

        total_loss += loss.item()
        num_batches += 1

        # Generate predictions (greedy decode for speed)
        pred_indices = model.greedy_decode(
            imgs, mask,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            max_len=config.model.max_len,
        )

        # Convert to strings for metric computation
        for pred_idx in pred_indices:
            all_preds.append(vocab.decode(pred_idx, remove_special=True))
        for idx_list in indices:
            all_targets.append(vocab.decode(idx_list, remove_special=True))

    avg_loss = total_loss / max(num_batches, 1)

    # Compute metrics
    exprate_dict = compute_exprate(all_preds, all_targets)
    bleu_dict = compute_bleu(all_preds, all_targets)
    token_acc = compute_token_accuracy(all_preds, all_targets)

    return avg_loss, exprate_dict, bleu_dict, token_acc


def main():
    parser = argparse.ArgumentParser(description="ICAL HMER Training")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--overfit", action="store_true", help="Overfit test on small subset")
    parser.add_argument("--num-samples", type=int, default=5, help="Samples for overfit test")
    args = parser.parse_args()

    config = Config()

    # Seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Vocab
    vocab = Vocab.from_file(config.data.vocab_path)
    structural_indices = get_structural_indices(vocab)
    print(f"Vocab size: {len(vocab)}")
    print(f"Structural indices ({len(structural_indices)}): {structural_indices}")

    # Data
    if args.overfit:
        # For overfit test, create a simple DataLoader without SizeGroupedBatchSampler
        from data.dataset import CROHMEDataset, collate_fn as data_collate_fn
        overfit_dataset = CROHMEDataset(
            csv_path=f"{config.data.processed_dir}/train.csv",
            vocab=vocab,
            max_seq_len=config.data.max_seq_len,
            augment=False,
        )
        overfit_dataset.entries = overfit_dataset.entries[:args.num_samples]
        train_loader = torch.utils.data.DataLoader(
            overfit_dataset,
            batch_size=args.num_samples,
            shuffle=False,
            num_workers=0,
            collate_fn=data_collate_fn,
        )
        val_loader = train_loader
        print(f"\n=== OVERFIT TEST: {args.num_samples} samples ===\n")
    else:
        train_loader = get_dataloader('train', vocab, config)
        val_loader = get_dataloader('val', vocab, config)

    print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    # Model
    model = build_model(config, len(vocab))
    model = model.to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Optimizer — ICAL uses SGD
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.train.lr,
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay,
    )

    # Scheduler — ICAL uses ReduceLROnPlateau on val_ExpRate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.train.lr_factor,
        patience=config.train.patience // config.train.val_every_n_epoch,
    )

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda', enabled=config.train.use_amp)

    # History
    history_path = os.path.join(config.train.output_dir, "history.json")
    history = {
        'train_loss': [], 'train_exp_loss': [], 'train_imp_loss': [], 'train_fusion_loss': [],
        'val_loss': [],
        'exprate': [], 'exprate_1': [], 'exprate_2': [],
        'bleu': [], 'bleu_1': [], 'bleu_4': [],
        'token_accuracy': [], 'lr': [],
    }

    start_epoch = 0
    best_exprate = 0.0

    # Resume
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_exprate = ckpt.get('best_exprate', 0.0)
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        print(f"Resumed at epoch {start_epoch}, best ExpRate: {best_exprate:.2f}%")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training: {config.train.epochs} epochs")
    print(f"Optimizer: SGD(lr={config.train.lr}, momentum={config.train.momentum})")
    print(f"Scheduler: ReduceLROnPlateau(patience={config.train.patience}, factor={config.train.lr_factor})")
    print(f"Val every {config.train.val_every_n_epoch} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, config.train.epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1}/{config.train.epochs} | LR: {current_lr:.6f}")

        # Train
        train_loss, exp_loss, imp_loss, fus_loss = train_one_epoch(
            model, train_loader, optimizer, config, vocab,
            structural_indices, scaler, epoch,
        )

        epoch_time = time.time() - epoch_start
        print(f"  Train Loss: {train_loss:.4f} "
              f"(exp={exp_loss:.3f} imp={imp_loss:.3f} fus={fus_loss:.3f}) "
              f"| {epoch_time:.1f}s")

        history['train_loss'].append(train_loss)
        history['train_exp_loss'].append(exp_loss)
        history['train_imp_loss'].append(imp_loss)
        history['train_fusion_loss'].append(fus_loss)
        history['lr'].append(current_lr)

        # Validate every N epochs (ICAL: check_val_every_n_epoch=2)
        if (epoch + 1) % config.train.val_every_n_epoch == 0 or epoch == 0:
            val_loss, exprate_dict, bleu_dict, token_acc = validate(
                model, val_loader, config, vocab, structural_indices,
            )

            exprate = exprate_dict['exprate']
            exprate_1 = exprate_dict['exprate_1']
            exprate_2 = exprate_dict['exprate_2']

            print(f"  Val Loss: {val_loss:.4f} | "
                  f"ExpRate: {exprate:.2f}% (<=1: {exprate_1:.2f}%, <=2: {exprate_2:.2f}%) | "
                  f"BLEU-4: {bleu_dict['bleu_4']:.2f} | TokenAcc: {token_acc:.2f}%")

            history['val_loss'].append(val_loss)
            history['exprate'].append(exprate)
            history['exprate_1'].append(exprate_1)
            history['exprate_2'].append(exprate_2)
            history['bleu'].append(bleu_dict['bleu'])
            history['bleu_1'].append(bleu_dict.get('bleu_1', 0.0))
            history['bleu_4'].append(bleu_dict.get('bleu_4', 0.0))
            history['token_accuracy'].append(token_acc)

            # Step scheduler on ExpRate
            scheduler.step(exprate)

            # Save best
            if exprate > best_exprate:
                best_exprate = exprate
                ckpt_path = os.path.join(config.train.checkpoint_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_exprate': best_exprate,
                    'config': str(config),
                }, ckpt_path)
                print(f"  ★ New best ExpRate: {best_exprate:.2f}% — saved to {ckpt_path}")

        # Save history
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best ExpRate: {best_exprate:.2f}%")


if __name__ == "__main__":
    main()
