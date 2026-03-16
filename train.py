"""
CoMER HMER training script (unidirectional, fast).

Architecture: DenseNet Encoder + Transformer Decoder with ARM.
Loss: Single CE (l2r only).
Optimizer: Adam + ReduceLROnPlateau.
Validation: Separate CROHME 2014/2016/2019 test sets.
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import Config
from data.dataset import get_dataloader, collate_fn, CROHMEDataset
from data.vocab import Vocab
from models.model import build_model
from utils.metrics import compute_exprate, compute_bleu, compute_token_accuracy

VAL_SPLITS = ['2014', '2016', '2019']


def build_targets(indices, device, pad_idx, sos_idx, eos_idx):
    """Build simple l2r teacher-forcing targets.

    Args:
        indices: List[List[int]] - raw token indices (no SOS/EOS)

    Returns:
        tgt: [b, max_len] - input to decoder (SOS + tokens)
        out: [b, max_len] - expected output (tokens + EOS)
    """
    batch_size = len(indices)
    # Add SOS/EOS
    seqs = []
    for idx_list in indices:
        seqs.append([sos_idx] + list(idx_list) + [eos_idx])

    max_len = max(len(s) for s in seqs)

    tgt = torch.full((batch_size, max_len - 1), pad_idx, dtype=torch.long, device=device)
    out = torch.full((batch_size, max_len - 1), pad_idx, dtype=torch.long, device=device)

    for i, seq in enumerate(seqs):
        tgt[i, :len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)
        out[i, :len(seq) - 1] = torch.tensor(seq[1:], dtype=torch.long)

    return tgt, out


def train_one_epoch(model, train_loader, optimizer, config, vocab, scaler, epoch):
    """Train one epoch (l2r only, single CE loss)."""
    model.train()
    device = config.device

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"  Train E{epoch+1}", leave=True, ncols=120)
    for batch in pbar:
        imgs = batch['image'].to(device)
        mask = batch['padding_mask'].to(device)
        indices = batch['indices']

        tgt, out = build_targets(indices, device, vocab.pad_idx, vocab.sos_idx, vocab.eos_idx)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=config.train.use_amp):
            out_hat = model(imgs, mask, tgt)  # [b, l, vocab_size]
            loss = F.cross_entropy(
                out_hat.reshape(-1, out_hat.size(-1)),
                out.reshape(-1),
                ignore_index=vocab.pad_idx,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, val_loader, config, vocab, split_name='val'):
    """Validate: loss + greedy decode metrics."""
    model.eval()
    device = config.device

    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(val_loader, desc=f"  Val {split_name}", leave=True, ncols=120)
    for batch in pbar:
        imgs = batch['image'].to(device)
        mask = batch['padding_mask'].to(device)
        indices = batch['indices']

        tgt, out = build_targets(indices, device, vocab.pad_idx, vocab.sos_idx, vocab.eos_idx)

        with torch.amp.autocast('cuda', enabled=config.train.use_amp):
            out_hat = model(imgs, mask, tgt)
            loss = F.cross_entropy(
                out_hat.reshape(-1, out_hat.size(-1)),
                out.reshape(-1),
                ignore_index=vocab.pad_idx,
            )

        total_loss += loss.item()
        num_batches += 1

        pred_indices = model.greedy_decode(
            imgs, mask,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            max_len=150,
        )
        for pred_idx in pred_indices:
            all_preds.append(vocab.decode(pred_idx, remove_special=True))
        for idx_list in indices:
            all_targets.append(vocab.decode(idx_list, remove_special=True))

    avg_loss = total_loss / max(num_batches, 1)
    exprate_dict = compute_exprate(all_preds, all_targets)
    bleu_dict = compute_bleu(all_preds, all_targets)
    token_acc = compute_token_accuracy(all_preds, all_targets)

    return avg_loss, exprate_dict, bleu_dict, token_acc


def plot_training_curves(history, output_dir):
    """Plot training metrics with 3 val splits."""
    n_train = len(history['train_loss'])
    if n_train == 0:
        return

    train_epochs = list(range(1, n_train + 1))
    n_val = len(history.get('2014_exprate', []))

    if n_val > 0:
        val_interval = max(1, round(n_train / n_val)) if n_val < n_train else 1
        val_epochs = [i * val_interval for i in range(1, n_val + 1)]
        val_epochs = [min(e, n_train) for e in val_epochs]
    else:
        val_epochs = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CoMER Training Progress', fontsize=16, fontweight='bold')

    colors = {'2014': '#e74c3c', '2016': '#2ecc71', '2019': '#3498db'}

    # 1. Loss
    ax = axes[0, 0]
    ax.plot(train_epochs, history['train_loss'], 'b-', label='Train', linewidth=1.5)
    for split in VAL_SPLITS:
        key = f'{split}_val_loss'
        if val_epochs and history.get(key):
            ax.plot(val_epochs, history[key], '-o', color=colors[split],
                    label=f'Val {split}', markersize=3, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. ExpRate
    ax = axes[0, 1]
    for split in VAL_SPLITS:
        key = f'{split}_exprate'
        if val_epochs and history.get(key):
            ax.plot(val_epochs, history[key], '-o', color=colors[split],
                    label=f'{split}', markersize=3, linewidth=1.5)
            if history[key]:
                best = max(history[key])
                best_idx = history[key].index(best)
                ax.annotate(f'{best:.1f}%', xy=(val_epochs[best_idx], best),
                            fontsize=8, fontweight='bold', color=colors[split],
                            xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('%')
    ax.set_title('ExpRate by Split')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. BLEU-4
    ax = axes[1, 0]
    for split in VAL_SPLITS:
        bleu_key = f'{split}_bleu_4'
        if val_epochs and history.get(bleu_key):
            ax.plot(val_epochs, history[bleu_key], '-o', color=colors[split],
                    label=f'BLEU-4 {split}', markersize=2, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('BLEU-4')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Learning Rate
    ax = axes[1, 1]
    if history['lr']:
        ax.plot(train_epochs, history['lr'], 'k-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.set_title('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CoMER HMER Training")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--overfit", action="store_true", help="Overfit test")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    config = Config()

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True

    vocab = Vocab.from_file(config.data.vocab_path)
    print(f"Vocab size: {len(vocab)}")

    # Data
    if args.overfit:
        overfit_dataset = CROHMEDataset(
            vocab=vocab,
            max_seq_len=config.data.max_seq_len,
            augment=False,
            caption_path=os.path.join(config.data.comer_data_dir, 'train', 'caption.txt'),
            img_dir=os.path.join(config.data.comer_data_dir, 'train', 'img'),
        )
        overfit_dataset.entries = overfit_dataset.entries[:args.num_samples]
        overfit_dataset._image_cache = {i: overfit_dataset._image_cache[i] for i in range(args.num_samples)}
        train_loader = torch.utils.data.DataLoader(
            overfit_dataset, batch_size=args.num_samples, shuffle=False,
            num_workers=0, collate_fn=collate_fn,
        )
        val_loaders = {'2014': train_loader}
        print(f"\n=== OVERFIT TEST: {args.num_samples} samples ===\n")
    else:
        train_loader = get_dataloader('train', vocab, config)
        val_loaders = {split: get_dataloader(split, vocab, config) for split in VAL_SPLITS}

    print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    for split, loader in val_loaders.items():
        print(f"Val {split}: {len(loader.dataset)} samples, {len(loader)} batches")

    # Model
    model = build_model(config, len(vocab)).to(config.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    optimizer = optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.train.lr_factor,
        patience=config.train.patience // config.train.val_every_n_epoch,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=config.train.use_amp)

    # History
    history_path = os.path.join(config.train.output_dir, "history.json")
    history = {'train_loss': [], 'lr': []}
    for split in VAL_SPLITS:
        for metric in ['val_loss', 'exprate', 'exprate_1', 'exprate_2', 'bleu_4', 'token_accuracy']:
            history[f'{split}_{metric}'] = []

    start_epoch = 0
    best_exprate = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_exprate = ckpt.get('best_exprate', 0.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config.train.lr
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        print(f"Resumed at epoch {start_epoch}, best ExpRate: {best_exprate:.2f}%")

    print(f"\n{'='*60}")
    print(f"Training: {config.train.epochs} epochs | Adam(lr={config.train.lr})")
    print(f"Val splits: {', '.join(val_loaders.keys())} every {config.train.val_every_n_epoch} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, config.train.epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{config.train.epochs} | LR: {current_lr:.6f}")

        train_loss = train_one_epoch(model, train_loader, optimizer, config, vocab, scaler, epoch)
        epoch_time = time.time() - epoch_start
        print(f"  Train Loss: {train_loss:.4f} | {epoch_time:.1f}s")

        history['train_loss'].append(train_loss)
        history['lr'].append(current_lr)

        if (epoch + 1) % config.train.val_every_n_epoch == 0 or epoch == 0:
            avg_exprate = 0.0
            for split, loader in val_loaders.items():
                val_loss, exprate_dict, bleu_dict, token_acc = validate(
                    model, loader, config, vocab, split_name=split,
                )
                exprate = exprate_dict['exprate']
                print(f"  {split}: Loss={val_loss:.4f} | "
                      f"ExpRate={exprate:.2f}% (<=1:{exprate_dict['exprate_1']:.2f}% "
                      f"<=2:{exprate_dict['exprate_2']:.2f}%) | "
                      f"BLEU-4={bleu_dict['bleu_4']:.2f} | TokAcc={token_acc:.2f}%")

                history[f'{split}_val_loss'].append(val_loss)
                history[f'{split}_exprate'].append(exprate)
                history[f'{split}_exprate_1'].append(exprate_dict['exprate_1'])
                history[f'{split}_exprate_2'].append(exprate_dict['exprate_2'])
                history[f'{split}_bleu_4'].append(bleu_dict.get('bleu_4', 0.0))
                history[f'{split}_token_accuracy'].append(token_acc)
                avg_exprate += exprate

            avg_exprate /= len(val_loaders)
            scheduler.step(avg_exprate)

            if avg_exprate > best_exprate:
                best_exprate = avg_exprate
                ckpt_path = os.path.join(config.train.checkpoint_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_exprate': best_exprate,
                }, ckpt_path)
                print(f"  * New best avg ExpRate: {best_exprate:.2f}% -- saved")

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        plot_training_curves(history, config.train.output_dir)

    print(f"\nTraining complete. Best avg ExpRate: {best_exprate:.2f}%")


if __name__ == "__main__":
    main()
