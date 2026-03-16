"""
Evaluate best checkpoint with beam search on all val splits.
Saves results and comparison plot.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from config import Config
from data.dataset import get_dataloader
from data.vocab import Vocab
from models.model import build_model
from utils.metrics import compute_exprate, compute_bleu, compute_token_accuracy


def evaluate_split(model, loader, vocab, config, split_name, beam_size=10):
    """Evaluate a single split with both greedy and beam search."""
    model.eval()
    device = config.device

    greedy_preds = []
    beam_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"  Eval {split_name}", leave=True, ncols=120)
    for batch in pbar:
        imgs = batch['image'].to(device)
        mask = batch['padding_mask'].to(device)
        indices = batch['indices']

        # Greedy decode
        g_indices = model.greedy_decode(
            imgs, mask,
            sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx, max_len=150,
        )
        for idx in g_indices:
            greedy_preds.append(vocab.decode(idx, remove_special=True))

        # Beam search decode
        with torch.amp.autocast('cuda', enabled=config.train.use_amp):
            b_indices = model.beam_search_decode(
                imgs, mask,
                sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
                pad_idx=vocab.pad_idx,
                beam_size=beam_size, max_len=150, alpha=1.0,
            )
        for idx in b_indices:
            beam_preds.append(vocab.decode(idx, remove_special=True))

        for idx_list in indices:
            all_targets.append(vocab.decode(idx_list, remove_special=True))

    # Compute metrics
    results = {}
    for name, preds in [('greedy', greedy_preds), (f'beam_{beam_size}', beam_preds)]:
        exprate = compute_exprate(preds, all_targets)
        bleu = compute_bleu(preds, all_targets)
        tok_acc = compute_token_accuracy(preds, all_targets)
        results[name] = {
            'exprate': exprate['exprate'],
            'exprate_1': exprate['exprate_1'],
            'exprate_2': exprate['exprate_2'],
            'bleu_4': bleu.get('bleu_4', 0.0),
            'token_accuracy': tok_acc,
        }

    return results


def plot_comparison(all_results, beam_size, output_path):
    """Plot greedy vs beam search comparison."""
    splits = list(all_results.keys())
    metrics = ['exprate', 'exprate_1', 'exprate_2', 'bleu_4', 'token_accuracy']
    labels = ['ExpRate', 'ExpRate <=1', 'ExpRate <=2', 'BLEU-4', 'Token Acc']

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    fig.suptitle(f'Greedy vs Beam Search (size={beam_size})', fontsize=14, fontweight='bold')

    x = range(len(splits))
    width = 0.35

    for ax, metric, label in zip(axes, metrics, labels):
        greedy_vals = [all_results[s]['greedy'][metric] for s in splits]
        beam_vals = [all_results[s][f'beam_{beam_size}'][metric] for s in splits]

        bars1 = ax.bar([i - width/2 for i in x], greedy_vals, width, label='Greedy', color='#3498db', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], beam_vals, width, label=f'Beam {beam_size}', color='#e74c3c', alpha=0.8)

        # Value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Split')
        ax.set_ylabel('%')
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def main():
    config = Config()
    vocab = Vocab.from_file(config.data.vocab_path)
    beam_size = config.model.beam_size  # 10

    # Load best model
    ckpt_path = os.path.join(config.train.checkpoint_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint at {ckpt_path}")
        return

    model = build_model(config, len(vocab)).to(config.device)
    ckpt = torch.load(ckpt_path, map_location=config.device)
    model.load_state_dict(ckpt['model_state_dict'])
    epoch = ckpt.get('epoch', '?')
    best_exprate = ckpt.get('best_exprate', 0.0)
    print(f"Loaded best model from epoch {epoch} (avg ExpRate: {best_exprate:.2f}%)")
    print(f"Beam size: {beam_size}")
    print(f"Vocab size: {len(vocab)}")
    print()

    # Evaluate all splits
    splits = ['2014', '2016', '2019']
    all_results = {}

    for split in splits:
        loader = get_dataloader(split, vocab, config)
        print(f"Evaluating {split}: {len(loader.dataset)} samples")
        results = evaluate_split(model, loader, vocab, config, split, beam_size)
        all_results[split] = results

        for method in ['greedy', f'beam_{beam_size}']:
            r = results[method]
            print(f"  {method:>10}: ExpRate={r['exprate']:.2f}% "
                  f"(<=1:{r['exprate_1']:.2f}% <=2:{r['exprate_2']:.2f}%) "
                  f"BLEU-4={r['bleu_4']:.2f} TokAcc={r['token_accuracy']:.2f}%")
        print()

    # Save results
    results_path = os.path.join(config.train.output_dir, "eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved: {results_path}")

    # Plot comparison
    plot_path = os.path.join(config.train.output_dir, "eval_greedy_vs_beam.png")
    plot_comparison(all_results, beam_size, plot_path)


if __name__ == "__main__":
    main()
