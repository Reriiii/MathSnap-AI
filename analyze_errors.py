"""
Error analysis script: run model on validation set and categorize errors.
Helps identify what types of expressions the model struggles with.
"""

import torch
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from config import Config
from data.vocab import Vocab
from data.dataset import get_dataloader
from models.model import build_model
from tqdm import tqdm


def categorize_expression(latex: str) -> list:
    """Categorize a LaTeX expression by its structural features."""
    cats = []
    if '\\frac' in latex:
        cats.append('fraction')
    if '\\sqrt' in latex:
        cats.append('sqrt')
    if '_' in latex:
        cats.append('subscript')
    if '^' in latex:
        cats.append('superscript')
    if '\\sum' in latex or '\\prod' in latex or '\\int' in latex:
        cats.append('big_operator')
    if '\\left' in latex or '\\right' in latex:
        cats.append('delimiters')
    if '\\matrix' in latex or '\\pmatrix' in latex or '\\begin{array}' in latex:
        cats.append('matrix')
    if '\\lim' in latex or '\\log' in latex or '\\sin' in latex or '\\cos' in latex:
        cats.append('function')

    tokens = latex.split()
    if len(tokens) <= 5:
        cats.append('short (<=5)')
    elif len(tokens) <= 15:
        cats.append('medium (6-15)')
    elif len(tokens) <= 30:
        cats.append('long (16-30)')
    else:
        cats.append('very_long (>30)')

    if not cats or all('(' in c for c in cats):
        cats.append('simple')
    return cats


def edit_distance(s1, s2):
    """Levenshtein edit distance between two token lists."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def analyze_error_types(pred_tokens, gt_tokens):
    """Analyze what kind of errors are present."""
    errors = []
    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)

    # Missing tokens
    missing = gt_set - pred_set
    if missing:
        errors.append(('missing_tokens', missing))

    # Extra tokens
    extra = pred_set - gt_set
    if extra:
        errors.append(('extra_tokens', extra))

    # Length mismatch
    len_diff = len(pred_tokens) - len(gt_tokens)
    if len_diff > 3:
        errors.append(('too_long', len_diff))
    elif len_diff < -3:
        errors.append(('too_short', len_diff))

    # Repetition detection
    if len(pred_tokens) > 3:
        for i in range(len(pred_tokens) - 2):
            if pred_tokens[i] == pred_tokens[i+1] == pred_tokens[i+2]:
                errors.append(('repetition', pred_tokens[i]))
                break

    # Empty prediction
    if len(pred_tokens) == 0:
        errors.append(('empty_pred', None))

    return errors


def main():
    config = Config()
    if not torch.cuda.is_available():
        config.device = 'cpu'

    # Load vocab
    vocab = Vocab.from_file(str(config.data.vocab_path))
    print(f"Vocab size: {len(vocab)}")

    # Load model
    model = build_model(vocab_size=len(vocab), config=config)
    ckpt_path = Path(config.train.checkpoint_dir) / 'best.pt'
    if not ckpt_path.exists():
        ckpt_path = Path(config.train.checkpoint_dir) / 'latest.pt'

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=config.device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(config.device)
    model.eval()

    # BN calibration
    print("Calibrating BatchNorm...")
    train_loader = get_dataloader('train', vocab, config)
    import torch.nn as nn
    model.train()
    for m in model.encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= 30:
                break
            images = batch['image'].to(config.device)
            pad_mask = batch['padding_mask'].to(config.device) if 'padding_mask' in batch else None
            model.encode(images, padding_mask=pad_mask)
    for m in model.encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.05
    model.eval()

    # Run validation
    val_loader = get_dataloader('val', vocab, config)

    all_preds = []
    all_gts = []
    all_paths = []

    print("Generating predictions on full val set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Generating"):
            images = batch['image'].to(config.device)
            pad_mask = batch['padding_mask'].to(config.device) if 'padding_mask' in batch else None
            preds = model.generate(
                images,
                sos_idx=vocab.sos_idx,
                eos_idx=vocab.eos_idx,
                max_len=config.data.max_seq_len,
                padding_mask=pad_mask,
            )
            for i in range(preds.size(0)):
                pred_str = vocab.decode(preds[i].cpu().tolist())
                gt_str = batch['latex'][i]
                all_preds.append(pred_str)
                all_gts.append(gt_str)
                all_paths.append(batch['image_path'][i])

    # Analysis
    total = len(all_preds)
    correct = sum(1 for p, g in zip(all_preds, all_gts) if p.strip() == g.strip())
    print(f"\n{'='*70}")
    print(f"RESULTS: {correct}/{total} correct = {100*correct/total:.2f}% ExpRate")
    print(f"{'='*70}")

    # 1. Error by expression category
    cat_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for pred, gt in zip(all_preds, all_gts):
        cats = categorize_expression(gt)
        is_correct = pred.strip() == gt.strip()
        for cat in cats:
            cat_stats[cat]['total'] += 1
            if is_correct:
                cat_stats[cat]['correct'] += 1

    print(f"\n--- ExpRate by Category ---")
    for cat, stats in sorted(cat_stats.items(), key=lambda x: x[1]['correct']/max(x[1]['total'],1)):
        rate = 100 * stats['correct'] / max(stats['total'], 1)
        print(f"  {cat:20s}: {rate:5.1f}% ({stats['correct']:4d}/{stats['total']:4d})")

    # 2. Error type distribution
    error_type_counts = Counter()
    missing_token_counts = Counter()
    extra_token_counts = Counter()
    repetition_counts = Counter()

    wrong_samples = []
    for pred, gt, path in zip(all_preds, all_gts, all_paths):
        if pred.strip() != gt.strip():
            pred_tokens = pred.strip().split()
            gt_tokens = gt.strip().split()
            errors = analyze_error_types(pred_tokens, gt_tokens)
            ed = edit_distance(pred_tokens, gt_tokens)

            for etype, edata in errors:
                error_type_counts[etype] += 1
                if etype == 'missing_tokens':
                    for t in edata:
                        missing_token_counts[t] += 1
                elif etype == 'extra_tokens':
                    for t in edata:
                        extra_token_counts[t] += 1
                elif etype == 'repetition':
                    repetition_counts[edata] += 1

            wrong_samples.append({
                'path': path,
                'gt': gt,
                'pred': pred,
                'edit_dist': ed,
                'gt_len': len(gt_tokens),
                'pred_len': len(pred_tokens),
                'errors': [(e, str(d)) for e, d in errors],
            })

    print(f"\n--- Error Type Distribution (out of {total - correct} wrong) ---")
    for etype, count in error_type_counts.most_common():
        print(f"  {etype:20s}: {count:4d} ({100*count/(total-correct):.1f}%)")

    print(f"\n--- Most Commonly Missing Tokens ---")
    for tok, count in missing_token_counts.most_common(15):
        print(f"  '{tok}': {count}")

    print(f"\n--- Most Commonly Extra Tokens ---")
    for tok, count in extra_token_counts.most_common(15):
        print(f"  '{tok}': {count}")

    if repetition_counts:
        print(f"\n--- Repetition Tokens ---")
        for tok, count in repetition_counts.most_common(10):
            print(f"  '{tok}': {count}")

    # 3. Edit distance distribution
    eds = [s['edit_dist'] for s in wrong_samples]
    print(f"\n--- Edit Distance Distribution (wrong samples) ---")
    print(f"  Mean: {sum(eds)/len(eds):.1f}")
    print(f"  Median: {sorted(eds)[len(eds)//2]}")
    bins = [(1,1), (2,3), (4,7), (8,15), (16,999)]
    for lo, hi in bins:
        count = sum(1 for e in eds if lo <= e <= hi)
        label = f"{lo}" if lo == hi else f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        print(f"  ED {label:5s}: {count:4d} ({100*count/len(eds):.1f}%)")

    # 4. Near-misses (ED=1) - most actionable
    near_misses = [s for s in wrong_samples if s['edit_dist'] == 1]
    print(f"\n--- Near Misses (ED=1): {len(near_misses)} samples ---")
    for s in near_misses[:10]:
        print(f"  GT:   {s['gt'][:70]}")
        print(f"  Pred: {s['pred'][:70]}")
        print()

    # 5. Worst predictions (highest ED)
    wrong_samples.sort(key=lambda x: x['edit_dist'], reverse=True)
    print(f"\n--- Worst Predictions (highest ED) ---")
    for s in wrong_samples[:10]:
        print(f"  ED={s['edit_dist']:3d} | GT({s['gt_len']:3d}): {s['gt'][:50]}")
        print(f"        | Pred({s['pred_len']:3d}): {s['pred'][:50]}")
        print()

    # Save full results
    output = {
        'summary': {
            'total': total,
            'correct': correct,
            'exprate': 100 * correct / total,
        },
        'category_stats': {k: v for k, v in cat_stats.items()},
        'error_types': dict(error_type_counts),
        'missing_tokens_top20': missing_token_counts.most_common(20),
        'extra_tokens_top20': extra_token_counts.most_common(20),
    }

    with open('outputs/error_analysis.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFull analysis saved to outputs/error_analysis.json")


if __name__ == '__main__':
    main()
