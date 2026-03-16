"""
Evaluate best CoMER checkpoint on external datasets:
- HMER100K_new (offline images)
- MathWriting-2024 (online ink → rendered images)

Reports ExpRate, <=1, <=2, <=3, BLEU-4 for each dataset.
Only evaluates on samples where ALL ground-truth tokens exist in our vocab.
"""

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import Config
from data.vocab import Vocab
from models.model import build_model
from utils.metrics import compute_exprate, compute_bleu, compute_token_accuracy


# ============================================================
# Label tokenization
# ============================================================

def tokenize_spaced_label(label_str, vocab):
    """Tokenize space-separated label (HMER100K, CROHME style).
    Returns list of token indices, or None if any OOV token found.
    """
    tokens = label_str.strip().split()
    indices = []
    for t in tokens:
        idx = vocab.token2idx.get(t, None)
        if idx is None or idx == vocab.unk_idx:
            return None  # OOV — skip this sample
        indices.append(idx)
    return indices


def tokenize_compact_label(label_str, vocab):
    """Tokenize compact LaTeX label (MathWriting style).
    Splits '\frac{x}{y}' into ['\frac', '{', 'x', '}', '{', 'y', '}']
    then maps to vocab indices. Returns None if any OOV.
    """
    tokens = _split_latex(label_str.strip())
    indices = []
    for t in tokens:
        idx = vocab.token2idx.get(t, None)
        if idx is None or idx == vocab.unk_idx:
            return None  # OOV
        indices.append(idx)
    return indices


def _split_latex(s):
    """Split compact LaTeX into token list."""
    tokens = []
    i = 0
    while i < len(s):
        if s[i] == ' ':
            i += 1
            continue
        if s[i] == '\\':
            # LaTeX command
            j = i + 1
            if j < len(s) and not s[j].isalpha():
                # Single-char command like \{ \} \, etc.
                tokens.append(s[i:j+1])
                i = j + 1
            else:
                while j < len(s) and s[j].isalpha():
                    j += 1
                tokens.append(s[i:j])
                i = j
        elif s[i] in '{}^_()[]+=<>!,.:;|/\'"':
            tokens.append(s[i])
            i += 1
        elif s[i] == '-':
            tokens.append('-')
            i += 1
        elif s[i].isalnum():
            tokens.append(s[i])
            i += 1
        else:
            tokens.append(s[i])
            i += 1
    return tokens


# ============================================================
# Image preprocessing
# ============================================================

def preprocess_image(img_gray, h_hi=128, w_hi=512, target_bg='black'):
    """Preprocess grayscale image to match CoMER training format.
    - Binarize with Otsu
    - Ensure black background / white foreground
    - Scale to fit within h_hi x w_hi
    - Return float32 [0, 1] image
    """
    # Binarize
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect background: check border pixels
    h, w = binary.shape
    border = np.concatenate([
        binary[0, :], binary[-1, :],
        binary[:, 0], binary[:, -1]
    ])
    bg_is_white = np.mean(border) > 128

    if target_bg == 'black' and bg_is_white:
        binary = 255 - binary
    elif target_bg == 'white' and not bg_is_white:
        binary = 255 - binary

    # Scale to fit within bounds (preserve aspect ratio)
    h, w = binary.shape
    scale = min(h_hi / h, w_hi / w, 1.0)  # only downscale
    if scale < 1.0:
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        binary = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return binary.astype(np.float32) / 255.0


def render_ink_to_image(traces, img_height=128, line_width=2, padding=10):
    """Render InkML traces (strokes) to a grayscale image.
    Black background, white strokes (matching CoMER format).
    """
    # Parse all points
    all_x, all_y = [], []
    strokes = []
    for trace in traces:
        points = []
        for pt_str in trace.text.strip().split(','):
            coords = pt_str.strip().split()
            if len(coords) >= 2:
                x, y = float(coords[0]), float(coords[1])
                points.append((x, y))
                all_x.append(x)
                all_y.append(y)
        if points:
            strokes.append(points)

    if not all_x:
        return np.zeros((img_height, img_height), dtype=np.float32)

    # Compute bounds
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    ink_w = max_x - min_x + 1e-6
    ink_h = max_y - min_y + 1e-6

    # Scale to target height
    scale = (img_height - 2 * padding) / ink_h
    img_w = max(int(ink_w * scale + 2 * padding), img_height)

    # Create image
    img = np.zeros((img_height, img_w), dtype=np.uint8)

    # Draw strokes
    for stroke in strokes:
        pts = []
        for x, y in stroke:
            px = int((x - min_x) * scale + padding)
            py = int((y - min_y) * scale + padding)
            pts.append((px, py))
        for k in range(len(pts) - 1):
            cv2.line(img, pts[k], pts[k+1], 255, line_width, cv2.LINE_AA)

    return img.astype(np.float32) / 255.0


# ============================================================
# Datasets
# ============================================================

class HMER100KTestDataset(Dataset):
    """HMER100K test set."""

    def __init__(self, data_dir, vocab, h_hi=128, w_hi=512):
        self.img_dir = os.path.join(data_dir, 'test', 'img')
        self.vocab = vocab
        self.h_hi = h_hi
        self.w_hi = w_hi
        self.samples = []  # (img_id, token_indices)

        caption_path = os.path.join(data_dir, 'test', 'caption.txt')
        skipped = 0
        with open(caption_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                img_id = parts[0]
                label = parts[1]
                indices = tokenize_spaced_label(label, vocab)
                if indices is not None:
                    self.samples.append((img_id, indices))
                else:
                    skipped += 1

        print(f"  HMER100K test: {len(self.samples)} compatible, {skipped} skipped (OOV)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, indices = self.samples[idx]
        img_path = os.path.join(self.img_dir, f'{img_id}.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback: black image
            img = np.zeros((self.h_hi, self.h_hi), dtype=np.float32)
        else:
            img = preprocess_image(img, self.h_hi, self.w_hi, target_bg='black')
        return img, indices


class MathWritingTestDataset(Dataset):
    """MathWriting-2024 test set (online ink → rendered images)."""

    def __init__(self, data_dir, vocab, h_hi=128, w_hi=512):
        self.vocab = vocab
        self.h_hi = h_hi
        self.w_hi = w_hi
        self.samples = []  # (inkml_path, token_indices)

        test_dir = os.path.join(data_dir, 'test')
        ns = '{http://www.w3.org/2003/InkML}'
        skipped_oov = 0
        skipped_parse = 0

        files = sorted([f for f in os.listdir(test_dir) if f.endswith('.inkml')])
        for fname in tqdm(files, desc="  Parsing MathWriting", ncols=100):
            fpath = os.path.join(test_dir, fname)
            try:
                tree = ET.parse(fpath)
                root = tree.getroot()
            except ET.ParseError:
                skipped_parse += 1
                continue

            # Get normalized label
            label = None
            for ann in root.findall(f'{ns}annotation'):
                if ann.get('type') == 'normalizedLabel':
                    label = ann.text
                    break
            if not label:
                for ann in root.findall(f'{ns}annotation'):
                    if ann.get('type') == 'label':
                        label = ann.text
                        break
            if not label:
                skipped_parse += 1
                continue

            # Tokenize
            indices = tokenize_compact_label(label, vocab)
            if indices is None:
                skipped_oov += 1
                continue

            self.samples.append((fpath, indices))

        print(f"  MathWriting test: {len(self.samples)} compatible, "
              f"{skipped_oov} OOV skipped, {skipped_parse} parse errors")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, indices = self.samples[idx]
        ns = '{http://www.w3.org/2003/InkML}'

        tree = ET.parse(fpath)
        root = tree.getroot()
        traces = root.findall(f'{ns}trace')

        img = render_ink_to_image(traces, img_height=self.h_hi)

        # Scale width if needed
        h, w = img.shape
        if w > self.w_hi:
            scale = self.w_hi / w
            new_w = self.w_hi
            new_h = max(1, int(h * scale))
            img_uint8 = (img * 255).astype(np.uint8)
            img_uint8 = cv2.resize(img_uint8, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = img_uint8.astype(np.float32) / 255.0

        return img, indices


def collate_external(batch):
    """Collate variable-size images into padded batch."""
    images, indices_list = zip(*batch)

    # Find max dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    # Pad images (black padding = 0)
    batch_imgs = torch.zeros(len(images), 1, max_h, max_w, dtype=torch.float32)
    batch_masks = torch.ones(len(images), max_h, max_w, dtype=torch.bool)

    for i, img in enumerate(images):
        h, w = img.shape
        batch_imgs[i, 0, :h, :w] = torch.from_numpy(img)
        batch_masks[i, :h, :w] = False  # False = not masked

    return {
        'image': batch_imgs,
        'padding_mask': batch_masks,
        'indices': list(indices_list),
    }


# ============================================================
# Metrics (with <=3 added)
# ============================================================

def compute_exprate_extended(preds, targets):
    """Compute ExpRate, <=1, <=2, <=3."""
    exact = 0
    off_1 = 0
    off_2 = 0
    off_3 = 0
    total = len(preds)

    for pred, target in zip(preds, targets):
        pred_tokens = pred.split() if isinstance(pred, str) else pred
        tgt_tokens = target.split() if isinstance(target, str) else target

        if pred_tokens == tgt_tokens:
            exact += 1
            off_1 += 1
            off_2 += 1
            off_3 += 1
        else:
            # Edit distance
            dist = _edit_distance(pred_tokens, tgt_tokens)
            if dist <= 1:
                off_1 += 1
            if dist <= 2:
                off_2 += 1
            if dist <= 3:
                off_3 += 1

    return {
        'exprate': 100.0 * exact / max(total, 1),
        'exprate_1': 100.0 * off_1 / max(total, 1),
        'exprate_2': 100.0 * off_2 / max(total, 1),
        'exprate_3': 100.0 * off_3 / max(total, 1),
    }


def _edit_distance(s1, s2):
    """Token-level edit distance."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


# ============================================================
# Evaluation
# ============================================================

def evaluate_dataset(model, dataloader, vocab, config, dataset_name):
    """Evaluate model on a dataset using greedy decode."""
    model.eval()
    device = config.device

    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"  Eval {dataset_name}", ncols=120)
    for batch in pbar:
        imgs = batch['image'].to(device)
        mask = batch['padding_mask'].to(device)
        indices = batch['indices']

        with torch.amp.autocast('cuda', enabled=config.train.use_amp):
            pred_indices = model.greedy_decode(
                imgs, mask,
                sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx, max_len=150,
            )

        for pidx in pred_indices:
            all_preds.append(vocab.decode(pidx, remove_special=True))
        for idx_list in indices:
            all_targets.append(vocab.decode(idx_list, remove_special=True))

    # Metrics
    exprate = compute_exprate_extended(all_preds, all_targets)
    bleu = compute_bleu(all_preds, all_targets)

    return exprate, bleu, all_preds, all_targets


def plot_results(results, output_path):
    """Plot comparison across datasets."""
    datasets = list(results.keys())
    metrics = ['exprate', 'exprate_1', 'exprate_2', 'exprate_3', 'bleu_4']
    labels = ['ExpRate', '<=1', '<=2', '<=3', 'BLEU-4']
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.15

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results[d][metric] for d in datasets]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('%')
    ax.set_title('CoMER Evaluation on External Datasets (Greedy Decode)')
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def main():
    config = Config()
    vocab = Vocab.from_file(config.data.vocab_path)

    # Load best model
    ckpt_path = os.path.join(config.train.checkpoint_dir, "best_model.pt")
    model = build_model(config, len(vocab)).to(config.device)
    ckpt = torch.load(ckpt_path, map_location=config.device)
    model.load_state_dict(ckpt['model_state_dict'])
    epoch = ckpt.get('epoch', '?')
    print(f"Loaded best model from epoch {epoch}")
    print(f"Vocab: {len(vocab)} tokens")
    print()

    results = {}

    # --- CROHME (baseline) ---
    from data.dataset import get_dataloader
    for split in ['2014', '2016', '2019']:
        loader = get_dataloader(split, vocab, config)
        print(f"CROHME {split}: {len(loader.dataset)} samples")
        exprate, bleu, _, _ = evaluate_dataset(model, loader, vocab, config, f"CROHME-{split}")
        results[f'CROHME-{split}'] = {**exprate, 'bleu_4': bleu.get('bleu_4', 0)}
        print(f"  ExpRate={exprate['exprate']:.2f}% <=1={exprate['exprate_1']:.2f}% "
              f"<=2={exprate['exprate_2']:.2f}% <=3={exprate['exprate_3']:.2f}% "
              f"BLEU-4={bleu.get('bleu_4', 0):.2f}")
        print()

    # --- HMER100K ---
    hmer100k_dir = 'D:/dataset/HMER100K_new'
    if os.path.exists(hmer100k_dir):
        print("Loading HMER100K test set...")
        ds = HMER100KTestDataset(hmer100k_dir, vocab, h_hi=config.data.h_hi, w_hi=config.data.w_hi)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_external)
        exprate, bleu, _, _ = evaluate_dataset(model, loader, vocab, config, "HMER100K")
        results['HMER100K'] = {**exprate, 'bleu_4': bleu.get('bleu_4', 0)}
        print(f"  ExpRate={exprate['exprate']:.2f}% <=1={exprate['exprate_1']:.2f}% "
              f"<=2={exprate['exprate_2']:.2f}% <=3={exprate['exprate_3']:.2f}% "
              f"BLEU-4={bleu.get('bleu_4', 0):.2f}")
        print()

    # --- MathWriting ---
    mathwriting_dir = 'D:/dataset/mathwriting-2024'
    if os.path.exists(mathwriting_dir):
        print("Loading MathWriting test set...")
        ds = MathWritingTestDataset(mathwriting_dir, vocab, h_hi=config.data.h_hi, w_hi=config.data.w_hi)
        if len(ds) > 0:
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_external)
            exprate, bleu, _, _ = evaluate_dataset(model, loader, vocab, config, "MathWriting")
            results['MathWriting'] = {**exprate, 'bleu_4': bleu.get('bleu_4', 0)}
            print(f"  ExpRate={exprate['exprate']:.2f}% <=1={exprate['exprate_1']:.2f}% "
                  f"<=2={exprate['exprate_2']:.2f}% <=3={exprate['exprate_3']:.2f}% "
                  f"BLEU-4={bleu.get('bleu_4', 0):.2f}")
        else:
            print("  No compatible samples found!")
        print()

    # Save results
    out_path = os.path.join(config.train.output_dir, "eval_external_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_path}")

    # Plot
    plot_path = os.path.join(config.train.output_dir, "eval_external.png")
    plot_results(results, plot_path)

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Dataset':<20} {'Samples':>8} {'ExpRate':>8} {'<=1':>8} {'<=2':>8} {'<=3':>8} {'BLEU-4':>8}")
    print(f"{'='*80}")
    for name, r in results.items():
        print(f"{name:<20} {'':>8} {r['exprate']:>7.2f}% {r['exprate_1']:>7.2f}% "
              f"{r['exprate_2']:>7.2f}% {r['exprate_3']:>7.2f}% {r['bleu_4']:>7.2f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
