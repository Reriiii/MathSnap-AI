import random
from functools import partial
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from .vocab import Vocabulary

# ── Data cleaning ─────────────────────────────────────────────────────────────
_NORMALIZE_TOK = {
    '√': '\\sqrt', '×': '\\times', '÷': '\\div',  '±': '\\pm',
    '≤': '\\leq',  '≥': '\\geq',   '≠': '\\neq',  '≈': '\\approx',
    '∞': '\\infty','→': '\\rightarrow', '←': '\\leftarrow',
    '∈': '\\in',   '∩': '\\cap',   '∪': '\\cup',
    '∑': '\\sum',  '∏': '\\prod',  '∫': '\\int',
    'π': '\\pi',   'α': '\\alpha', 'β': '\\beta',  'γ': '\\gamma',
    'θ': '\\theta','λ': '\\lambda','μ': '\\mu',    'σ': '\\sigma',
    'φ': '\\phi',  'ω': '\\omega',
    '²': '^{2}',   '³': '^{3}',
    '、': ',',      '。': '.', '，': ',', '．': '.', '！': '!', '？': '?',
}
_NOISE_TOKENS = {'"', "'", '…', '—', '–'}


def _is_chinese(token: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf'
               for c in token)


def _clean_tokens(tokens: list) -> list:
    return [_NORMALIZE_TOK.get(t, t) for t in tokens]


def _parse_label_file(path: str, data_root: str, max_len: int):
    root = Path(data_root)
    samples = []
    n_notab = n_miss = n_long = n_empty = n_chinese = n_noise = 0

    with open(path, encoding='utf-8') as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip('\n')
            if not line.strip():
                continue
            if '\t' not in line:
                n_notab += 1
                if n_notab <= 3:
                    tqdm.write(f"  WARNING line {lineno}: no TAB → {line[:60]!r}")
                continue
            img_name, latex = line.split('\t', 1)
            img_path = root / img_name.strip()
            if not img_path.exists():
                n_miss += 1
                continue
            tokens = _clean_tokens(latex.strip().split())
            if not tokens:
                n_empty += 1
                continue
            if len(tokens) > max_len - 2:
                n_long += 1
                continue
            if any(_is_chinese(t) for t in tokens):
                n_chinese += 1
                continue
            if set(tokens) & _NOISE_TOKENS:
                n_noise += 1
                continue
            samples.append({'img_path': str(img_path),
                            'latex':    ' '.join(tokens),
                            'tokens':   tokens})

    tqdm.write(
        f"Parsed {Path(path).name}: {len(samples):,} ok | "
        f"no-tab={n_notab} | missing={n_miss} | too-long={n_long} | "
        f"empty={n_empty} | chinese={n_chinese} | noise={n_noise}"
    )
    return samples


def _split3(samples, train_r, val_r, seed):
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n  = len(idx)
    n1 = int(n * train_r)
    n2 = int(n * val_r)
    train_s = [samples[i] for i in idx[:n1]]
    val_s   = [samples[i] for i in idx[n1:n1+n2]]
    test_s  = [samples[i] for i in idx[n1+n2:]]
    test_r  = max(0.0, 1.0 - train_r - val_r)
    tqdm.write(
        f"Split {train_r:.0%}/{val_r:.0%}/{test_r:.0%} → "
        f"train={len(train_s):,} | val={len(val_s):,} | test={len(test_s):,}"
    )
    return train_s, val_s, test_s


class HME100KDataset(Dataset):
    def __init__(self, samples, vocab: Vocabulary, img_h, img_w,
                 augment=False, name='?'):
        self.samples = samples
        self.vocab   = vocab
        self.name    = name
        self.tf      = self._build_tf(img_h, img_w, augment)
        tqdm.write(f"Dataset [{name}]: {len(samples):,} samples")

    @staticmethod
    def _build_tf(h, w, augment):
        ops = []
        if augment:
            ops += [transforms.RandomAffine(
                degrees=10, scale=(0.7, 1.1), fill=255,
            )]
        ops += [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        return transforms.Compose(ops)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            # Ảnh toán học là chữ đen giấy trắng -> load trực tiếp thành Grayscale (kênh L)
            # Sau đó convert sang RGB vì mạng DenseNet yêu cầu input 3 kênh
            img = Image.open(s['img_path']).convert('L').convert('RGB')
        except Exception:
            img = Image.new('RGB', (512, 128), 255)
        tids = ([self.vocab.sos_idx]
                + self.vocab.encode(s['tokens'])
                + [self.vocab.eos_idx])
        return {
            'image':     self.tf(img),
            'token_ids': torch.tensor(tids, dtype=torch.long),
            'latex':     s['latex'],
            'tokens':    s['tokens'],
            'img_path':  s['img_path']
        }


def _collate(batch, pad_idx):
    imgs  = torch.stack([b['image'] for b in batch])
    maxL  = max(b['token_ids'].size(0) for b in batch)
    tids  = torch.full((len(batch), maxL), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b['token_ids'].size(0)
        tids[i, :L] = b['token_ids']
    return {
        'image':     imgs,
        'token_ids': tids,
        'latex':     [b['latex']  for b in batch],
        'tokens':    [b['tokens'] for b in batch],
        'img_path':  [b['img_path'] for b in batch],
    }


def build_datasets(cfg):
    """Parse label file, split 80/10/10, load/build vocab."""
    import os
    all_s = _parse_label_file(cfg.label_file, cfg.data_root, cfg.max_len)
    train_s, val_s, test_s = _split3(all_s, cfg.train_ratio, cfg.val_ratio, cfg.seed)

    if os.path.exists(cfg.vocab_path):
        vocab = Vocabulary.load(cfg.vocab_path)
        current_tokens = {t for s in all_s for t in s['tokens']}
        unknown = current_tokens - set(vocab.t2i.keys())
        if unknown:
            tqdm.write(
                f"  WARNING: {len(unknown)} tokens in data NOT in vocab. "
                f"Delete {cfg.vocab_path} and re-run to rebuild."
            )
    else:
        vocab = Vocabulary.build([s['tokens'] for s in all_s])
        vocab.save(cfg.vocab_path)

    kw = dict(img_h=cfg.img_h, img_w=cfg.img_w)
    return (
        HME100KDataset(train_s, vocab, augment=cfg.augment, name='train', **kw),
        HME100KDataset(val_s,   vocab, augment=False,       name='val',   **kw),
        HME100KDataset(test_s,  vocab, augment=False,       name='test',  **kw),
        vocab,
    )


def build_loaders(cfg, vocab):
    """Build train/val/test DataLoaders given pre-built datasets."""
    all_s = _parse_label_file(cfg.label_file, cfg.data_root, cfg.max_len)
    train_s, val_s, test_s = _split3(all_s, cfg.train_ratio, cfg.val_ratio, cfg.seed)
    kw   = dict(img_h=cfg.img_h, img_w=cfg.img_w)
    col  = partial(_collate, pad_idx=vocab.pad_idx)
    dkw  = dict(collate_fn=col, pin_memory=True,
                num_workers=getattr(cfg, 'num_workers', 2))
    train_ds = HME100KDataset(train_s, vocab, augment=getattr(cfg,'augment',False),
                               name='train', **kw)
    val_ds   = HME100KDataset(val_s,   vocab, augment=False, name='val',   **kw)
    test_ds  = HME100KDataset(test_s,  vocab, augment=False, name='test',  **kw)
    train_ldr = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                           drop_last=True, **dkw)
    val_ldr   = DataLoader(val_ds,   cfg.batch_size, shuffle=False, **dkw)
    test_ldr  = DataLoader(test_ds,  cfg.batch_size, shuffle=False, **dkw)
    return train_ldr, val_ldr, test_ldr
