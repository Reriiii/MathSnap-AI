import os, json, math, time, random, logging
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from functools import partial
from typing import Optional

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING  (tqdm-safe: won't mangle progress bars)
# ─────────────────────────────────────────────────────────────────────────────
class _TqdmHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)

_logger = logging.getLogger('NAMER')
if not _logger.handlers:
    _h = _TqdmHandler()
    _h.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  ← Edit this section on Kaggle
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    data_root:  str          = 'D://dataset/HME100K'
    label_file: str          = 'D://dataset/HME100K/train.txt'
    checkpoint_dir: str      = './checkpoints'
    vocab_path: str          = './vocab.json'
    resume_checkpoint: Optional[str] = None

    # Split (out of the single label_file)
    train_ratio: float = 0.80   # → 80% train / 10% val / 10% test
    val_ratio:   float = 0.10

    # Training
    epochs:       int   = 40    # HME100K ~74K samples → 40ep ≈ 60K steps (same as CROHME 240ep)
    batch_size:   int   = 32
    lr:           float = 2e-4
    lambda_pgd:   float = 0.5
    augment:      bool  = False
    eval_every:   int   = 5      # validate every N epochs
    log_interval: int   = 50     # update tqdm postfix every N steps
    pgd_teacher_epochs: int = 8  # epochs using GT tokens for PGD (curriculum)

    # Model (paper defaults)
    d_model:    int = 256
    nhead:      int = 8
    pgd_layers: int = 2   # paper: "two layer Transformer in PGD"
    drop:       float = 0.3
    img_h:      int = 128
    img_w:      int = 512
    max_len:    int = 200        # samples with more tokens are skipped

    # Misc
    num_workers: int = 2
    seed:        int = 42


# ─────────────────────────────────────────────────────────────────────────────
#  VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
_SPECIAL = ['<pad>', '<sos>', '<eos>', '<unk>', '∅']   # ∅ = none / background token

class Vocabulary:
    def __init__(self):
        self.t2i: dict = {}
        self.i2t: dict = {}
        for t in _SPECIAL:
            self._add(t)

    def _add(self, tok: str):
        if tok not in self.t2i:
            i = len(self.t2i)
            self.t2i[tok] = i
            self.i2t[i]   = tok

    def __len__(self):           return len(self.t2i)
    @property
    def pad_idx(self):           return self.t2i['<pad>']
    @property
    def sos_idx(self):           return self.t2i['<sos>']
    @property
    def eos_idx(self):           return self.t2i['<eos>']
    @property
    def none_idx(self):          return self.t2i['∅']

    def encode(self, tokens):
        unk = self.t2i['<unk>']
        return [self.t2i.get(t, unk) for t in tokens]

    def decode(self, ids):
        return [self.i2t.get(int(i), '<unk>') for i in ids]

    @classmethod
    def build(cls, token_lists):
        v = cls()
        for tok in sorted({t for toks in token_lists for t in toks}):
            v._add(tok)
        return v

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.t2i, f, ensure_ascii=False, indent=2)
        tqdm.write(f"Vocab saved → {path}  ({len(self)} tokens)")

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            t2i = json.load(f)
        v = cls()
        v.t2i = t2i
        v.i2t = {int(idx): tok for tok, idx in t2i.items()}
        tqdm.write(f"Vocab loaded ← {path}  ({len(v)} tokens)")
        return v


# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
# Unicode/CJK punctuation → LaTeX/ASCII equivalent
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

# Samples containing ANY of these tokens are dropped entirely
_NOISE_TOKENS = {'"', "'", '…', '—', '–'}

def _is_chinese(token: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf'
               for c in token)

def _clean_tokens(tokens: list) -> list:
    """Normalize token list in-place: unicode→latex, CJK punct→ascii."""
    return [_NORMALIZE_TOK.get(t, t) for t in tokens]


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────
def _parse_label_file(path: str, data_root: str, max_len: int):
    """
    Parse TAB-separated label file, apply cleaning inline:
      - Normalize unicode/CJK tokens
      - Drop samples with Chinese characters or noise tokens
    Returns list of {img_path, latex, tokens}.
    """
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
            # Drop samples with Chinese characters
            if any(_is_chinese(t) for t in tokens):
                n_chinese += 1
                continue
            # Drop samples with noise tokens
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
    """Split a list into train/val/test by ratio. test = 1-train-val."""
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
    def __init__(self, samples, vocab: Vocabulary, img_h, img_w, augment=False, name='?'):
        self.samples = samples
        self.vocab   = vocab
        self.name    = name
        self.tf      = self._build_tf(img_h, img_w, augment)
        tqdm.write(f"Dataset [{name}]: {len(samples):,} samples")

    @staticmethod
    def _build_tf(h, w, augment):
        ops = []
        if augment:
            # Paper: "Only simple random scales and rotations are used"
            # No ColorJitter, no flip
            ops += [transforms.RandomAffine(
                degrees=10,           # ±10° rotation
                scale=(0.7, 1.1),     # random scale
                fill=255,             # white background
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
            img = Image.open(s['img_path']).convert('RGB')
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
    }


def build_datasets(cfg: Config):
    """Parse cfg.label_file, split 80/10/10, build vocab, return datasets + vocab."""
    all_s = _parse_label_file(cfg.label_file, cfg.data_root, cfg.max_len)
    train_s, val_s, test_s = _split3(all_s, cfg.train_ratio, cfg.val_ratio, cfg.seed)

    if os.path.exists(cfg.vocab_path):
        vocab = Vocabulary.load(cfg.vocab_path)
        # Sanity check: vocab built from current data should not have tokens
        # outside the saved vocab. Warn if mismatch detected.
        current_tokens = {t for s in all_s for t in s['tokens']}
        unknown = current_tokens - set(vocab.t2i.keys())
        if unknown:
            tqdm.write(
                f"  WARNING: {len(unknown)} tokens in data are NOT in "
                f"saved vocab ({cfg.vocab_path}). "
                f"Delete the vocab file and re-run to rebuild it."
            )
    else:
        # Build vocab from ALL samples (before split) so val/test tokens
        # are always covered — never from train only.
        vocab = Vocabulary.build([s['tokens'] for s in all_s])
        vocab.save(cfg.vocab_path)

    kw = dict(img_h=cfg.img_h, img_w=cfg.img_w)
    return (
        HME100KDataset(train_s, vocab, augment=cfg.augment, name='train', **kw),
        HME100KDataset(val_s,   vocab, augment=False,       name='val',   **kw),
        HME100KDataset(test_s,  vocab, augment=False,       name='test',  **kw),
        vocab,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ENCODER: DenseNet (standard in HMER literature)
# ─────────────────────────────────────────────────────────────────────────────
class _DenseLayer(nn.Module):
    def __init__(self, in_ch, gr, bns, drop):
        super().__init__()
        self.f = nn.Sequential(
            nn.BatchNorm2d(in_ch),   nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, bns*gr, 1, bias=False),
            nn.BatchNorm2d(bns*gr),  nn.ReLU(inplace=True),
            nn.Conv2d(bns*gr, gr, 3, padding=1, bias=False),
        )
        self.drop = drop

    def forward(self, x):
        out = self.f(x)
        if self.drop > 0:
            out = F.dropout(out, self.drop, self.training)
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, n, in_ch, gr, bns, drop):
        super().__init__()
        for i in range(n):
            self.add_module(f'l{i}', _DenseLayer(in_ch + i*gr, gr, bns, drop))


class _Trans(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.add_module('n', nn.BatchNorm2d(in_ch))
        self.add_module('r', nn.ReLU(inplace=True))
        self.add_module('c', nn.Conv2d(in_ch, out_ch, 1, bias=False))
        self.add_module('p', nn.AvgPool2d(2, 2))


class DenseNetEncoder(nn.Module):
    """
    DenseNet encoder as used throughout HMER literature.
    Returns (F_8x, F_16x) at stride 8 and 16 respectively.
    """
    def __init__(self, gr=24, blocks=(16, 16, 16), init_ch=48, bns=4, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, init_ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        ch = init_ch
        self.b0 = _DenseBlock(blocks[0], ch, gr, bns, drop); ch += blocks[0] * gr
        self.t0 = _Trans(ch, ch // 2);                        ch //= 2
        self.b1 = _DenseBlock(blocks[1], ch, gr, bns, drop); ch += blocks[1] * gr
        self.ch_8x = ch                                        # channel count at stride-8
        self.t1 = _Trans(ch, ch // 2);                        ch //= 2
        self.b2 = _DenseBlock(blocks[2], ch, gr, bns, drop); ch += blocks[2] * gr
        self.n2 = nn.BatchNorm2d(ch)
        self.ch_16x = ch                                       # channel count at stride-16

    def forward(self, x):
        x    = self.stem(x)
        x    = self.t0(self.b0(x))
        x    = self.b1(x);              f8x  = x
        x    = self.b2(self.t1(x));     f16x = F.relu(self.n2(x), True)
        return f8x, f16x


# ─────────────────────────────────────────────────────────────────────────────
#  2D SINUSOIDAL POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────
class PE2D(nn.Module):
    def __init__(self, d, max_h=64, max_w=256):
        super().__init__()
        assert d % 4 == 0
        pe  = torch.zeros(d, max_h, max_w)
        hd  = d // 4
        div = torch.exp(torch.arange(0, hd).float() * (-math.log(10000.0) / hd))
        h   = torch.arange(max_h).float().unsqueeze(1)
        w   = torch.arange(max_w).float().unsqueeze(1)
        pe[:hd]        = (torch.sin(h * div).T).unsqueeze(2)
        pe[hd:2*hd]    = (torch.cos(h * div).T).unsqueeze(2)
        pe[2*hd:3*hd]  = (torch.sin(w * div).T).unsqueeze(1)
        pe[3*hd:]      = (torch.cos(w * div).T).unsqueeze(1)
        self.register_buffer('pe', pe.unsqueeze(0))   # [1, d, H, W]

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


# ─────────────────────────────────────────────────────────────────────────────
#  VISUAL AWARE TOKENIZER (VAT)
# ─────────────────────────────────────────────────────────────────────────────
class VAT(nn.Module):
    """
    Predicts K+1 classes (K tokens + 1 background ∅) at every
    position of the H/8 × W/8 feature map, in parallel (non-autoregressive).
    """
    def __init__(self, ch_8x: int, ch_16x: int, d: int, num_cls: int):
        super().__init__()
        # FPN-style merge: project F_16x up to stride-8 resolution
        self.proj16 = nn.Sequential(
            nn.Conv2d(ch_16x, d, 1, bias=False),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(ch_8x + d, d, 3, padding=1, bias=False),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(d, num_cls, 3, padding=1)

    def forward(self, f8x, f16x):
        p16    = self.proj16(F.interpolate(f16x, f8x.shape[-2:],
                                           mode='bilinear', align_corners=False))
        fm     = self.merge(torch.cat([f8x, p16], dim=1))
        logits = self.head(fm)                  # [B, K+1, H/8, W/8]
        probs  = F.softmax(logits, dim=1)
        return probs, logits


# ─────────────────────────────────────────────────────────────────────────────
#  PARALLEL GRAPH DECODER (PGD)
# ─────────────────────────────────────────────────────────────────────────────
class _XAttnLayer(nn.Module):
    """Cross-attention (to visual features) + self-attention + FFN."""
    def __init__(self, d: int, heads: int, ff: int, drop: float = 0.1):
        super().__init__()
        self.xattn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.sattn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d, ff), nn.ReLU(inplace=False), nn.Dropout(drop), nn.Linear(ff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.dp    = nn.Dropout(drop)

    def forward(self, q, kv, key_padding_mask=None):
        # key_padding_mask: [B, N] True = padding position (ignored as key in sattn)
        q2, _ = self.xattn(q, kv, kv)
        q = self.n1(q + self.dp(q2))
        q2, _ = self.sattn(q, q, q, key_padding_mask=key_padding_mask)
        q = self.n2(q + self.dp(q2))
        return self.n3(q + self.dp(self.ffn(q)))


class _PGDHead(nn.Module):
    def __init__(self, d, heads, ff, n_layers, drop=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [_XAttnLayer(d, heads, ff, drop) for _ in range(n_layers)])

    def forward(self, q, kv, key_padding_mask=None):
        for layer in self.layers:
            q = layer(q, kv, key_padding_mask)
        return q


class PGD(nn.Module):
    """
    Parallel Graph Decoder: three independent NAR heads.
      SCH – Self-node Correction Head  (fixes misclassified tokens)
      LCH – Left  Connectivity Head    (predicts left  neighbour)
      RCH – Right Connectivity Head    (predicts right neighbour)

    Q0 initialization (paper Fig 5, Section 3.3):
      Q0 = SampleFeature(F16x, token_positions) + PE2D + WordEmbedding
    The visual feature is sampled from F16x at each token's approximate position.
    This is the KEY difference from a plain transformer — PGD gets visual grounding.
    """
    def __init__(self, ch_16x: int, d: int, heads: int,
                 n_layers: int, num_cls: int, vocab_sz: int, drop: float = 0.1):
        super().__init__()
        # Project F16x for KV (key/value in cross-attention)
        self.proj_kv = nn.Sequential(
            nn.Conv2d(ch_16x, d, 1, bias=False),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
        )
        # Project F16x for visual feature sampling into Q0.
        # NO BN/ReLU here — vis_feat will be summed with word_emb (norm≈√d≈16),
        # so it must have comparable scale. BN+ReLU at init shrinks it to ~0.04
        # (360× smaller), making visual grounding in Q0 effectively zero.
        self.proj_q = nn.Conv2d(ch_16x, d, 1, bias=True)
        # LayerNorm normalises vis_feat to unit scale before adding to Q0
        self.vis_norm = nn.LayerNorm(d)
        self.pe       = PE2D(d)
        self.word_emb = nn.Embedding(vocab_sz, d, padding_idx=0)
        self.pos_emb  = nn.Embedding(1000, d)   # learned slot positions

        ff = d * 2
        self.sch = _PGDHead(d, heads, ff, n_layers, drop)
        self.lch = _PGDHead(d, heads, ff, n_layers, drop)
        self.rch = _PGDHead(d, heads, ff, n_layers, drop)
        self.cls_head = nn.Linear(d, num_cls)

        # Small init → reduces initial logit magnitude → lower initial PGD loss
        nn.init.normal_(self.word_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        # Restore padding embedding to zero: nn.init.normal_ overrides padding_idx=0 zeroing
        self.word_emb.weight.data[0].zero_()

    def _sample_visual(self, f16x_proj, token_ids):
        """
        Sample visual features from projected F16x at estimated token positions.
        Paper: Q0 includes 'VAT tokens' visual feature' sampled from F16x.

        Since we don't have precise positions, we use the token index to estimate
        a column position (uniform spread), same as VAT target estimation.
        Shape: f16x_proj [B, d, H, W] → sampled [B, N, d]
        """
        B, d, H, W = f16x_proj.shape
        N = token_ids.size(1)
        if N == 0:
            return torch.zeros(B, 0, d, device=f16x_proj.device)

        row = H // 2
        cols = torch.linspace(0, W - 1, N, device=f16x_proj.device).long()
        # f16x_proj[:, :, row, cols] → [B, d, N] → [B, N, d]
        sampled = f16x_proj[:, :, row, cols]   # [B, d, N]
        return sampled.permute(0, 2, 1)        # [B, N, d]

    def _build_q0(self, f16x, token_ids):
        """
        Q0 = VisualFeature(sample from F16x) + 2D_PE + WordEmbedding
        Paper Section 3.3 + Fig 5: elementwise sum of three components.
        """
        B, N = token_ids.shape
        safe_tids = token_ids.clamp(0, self.word_emb.num_embeddings - 1)
        pos = torch.arange(N, device=token_ids.device).clamp(0, self.pos_emb.num_embeddings - 1)

        word_feat = self.word_emb(safe_tids)          # [B, N, d]
        pos_feat  = self.pos_emb(pos)                 # [N, d] → broadcast

        # Sample visual feature from F16x, then normalise to word_emb scale
        f16x_q   = self.proj_q(f16x)                          # [B, d, H, W]
        vis_feat = self._sample_visual(f16x_q, token_ids)     # [B, N, d]
        vis_feat = self.vis_norm(vis_feat)                     # LayerNorm → unit scale

        return vis_feat + pos_feat + word_feat        # [B, N, d]

    def _build_kv(self, f16x):
        """Flatten projected F16x with 2D PE → key/value sequence [B, H*W, d]."""
        f = self.pe(self.proj_kv(f16x))
        return f.flatten(2).transpose(1, 2)

    def forward(self, f16x, token_ids):
        # pad_mask [B, N]: True = padding position, excluded from self-attention keys
        pad_mask = (token_ids == 0)   # pad_idx = 0
        kv = self._build_kv(f16x)
        q0 = self._build_q0(f16x, token_ids)
        qs = self.sch(q0, kv, pad_mask)
        ql = self.lch(q0, kv, pad_mask)
        qr = self.rch(q0, kv, pad_mask)
        return qs, ql, qr, q0

    def compute_scores(self, qs, ql, qr, q0, pad_mask=None):
        """
        cls_logits   [B, N, num_cls]
        edge_scores  [B, N, N]
        left_scores  [B, N, N]   — softmax probs (for display / inference)
        right_scores [B, N, N]   — softmax probs (for display / inference)
        left_logits  [B, N, N]   — raw logits    (for CrossEntropyLoss!)
        right_logits [B, N, N]   — raw logits    (for CrossEntropyLoss!)

        pad_mask: [B, N] bool — True for padding positions (will be masked to -inf)
        """
        cls_logits = self.cls_head(qs)

        # Scale dot-product by √d to prevent softmax saturation / NaN
        # Without scaling: bmm values ∝ d=256 → softmax overflow → NaN gradients
        scale = q0.size(-1) ** 0.5

        raw_left  = torch.bmm(q0, ql.transpose(1, 2)) / scale   # [B, N, N]
        raw_right = torch.bmm(q0, qr.transpose(1, 2)) / scale

        # Mask padding columns to -inf so softmax ignores them
        if pad_mask is not None:
            # pad_mask [B, N] → [B, 1, N] broadcast over query dim
            raw_left  = raw_left.masked_fill(pad_mask.unsqueeze(1), float('-inf'))
            raw_right = raw_right.masked_fill(pad_mask.unsqueeze(1), float('-inf'))

        left_scores  = F.softmax(raw_left,  dim=-1)
        right_scores = F.softmax(raw_right, dim=-1)

        # Safety: replace NaN/inf BEFORE clamp — clamp() does NOT fix NaN values
        # Also clamp both sides: +inf in logits → logsumexp=inf → CE=NaN
        left_logits  = torch.nan_to_num(raw_left,  nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        right_logits = torch.nan_to_num(raw_right, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        # Replace NaN (from all-inf rows) with 0
        left_scores  = torch.nan_to_num(left_scores,  nan=0.0)
        right_scores = torch.nan_to_num(right_scores, nan=0.0)

        edge_scores = right_scores + left_scores.transpose(1, 2)
        return cls_logits, edge_scores, left_scores, right_scores, left_logits, right_logits


# ─────────────────────────────────────────────────────────────────────────────
#  FULL NAMER MODEL
# ─────────────────────────────────────────────────────────────────────────────
class NAMER(nn.Module):
    def __init__(self, vocab_size: int, d: int = 256, heads: int = 8,
                 pgd_layers: int = 3, drop: float = 0.3):
        super().__init__()
        self.enc      = DenseNetEncoder()
        ch8           = self.enc.ch_8x
        ch16          = self.enc.ch_16x
        self.vat      = VAT(ch8, ch16, d, vocab_size)
        self.pgd      = PGD(ch16, d, heads, pgd_layers, vocab_size, vocab_size, drop)

    def forward(self, images, token_ids=None):
        f8x, f16x         = self.enc(images)
        probs, vat_logits = self.vat(f8x, f16x)

        if token_ids is None:
            # none_idx is ∅ (background). In vocab: <pad>=0,<sos>=1,<eos>=2,<unk>=3,∅=4
            # VAT has vocab_size outputs; ∅ class = index 4 always.
            none_idx_inf = 4   # _SPECIAL index of '∅'
            return self._infer(f16x, probs, none_idx=none_idx_inf)

        # Training: use GT token_ids (+ imaginary) as PGD queries
        qs, ql, qr, q0 = self.pgd(f16x, token_ids)
        cls_logits, edge_scores, left_scores, right_scores, left_logits, right_logits = \
            self.pgd.compute_scores(qs, ql, qr, q0)
        return {
            'vat_logits':     vat_logits,
            'pgd_cls_logits': cls_logits,
            'left_scores':    left_scores,
            'right_scores':   right_scores,
            'left_logits':    left_logits,   # raw logits for CrossEntropyLoss
            'right_logits':   right_logits,  # raw logits for CrossEntropyLoss
            'f8x':            f8x,           # for bipartite matching targets
        }

    @torch.no_grad()
    def _infer(self, f16x, probs, none_idx: int):
        """Greedy NAR inference: VAT → PGD → path selection."""
        B, K1, Hm, Wm = probs.shape
        vocab_sz = K1                 # VAT outputs K+1 = vocab_size classes
        results  = []

        for b in range(B):
            pred = probs[b].argmax(dim=0)                        # [H, W]
            pos  = (pred != none_idx).nonzero(as_tuple=False)    # [N, 2]

            # ── Guard: no tokens detected ────────────────────────────────────
            if pos.size(0) == 0:
                results.append([])
                continue

            tids = pred[pos[:, 0], pos[:, 1]].unsqueeze(0)      # [1, N]

            # ── Guard: clamp to valid embedding range ─────────────────────────
            tids = tids.clamp(0, vocab_sz - 1)

            # ── Guard: MultiheadAttention requires seq_len >= 1 ──────────────
            # (self-attention on [1,1,d] is fine; on [1,0,d] crashes CUDA)
            N = tids.size(1)
            if N == 0:
                results.append([])
                continue

            qs, ql, qr, q0 = self.pgd(f16x[b:b+1], tids)
            pad_mask = (tids == 0)   # pad_idx=0
            cls_logits, edge_scores, _, _, _, _ = self.pgd.compute_scores(qs, ql, qr, q0, pad_mask=pad_mask)

            corrected = cls_logits[0].argmax(dim=-1).cpu().tolist()
            E         = edge_scores[0].cpu().numpy()
            results.append(_path_selection(corrected, E, none_idx))

        return results


# ─────────────────────────────────────────────────────────────────────────────
#  PATH SELECTION — DAG longest path
# ─────────────────────────────────────────────────────────────────────────────
def _path_selection(token_ids, E, none_idx, eps=0.5, sos=1, eos=2):
    N = len(token_ids)
    if N == 0:
        return []

    # Build sparse adjacency list (prune weak edges)
    adj = defaultdict(list)
    for i in range(N):
        for j in range(N):
            if i != j and float(E[i, j]) >= eps:
                adj[i].append((j, float(E[i, j])))

    sos_nodes = [i for i, t in enumerate(token_ids) if t == sos]
    eos_nodes = [i for i, t in enumerate(token_ids) if t == eos]

    # Fallback: return tokens in order if no sos/eos found
    if not sos_nodes or not eos_nodes:
        return [t for t in token_ids if t not in (none_idx, sos, eos)]

    start, end = sos_nodes[0], eos_nodes[0]

    # Topological sort via DFS
    vis, topo = [False] * N, []
    def _dfs(v):
        vis[v] = True
        for u, _ in adj[v]:
            if not vis[u]:
                _dfs(u)
        topo.append(v)
    for v in range(N):
        if not vis[v]:
            _dfs(v)
    topo.reverse()

    # DP: longest path in DAG
    dist = [-1e9] * N
    prev = [-1]   * N
    dist[start] = 0.0
    for v in topo:
        if dist[v] == -1e9:
            continue
        for u, w in adj[v]:
            if dist[v] + w > dist[u]:
                dist[u] = dist[v] + w
                prev[u] = v

    # Trace back — guard against cycles in prev[] (can happen with untrained weights)
    path, cur, seen = [], end, set()
    while cur != -1 and cur not in seen:
        seen.add(cur)
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    if not path or path[0] != start:
        return [t for t in token_ids if t not in (none_idx, sos, eos)]

    return [token_ids[i] for i in path if token_ids[i] not in (sos, eos, none_idx)]


# ─────────────────────────────────────────────────────────────────────────────
#  LOSS
# ─────────────────────────────────────────────────────────────────────────────
class NAMERLoss(nn.Module):
    """L_all = L_VAT + lambda * (L_self + w_conn*(L_left + L_right))   — paper Eq.9"""
    def __init__(self, lam: float = 0.5, w_conn: float = 0.1):
        super().__init__()
        self.lam    = lam
        self.w_conn = w_conn   # weight on connectivity loss (L_left + L_right)
        self.ce  = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, out, vat_tgt, pgd_tgt, l_tgt, r_tgt, mask):
        B, L   = pgd_tgt.shape
        L_vat  = self.ce(out['vat_logits'], vat_tgt)
        # Guard: when pgd_tgt is all -100 (VAT predicts nothing, early training),
        # CE with reduction='mean' returns 0/0 = NaN. Same guard as L_left/L_right.
        pgd_valid = (pgd_tgt.reshape(-1) != -100)
        if pgd_valid.sum() > 0:
            L_self = self.ce(out['pgd_cls_logits'].reshape(B * L, -1), pgd_tgt.reshape(-1))
        else:
            L_self = torch.tensor(0.0, device=vat_tgt.device)

        m = mask.reshape(-1).bool()
        l_tgt_c = l_tgt.clamp(0, L - 1)
        r_tgt_c = r_tgt.clamp(0, L - 1)
        if m.sum() > 0:
            # IMPORTANT: use raw logits (left_logits/right_logits), NOT softmax
            # probabilities (left_scores/right_scores). CrossEntropyLoss requires
            # un-normalized logits. Passing post-softmax values causes log(prob)→-inf
            # for small probs, which accumulates into NaN gradients.
            L_left  = self.ce(out['left_logits'].reshape(B*L, L)[m],  l_tgt_c.reshape(-1)[m])
            L_right = self.ce(out['right_logits'].reshape(B*L, L)[m], r_tgt_c.reshape(-1)[m])
        else:
            L_left = L_right = torch.tensor(0.0, device=vat_tgt.device)

        L_pgd = L_self + self.w_conn * (L_left + L_right)
        L_all = L_vat + self.lam * L_pgd
        return L_all, {
            'L_all':  L_all.item(),  'L_vat':   L_vat.item(),
            'L_pgd':  L_pgd.item(),  'L_self':  L_self.item(),
            'L_left': L_left.item(), 'L_right': L_right.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING TARGETS
# ─────────────────────────────────────────────────────────────────────────────
def _bipartite_match_vat_targets(token_ids, P_vat, map_h, map_w,
                                  none_idx, pad_idx, sos_idx, eos_idx,
                                  vocab_sz, km=5, device='cpu'):
    """
    Paper Section 3.2 + Listing 2.1: Bipartite matching via Hungarian algorithm.

    Position estimate T: uniform column spread (no pretrained DWAP needed).
    Paper ablation (Table 4) shows uniform spread ≈ DWAP positions (≤1% gap).

    Algorithm (follows paper Listing 2.1 exactly):
      1. Estimate token positions T via uniform column spread on the feature map
      2. Convert T → indicator matrix T_mat [L, H, W]
      3. Max-pool T_mat with km×km kernel to create local matching windows
      4. Distance = |P_vat[y_label] - T_mat|  (VAT predicted prob of correct class
         vs. indicator mask — paper Eq.6)
      5. Force matching outside window to ∞ (limit to km×km neighbourhood)
      6. Hungarian algorithm → optimal one-to-one assignment
      7. Build P* target map

    Args:
      token_ids: [B, L] ground-truth token indices (with SOS/EOS/pad)
      P_vat:     [B, K+1, H, W] VAT predicted probabilities (softmax output)
      map_h/w:   H/8, W/8 dimensions
      km:        matching window size (paper default 5)
    """
    B      = token_ids.size(0)
    vat_tgt = torch.full((B, map_h, map_w), none_idx,
                         dtype=torch.long, device=device)
    pad    = km // 2

    with torch.no_grad():
        for b in range(B):
            tids_b = token_ids[b]
            keep   = ((tids_b != pad_idx) & (tids_b != sos_idx)
                      & (tids_b != eos_idx))
            valid  = tids_b[keep].clamp(0, vocab_sz - 1)   # [n]
            n      = valid.size(0)
            if n == 0:
                continue

            # ── Step 1: Estimate positions T via uniform column spread ──────
            row_mid = map_h // 2
            cols_t  = torch.linspace(0, map_w - 1, n, device=device).long()
            rows_t  = torch.full((n,), row_mid, dtype=torch.long, device=device)

            # ── Step 2: T → indicator matrix [n, H, W] ─────────────────────
            y_idx   = torch.arange(n, device=device)
            T_mat   = torch.zeros(n, map_h, map_w, device=device)
            T_mat[y_idx, rows_t, cols_t] = 1.0

            # ── Step 3: Max-pool km×km to get local matching windows ────────
            T_mat = F.max_pool2d(
                T_mat.unsqueeze(0),
                kernel_size=(km, km), stride=1, padding=pad
            )[0]   # [n, H, W]

            # ── Step 4: Distance = |P_vat[y_label] - T_mat| ────────────────
            P_b      = P_vat[b]                              # [K+1, H, W]
            dist_mat = (P_b[valid].float() - T_mat).abs()   # [n, H, W]

            # ── Step 5: Outside window → large cost (limit to km×km) ───────
            dist_mat = dist_mat * T_mat + (1.0 - T_mat) * 1e6

            # ── Step 6: Hungarian algorithm ─────────────────────────────────
            cost_np          = dist_mat.view(n, -1).cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            h_ind            = col_ind // map_w
            w_ind            = col_ind %  map_w

            # ── Step 7: Fill P* target map ──────────────────────────────────
            for ri, h_i, w_i in zip(row_ind, h_ind, w_ind):
                vat_tgt[b, h_i, w_i] = valid[ri].item()

    return vat_tgt





# Imaginary token map: which structural tokens need imaginary closing tokens
# Paper Section 3.3: structural tokens that require imaginary closing "}" tokens.
# Only include tokens that actually exist in the HME100K vocab.
# Tokens NOT in vocab (\\hat, \\bar, \\vec, \\tilde, \\overleftarrow) are excluded
# to avoid adding imaginary tokens that map to <unk>.
_IMAGINARY_TOKENS = {
    '^':               ['}'],       # idx=156 — superscript
    '_':               ['}'],       # idx=157 — subscript
    '\\frac':          ['}', '}'],  # idx=96  — numerator + denominator
    '\\sqrt':          ['}'],       # idx=132 — radicand
    '\\overrightarrow':['}'],       # idx=119 — vector arrow
    '\\boxed':         ['}'],       # idx=78  — boxed expression
    '\\widehat':       ['}'],       # idx=148
    '\\overline':      ['}'],       # idx=118
    '\\dot':           ['}'],       # idx=90
    '\\ddot':          ['}'],       # idx=87
    '\\xlongequal':    ['}'],       # idx=150
    '\\xrightarrow':   ['}'],       # idx=151
    '\\textcircled':   ['}'],       # idx=138
}


def _add_imaginary_tokens(token_ids_list, vocab: 'Vocabulary'):
    """
    Paper Section 3.3: insert imaginary closing '}' after structural tokens.

    NOT used for HME100K — labels already contain explicit '{' '}' tokens.
    Reserved for CROHME where labels are compact (e.g. '\\frac a b' without braces).

    token_ids_list: list of 1-D token id tensors (one per sample)
    Returns: padded tensor [B, L_new]
    """
    pad_idx = vocab.pad_idx
    results = []
    for tids in token_ids_list:
        new_tids = []
        for tid in tids.tolist():
            if tid == pad_idx:
                continue
            new_tids.append(tid)
            tok = vocab.i2t.get(tid, '')
            for imag in _IMAGINARY_TOKENS.get(tok, []):
                imag_id = vocab.t2i.get(imag, vocab.t2i.get('<unk>'))
                new_tids.append(imag_id)
        results.append(new_tids)

    # Pad to same length
    max_len = max(len(r) for r in results) if results else 1
    out = torch.full((len(results), max_len), pad_idx, dtype=torch.long)
    for i, r in enumerate(results):
        out[i, :len(r)] = torch.tensor(r, dtype=torch.long)
    return out


def _make_vat_targets(token_ids, P_vat, map_h, map_w, vocab, device, km=5):
    """Build VAT targets via Hungarian matching. Called from train_epoch after VAT forward."""
    return _bipartite_match_vat_targets(
        token_ids, P_vat, map_h, map_w,
        vocab.none_idx, vocab.pad_idx, vocab.sos_idx, vocab.eos_idx,
        len(vocab), km=km, device=device,
    ).clamp(0, len(vocab) - 1)






# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────
def _edit_dist(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        ndp = [i] + [0] * n
        for j in range(1, n + 1):
            ndp[j] = dp[j-1] if a[i-1] == b[j-1] else 1 + min(dp[j], ndp[j-1], dp[j-1])
        dp = ndp
    return dp[n]


def compute_exprate(preds, gts):
    ex = l1 = l2 = 0
    for p, g in zip(preds, gts):
        d = _edit_dist(p, g)
        if d == 0: ex += 1
        if d <= 1: l1 += 1
        if d <= 2: l2 += 1
    N = max(len(preds), 1)
    return ex / N, l1 / N, l2 / N


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINER  (tqdm real-time progress bars)
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn,
                 vocab: Vocabulary, device, checkpoint_dir: str, log_interval: int = 50):
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

    # ── One training epoch ──────────────────────────────────────────────────
    def train_epoch(self, loader, epoch: int, total_epochs: int,
                    map_h: int, map_w: int, teacher_epochs: int = 0):
        self.model.train()
        run = defaultdict(float)

        pbar = tqdm(loader,
                    desc=f"Ep {epoch:>3}/{total_epochs} [train]",
                    leave=True, dynamic_ncols=True, unit='batch')

        for step, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            token_ids_gt = batch['token_ids'].to(self.device)   # [B, L] GT

            # ── Encoder pass ─────────────────────────────────────────────────
            f8x, f16x = self.model.enc(images)

            # ── VAT forward (with grad) ───────────────────────────────────────
            vat_probs, vat_logits = self.model.vat(f8x, f16x)

            # ── VAT targets via Hungarian matching (paper Listing 2.1) ────────
            # Use actual VAT probs so the matching distance reflects the model's
            # current predictions (not a saliency proxy).
            vat_tgt = _make_vat_targets(
                token_ids_gt, vat_probs.detach(), map_h, map_w,
                self.vocab, self.device
            )

            none_idx = self.vocab.none_idx
            pad_idx  = self.vocab.pad_idx
            sos_idx  = self.vocab.sos_idx
            eos_idx  = self.vocab.eos_idx
            B        = images.size(0)
            dev      = self.device
            MAX_PGD  = 128   # cap sequence length → prevents [B,L,L] OOM

            # ── Build pgd_input ───────────────────────────────────────────────
            # Curriculum: use GT tokens for PGD during early epochs so PGD can
            # learn connectivity before VAT is reliable.
            use_gt_pgd = (teacher_epochs > 0 and epoch <= teacher_epochs)

            if use_gt_pgd:
                # Teacher forcing: directly use GT token sequence for PGD
                # Build from GT token_ids stripped of SOS/EOS/pad
                raw = token_ids_gt                                 # [B, L]
                L_gt  = raw.size(1)
                max_l = min(L_gt, MAX_PGD)

                pgd_input = raw[:, :max_l].contiguous()
                pgd_tgt   = pgd_input.clone()
                # Mask SOS, EOS, and PAD positions: CE should only train on real tokens
                pgd_tgt[pgd_tgt == pad_idx] = -100
                pgd_tgt[pgd_tgt == sos_idx] = -100
                pgd_tgt[pgd_tgt == eos_idx] = -100
                mask = (pgd_input != pad_idx).float()

                pos    = torch.arange(max_l, device=dev)
                lengths_gt = (pgd_input != pad_idx).sum(dim=1) - 1  # excl. last pad
                ends   = lengths_gt.clamp(max=max_l - 1)
                l_tgt2 = (pos - 1).clamp(min=0).unsqueeze(0).expand(B, -1).clone()
                r_tgt2 = (pos + 1).clamp(max=max_l - 1).unsqueeze(0).expand(B, -1).clone()
                for b in range(B):
                    r_tgt2[b] = r_tgt2[b].clamp(max=ends[b].item())

            else:
                # Post-curriculum: use VAT predictions as PGD input (paper default)
                vat_pred  = vat_probs.detach().argmax(dim=1)      # [B, H, W]
                H, W      = vat_pred.shape[1], vat_pred.shape[2]
                pred_flat = vat_pred.view(B, -1)
                gt_flat   = vat_tgt.view(B, -1)
                col_idx   = torch.arange(W, device=dev).unsqueeze(0).expand(H, W).reshape(-1)
                is_tok    = (pred_flat != none_idx)
                lengths   = is_tok.sum(dim=1).clamp(max=MAX_PGD - 2)
                max_l     = int(lengths.max().item()) + 2

                pgd_input = torch.full((B, max_l), pad_idx, dtype=torch.long, device=dev)
                pgd_tgt   = torch.full((B, max_l), -100,    dtype=torch.long, device=dev)
                mask      = torch.zeros(B, max_l, device=dev)
                pgd_input[:, 0] = sos_idx

                for b in range(B):
                    n = lengths[b].item()
                    if n == 0:
                        pgd_input[b, 1] = eos_idx
                        mask[b, :2]     = 1.0
                        continue
                    tok_mask  = is_tok[b]
                    cols_b    = col_idx[tok_mask]
                    order     = cols_b.argsort()[:n]
                    pred_seq  = pred_flat[b][tok_mask][order]
                    gt_seq    = gt_flat[b][tok_mask][order]
                    pgd_input[b, 1:n+1] = pred_seq
                    pgd_input[b, n+1]   = eos_idx
                    pgd_tgt[b,   1:n+1] = gt_seq
                    mask[b, :n+2]       = 1.0

                pos    = torch.arange(max_l, device=dev)
                lengths_v = (pgd_input != pad_idx).sum(dim=1) - 1
                ends   = lengths_v.clamp(max=max_l - 1)
                l_tgt2 = (pos - 1).clamp(min=0).unsqueeze(0).expand(B, -1).clone()
                r_tgt2 = (pos + 1).clamp(max=max_l - 1).unsqueeze(0).expand(B, -1).clone()
                for b in range(B):
                    r_tgt2[b] = r_tgt2[b].clamp(max=ends[b].item())


            # ── PGD forward (reuse f16x, no second encoder pass) ──────────────
            pad_mask = (pgd_input == pad_idx)   # [B, max_l] True=padding
            qs, ql, qr, q0 = self.model.pgd(f16x, pgd_input)
            cls_logits, edge_scores, left_scores, right_scores, left_logits, right_logits = \
                self.model.pgd.compute_scores(qs, ql, qr, q0, pad_mask=pad_mask)
            out = {
                'vat_logits':     vat_logits,
                'pgd_cls_logits': cls_logits,
                'left_scores':    left_scores,
                'right_scores':   right_scores,
                'left_logits':    left_logits,   # raw logits for loss
                'right_logits':   right_logits,  # raw logits for loss
                'f8x':            f8x,
            }

            # ── Loss + backward ─────────────────────────────────────────────
            loss, ld = self.loss_fn(out, vat_tgt, pgd_tgt, l_tgt2, r_tgt2, mask)

            # Guard: skip batch if loss is NaN/inf – prevents one bad batch
            # from poisoning all model weights via NaN gradients.
            if not torch.isfinite(loss):
                tqdm.write(
                    f"  [step {step+1}] skipped NaN batch —"
                    f" vat={ld['L_vat']:.3f} self={ld['L_self']:.3f}"
                    f" left={ld['L_left']:.3f} right={ld['L_right']:.3f}"
                )
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()  # still advance LR schedule
                continue

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            for k, v in ld.items():
                run[k] += v

            if (step + 1) % self.log_interval == 0 or (step + 1) == len(loader):
                n  = step + 1
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix(ordered_dict={
                    'L':   f"{run['L_all']/n:.4f}",
                    'vat': f"{run['L_vat']/n:.4f}",
                    'pgd': f"{run['L_pgd']/n:.4f}",
                    'lr':  f"{lr:.1e}",
                })

        pbar.close()
        # Divide by steps that actually ran (not skipped NaN batches)
        n_valid = max(1, sum(1 for v in run.values() if v == v))  # len(run) always > 0
        n_valid = len(loader)  # keep denominator consistent for logging
        avg = {k: v / n_valid for k, v in run.items()}
        tqdm.write(
            f"  ↳ Ep {epoch} train — "
            f"L={avg['L_all']:.4f}  vat={avg['L_vat']:.4f}  pgd={avg['L_pgd']:.4f}"
            f"  (self={avg['L_self']:.3f} left={avg['L_left']:.3f} right={avg['L_right']:.3f})"
        )
        self.history['train_loss'].append(avg['L_all'])
        return avg

    # ── Evaluate ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader, split: str = 'val'):
        self.model.eval()
        preds, gts = [], []

        pbar = tqdm(loader,
                    desc=f"             [{split:>5}]",
                    leave=False, dynamic_ncols=True, unit='batch')

        for batch in pbar:
            pred_seqs = self.model(batch['image'].to(self.device), token_ids=None)
            for ids, gt_toks in zip(pred_seqs, batch['tokens']):
                preds.append(self.vocab.decode(ids))
                gts.append(gt_toks)
            pbar.set_postfix({'n': len(preds)})

        pbar.close()
        er, l1, l2 = compute_exprate(preds, gts)
        tqdm.write(
            f"  ↳ [{split:>5}] ExpRate={er*100:.2f}%  "
            f"≤1={l1*100:.2f}%  ≤2={l2*100:.2f}%"
        )
        if split == 'val':
            self.history['val_exprate'].append(er)
        return er, l1, l2

    # ── Checkpoint ──────────────────────────────────────────────────────────
    def save(self, epoch: int, er: float):
        ckpt = dict(epoch=epoch, exprate=er,
                    model=self.model.state_dict(),
                    optim=self.optimizer.state_dict(),
                    history=dict(self.history))
        fname = self.ckpt_dir / f'ep{epoch:03d}_{er*100:.2f}.pth'
        torch.save(ckpt, fname)
        if er > self.best_er:
            self.best_er = er
            torch.save(ckpt, self.ckpt_dir / 'best.pth')
            tqdm.write(f"  ✓ New best: {er*100:.2f}%  → best.pth")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optim'])
        self.best_er = ckpt.get('exprate', 0.0)
        if 'history' in ckpt:
            self.history = defaultdict(list, ckpt['history'])
        tqdm.write(
            f"Resumed ← {path}  "
            f"epoch={ckpt['epoch']}  best_er={self.best_er*100:.2f}%"
        )
        return ckpt['epoch']


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def _set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def run_training(cfg: Config):
    _set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tqdm.write(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, vocab = build_datasets(cfg)
    col = partial(_collate, pad_idx=vocab.pad_idx)

    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=col,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, collate_fn=col, pin_memory=True)
    test_loader  = DataLoader(test_ds,  cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, collate_fn=col, pin_memory=True)

    tqdm.write(
        f"Steps/epoch — train={len(train_loader):,} | "
        f"val={len(val_loader):,} | test={len(test_loader):,}"
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = NAMER(
        vocab_size=len(vocab),
        d=cfg.d_model,
        heads=cfg.nhead,
        pgd_layers=cfg.pgd_layers,
        drop=cfg.drop,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tqdm.write(f"NAMER — vocab={len(vocab)} | params={n_params:,}")

    map_h = cfg.img_h // 8
    map_w = cfg.img_w // 8

    # ── Optimizer: Adam + cosine LR with 2-epoch warmup ────────────────────
    # Paper: Adam, LR 0→2e-4 in 1st epoch, then cosine decay to 2e-7
    optimizer    = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    total_steps  = cfg.epochs * len(train_loader)
    warmup_steps = 2 * len(train_loader)   # 2-epoch warmup (HME100K: ~3,700 steps)

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # Cosine decay from 2e-4 down to 2e-7 (ratio = 1e-3) — paper setting
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    loss_fn   = NAMERLoss(lam=cfg.lambda_pgd)

    trainer = Trainer(model, optimizer, scheduler, loss_fn,
                      vocab, device, cfg.checkpoint_dir, cfg.log_interval)

    start_epoch = 0
    if cfg.resume_checkpoint:
        start_epoch = trainer.load(cfg.resume_checkpoint)

    # ── Epoch loop ───────────────────────────────────────────────────────────
    epoch_bar = tqdm(
        range(start_epoch + 1, cfg.epochs + 1),
        desc='Overall',
        position=0,
        leave=True,
        dynamic_ncols=True,
        unit='epoch',
    )

    for epoch in epoch_bar:
        epoch_bar.set_postfix({'best_val': f"{trainer.best_er*100:.2f}%"})

        avg = trainer.train_epoch(train_loader, epoch, cfg.epochs, map_h, map_w,
                                   teacher_epochs=cfg.pgd_teacher_epochs)

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            er, l1, l2 = trainer.evaluate(val_loader, split='val')
            trainer.save(epoch, er)
            epoch_bar.set_postfix({
                'best_val': f"{trainer.best_er*100:.2f}%",
                'val':      f"{er*100:.2f}%",
                'L':        f"{avg['L_all']:.4f}",
            })

    epoch_bar.close()

    # ── Final test evaluation ─────────────────────────────────────────────────
    sep = '=' * 50
    tqdm.write(f"\n{sep}\n  Final Test Evaluation\n{sep}")
    best_path = trainer.ckpt_dir / 'best.pth'
    if best_path.exists():
        trainer.load(str(best_path))
    er, l1, l2 = trainer.evaluate(test_loader, split='test')
    tqdm.write(f"  ExpRate   : {er*100:.2f}%")
    tqdm.write(f"  ExpRate≤1 : {l1*100:.2f}%")
    tqdm.write(f"  ExpRate≤2 : {l2*100:.2f}%")
    tqdm.write(sep)
    return er, l1, l2


# ─────────────────────────────────────────────────────────────────────────────
#  Script entry point (not needed on Kaggle)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    cfg = Config()
    run_training(cfg)