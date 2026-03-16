"""
CROHME dataset for ICAL HMER.

Loads image-LaTeX pairs from CSV files. Uses ICAL-style:
- No pre-padding: images are resized, collate_fn pads per-batch
- ScaleAugmentation(0.7, 1.4) for training augmentation
- SizeGroupedBatchSampler for efficient batching
"""

import csv
import numpy as np
from typing import List, Dict

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import torchvision.transforms as T

from data.vocab import Vocab


class ScaleAugmentation:
    """ICAL-style scale augmentation."""
    def __init__(self, lo: float = 0.7, hi: float = 1.4):
        self.lo = lo
        self.hi = hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        k = np.random.uniform(self.lo, self.hi)
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        return img


class ScaleToLimitRange:
    """ICAL-style: ensure image fits within (h_lo..h_hi, w_lo..w_hi)."""
    def __init__(self, w_lo: int = 16, w_hi: int = 1024, h_lo: int = 16, h_hi: int = 256):
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.h_lo = h_lo
        self.h_hi = h_hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        scale_r = min(self.h_hi / h, self.w_hi / w)
        if scale_r < 1.0:
            img = cv2.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR)
            return img

        scale_r = max(self.h_lo / h, self.w_lo / w)
        if scale_r > 1.0:
            img = cv2.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR)
            return img

        return img


class SizeGroupedBatchSampler(Sampler):
    """
    Pre-grouped batch sampler (ICAL-style).
    Sorts samples by image area, groups similarly-sized images.
    Batch ORDER is shuffled each epoch.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        max_pixels: int = 320000,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_pixels = max_pixels
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batches = self._build_batches()

    def _get_image_area(self, idx: int) -> int:
        entry = self.dataset.entries[idx]
        try:
            with Image.open(entry['image_path']) as img:
                w, h = img.size
            return h * w
        except Exception:
            return 256 * 1024  # fallback

    def _build_batches(self):
        n = len(self.dataset)
        areas = [(i, self._get_image_area(i)) for i in range(n)]
        areas.sort(key=lambda x: x[1])

        batches = []
        current_batch = []
        biggest_area = 0

        for idx, area in areas:
            if area > biggest_area:
                biggest_area = area
            batch_cost = biggest_area * (len(current_batch) + 1)

            if batch_cost > self.max_pixels or len(current_batch) >= self.batch_size:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                biggest_area = area
            else:
                current_batch.append(idx)

        if current_batch and not self.drop_last:
            batches.append(current_batch)

        return batches

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(len(self.batches)).tolist()
        else:
            order = list(range(len(self.batches)))
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


class CROHMEDataset(Dataset):
    """
    CROHME dataset for ICAL HMER.
    Returns variable-size images (no pre-padding) and raw label indices (no SOS/EOS).
    """

    def __init__(
        self,
        csv_path: str,
        vocab: Vocab,
        max_seq_len: int = 200,
        augment: bool = False,
        scale_aug: bool = True,
        scale_lo: float = 0.7,
        scale_hi: float = 1.4,
        h_lo: int = 16,
        h_hi: int = 256,
        w_lo: int = 16,
        w_hi: int = 1024,
    ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.augment = augment

        # Build transforms (ICAL-style)
        trans_list = []
        if augment and scale_aug:
            trans_list.append(ScaleAugmentation(scale_lo, scale_hi))
        trans_list.append(ScaleToLimitRange(w_lo=w_lo, w_hi=w_hi, h_lo=h_lo, h_hi=h_hi))
        trans_list.append(T.ToTensor())  # [0, 1] range
        self.transform = T.Compose(trans_list)

        # Load data entries
        self.entries = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append({
                    'image_path': row['image_path'],
                    'latex': row['latex']
                })

        print(f"Loaded {len(self.entries)} samples from {csv_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> Dict:
        entry = self.entries[idx]

        # Load image as grayscale numpy array
        img = cv2.imread(entry['image_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback: try PIL
            pil_img = Image.open(entry['image_path']).convert('L')
            img = np.array(pil_img)

        # Apply transforms (scale aug + limit range + to_tensor)
        img_tensor = self.transform(img)  # [1, H, W]

        # Encode label — raw indices WITHOUT SOS/EOS
        # (ICAL's plicit_tgt_out adds SOS/EOS in training loop)
        raw_indices = self.vocab.encode(entry['latex'], add_sos=False, add_eos=False)

        # Truncate if needed
        if len(raw_indices) > self.max_seq_len:
            raw_indices = raw_indices[:self.max_seq_len]

        return {
            'image': img_tensor,       # [1, H, W] variable size
            'indices': raw_indices,     # List[int] raw token indices
            'latex': entry['latex'],
            'image_path': entry['image_path'],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    ICAL-style collate: dynamic padding to max size in batch.
    Returns images [B, 1, max_H, max_W], mask [B, max_H, max_W], indices List[List[int]].
    """
    images_x = [item['image'] for item in batch]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    # Pad images (ICAL: zero-padded)
    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    for idx, s_x in enumerate(images_x):
        x[idx, :, :heights_x[idx], :widths_x[idx]] = s_x
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 0

    indices = [item['indices'] for item in batch]

    return {
        'image': x,                # [B, 1, H, W]
        'padding_mask': x_mask,    # [B, H, W] True=padding
        'indices': indices,        # List[List[int]] raw token indices
        'latex': [item['latex'] for item in batch],
        'image_path': [item['image_path'] for item in batch],
    }


def get_dataloader(
    split: str,
    vocab: Vocab,
    config=None,
    shuffle: bool = None,
) -> DataLoader:
    """Create a DataLoader for a given split."""
    from config import Config

    if config is None:
        config = Config()

    csv_path = f"{config.data.processed_dir}/{split}.csv"

    if shuffle is None:
        shuffle = (split == 'train')

    is_train = (split == 'train')

    dataset = CROHMEDataset(
        csv_path=csv_path,
        vocab=vocab,
        max_seq_len=config.data.max_seq_len,
        augment=(is_train and config.data.augment),
        scale_aug=config.data.scale_aug,
        scale_lo=config.data.scale_lo,
        scale_hi=config.data.scale_hi,
        h_lo=config.data.h_lo,
        h_hi=config.data.h_hi,
        w_lo=config.data.w_lo,
        w_hi=config.data.w_hi,
    )

    # Use pre-grouped batching for training
    if is_train:
        batch_sampler = SizeGroupedBatchSampler(
            dataset,
            batch_size=config.data.batch_size,
            max_pixels=config.data.max_batch_pixels,
            shuffle=shuffle,
            drop_last=True,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=collate_fn,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=shuffle,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )
