"""
CROHME dataset for CoMER HMER.

Supports two formats:
- CoMER: caption.txt (tab-separated: image_id<TAB>token1 token2 ...) + img/ folder with BMP files
- CSV: image_path,latex (legacy format)

Pipeline:
- Variable image sizes (preserve natural aspect ratio)
- ScaleAugmentation(0.7, 1.4) + ScaleToLimitRange for training
- SizeGroupedBatchSampler: group similar-sized images -> minimal padding
- collate_fn: dynamic per-batch padding
"""

import os
import csv
import random
import numpy as np
from typing import List, Dict, Optional

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image

from data.vocab import Vocab
from data import augmentations as aug


class ScaleAugmentation:
    """Scale augmentation: random resize by factor in [lo, hi]."""
    def __init__(self, lo: float = 0.7, hi: float = 1.4):
        self.lo = lo
        self.hi = hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        k = np.random.uniform(self.lo, self.hi)
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        return img


class ScaleToLimitRange:
    """Ensure image fits within (h_lo..h_hi, w_lo..w_hi)."""
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
    Batch sampler that groups similar-sized images.
    Sorts by image area, groups into batches respecting pixel budget.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        max_pixels: int = 2_000_000,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_pixels = max_pixels
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batches = self._build_batches()

    def _build_batches(self):
        n = len(self.dataset)
        areas = [(i, self.dataset.get_image_area(i)) for i in range(n)]
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
    CROHME dataset with variable image sizes.

    Supports:
    - CoMER format: caption_path + img_dir (BMP files, caption.txt)
    - CSV format: csv_path with image_path,latex columns
    """

    def __init__(
        self,
        vocab: Vocab,
        max_seq_len: int = 200,
        augment: bool = False,
        scale_aug: bool = True,
        # optional extra image-level augmentations (offline pixel/geometric)
        image_aug: bool = False,
        scale_lo: float = 0.7,
        scale_hi: float = 1.4,
        h_lo: int = 16,
        h_hi: int = 256,
        w_lo: int = 16,
        w_hi: int = 1024,
        # CoMER format
        caption_path: Optional[str] = None,
        img_dir: Optional[str] = None,
        # CSV format (legacy)
        csv_path: Optional[str] = None,
    ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.augment = augment

        self.scale_aug = None
        if augment and scale_aug:
            self.scale_aug = ScaleAugmentation(scale_lo, scale_hi)

        # Extra image augmentation toggle (does not change existing defaults)
        self.image_aug = image_aug and augment

        self.scale_limit = ScaleToLimitRange(w_lo, w_hi, h_lo, h_hi)

        # Load data entries
        self.entries = []

        if caption_path is not None and img_dir is not None:
            # CoMER format: caption.txt + img/ directory
            self._load_comer_format(caption_path, img_dir)
        elif csv_path is not None:
            # Legacy CSV format
            self._load_csv_format(csv_path)
        else:
            raise ValueError("Must provide either (caption_path, img_dir) or csv_path")

        self._image_areas = None

        # Pre-load all images into RAM (8834 BMP ~ 200MB in numpy)
        self._image_cache = {}
        self._preload_images()

        print(f"Loaded {len(self.entries)} samples ({len(self._image_cache)} cached)")

    def _preload_images(self):
        """Load all images into RAM to eliminate disk I/O during training."""
        print("  Pre-loading images into RAM...")
        for i, entry in enumerate(self.entries):
            try:
                img = cv2.imread(entry['image_path'], cv2.IMREAD_GRAYSCALE)
                if img is None:
                    pil_img = Image.open(entry['image_path']).convert('L')
                    img = np.array(pil_img)
                self._image_cache[i] = img
            except Exception:
                self._image_cache[i] = np.zeros((32, 32), dtype=np.uint8)

    def _load_comer_format(self, caption_path: str, img_dir: str):
        """Load CoMER format: tab-separated caption.txt + BMP images."""
        with open(caption_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) < 2:
                    continue
                image_id = parts[0]
                label = parts[1]

                # Find image file (try .bmp first, then others)
                img_path = None
                for ext in ['.bmp', '.png', '.jpg']:
                    candidate = os.path.join(img_dir, image_id + ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break

                if img_path is None:
                    continue

                self.entries.append({
                    'image_path': img_path,
                    'latex': label,
                })

    def _load_csv_format(self, csv_path: str):
        """Load legacy CSV format."""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append({
                    'image_path': row['image_path'],
                    'latex': row['latex'],
                })

    def _cache_image_areas(self):
        """Pre-compute image areas for batch grouping."""
        print("  Caching image sizes for batch grouping...")
        self._image_areas = []
        for entry in self.entries:
            try:
                with Image.open(entry['image_path']) as img:
                    w, h = img.size
                scale_r = min(self.scale_limit.h_hi / h, self.scale_limit.w_hi / w)
                if scale_r < 1.0:
                    h, w = int(h * scale_r), int(w * scale_r)
                else:
                    scale_r = max(self.scale_limit.h_lo / h, self.scale_limit.w_lo / w)
                    if scale_r > 1.0:
                        h, w = int(h * scale_r), int(w * scale_r)
                self._image_areas.append(h * w)
            except Exception:
                self._image_areas.append(128 * 512)
        return self._image_areas

    def get_image_area(self, idx: int) -> int:
        if self._image_areas is None:
            self._cache_image_areas()
        return self._image_areas[idx]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> Dict:
        entry = self.entries[idx]

        # Get image from RAM cache (no disk I/O)
        img = self._image_cache[idx].copy()  # copy for augmentation safety

        # Augmentation pipeline
        if self.scale_aug is not None:
            img = self.scale_aug(img)
        img = self.scale_limit(img)

        # Additional pixel/geometric augmentations (optional)
        if self.image_aug:
            # small affine (translation/scale/limited rotation)
            if np.random.rand() < 0.9:
                img = aug.random_affine_small(img)

            # elastic deformation sometimes
            if np.random.rand() < 0.5:
                img = aug.elastic_transform(img, alpha=np.random.uniform(20, 40), sigma=np.random.uniform(4, 8))

            # blur variants
            if np.random.rand() < 0.3:
                img = aug.gaussian_blur(img)
            if np.random.rand() < 0.15:
                img = aug.motion_blur(img, degree=random.randint(5, 12))

            # random erasing / cutout
            img = aug.random_erasing(img, p=0.5)

            # pairwise intensity transfer with another random sample
            if np.random.rand() < 0.2:
                j = np.random.randint(0, len(self.entries))
                other = self._image_cache[j]
                img = aug.pairwise_intensity_transfer(img, other)

            # Mixup / CutMix with small probability
            if np.random.rand() < 0.05:
                j = np.random.randint(0, len(self.entries))
                other = self._image_cache[j]
                if np.random.rand() < 0.5:
                    img = aug.mixup(img, other)
                else:
                    img = aug.cutmix(img, other)

        # To tensor [1, H, W] in [0, 1]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        # Encode label (raw indices, no SOS/EOS)
        raw_indices = self.vocab.encode(entry['latex'], add_sos=False, add_eos=False)
        if len(raw_indices) > self.max_seq_len:
            raw_indices = raw_indices[:self.max_seq_len]

        return {
            'image': img_tensor,
            'indices': raw_indices,
            'latex': entry['latex'],
            'image_path': entry['image_path'],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Dynamic padding collate function."""
    images_x = [item['image'] for item in batch]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    for idx, s_x in enumerate(images_x):
        x[idx, :, :heights_x[idx], :widths_x[idx]] = s_x
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 0

    indices = [item['indices'] for item in batch]

    return {
        'image': x,
        'padding_mask': x_mask,
        'indices': indices,
        'latex': [item['latex'] for item in batch],
        'image_path': [item['image_path'] for item in batch],
    }


def get_dataloader(
    split: str,
    vocab: Vocab,
    config=None,
) -> DataLoader:
    """Create DataLoader for a given split.

    split: 'train', '2014', '2016', '2019' (or legacy 'val', 'test')
    """
    from config import Config

    if config is None:
        config = Config()

    is_train = (split == 'train')

    # Determine data source
    comer_data_dir = config.data.comer_data_dir
    caption_path = os.path.join(comer_data_dir, split, 'caption.txt')
    img_dir = os.path.join(comer_data_dir, split, 'img')

    if os.path.exists(caption_path):
        # CoMER format
        dataset = CROHMEDataset(
            vocab=vocab,
            max_seq_len=config.data.max_seq_len,
                augment=(is_train and config.data.augment),
                scale_aug=config.data.scale_aug,
                image_aug=(is_train and config.data.image_aug),
            scale_lo=config.data.scale_lo,
            scale_hi=config.data.scale_hi,
            h_lo=config.data.h_lo,
            h_hi=config.data.h_hi,
            w_lo=config.data.w_lo,
            w_hi=config.data.w_hi,
            caption_path=caption_path,
            img_dir=img_dir,
        )
    else:
        # Legacy CSV format
        csv_path = os.path.join(config.data.processed_dir, f"{split}.csv")
        dataset = CROHMEDataset(
            vocab=vocab,
            max_seq_len=config.data.max_seq_len,
                augment=(is_train and config.data.augment),
                image_aug=(is_train and config.data.image_aug),
            csv_path=csv_path,
        )

    if is_train:
        batch_sampler = SizeGroupedBatchSampler(
            dataset,
            batch_size=config.data.batch_size,
            max_pixels=config.data.max_batch_pixels,
            shuffle=True,
            drop_last=True,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.num_workers > 0,
            collate_fn=collate_fn,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.num_workers > 0,
            collate_fn=collate_fn,
        )
