"""
CROHME dataset with augmentation for HMER.

Loads image-LaTeX pairs from preprocessed CSV files.
Applies paper-simulation augmentations for training.
"""

import csv
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from data.vocab import Vocab


class CROHMEDataset(Dataset):
    """
    CROHME dataset for handwritten math expression recognition.

    Loads grayscale images and their LaTeX label sequences.
    Applies augmentation during training to simulate real paper conditions.
    """

    def __init__(
        self,
        csv_path: str,
        vocab: Vocab,
        img_height: int = 128,
        img_max_width: int = 512,
        max_seq_len: int = 200,
        augment: bool = False,
        aug_config: dict = None,
    ):
        """
        Args:
            csv_path: path to processed CSV file with (image_path, latex) columns
            vocab: Vocab instance for encoding labels
            img_height: target image height
            img_max_width: maximum image width (pad/crop to this)
            max_seq_len: maximum label sequence length
            augment: whether to apply augmentation
            aug_config: augmentation parameters dict
        """
        self.vocab = vocab
        self.img_height = img_height
        self.img_max_width = img_max_width
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.aug_config = aug_config or {}

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

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]

        # Load image
        img = Image.open(entry['image_path']).convert('L')  # grayscale

        # Apply augmentation
        if self.augment:
            img = self._augment(img)

        # Resize and pad
        img = self._resize_and_pad(img)

        # Convert to tensor and normalize
        img_tensor = TF.to_tensor(img)  # [1, H, W], values in [0, 1]
        img_tensor = TF.normalize(img_tensor, mean=[0.5], std=[0.5])  # [-1, 1]

        # Encode label
        label_indices = self.vocab.encode(entry['latex'], add_sos=True, add_eos=True)

        # Truncate if needed
        if len(label_indices) > self.max_seq_len:
            label_indices = label_indices[:self.max_seq_len - 1] + [self.vocab.eos_idx]

        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        return {
            'image': img_tensor,
            'target': label_tensor,
            'latex': entry['latex'],
            'image_path': entry['image_path']
        }

    def _resize_and_pad(self, img: Image.Image) -> Image.Image:
        """Resize image to target height maintaining aspect ratio, then pad width."""
        w, h = img.size
        # Calculate new width maintaining aspect ratio
        new_h = self.img_height
        new_w = int(w * (new_h / h))

        # Cap width
        if new_w > self.img_max_width:
            new_w = self.img_max_width

        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Pad to max width (right padding with white)
        if new_w < self.img_max_width:
            padded = Image.new('L', (self.img_max_width, new_h), 255)
            padded.paste(img, (0, 0))
            img = padded

        return img

    def _augment(self, img: Image.Image) -> Image.Image:
        """
        Apply augmentation to simulate handwritten expressions on paper.

        Includes: rotation, affine, perspective, noise, erosion/dilation,
        brightness/contrast variation, and paper texture simulation.

        global_prob in aug_config acts as a curriculum gate: during early
        training it is set below 1.0 so augmentations are applied less
        frequently, letting the model first learn from cleaner images.
        """
        cfg = self.aug_config

        # Curriculum gate — scales all per-augmentation probabilities uniformly.
        # Set to 1.0 (full augmentation) once the model has warmed up.
        global_prob = cfg.get('global_prob', 1.0)

        def should_apply(base_prob: float) -> bool:
            return random.random() < base_prob * global_prob

        # 1. Random rotation
        rotation_range = cfg.get('rotation_range', 5.0)
        if should_apply(0.5):
            angle = random.uniform(-rotation_range, rotation_range)
            img = img.rotate(angle, fillcolor=255, expand=False)

        # 2. Random affine (scale + shear)
        scale_range = cfg.get('scale_range', (0.9, 1.1))
        shear_range = cfg.get('shear_range', 0.1)
        if should_apply(0.5):
            scale = random.uniform(*scale_range)
            shear_x = random.uniform(-shear_range, shear_range)
            shear_y = random.uniform(-shear_range, shear_range)
            w, h = img.size
            img = TF.affine(
                img,
                angle=0,
                translate=(0, 0),
                scale=scale,
                shear=(shear_x * 180, shear_y * 180),
                fill=255
            )

        # 3. Random perspective distortion
        if should_apply(0.3):
            img = TF.to_tensor(img)
            img = T.RandomPerspective(distortion_scale=0.1, p=1.0, fill=1.0)(img)
            img = TF.to_pil_image(img)

        # 4. Brightness and contrast jitter
        brightness_range = cfg.get('brightness_range', (0.7, 1.3))
        contrast_range = cfg.get('contrast_range', (0.7, 1.3))
        if should_apply(0.5):
            brightness_factor = random.uniform(*brightness_range)
            img = TF.adjust_brightness(img, brightness_factor)
        if should_apply(0.5):
            contrast_factor = random.uniform(*contrast_range)
            img = TF.adjust_contrast(img, contrast_factor)

        # 5. Erosion/dilation (stroke thickness variation)
        erosion_dilation_prob = cfg.get('erosion_dilation_prob', 0.15)
        kernel_size = cfg.get('erosion_dilation_kernel', 2)
        if should_apply(erosion_dilation_prob):
            if random.random() < 0.5:
                # Dilation (thicker strokes) - using MinFilter
                img = img.filter(ImageFilter.MinFilter(kernel_size + 1))
            else:
                # Erosion (thinner strokes) - using MaxFilter
                img = img.filter(ImageFilter.MaxFilter(kernel_size + 1))

        # 6. Gaussian noise
        noise_std = cfg.get('noise_std', 0.02)
        if should_apply(0.4):
            img_np = np.array(img, dtype=np.float32) / 255.0
            noise = np.random.normal(0, noise_std, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 1)
            img = Image.fromarray((img_np * 255).astype(np.uint8))

        # 7. Gaussian blur (simulate slight paper texture / scan blur)
        if should_apply(0.2):
            radius = random.uniform(0.3, 1.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # 8. Random elastic deformation
        elastic_alpha = cfg.get('elastic_alpha', 8.0)
        elastic_sigma = cfg.get('elastic_sigma', 6.0)
        if should_apply(0.3):
            img = self._elastic_transform(img, elastic_alpha, elastic_sigma)

        return img

    def _elastic_transform(
        self, img: Image.Image, alpha: float, sigma: float
    ) -> Image.Image:
        """Apply elastic deformation to simulate natural handwriting variation."""
        from scipy.ndimage import gaussian_filter, map_coordinates

        img_np = np.array(img, dtype=np.float32)
        shape = img_np.shape

        # Generate random displacement field
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        ) * alpha
        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        ) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = [np.clip(y + dy, 0, shape[0] - 1), np.clip(x + dx, 0, shape[1] - 1)]

        distorted = map_coordinates(img_np, indices, order=1, mode='constant', cval=255)

        return Image.fromarray(distorted.astype(np.uint8))


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function to handle variable-length sequences.

    Pads targets to the same length within the batch.
    Images are already padded to img_max_width.
    """
    images = torch.stack([item['image'] for item in batch])

    # Get max target length in this batch
    max_len = max(item['target'].size(0) for item in batch)

    # Pad targets (with PAD=0 index)
    padded_targets = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        tgt = item['target']
        padded_targets[i, :tgt.size(0)] = tgt

    return {
        'image': images,
        'target': padded_targets,
        'latex': [item['latex'] for item in batch],
        'image_path': [item['image_path'] for item in batch]
    }


def get_dataloader(
    split: str,
    vocab: Vocab,
    config=None,
    shuffle: bool = None,
) -> DataLoader:
    """
    Create a DataLoader for a given split.

    Args:
        split: 'train', 'val', or 'test'
        vocab: Vocab instance
        config: Config instance (uses defaults if None)
        shuffle: override shuffle (default: True for train, False otherwise)
    """
    from config import Config

    if config is None:
        config = Config()

    csv_path = f"{config.data.processed_dir}/{split}.csv"

    if shuffle is None:
        shuffle = (split == 'train')

    aug_config = {
        'rotation_range': config.data.rotation_range,
        'scale_range': config.data.scale_range,
        'shear_range': config.data.shear_range,
        'brightness_range': config.data.brightness_range,
        'contrast_range': config.data.contrast_range,
        'noise_std': config.data.noise_std,
        'elastic_alpha': config.data.elastic_alpha,
        'elastic_sigma': config.data.elastic_sigma,
        'erosion_dilation_prob': config.data.erosion_dilation_prob,
        'erosion_dilation_kernel': config.data.erosion_dilation_kernel,
    }

    dataset = CROHMEDataset(
        csv_path=csv_path,
        vocab=vocab,
        img_height=config.data.img_height,
        img_max_width=config.data.img_max_width,
        max_seq_len=config.data.max_seq_len,
        augment=(split == 'train' and config.data.augment),
        aug_config=aug_config,
    )

    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_fn,
        drop_last=(split == 'train'),
    )