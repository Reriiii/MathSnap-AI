"""
DenseNet encoder for HMER.

Multi-scale dense feature extractor adapted for single-channel
handwritten math expression images. Outputs a flattened sequence
of 2D features with positional encoding for the Transformer decoder.

Key design choice — number of transition layers:
  With img_height=128, img_max_width=512 and the standard initial
  conv (stride=2) + maxpool (stride=2), the feature map entering the
  dense blocks is already 32×128. Each transition halves both dims:
    3 transitions → 4×16 = 64 tokens  (too sparse for 200-token exprs)
    2 transitions → 8×32 = 256 tokens (recommended)
  Use num_transitions=2 to preserve spatial resolution.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Module):
    """Single dense layer: BN -> ReLU -> 1x1 Conv -> BN -> ReLU -> 3x3 Conv.

    BatchNorm momentum is set to 0.05 (half the PyTorch default of 0.1).
    Lower momentum makes running_mean/var track distribution shifts more
    smoothly — important when training data distribution changes gradually
    due to curriculum augmentation ramp-up.
    """
    _BN_MOMENTUM = 0.05

    def __init__(self, in_channels: int, growth_rate: int, bn_size: int, drop_rate: float):
        super().__init__()
        inter_channels = bn_size * growth_rate

        self.norm1 = nn.BatchNorm2d(in_channels, momentum=self._BN_MOMENTUM)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_channels, momentum=self._BN_MOMENTUM)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    """A block of multiple dense layers."""

    def __init__(self, num_layers: int, in_channels: int, growth_rate: int,
                 bn_size: int, drop_rate: float):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                in_channels + i * growth_rate,
                growth_rate, bn_size, drop_rate
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    """Transition layer: BN -> 1x1 Conv -> AvgPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels, momentum=_DenseLayer._BN_MOMENTUM)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.norm(x), inplace=True))
        x = self.pool(x)
        return x


class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for feature maps.

    The d_model channels are split evenly between height and width
    encodings, then concatenated. Each half uses the standard
    sin/cos formulation independently.
    """

    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 256, dropout: float = 0.1):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for 2D positional encoding"
        self.dropout = nn.Dropout(p=dropout)
        half = d_model // 2

        pe_h = torch.zeros(max_h, half)
        pe_w = torch.zeros(max_w, half)

        pos_h = torch.arange(0, max_h).unsqueeze(1).float()
        pos_w = torch.arange(0, max_w).unsqueeze(1).float()

        # div_term spans the full half dimension (step=2 for sin/cos pairs)
        div_term = torch.exp(
            torch.arange(0, half, 2).float() * -(math.log(10000.0) / half)
        )

        pe_h[:, 0::2] = torch.sin(pos_h * div_term)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term[:half // 2 + half % 2])
        pe_w[:, 0::2] = torch.sin(pos_w * div_term)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term[:half // 2 + half % 2])

        self.register_buffer('pe_h', pe_h)  # [max_h, half]
        self.register_buffer('pe_w', pe_w)  # [max_w, half]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            x with positional encoding added: [B, C, H, W]
        """
        B, C, H, W = x.shape
        # [H, W, half] — broadcast height enc across all columns
        pos_h = self.pe_h[:H, :].unsqueeze(1).expand(-1, W, -1)
        # [H, W, half] — broadcast width enc across all rows
        pos_w = self.pe_w[:W, :].unsqueeze(0).expand(H, -1, -1)
        # Concatenate → [H, W, C], permute → [C, H, W], broadcast over batch
        pos = torch.cat([pos_h, pos_w], dim=-1).permute(2, 0, 1)
        return self.dropout(x + pos.unsqueeze(0))


class DenseNetEncoder(nn.Module):
    """
    DenseNet-based encoder for HMER.

    Takes grayscale images and produces a sequence of feature vectors
    with 2D positional encoding.

    The num_transitions parameter controls how many transition (halving)
    layers are inserted between dense blocks. Fewer transitions preserve
    spatial resolution and produce more encoder tokens for the decoder
    to attend to:
        num_transitions=3 (default DenseNet) → 64 tokens  for 128×512 input
        num_transitions=2 (recommended)      → 256 tokens for 128×512 input
    """

    def __init__(
        self,
        in_channels: int = 1,
        growth_rate: int = 24,
        block_config: tuple = (6, 12, 16, 8),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.2,
        compression: float = 0.8,
        d_model: int = 256,
        num_transitions: int = 2,
        pos_dropout: float = 0.1,
    ):
        super().__init__()

        # Initial convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7,
                                stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features, momentum=_DenseLayer._BN_MOMENTUM)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Dense blocks and transitions
        # Transitions are inserted after the first num_transitions blocks only.
        # This preserves spatial resolution in later blocks.
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate

            if i < num_transitions:  # Only insert transition after first num_transitions blocks
                out_features = int(num_features * compression)
                trans = _Transition(num_features, out_features)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = out_features

        # Final batch norm
        self.features.add_module('norm_final',
                                  nn.BatchNorm2d(num_features, momentum=_DenseLayer._BN_MOMENTUM))

        # Project to d_model
        self.projection = nn.Conv2d(num_features, d_model, kernel_size=1)

        # 2D positional encoding with dropout
        self.pos_encoding = PositionalEncoding2D(d_model, dropout=pos_dropout)

        self.d_model = d_model
        self._num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] grayscale image

        Returns:
            features: [B, S, d_model] where S = H' * W' (flattened feature map)
        """
        # Extract features
        features = self.features(x)  # [B, C, H', W']
        features = F.relu(features, inplace=True)

        # Project to d_model
        features = self.projection(features)  # [B, d_model, H', W']

        # Add 2D positional encoding
        features = self.pos_encoding(features)  # [B, d_model, H', W']

        # Flatten spatial dimensions to sequence
        B, C, H, W = features.shape
        features = features.view(B, C, H * W)  # [B, d_model, S]
        features = features.permute(0, 2, 1)    # [B, S, d_model]

        return features