"""
DenseNet encoder for HMER.

Multi-scale dense feature extractor adapted for single-channel
handwritten math expression images. Outputs a flattened sequence
of 2D features with positional encoding for the Transformer decoder.

Key design choice — number of transition layers:
  With img_height=128, img_max_width=512 and the standard initial
  conv (stride=2) + maxpool (stride=2), the feature map entering the
  dense blocks is already 32x128. Each transition halves both dims:
    3 transitions -> 4x16 = 64 tokens  (too sparse for 200-token exprs)
    2 transitions -> 8x32 = 256 tokens (recommended)
  Use num_transitions=2 to preserve spatial resolution.

GroupNorm is used instead of BatchNorm to avoid distribution shift
issues during curriculum augmentation (BN running stats diverge when
aug_prob ramps up gradually, causing validation spikes).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple, List


def _get_num_groups(channels: int, target_groups: int = 32) -> int:
    """Pick the largest divisor of channels that is <= target_groups."""
    for g in range(target_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


class _DenseLayer(nn.Module):
    """Single dense layer: GN -> ReLU -> 1x1 Conv -> GN -> ReLU -> 3x3 Conv."""

    def __init__(self, in_channels: int, growth_rate: int, bn_size: int,
                 drop_rate: float, num_groups: int = 32):
        super().__init__()
        inter_channels = bn_size * growth_rate

        self.norm1 = nn.GroupNorm(_get_num_groups(in_channels, num_groups), in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(_get_num_groups(inter_channels, num_groups), inter_channels)
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
                 bn_size: int, drop_rate: float, num_groups: int = 32):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                in_channels + i * growth_rate,
                growth_rate, bn_size, drop_rate, num_groups
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    """Transition layer: GN -> 1x1 Conv -> AvgPool."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(_get_num_groups(in_channels, num_groups), in_channels)
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
        # [H, W, half] -- broadcast height enc across all columns
        pos_h = self.pe_h[:H, :].unsqueeze(1).expand(-1, W, -1)
        # [H, W, half] -- broadcast width enc across all rows
        pos_w = self.pe_w[:W, :].unsqueeze(0).expand(H, -1, -1)
        # Concatenate -> [H, W, C], permute -> [C, H, W], broadcast over batch
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
        num_transitions=3 (default DenseNet) -> 64 tokens  for 128x512 input
        num_transitions=2 (recommended)      -> 256 tokens for 128x512 input

    Also exposes intermediate feature maps from each dense block for use
    by the multi-scale counting module (CAN, ECCV 2022).
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
        num_groups: int = 32,
    ):
        super().__init__()
        self.num_blocks = len(block_config)
        self.num_transitions = num_transitions

        # Initial convolution
        self.init_conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7,
                                stride=2, padding=3, bias=False)),
            ('norm0', nn.GroupNorm(_get_num_groups(num_init_features, num_groups),
                                   num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Dense blocks and transitions (stored separately to capture intermediates)
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self._block_out_channels = []

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                num_groups=num_groups,
            )
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            self._block_out_channels.append(num_features)

            if i < num_transitions:
                out_features = int(num_features * compression)
                # Round to nearest multiple of num_groups for clean GroupNorm
                out_features = max(num_groups, (out_features // num_groups) * num_groups)
                trans = _Transition(num_features, out_features, num_groups)
                self.transitions.append(trans)
                num_features = out_features

        # Final norm
        self.norm_final = nn.GroupNorm(_get_num_groups(num_features, num_groups),
                                       num_features)

        # Project to d_model
        self.projection = nn.Conv2d(num_features, d_model, kernel_size=1)

        # 2D positional encoding with dropout
        self.pos_encoding = PositionalEncoding2D(d_model, dropout=pos_dropout)

        self.d_model = d_model
        self._num_features = num_features

    @property
    def block_out_channels(self) -> List[int]:
        """Channel counts after each dense block (before transition)."""
        return list(self._block_out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, List[torch.Tensor]]:
        """
        Args:
            x: [B, 1, H, W] grayscale image

        Returns:
            features:       [B, S, d_model] where S = feat_h * feat_w
            feat_h:         encoder feature map height
            feat_w:         encoder feature map width
            intermediates:  list of [B, C_i, H_i, W_i] feature maps from each dense block
        """
        x = self.init_conv(x)

        intermediates = []
        trans_idx = 0
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            intermediates.append(x)
            if i < self.num_transitions:
                x = self.transitions[trans_idx](x)
                trans_idx += 1

        x = self.norm_final(x)
        x = F.relu(x, inplace=True)
        x = self.projection(x)        # [B, d_model, H', W']
        x = self.pos_encoding(x)      # [B, d_model, H', W']

        B, C, H, W = x.shape
        features = x.view(B, C, H * W).permute(0, 2, 1)  # [B, S, d_model]
        return features, H, W, intermediates
