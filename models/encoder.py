"""
DenseNet encoder for HMER.

Multi-scale dense feature extractor adapted for single-channel
handwritten math expression images. Outputs a flattened sequence
of 2D features with positional encoding for the Transformer decoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Module):
    """Single dense layer: BN -> ReLU -> 1x1 Conv -> BN -> ReLU -> 3x3 Conv."""

    def __init__(self, in_channels: int, growth_rate: int, bn_size: int, drop_rate: float):
        super().__init__()
        inter_channels = bn_size * growth_rate

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_channels)
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
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.norm(x), inplace=True))
        x = self.pool(x)
        return x


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for feature maps."""

    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 256):
        super().__init__()
        self.d_model = d_model

        # Create position encodings
        pe_h = torch.zeros(max_h, d_model // 2)
        pe_w = torch.zeros(max_w, d_model // 2)

        pos_h = torch.arange(0, max_h).unsqueeze(1).float()
        pos_w = torch.arange(0, max_w).unsqueeze(1).float()

        div_term_h = torch.exp(
            torch.arange(0, d_model // 2, 2).float() * -(math.log(10000.0) / (d_model // 2))
        )
        div_term_w = torch.exp(
            torch.arange(0, d_model // 2, 2).float() * -(math.log(10000.0) / (d_model // 2))
        )

        pe_h[:, 0::2] = torch.sin(pos_h * div_term_h)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term_h)
        pe_w[:, 0::2] = torch.sin(pos_w * div_term_w)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term_w)

        self.register_buffer('pe_h', pe_h)
        self.register_buffer('pe_w', pe_w)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            x with positional encoding added: [B, C, H, W]
        """
        B, C, H, W = x.shape
        # Expand height encoding to [H, W, C//2]
        pos_h = self.pe_h[:H, :].unsqueeze(1).expand(-1, W, -1)
        # Expand width encoding to [H, W, C//2]
        pos_w = self.pe_w[:W, :].unsqueeze(0).expand(H, -1, -1)
        # Concatenate and permute to [C, H, W]
        pos = torch.cat([pos_h, pos_w], dim=-1).permute(2, 0, 1)
        return x + pos.unsqueeze(0)


class DenseNetEncoder(nn.Module):
    """
    DenseNet-based encoder for HMER.

    Takes grayscale images and produces a sequence of feature vectors
    with 2D positional encoding.
    """

    def __init__(
        self,
        in_channels: int = 1,
        growth_rate: int = 24,
        block_config: tuple = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.2,
        compression: float = 0.5,
        d_model: int = 256,
    ):
        super().__init__()

        # Initial convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7,
                                stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Dense blocks and transitions
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

            if i != len(block_config) - 1:
                out_features = int(num_features * compression)
                trans = _Transition(num_features, out_features)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = out_features

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Project to d_model
        self.projection = nn.Conv2d(num_features, d_model, kernel_size=1)

        # 2D positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model)

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
