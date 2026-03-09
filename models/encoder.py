import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Returns (F_8x, F_16x) at stride-8 and stride-16 respectively.
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
        self.ch_8x = ch
        self.t1 = _Trans(ch, ch // 2);                        ch //= 2
        self.b2 = _DenseBlock(blocks[2], ch, gr, bns, drop); ch += blocks[2] * gr
        self.n2 = nn.BatchNorm2d(ch)
        self.ch_16x = ch

    def forward(self, x):
        x    = self.stem(x)
        x    = self.t0(self.b0(x))
        x    = self.b1(x);              f8x  = x
        x    = self.b2(self.t1(x));     f16x = F.relu(self.n2(x), True)
        return f8x, f16x


class PE2D(nn.Module):
    """2D sinusoidal positional encoding."""
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
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2), :x.size(3)]
