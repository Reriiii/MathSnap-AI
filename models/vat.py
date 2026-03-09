import torch
import torch.nn as nn
import torch.nn.functional as F


class VAT(nn.Module):
    """
    Visual Aware Tokenizer — paper Section 3.2.

    Predicts K+1 classes (K vocab tokens + 1 background ∅) at every
    position of the H/8 × W/8 feature map, in parallel (non-autoregressive).
    """
    def __init__(self, ch_8x: int, ch_16x: int, d: int, num_cls: int):
        super().__init__()
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
        logits = self.head(fm)           # [B, K+1, H/8, W/8]
        probs  = F.softmax(logits, dim=1)
        return probs, logits
