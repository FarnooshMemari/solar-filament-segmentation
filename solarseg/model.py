"""A compact U-Net written from scratch (no pretrained weights or external downloads)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class UNet(nn.Module):
    """U-Net with ``depth`` downsampling steps; widths base, 2*base, 4*base, ..."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        widths = [base * 2**i for i in range(depth + 1)]
        self.encoders = nn.ModuleList(
            [ConvBlock(in_ch, widths[0])]
            + [ConvBlock(widths[i - 1], widths[i], dropout if i == depth else 0.0) for i in range(1, depth + 1)]
        )
        self.decoders = nn.ModuleList(
            [ConvBlock(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(depth))]
        )
        self.head = nn.Conv2d(widths[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for dec in self.decoders:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


def build_model(base: int = 32, depth: int = 4) -> UNet:
    return UNet(in_ch=1, out_ch=1, base=base, depth=depth)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
