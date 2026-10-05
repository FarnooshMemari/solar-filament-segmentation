"""Exponential moving average (EMA) of model weights.

The averaged weights change slowly and usually predict a little better than the last
weights of training, at no extra training cost.
"""
from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # warm up: average over fewer steps at the start, when the weights move fast
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        current = model.state_dict()
        for name, value in self.model.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(d).add_(current[name].detach(), alpha=1 - d)
            else:
                value.copy_(current[name])
