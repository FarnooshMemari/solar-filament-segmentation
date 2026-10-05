"""Loss for thin, sparse structures: weighted BCE plus a batch-level soft Dice."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    denom = prob.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 3.0, dice_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        return bce + self.dice_weight * soft_dice_loss(logits, target)


@torch.no_grad()
def dice_score(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (prob >= threshold).float()
    tgt = (target >= 0.5).float()
    inter = (pred * tgt).sum()
    denom = pred.sum() + tgt.sum()
    return float((2 * inter / denom).item()) if denom > 0 else 1.0
