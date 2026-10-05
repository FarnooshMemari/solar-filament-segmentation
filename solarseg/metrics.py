"""Panoptic Quality (PQ), pooled over every (image, annotator) pair.

A predicted filament matches a ground-truth filament when their IoU is above 0.5.
Across all pairs:

    PQ = sum(IoU of matched pairs) / (TP + 0.5 * FP + 0.5 * FN)

This mirrors the host's description. Use the organizers' self-evaluation notebook on
Kaggle as the final word; this local version is for fast tuning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_utils


@dataclass
class PQStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0

    def __iadd__(self, other: "PQStats") -> "PQStats":
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.iou_sum += other.iou_sum
        return self

    @property
    def pq(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / denom if denom else 0.0

    @property
    def sq(self) -> float:
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / denom if denom else 0.0

    def as_dict(self) -> dict:
        return {
            "pq": round(self.pq, 5),
            "sq": round(self.sq, 5),
            "rq": round(self.rq, 5),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def match(pred_rles: list[dict], gt_rles: list[dict], threshold: float = 0.5) -> PQStats:
    """Match one image's predictions to one annotator's filaments."""
    if not pred_rles and not gt_rles:
        return PQStats()
    if not pred_rles:
        return PQStats(fn=len(gt_rles))
    if not gt_rles:
        return PQStats(fp=len(pred_rles))
    ious = np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=np.float64)
    ious = ious.reshape(len(pred_rles), len(gt_rles))
    # greedy one-to-one matching; with IoU > 0.5 and non-overlapping masks it is unique
    pairs = np.argwhere(ious > threshold)
    order = np.argsort(-ious[pairs[:, 0], pairs[:, 1]]) if len(pairs) else []
    used_p, used_g = set(), set()
    stats = PQStats()
    for idx in order:
        p, g = pairs[idx]
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        stats.tp += 1
        stats.iou_sum += float(ious[p, g])
    stats.fp = len(pred_rles) - stats.tp
    stats.fn = len(gt_rles) - stats.tp
    return stats
