"""Panoptic Quality (PQ), pooled over every (image, annotator) pair.

Counting follows the host's self-evaluation notebook exactly:

* every annotator's labels for an image are scored separately against the same predictions,
  and all (image, annotator) pairs are pooled;
* a (ground truth, prediction) pair is a hit when their IoU is above 0.5;
* TP = number of hits, FP = predictions with no hit, FN = ground-truth filaments with no hit;

    PQ = sum(IoU of hits) / (TP + 0.5 * FP + 0.5 * FN)

With non-overlapping predictions (which this pipeline always produces) a filament can be hit
by at most one prediction, so this equals the usual one-to-one PQ.
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


def score_ious(ious: np.ndarray, threshold: float = 0.5) -> PQStats:
    """PQ counts for one (image, annotator) pair from an (n_pred, n_gt) IoU matrix."""
    n_pred, n_gt = ious.shape
    if n_gt == 0:
        return PQStats(fp=n_pred)
    if n_pred == 0:
        return PQStats(fn=n_gt)
    hit = ious > threshold
    return PQStats(
        tp=int(hit.sum()),
        fp=int((~hit.any(axis=1)).sum()),
        fn=int((~hit.any(axis=0)).sum()),
        iou_sum=float(ious[hit].sum()),
    )


def iou_matrix(pred_rles: list[dict], gt_rles: list[dict]) -> np.ndarray:
    """(n_pred, n_gt) IoU matrix between two lists of COCO RLEs."""
    if not pred_rles or not gt_rles:
        return np.zeros((len(pred_rles), len(gt_rles)))
    ious = mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles))
    return np.asarray(ious, dtype=np.float64).reshape(len(pred_rles), len(gt_rles))


def match(pred_rles: list[dict], gt_rles: list[dict], threshold: float = 0.5) -> PQStats:
    """Score one image's predictions against one annotator's filaments."""
    return score_ious(iou_matrix(pred_rles, gt_rles), threshold)
