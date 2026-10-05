"""Probability map -> separate filament instances.

Steps: threshold, keep pixels on the solar disk, optionally join fragments that are
closer than ``merge_dist`` pixels, label connected pieces, drop tiny ones.
All sizes are in full-resolution (2048 x 2048) pixels.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PostConfig:
    threshold: float = 0.5
    min_area: int = 150
    merge_dist: int = 0
    use_disk: bool = True
    max_instances: int = 60

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        data = asdict(self)
        if extra:
            data["validation"] = extra
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PostConfig":
        data = json.loads(Path(path).read_text())
        fields = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**fields)


def label_components(
    prob: np.ndarray, disk: np.ndarray | None, threshold: float, merge_dist: int, use_disk: bool = True
) -> np.ndarray:
    """Threshold + optional fragment merging -> raw int32 labels (before size filtering)."""
    fg = prob >= threshold
    if use_disk and disk is not None:
        fg &= disk
    fg_u8 = fg.astype(np.uint8)
    if not fg_u8.any():
        return np.zeros(prob.shape, np.int32)
    if merge_dist > 0:
        r = int(merge_dist)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        grown = cv2.dilate(fg_u8, kernel)
        _, labels = cv2.connectedComponents(grown, connectivity=8)
        return labels.astype(np.int32) * fg_u8  # fragments share a label, gaps stay empty
    _, labels = cv2.connectedComponents(fg_u8, connectivity=8)
    return labels.astype(np.int32)


def filter_components(labels: np.ndarray, min_area: int, max_instances: int = 60) -> np.ndarray:
    """Drop pieces smaller than ``min_area``, keep at most ``max_instances``, relabel 1..K."""
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    ids = np.flatnonzero(keep)
    if len(ids) > max_instances:  # keep the largest ones
        ids = ids[np.argsort(-areas[ids])[:max_instances]]
    remap = np.zeros(len(areas), np.int32)
    remap[np.sort(ids)] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return remap[labels]


def instances_from_prob(
    prob: np.ndarray, disk: np.ndarray | None = None, cfg: PostConfig | None = None
) -> np.ndarray:
    """Return an int32 label map: 0 = background, 1..K = filaments (disjoint by design)."""
    cfg = cfg or PostConfig()
    raw = label_components(prob, disk, cfg.threshold, cfg.merge_dist, cfg.use_disk)
    return filter_components(raw, cfg.min_area, cfg.max_instances)
