"""Probability map -> separate filament instances.

Steps:
1. keep pixels above the low threshold that lie on the solar disk;
2. optionally close small gaps (morphological closing) and fill holes inside pieces;
3. optionally give fragments closer than ``merge_dist`` pixels the same label;
4. label connected pieces, then keep a piece only if its strongest pixel reaches the high
   threshold (hysteresis) and it covers at least ``min_area`` pixels.

With the defaults (low threshold = high threshold, no closing, no hole filling) this is a
plain threshold followed by connected components. All sizes are in full-resolution
(2048 x 2048) pixels.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PostConfig:
    threshold: float = 0.5  # high threshold: a piece needs at least one pixel this sure
    min_area: int = 150
    merge_dist: int = 0
    use_disk: bool = True
    max_instances: int = 60
    low_threshold: float | None = None  # extent threshold; None means the same as `threshold`
    close_radius: int = 0
    fill_holes: bool = False

    @property
    def low(self) -> float:
        if self.low_threshold is None:
            return self.threshold
        return min(self.low_threshold, self.threshold)

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


def tuned_tta(path: str | Path) -> int:
    """Test-time augmentation used when the settings in `path` were tuned (0 if unknown)."""
    path = Path(path)
    if not path.exists():
        return 0
    return int(json.loads(path.read_text()).get("validation", {}).get("tta", 0) or 0)


def _round_kernel(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions that are completely enclosed by foreground (uint8 0/1 mask)."""
    h, w = mask.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    flood[1:-1, 1:-1] = mask
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)  # outside background -> 1
    holes = flood[1:-1, 1:-1] == 0
    return mask | holes.astype(np.uint8)


def foreground(
    prob: np.ndarray,
    disk: np.ndarray | None,
    threshold: float,
    close_radius: int = 0,
    fill_holes: bool = False,
    use_disk: bool = True,
) -> np.ndarray:
    """Thresholded (and optionally closed / hole-filled) foreground as a uint8 0/1 mask."""
    fg = prob >= threshold
    on_disk = use_disk and disk is not None
    if on_disk:
        fg &= disk
    fg = fg.astype(np.uint8)
    if not fg.any():
        return fg
    if close_radius > 0:
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _round_kernel(int(close_radius)))
        if on_disk:
            fg &= disk.astype(np.uint8)
    if fill_holes:
        fg = _fill_holes(fg)
    return fg


def label_components(
    prob: np.ndarray,
    disk: np.ndarray | None,
    threshold: float,
    merge_dist: int = 0,
    use_disk: bool = True,
    close_radius: int = 0,
    fill_holes: bool = False,
) -> np.ndarray:
    """Foreground -> raw int32 labels (before any piece is dropped)."""
    fg = foreground(prob, disk, threshold, close_radius, fill_holes, use_disk)
    if not fg.any():
        return np.zeros(prob.shape, np.int32)
    if merge_dist > 0:
        grown = cv2.dilate(fg, _round_kernel(int(merge_dist)))
        _, labels = cv2.connectedComponents(grown, connectivity=8)
        return labels.astype(np.int32) * fg  # fragments share a label, gaps stay empty
    _, labels = cv2.connectedComponents(fg, connectivity=8)
    return labels.astype(np.int32)


def component_stats(labels: np.ndarray, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Area and highest probability of every piece, indexed by label (index 0 is background)."""
    n = int(labels.max()) + 1
    flat = labels.ravel()
    area = np.bincount(flat, minlength=n)
    peak = np.zeros(n, np.float32)
    inside = flat > 0
    np.maximum.at(peak, flat[inside], prob.ravel()[inside].astype(np.float32))
    return area, peak


def select_components(
    area: np.ndarray, peak: np.ndarray, threshold: float, min_area: int, max_instances: int = 60
) -> np.ndarray:
    """Label ids of the pieces to keep: strong enough, big enough, at most `max_instances` (largest)."""
    keep = (area >= min_area) & (peak >= threshold)
    keep[0] = False
    ids = np.flatnonzero(keep)
    if len(ids) > max_instances:
        ids = np.sort(ids[np.argsort(-area[ids])[:max_instances]])
    return ids


def relabel(labels: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Keep only `ids` and number them 1..K in the same order."""
    remap = np.zeros(int(labels.max()) + 1, np.int32)
    remap[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return remap[labels]


def filter_components(labels: np.ndarray, min_area: int, max_instances: int = 60) -> np.ndarray:
    """Drop pieces smaller than ``min_area``, keep at most ``max_instances``, relabel 1..K."""
    area = np.bincount(labels.ravel())
    ids = select_components(area, np.ones(len(area), np.float32), 0.0, min_area, max_instances)
    return relabel(labels, ids)


def instances_from_prob(
    prob: np.ndarray, disk: np.ndarray | None = None, cfg: PostConfig | None = None
) -> np.ndarray:
    """Return an int32 label map: 0 = background, 1..K = filaments (disjoint by design)."""
    cfg = cfg or PostConfig()
    raw = label_components(
        prob, disk, cfg.low, cfg.merge_dist, cfg.use_disk, cfg.close_radius, cfg.fill_holes
    )
    if raw.max() == 0:
        return raw
    area, peak = component_stats(raw, prob)
    return relabel(raw, select_components(area, peak, cfg.threshold, cfg.min_area, cfg.max_instances))
