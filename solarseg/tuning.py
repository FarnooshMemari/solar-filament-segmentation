"""Score many post-processing settings on validation images at once (used by scripts/tune.py).

A setting has two parts:

* how pieces are formed: low threshold, gap closing, hole filling, fragment merging;
* which pieces are kept: high threshold (hysteresis), minimum area, instance cap.

The first part needs a full pass over the 2048 x 2048 probability map, so each combination
is computed once per image. Every piece is then encoded and compared with every annotator's
filaments once. The second part only drops pieces, so all of its combinations are scored by
picking rows of the same IoU table, which keeps a large grid cheap.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .metrics import PQStats, iou_matrix, score_ious
from .postprocess import PostConfig, component_stats, label_components, select_components
from .rle import labels_to_rles

Key = tuple  # (low, close_radius, fill_holes, merge_dist, threshold, min_area)


@dataclass(frozen=True)
class Grid:
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
    low_thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6)
    min_areas: tuple[int, ...] = (150, 300, 400, 600, 1000, 1500, 2500)
    merge_dists: tuple[int, ...] = (0, 10, 25)
    close_radii: tuple[int, ...] = (0, 6)
    fill_holes: tuple[bool, ...] = (False, True)
    max_instances: int = 60

    def piece_settings(self) -> list[tuple]:
        return list(itertools.product(self.low_thresholds, self.close_radii, self.fill_holes, self.merge_dists))

    def keep_settings(self, low: float) -> list[tuple]:
        return [(t, a) for t in self.thresholds if t >= low for a in self.min_areas]

    def size(self) -> int:
        return sum(len(self.keep_settings(p[0])) for p in self.piece_settings())


def key_to_config(key: Key, max_instances: int = 60) -> PostConfig:
    low, close, fill, merge, high, area = key
    return PostConfig(
        threshold=float(high),
        min_area=int(area),
        merge_dist=int(merge),
        max_instances=max_instances,
        low_threshold=None if low == high else float(low),
        close_radius=int(close),
        fill_holes=bool(fill),
    )


def config_to_key(cfg: PostConfig) -> Key:
    return (cfg.low, cfg.close_radius, cfg.fill_holes, cfg.merge_dist, cfg.threshold, cfg.min_area)


def score_image(prob: np.ndarray, disk: np.ndarray, gts: list[list[dict]], grid: Grid) -> dict[Key, tuple]:
    """PQ counts (iou_sum, tp, fp, fn) for every setting of the grid on one image.

    ``gts`` holds one list of filament RLEs per annotator of the image.
    """
    out: dict[Key, tuple] = {}
    for low, close, fill, merge in grid.piece_settings():
        raw = label_components(prob, disk, low, merge, True, close, fill)
        if raw.max() > 0:
            area, peak = component_stats(raw, prob)
            ids, rles = labels_to_rles(raw)
        else:
            area, peak = np.zeros(1, np.int64), np.zeros(1, np.float32)
            ids, rles = np.zeros(0, np.int64), []
        ious = [iou_matrix(rles, gt) for gt in gts]
        for high, min_area in grid.keep_settings(low):
            keep = select_components(area, peak, high, min_area, grid.max_instances)
            rows = np.searchsorted(ids, keep)
            total = PQStats()
            for iou in ious:
                total += score_ious(iou[rows])
            out[(low, close, fill, merge, high, min_area)] = (total.iou_sum, total.tp, total.fp, total.fn)
    return out


def add_up(results: list[dict[Key, tuple]]) -> dict[Key, PQStats]:
    """Pool the per-image counts of every setting."""
    totals: dict[Key, PQStats] = {}
    for res in results:
        for key, (iou_sum, tp, fp, fn) in res.items():
            st = totals.setdefault(key, PQStats())
            st += PQStats(tp=tp, fp=fp, fn=fn, iou_sum=iou_sum)
    return totals
