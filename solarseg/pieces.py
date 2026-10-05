"""Describe every candidate filament piece and score which ones to submit.

A piece either matches an annotated filament (IoU > 0.5) or counts as a false positive.
Adding one piece to an image with R annotators raises the PQ denominator by 0.5 for each
annotator (a match turns a miss into a hit, otherwise it adds a false positive) and the
numerator by its matched IoU. So a piece raises PQ only when

    mean over annotators of (its matched IoU, or 0 when unmatched)  >  PQ / 2.

``piece_table`` measures each piece (size, shape, confidence, brightness, position on the
disk); a gradient-boosted model learns that mean from out-of-fold pieces, and only pieces
whose predicted value passes a threshold tuned on out-of-fold PQ are kept.
"""
from __future__ import annotations

import numpy as np

from .data import STEM_RE
from .features import disk_geometry
from .metrics import PQStats, iou_matrix
from .postprocess import PostConfig, label_components

STATIONS = "BCLMTU"  # GONG sites: Big Bear, Cerro Tololo, Learmonth, Mauna Loa, Teide, Udaipur
BASE_FEATURES = [
    "log_area", "peak", "mean_prob", "core_frac", "high_frac", "r_norm",
    "major", "minor", "elongation", "n_pieces", "img_mean_prob", "station",
]


def feature_names(channels) -> list[str]:
    return BASE_FEATURES + [f"mean_{c}" for c in channels]


def station_of(name: str) -> float:
    """Index of the observing site in STATIONS from a GONG frame id in ``name``, or -1."""
    m = STEM_RE.search(name)
    return float(STATIONS.find(m.group(1)[-2])) if m else -1.0


def piece_table(
    labels: np.ndarray,
    prob: np.ndarray,
    disk: np.ndarray,
    x_small: np.ndarray | None,
    stem: str,
) -> tuple[np.ndarray, np.ndarray]:
    """(ids, features) for every piece of a full-resolution label map.

    ``x_small`` is the (C, h, w) network input at the working resolution; the mean of each
    channel inside the piece is added as a feature.
    """
    n = int(labels.max()) + 1
    flat_lab = labels.ravel()
    idx = np.flatnonzero(flat_lab)
    ids = np.unique(flat_lab[idx]) if idx.size else np.zeros(0, np.int64)
    n_ch = 0 if x_small is None else x_small.shape[0]
    if ids.size == 0:
        return ids, np.zeros((0, len(BASE_FEATURES) + n_ch), np.float32)
    lab = flat_lab[idx]
    h, w = labels.shape
    ys, xs = (idx // w).astype(np.float64), (idx % w).astype(np.float64)
    p = prob.ravel()[idx].astype(np.float32)

    area = np.bincount(lab, minlength=n).astype(np.float64)
    a = np.maximum(area, 1.0)
    peak = np.zeros(n, np.float32)
    np.maximum.at(peak, lab, p)
    mean_p = np.bincount(lab, weights=p, minlength=n) / a
    core = np.bincount(lab, weights=(p >= 0.5).astype(np.float64), minlength=n) / a
    high = np.bincount(lab, weights=(p >= 0.8).astype(np.float64), minlength=n) / a
    mx = np.bincount(lab, weights=xs, minlength=n) / a
    my = np.bincount(lab, weights=ys, minlength=n) / a
    cxx = np.bincount(lab, weights=xs * xs, minlength=n) / a - mx**2
    cyy = np.bincount(lab, weights=ys * ys, minlength=n) / a - my**2
    cxy = np.bincount(lab, weights=xs * ys, minlength=n) / a - mx * my
    half_tr, root = 0.5 * (cxx + cyy), np.sqrt(np.maximum(0.25 * (cxx - cyy) ** 2 + cxy**2, 0))
    major = 4 * np.sqrt(np.maximum(half_tr + root, 0))
    minor = 4 * np.sqrt(np.maximum(half_tr - root, 0))
    cx, cy, radius = disk_geometry(disk)
    r_norm = np.hypot(mx - cx, my - cy) / max(radius, 1.0)
    img_mean = float(prob[disk].mean()) if disk.any() else float(prob.mean())
    station = station_of(stem)

    cols = [
        np.log1p(area), peak, mean_p, core, high, r_norm, major, minor, major / (minor + 1.0),
        np.full(n, float(ids.size)), np.full(n, img_mean), np.full(n, station),
    ]
    if n_ch:
        f = max(1, h // x_small.shape[1])
        small = labels[::f, ::f][: x_small.shape[1], : x_small.shape[2]].ravel()
        cnt = np.bincount(small, minlength=n).astype(np.float64)
        for c in range(n_ch):
            s = np.bincount(small, weights=np.asarray(x_small[c], dtype=np.float64).ravel(), minlength=n)
            with np.errstate(invalid="ignore", divide="ignore"):
                cols.append(np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan))
    table = np.stack([np.asarray(c, dtype=np.float64)[ids] for c in cols], axis=1).astype(np.float32)
    return ids, table


def matched_ious(rles: list[dict], gts: list[list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    """(hits, n_gt): hits[k, r] is piece k's IoU with annotator r's filament when above 0.5, else 0."""
    hits = np.zeros((len(rles), len(gts)), np.float32)
    n_gt = np.array([len(g) for g in gts], np.int64)
    for r, gt in enumerate(gts):
        iou = iou_matrix(rles, gt)
        if iou.size:
            best = iou.max(axis=1)
            hits[:, r] = np.where(best > 0.5, best, 0.0)
    return hits, n_gt


def selection_pq(hits_list, n_gt_list, keep_list) -> PQStats:
    """Pooled PQ when only the pieces in ``keep_list`` (boolean masks) are submitted."""
    total = PQStats()
    for hits, n_gt, keep in zip(hits_list, n_gt_list, keep_list):
        h = hits[keep]
        for r in range(len(n_gt)):
            tp = int((h[:, r] > 0).sum())
            total += PQStats(tp=tp, fp=int(h.shape[0] - tp), fn=int(n_gt[r] - tp), iou_sum=float(h[:, r].sum()))
    return total


def keep_by_score(scores: np.ndarray, tau: float, max_instances: int = 60) -> np.ndarray:
    keep = scores >= tau
    if keep.sum() > max_instances:
        order = np.argsort(-scores)
        keep = np.zeros_like(keep)
        keep[order[:max_instances]] = True
    return keep


def best_tau(scores_list, hits_list, n_gt_list, taus=None) -> tuple[float, PQStats, list]:
    """Threshold on predicted scores that gives the highest pooled PQ."""
    taus = np.round(np.arange(0.02, 0.62, 0.01), 2) if taus is None else taus
    curve = []
    for t in taus:
        st = selection_pq(hits_list, n_gt_list, [keep_by_score(s, t) for s in scores_list])
        curve.append((float(t), st))
    t, st = max(curve, key=lambda c: c[1].pq)
    return t, st, curve


def make_pieces(prob: np.ndarray, disk: np.ndarray, x_small, stem: str, post: PostConfig):
    """Candidate pieces with the tuned low threshold / merging: (labels, ids, features)."""
    labels = label_components(prob, disk, post.low, post.merge_dist, post.use_disk, post.close_radius, post.fill_holes)
    ids, table = piece_table(labels, prob, disk, x_small, stem)
    return labels, ids, table
