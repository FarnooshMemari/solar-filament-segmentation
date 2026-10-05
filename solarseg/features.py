"""Input channels for the network, computed at the working resolution.

Available channels:

* ``z``        the image z-scored over the solar disk (the only channel before run 3)
* ``flat``     limb darkening removed: each pixel divided by the median brightness of all disk
               pixels at the same distance from the centre, minus 1, then scaled by its spread.
               The disk gets darker toward its edge, so without this a filament near the limb
               and the plain limb look alike.
* ``contrast`` local contrast: ``flat`` minus its local mean, divided by its local spread
* ``ridge``    dark-ridge strength: the largest Hessian eigenvalue of ``flat`` (positive across
               a dark line) at three widths, scale-normalized, so thin filaments and barbs stand out

Everything outside the disk is set to 0. Channels are stacked as (C, H, W) float32.
"""
from __future__ import annotations

import cv2
import numpy as np

CHANNELS = ("z", "flat", "contrast", "ridge")
RIDGE_SIGMAS = (1.5, 3.0, 5.0)  # in working-resolution pixels (filaments are ~3-15 px wide at 1024)


def parse_channels(spec: str | tuple | list | None) -> tuple[str, ...]:
    """'z,flat,contrast' -> ('z', 'flat', 'contrast'); None -> ('z',)."""
    if spec is None:
        return ("z",)
    names = tuple(s.strip() for s in spec.split(",")) if isinstance(spec, str) else tuple(spec)
    unknown = [c for c in names if c not in CHANNELS]
    if not names or unknown:
        raise ValueError(f"unknown channels {unknown}; choose from {CHANNELS}")
    return names


def disk_geometry(disk: np.ndarray) -> tuple[float, float, float]:
    """Centre (cx, cy) and radius of the circle with the same area and centroid as the mask."""
    m = cv2.moments(disk.astype(np.uint8), binaryImage=True)
    if m["m00"] <= 0:
        h, w = disk.shape
        return w / 2, h / 2, 0.45 * min(h, w)
    return m["m10"] / m["m00"], m["m01"] / m["m00"], float(np.sqrt(m["m00"] / np.pi))


def _radius_map(shape: tuple[int, int], cx: float, cy: float, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    return np.hypot(xx - cx, yy - cy) / max(radius, 1.0)


def _radial_median(img: np.ndarray, rr: np.ndarray, inside: np.ndarray, bins: int = 128) -> np.ndarray:
    """Median brightness at each normalized radius, mapped back to every pixel."""
    r = rr[inside]
    v = img[inside]
    idx = np.minimum((r * bins).astype(np.int32), bins - 1)
    order = np.argsort(idx, kind="stable")
    idx, v = idx[order], v[order]
    edges = np.searchsorted(idx, np.arange(bins + 1))
    centres = (np.arange(bins) + 0.5) / bins
    prof = np.full(bins, np.nan, np.float32)
    for b in range(bins):
        seg = v[edges[b] : edges[b + 1]]
        if seg.size >= 10:
            prof[b] = np.median(seg)
    ok = ~np.isnan(prof)
    if not ok.any():
        return np.full(img.shape, float(np.median(v)) if v.size else 1.0, np.float32)
    prof = np.interp(centres, centres[ok], prof[ok]).astype(np.float32)
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (7, 1), 1.5, borderType=cv2.BORDER_REPLICATE).ravel()
    return np.interp(rr, centres, prof).astype(np.float32)


def _masked_blur(x: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that only averages pixels where w > 0 (no bleeding from outside the disk)."""
    num = cv2.GaussianBlur(x * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)


def _robust_scale(x: np.ndarray, inside: np.ndarray) -> float:
    v = x[inside]
    if v.size == 0:
        return 1.0
    mad = float(np.median(np.abs(v - np.median(v))))
    return max(1.4826 * mad, 1e-4)


def _ridge(flat: np.ndarray, sigmas=RIDGE_SIGMAS) -> np.ndarray:
    out = np.zeros_like(flat)
    for s in sigmas:
        g = cv2.GaussianBlur(flat, (0, 0), s)
        ixx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        iyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        ixy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        half_tr = 0.5 * (ixx + iyy)
        root = np.sqrt(np.maximum(0.25 * (ixx - iyy) ** 2 + ixy**2, 0))
        out = np.maximum(out, (s**2) * np.maximum(half_tr + root, 0))
    return out


def compute_channels(img: np.ndarray, disk: np.ndarray, size: tuple[int, int], channels=("z",)) -> np.ndarray:
    """Full-resolution uint8 image + disk mask -> (C, H, W) float32 at ``size`` (height, width)."""
    channels = parse_channels(channels)
    h, w = size
    small = img.astype(np.float32)
    if small.shape != (h, w):
        small = cv2.resize(small, (w, h), interpolation=cv2.INTER_AREA)
    inside = cv2.resize(disk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not inside.any():
        inside = np.ones((h, w), bool)
    wgt = inside.astype(np.float32)
    out = {}

    vals = small[inside]
    out["z"] = np.clip((small - vals.mean()) / (vals.std() + 1e-6), -5, 5) * wgt

    if any(c in channels for c in ("flat", "contrast", "ridge")):
        cx, cy, radius = disk_geometry(inside)
        rr = _radius_map((h, w), cx, cy, radius)
        prof = _radial_median(small, rr, inside)
        flat = np.where(inside, small / np.maximum(prof, 1.0) - 1.0, 0.0).astype(np.float32)
        flat_s = _robust_scale(flat, inside)
        out["flat"] = np.clip(flat / flat_s, -5, 5) * wgt
        if "contrast" in channels:
            mu = _masked_blur(flat, wgt, 16.0)
            d = (flat - mu) * wgt
            sd = np.sqrt(np.maximum(_masked_blur(d * d, wgt, 32.0), 1e-8))
            out["contrast"] = np.clip(d / np.maximum(sd, 0.25 * flat_s), -5, 5) * wgt
        if "ridge" in channels:
            r = _ridge(flat) * wgt
            ref = float(np.median(r[inside & (r > 0)])) if (inside & (r > 0)).any() else 1.0
            out["ridge"] = np.clip(np.log1p(r / max(ref, 1e-6)), 0, 5) * wgt
    return np.stack([out[c].astype(np.float32) for c in channels])
