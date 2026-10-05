"""On-disk cache of network inputs, soft targets and disk masks.

Computing the extra input channels takes about half a second per image, and two training
runs share the same images, so everything is computed once and memory-mapped afterwards.

Layout of a cache folder::

    meta.json          {"channels": [...], "scale": 0.5, "train": [stems], "test": [stems]}
    x/<stem>.npy       (C, H, W) float16 network input at the working resolution
    y/<stem>.npy       (H, W) float16 soft target (training images only)
    disk/<stem>.npy    full-resolution solar-disk mask, bit-packed
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _path(root: str | Path, kind: str, stem: str) -> Path:
    return Path(root) / kind / f"{stem}.npy"


def save_item(root: str | Path, stem: str, x: np.ndarray, disk: np.ndarray, y: np.ndarray | None = None) -> None:
    root = Path(root)
    for kind in ("x", "y", "disk"):
        (root / kind).mkdir(parents=True, exist_ok=True)
    x = x if x.ndim == 3 else x[None]
    np.save(_path(root, "x", stem), x.astype(np.float16))
    np.save(_path(root, "disk", stem), np.packbits(disk.astype(bool), axis=None))
    if y is not None:
        np.save(_path(root, "y", stem), y.astype(np.float16))


def load_x(root: str | Path, stem: str) -> np.ndarray:
    return np.load(_path(root, "x", stem), mmap_mode="r")


def load_y(root: str | Path, stem: str) -> np.ndarray:
    return np.load(_path(root, "y", stem), mmap_mode="r")


def load_disk(root: str | Path, stem: str, shape: tuple[int, int] = (2048, 2048)) -> np.ndarray:
    bits = np.load(_path(root, "disk", stem))
    return np.unpackbits(bits, count=shape[0] * shape[1]).reshape(shape).astype(bool)


def write_meta(root: str | Path, meta: dict) -> None:
    Path(root, "meta.json").write_text(json.dumps(meta, indent=1))


def read_meta(root: str | Path) -> dict:
    return json.loads(Path(root, "meta.json").read_text())


def has_item(root: str | Path, stem: str, need_target: bool = False) -> bool:
    ok = _path(root, "x", stem).exists() and _path(root, "disk", stem).exists()
    return ok and (not need_target or _path(root, "y", stem).exists())


def load_prob(paths, shape: tuple[int, int] = (2048, 2048)) -> np.ndarray:
    """Average one or more saved probability maps and upsample to full resolution (float32)."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    acc = None
    for p in paths:
        m = np.load(p).astype(np.float32)
        acc = m if acc is None else acc + m
    prob = acc / len(paths)
    if prob.shape != shape:
        prob = cv2.resize(prob, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return prob
