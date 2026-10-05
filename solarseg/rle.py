"""Masks <-> the competition's submission format.

The CSV has two columns, ``filament_id`` and ``segmentation_rle``:

* ``filament_id`` is ``<image id>_<n>``, with the image file name minus its extension
  and ``n`` counting from 1 within each image;
* ``segmentation_rle`` is only the ``counts`` string of a pycocotools compressed RLE.
  The size is not written (every frame is 2048 x 2048) and values are never quoted.

Images with no predicted filament simply get no rows. Masks in the same image must
not share pixels, otherwise Kaggle rejects the file.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from pycocotools import mask as mask_utils

HEADER = "filament_id,segmentation_rle"
ID_RE = re.compile(r"^(?P<stem>.+)_(?P<n>\d+)$")


def encode_mask(mask: np.ndarray) -> str:
    """Binary H x W mask -> compressed RLE counts string."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def decode_counts(counts: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    rle = {"size": [height, width], "counts": counts.encode("ascii")}
    return mask_utils.decode(rle).astype(bool)


def counts_to_rle(counts: str, height: int = 2048, width: int = 2048) -> dict:
    return {"size": [height, width], "counts": counts.encode("ascii")}


def labels_to_counts(labels: np.ndarray) -> list[str]:
    """Instance label map (0 = background, 1..K = filaments) -> list of counts strings."""
    out = []
    for k in range(1, int(labels.max()) + 1):
        m = labels == k
        if m.any():
            out.append(encode_mask(m))
    return out


def write_submission(rows: Iterable[tuple[str, str]], path: str | Path) -> int:
    """Write rows of (filament_id, counts) without quotes. Returns the row count."""
    n = 0
    with open(path, "w", newline="\n") as f:
        f.write(HEADER + "\n")
        for fid, counts in rows:
            if any(c in counts for c in ",\"'\n"):
                raise ValueError(f"Unexpected character in RLE for {fid}")
            f.write(f"{fid},{counts}\n")
            n += 1
    return n


def read_submission(path: str | Path) -> list[tuple[str, str]]:
    rows = []
    with open(path) as f:
        header = f.readline().strip()
        if header != HEADER:
            raise ValueError(f"Bad header {header!r}; expected {HEADER!r}")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            fid, _, counts = line.partition(",")
            rows.append((fid, counts))
    return rows


def rows_by_image(rows: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for fid, counts in rows:
        m = ID_RE.match(fid)
        stem = m.group("stem") if m else fid
        out[stem].append(counts)
    return dict(out)
