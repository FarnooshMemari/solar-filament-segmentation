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


def labels_to_rles(labels: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Instance label map -> (label ids present, one compressed RLE per id), in increasing id order.

    COCO RLE stores alternating run lengths of 0s and 1s in column-major order. Instead of
    building a full 2048 x 2048 mask for every filament, the runs of the whole label map are
    found once and split by label, which is several times faster on images with many pieces.
    """
    h, w = labels.shape
    flat = np.ravel(labels, order="F")
    n = flat.size
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    values = flat[starts]
    fg = values > 0
    if not fg.any():
        return np.zeros(0, labels.dtype), []
    starts, ends, values = starts[fg], ends[fg], values[fg]
    order = np.argsort(values, kind="stable")  # group runs by label, keep their pixel order
    starts, ends, values = starts[order], ends[order], values[order]
    ids, first = np.unique(values, return_index=True)
    bounds = np.append(first, len(values))
    uncompressed = []
    for j in range(len(ids)):
        s, e = starts[bounds[j] : bounds[j + 1]], ends[bounds[j] : bounds[j + 1]]
        counts = np.empty(2 * len(s) + 1, np.int64)
        counts[0] = s[0]  # zeros before the first run
        counts[1::2] = e - s  # each run of ones
        counts[2:-1:2] = s[1:] - e[:-1]  # zeros between runs
        counts[-1] = n - e[-1]  # trailing zeros (left out when there are none)
        counts = counts if counts[-1] > 0 else counts[:-1]
        uncompressed.append({"size": [h, w], "counts": counts.tolist()})
    return ids, mask_utils.frPyObjects(uncompressed, h, w)


def labels_to_counts(labels: np.ndarray) -> list[str]:
    """Instance label map (0 = background, 1..K = filaments) -> list of counts strings."""
    _, rles = labels_to_rles(labels)
    return [r["counts"].decode("ascii") for r in rles]


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
