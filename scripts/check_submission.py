#!/usr/bin/env python
"""Validate a submission CSV before uploading it (each rejected upload still uses one
of the five daily submissions).

Checks: header, unique ids in the form <image id>_<n>, no quotes, every RLE decodes to a
non-empty 2048 x 2048 mask, no two masks in an image share a pixel, and (when the test
folder is available) every id refers to a real test image.

    python scripts/check_submission.py --csv submission.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from solarseg.data import list_images, resolve_paths  # noqa: E402
from solarseg.rle import ID_RE, decode_counts, read_submission, rows_by_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--images", default=None, help="test image folder (auto-detected if possible)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--height", type=int, default=2048)
    p.add_argument("--width", type=int, default=2048)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    raw = Path(args.csv).read_text()
    if '"' in raw or "'" in raw:
        errors.append("file contains quote characters; RLE values must not be quoted")

    rows = read_submission(args.csv)
    ids = [fid for fid, _ in rows]
    if len(ids) != len(set(ids)):
        errors.append(f"{len(ids) - len(set(ids))} duplicate filament_id values")
    bad_ids = [fid for fid in ids if not ID_RE.match(fid)]
    if bad_ids:
        errors.append(f"{len(bad_ids)} ids not in <image id>_<n> form, e.g. {bad_ids[:3]}")

    known = None
    folder = Path(args.images) if args.images else None
    if folder is None:
        try:
            folder = resolve_paths(args.data_root).test_images
        except FileNotFoundError:
            folder = None
    if folder is not None and folder.exists():
        known = {p.stem for p in list_images(folder)}

    per_image = rows_by_image(rows)
    overlaps = empties = 0
    for stem, counts_list in per_image.items():
        if known is not None and stem not in known:
            errors.append(f"unknown image id {stem}")
        cover = np.zeros((args.height, args.width), np.uint8)
        for counts in counts_list:
            try:
                m = decode_counts(counts, args.height, args.width)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"RLE for {stem} does not decode: {exc}")
                continue
            if not m.any():
                empties += 1
            cover += m
        if (cover > 1).any():
            overlaps += 1
    if empties:
        errors.append(f"{empties} empty masks")
    if overlaps:
        errors.append(f"{overlaps} images have masks that share pixels")

    print(f"rows: {len(rows)}   images with predictions: {len(per_image)}")
    if known is not None:
        print(f"test images: {len(known)}   without predictions: {len(known - set(per_image))}")
    if errors:
        print("PROBLEMS:")
        for e in errors[:20]:
            print("  -", e)
        sys.exit(1)
    print("submission=ok")


if __name__ == "__main__":
    main()
