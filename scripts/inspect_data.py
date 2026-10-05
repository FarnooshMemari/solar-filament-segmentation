#!/usr/bin/env python
"""Print what the loader finds in the competition data. Run this first.

    python scripts/inspect_data.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarseg.data import (  # noqa: E402
    group_by_stem,
    list_images,
    load_records,
    read_image,
    record_instance_rles,
    resolve_paths,
)


def short(obj, limit: int = 400) -> str:
    text = json.dumps(obj, default=str)
    return text if len(text) <= limit else text[:limit] + " ..."


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=None)
    args = p.parse_args()

    paths = resolve_paths(args.data_root)
    print("data root:     ", paths.root)
    print("annotations:   ", paths.annotations)
    print("train images:  ", paths.train_images, f"({len(list_images(paths.train_images))} files)")
    if paths.test_images:
        print("test images:   ", paths.test_images, f"({len(list_images(paths.test_images))} files)")

    coco = json.loads(paths.annotations.read_text())
    print("\nJSON top-level keys:")
    for k, v in coco.items():
        print(f"  {k}: {type(v).__name__}" + (f" with {len(v)} items" if isinstance(v, (list, dict)) else ""))
    if coco.get("images"):
        print("\nfirst image entry:     ", short(coco["images"][0]))
    if coco.get("annotations"):
        ann = dict(coco["annotations"][0])
        print("first annotation keys: ", sorted(ann))
        print("first annotation:      ", short(ann))
    if coco.get("categories"):
        print("categories:            ", short(coco["categories"]))

    records, report = load_records(paths.annotations, paths.train_images)
    print("\nloader report:", json.dumps(report, indent=2))
    groups = group_by_stem(records)
    per_image = Counter(len(v) for v in groups.values())
    print("annotators per image:", dict(sorted(per_image.items())))
    print("annotator ids (top 10):", Counter(r.annotator for r in records).most_common(10))
    print("filaments per record (min/mean/max):", end=" ")
    counts = [len(r.segmentations) for r in records]
    print(min(counts), round(sum(counts) / len(counts), 2), max(counts))
    print("images per year:", dict(sorted(Counter(s[:4] for s in groups).items())))

    first = records[0]
    img = read_image(first.image_path)
    rles = record_instance_rles(first)
    print(f"\nexample: {first.stem} annotator={first.annotator} image shape={img.shape} dtype={img.dtype}")
    print(f"  declared size in JSON: {first.height} x {first.width}; filaments rasterized: {len(rles)}")
    if img.shape != (first.height, first.width):
        print("  WARNING: image size differs from the JSON size; check polygon coordinates.")


if __name__ == "__main__":
    main()
