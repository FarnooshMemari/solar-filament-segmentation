#!/usr/bin/env python
"""Build a small fake dataset with the same folder layout as the competition data,
so the whole pipeline can be tested without downloading anything.

    python scripts/make_synthetic.py --out data/synthetic --size 512
    python scripts/train.py --data-root data/synthetic/MAGFiLO_1.0_Kaggle_2026 ...
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

SITES = "BCLMTU"


def sun(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size / 2 + rng.uniform(-0.02, 0.02) * size
    r = 0.44 * size
    d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / r
    disk = d < 1
    limb = 1 - 0.4 * (1 - np.sqrt(np.clip(1 - d**2, 0, 1)))
    img = np.where(disk, 165 * limb, 6).astype(np.float32)
    texture = cv2.GaussianBlur(rng.normal(0, 12, (size, size)).astype(np.float32), (0, 0), 2)
    return img + texture * disk, disk


def filament(rng: np.random.Generator, size: int, disk: np.ndarray) -> np.ndarray:
    c = size / 2
    r = rng.uniform(0, 0.75) * 0.44 * size
    a = rng.uniform(0, 2 * np.pi)
    pt = np.array([c + r * np.cos(a), c + r * np.sin(a)])
    heading = rng.uniform(0, 2 * np.pi)
    step = size * 0.006
    pts = [pt.copy()]
    for _ in range(int(rng.integers(15, 45))):
        heading += rng.normal(0, 0.25)
        pt = pt + step * np.array([np.cos(heading), np.sin(heading)])
        pts.append(pt.copy())
    mask = np.zeros((size, size), np.uint8)
    thickness = max(2, int(size * rng.uniform(0.004, 0.009)))
    cv2.polylines(mask, [np.round(np.array(pts)).astype(np.int32)], False, 1, thickness)
    return (mask.astype(bool) & disk)


def polygons(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1).astype(float).tolist() for c in contours if len(c) >= 3]


def make_frame(rng, size, when):
    img, disk = sun(rng, size)
    masks = []
    for _ in range(int(rng.integers(3, 8))):
        m = filament(rng, size, disk)
        if m.sum() < 20 or any((m & o).any() for o in masks):
            continue
        masks.append(m)
        img[m] -= rng.uniform(45, 75)
    img = cv2.GaussianBlur(np.clip(img, 0, 255), (0, 0), 1).astype(np.uint8)
    stem = when.strftime("%Y%m%d%H%M%S") + SITES[int(rng.integers(len(SITES)))] + "h"
    return stem, img, masks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--n-train", type=int, default=12)
    p.add_argument("--n-test", type=int, default=3)
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out) / "MAGFiLO_1.0_Kaggle_2026"
    train_dir = root / "train" / "train_images"
    test_dir = root / "test" / "test_images"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    start = datetime(2012, 1, 1)
    images, annotations = [], []
    kernel = np.ones((3, 3), np.uint8)
    for i in range(args.n_train):
        when = start + timedelta(days=int(rng.integers(0, 3500)), seconds=int(rng.integers(0, 86400)))
        stem, img, masks = make_frame(rng, args.size, when)
        cv2.imwrite(str(train_dir / f"{stem}.jpg"), img)
        n_annotators = int(rng.integers(1, 4))
        for a in range(n_annotators):
            annotator = f"A{a + 1}"
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": f"{annotator}-{stem}.jpg",
                           "height": args.size, "width": args.size})
            for m in masks:
                if a == 1:  # a second annotator draws slightly wider outlines
                    m = cv2.dilate(m.astype(np.uint8), kernel).astype(bool)
                if a == 2 and rng.random() < 0.2:  # a third one sometimes skips a filament
                    continue
                polys = polygons(m)
                if not polys:
                    continue
                ys, xs = np.nonzero(m)
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                    "segmentation": polys, "area": int(m.sum()), "iscrowd": 0,
                    "bbox": [int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1],
                })
    for i in range(args.n_test):
        when = start + timedelta(days=int(rng.integers(3600, 4000)), seconds=int(rng.integers(0, 86400)))
        stem, img, _ = make_frame(rng, args.size, when)
        cv2.imwrite(str(test_dir / f"{stem}.jpg"), img)

    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "filament"}]}
    ann_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    ann_path.write_text(json.dumps(coco))
    print(f"wrote {args.n_train} train and {args.n_test} test images to {root}")


if __name__ == "__main__":
    main()
