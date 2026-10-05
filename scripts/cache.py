#!/usr/bin/env python
"""Compute the network inputs (and soft targets) once and store them for training and
prediction, so parallel runs on two GPUs share one copy.

    python scripts/cache.py --out /kaggle/temp/cache --channels z,flat,contrast,ridge --workers 4
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarseg.cache import has_item, save_item, write_meta  # noqa: E402
from solarseg.data import (  # noqa: E402
    group_by_stem,
    list_images,
    load_records,
    preprocess,
    read_image,
    resize_target,
    resolve_paths,
    soft_target,
)
from solarseg.features import parse_channels  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--channels", default="z,flat,contrast,ridge")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-test", action="store_true", help="skip the test images")
    p.add_argument("--max-images", type=int, default=0, help="debug: only the first N training images")
    return p.parse_args()


def _work(job):
    out, stem, image_path, recs, scale, channels = job
    img = read_image(image_path)
    if not has_item(out, stem, need_target=recs is not None):
        x, disk = preprocess(img, scale, channels)
        y = resize_target(soft_target(recs), scale) if recs is not None else None
        save_item(out, stem, x, disk, y)
    return img.shape


def main() -> None:
    args = parse_args()
    channels = parse_channels(args.channels)
    paths = resolve_paths(args.data_root)
    records, _ = load_records(paths.annotations, paths.train_images)
    groups = group_by_stem(records)
    train = sorted(groups)
    if args.max_images:
        train = train[: args.max_images]
    jobs = [(args.out, s, groups[s][0].image_path, groups[s], args.scale, channels) for s in train]
    test = []
    if not args.no_test and paths.test_images is not None:
        for path in list_images(paths.test_images):
            # test images are stored under their file name, which is also the submission id
            test.append(path.stem)
            jobs.append((args.out, test[-1], path, None, args.scale, channels))
    clash = sorted(set(test) & set(train))
    if clash:
        sys.exit(f"{len(clash)} test file names equal training frame ids (e.g. {clash[0]}); use separate caches")

    print(f"caching {len(train)} training and {len(test)} test images with channels {channels}...", flush=True)
    t0 = time.time()
    shapes = set()
    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        for i, shape in enumerate(pool.map(_work, jobs, chunksize=4), 1):
            shapes.add(tuple(shape))
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    if len(shapes) != 1:
        sys.exit(f"images have different sizes: {sorted(shapes)}")
    meta = {"channels": list(channels), "scale": args.scale, "image_size": list(shapes.pop()),
            "train": train, "test": sorted(test)}
    write_meta(args.out, meta)
    print(f"saved cache to {args.out}")


if __name__ == "__main__":
    main()
