#!/usr/bin/env python
"""Predict the test images and write a Kaggle submission CSV.

    python scripts/predict.py --run runs/baseline --out submission.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarseg.data import list_images, resolve_paths  # noqa: E402
from solarseg.infer import get_device, load_checkpoint, predict_image  # noqa: E402
from solarseg.postprocess import PostConfig, instances_from_prob  # noqa: E402
from solarseg.rle import labels_to_counts, write_submission  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default=None, help="defaults to <run>/best.pt")
    p.add_argument("--params", default=None, help="post-processing JSON; defaults to <run>/postprocess.json")
    p.add_argument("--data-root", default=None)
    p.add_argument("--images", default=None, help="folder of images to predict; defaults to test_images")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N images")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run = Path(args.run)
    device = get_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint or run / "best.pt", device)

    params_path = Path(args.params) if args.params else run / "postprocess.json"
    post = PostConfig.load(params_path) if params_path.exists() else PostConfig()
    print(f"post-processing: {post}")

    if args.images:
        folder = Path(args.images)
    else:
        folder = resolve_paths(args.data_root).test_images
        if folder is None:
            sys.exit("No test_images folder found; pass --images")
    images = list_images(folder)
    if args.limit:
        images = images[: args.limit]

    rows, empty, t0 = [], [], time.time()
    for i, path in enumerate(images, 1):
        prob, disk = predict_image(
            model, path, device, cfg.get("scale", 0.5),
            tile=cfg.get("tile", 512), overlap=cfg.get("overlap", 128), tta=args.tta,
        )
        counts = labels_to_counts(instances_from_prob(prob, disk, post))
        if not counts:
            empty.append(path.stem)
        # the id is the image file name without its extension, then a running number
        rows.extend((f"{path.stem}_{k}", c) for k, c in enumerate(counts, 1))
        if i % 20 == 0 or i == len(images):
            print(f"  {i}/{len(images)} images, {len(rows)} filaments ({time.time() - t0:.0f}s)", flush=True)

    if not rows:
        sys.exit("No filaments predicted at all; check the threshold before submitting.")
    n = write_submission(rows, args.out)
    summary = {
        "images": len(images),
        "rows": n,
        "images_without_detections": empty,
        "postprocess": vars(post),
        "checkpoint": str(args.checkpoint or run / "best.pt"),
        "tta": args.tta,
    }
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}: {n} rows for {len(images)} images ({len(empty)} with no detections)")
    print(f"next: python scripts/check_submission.py --csv {args.out}")


if __name__ == "__main__":
    main()
