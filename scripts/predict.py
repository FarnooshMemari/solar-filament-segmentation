#!/usr/bin/env python
"""Predict the test images and write a Kaggle submission CSV.

With one trained run:

    python scripts/predict.py --run runs/baseline --out submission.csv

With the k-fold runs of run 3 (their test maps saved by scripts/oof.py are averaged) and,
optionally, the LightGBM filament picker from scripts/scorer.py:

    python scripts/predict.py --probs /kaggle/temp/probs --cache /kaggle/temp/cache \\
        --params runs/run3/postprocess.json --picker runs/run3 --out submission.csv
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarseg.postprocess import PostConfig, instances_from_prob, tuned_tta  # noqa: E402
from solarseg.rle import labels_to_counts, write_submission  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="run folder with best.pt")
    p.add_argument("--probs", default=None, help="folder from scripts/oof.py with test/<fold>/<stem>.npy maps")
    p.add_argument("--cache", default=None, help="cache folder (needed with --probs)")
    p.add_argument("--picker", default=None, help="folder with picker.txt/picker.json from scripts/scorer.py")
    p.add_argument("--checkpoint", default=None, help="defaults to <run>/best.pt")
    p.add_argument("--params", default=None, help="post-processing JSON; defaults to <run>/postprocess.json")
    p.add_argument("--data-root", default=None)
    p.add_argument("--images", default=None, help="folder of images to predict; defaults to test_images")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--tta", type=int, nargs="?", const=4, default=None,
                   help="views to average: 4 (flips) or 8 (flips and rotations); "
                        "defaults to what the post-processing was tuned with")
    p.add_argument("--workers", type=int, default=4, help="CPU processes for post-processing saved maps")
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N images")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if (args.run is None) == (args.probs is None):
        p.error("give either --run or --probs")
    if args.probs and (not args.cache or not args.params):
        p.error("--probs needs --cache and --params")
    return args


_PICKER = {}


def _init_worker(picker_dir) -> None:
    import cv2

    cv2.setNumThreads(1)
    if picker_dir:
        import lightgbm as lgb

        info = json.loads((Path(picker_dir) / "picker.json").read_text())
        _PICKER["model"] = lgb.Booster(model_file=str(Path(picker_dir) / "picker.txt"))
        _PICKER["info"] = info
        _PICKER["cols"] = [info["table_features"].index(f) for f in info["features"]]


def _from_saved(task):
    """Average the fold maps of one test image, cut it into pieces and keep the chosen ones."""
    from solarseg.cache import load_disk, load_prob, load_x
    from solarseg.pieces import keep_by_score, make_pieces
    from solarseg.postprocess import relabel

    stem, maps, cache, shape, post = task
    prob = load_prob(maps, shape)
    disk = load_disk(cache, stem, shape)
    if "model" not in _PICKER:
        return stem, labels_to_counts(instances_from_prob(prob, disk, post))
    import numpy as np

    info = _PICKER["info"]
    labels, ids, table = make_pieces(prob, disk, load_x(cache, stem), stem, post)
    if len(ids) == 0:
        return stem, []
    scores = _PICKER["model"].predict(table[:, _PICKER["cols"]])
    keep = keep_by_score(scores, info["tau"], info.get("max_instances", 60))
    return stem, labels_to_counts(relabel(labels, np.sort(ids[keep])))


def main() -> None:
    args = parse_args()
    rows, empty, t0 = [], [], time.time()
    params_path = Path(args.params) if args.params else Path(args.run) / "postprocess.json"
    post = PostConfig.load(params_path) if params_path.exists() else PostConfig()
    print(f"post-processing: {post}")

    if args.run:
        from solarseg.data import list_images, resolve_paths
        from solarseg.infer import get_device, load_checkpoint, predict_image

        run = Path(args.run)
        device = get_device(args.device)
        model, cfg = load_checkpoint(args.checkpoint or run / "best.pt", device)
        tta = args.tta if args.tta is not None else tuned_tta(params_path)
        print(f"test-time augmentation: {tta or 'off'}")
        folder = Path(args.images) if args.images else resolve_paths(args.data_root).test_images
        if folder is None:
            sys.exit("No test_images folder found; pass --images")
        images = list_images(folder)
        images = images[: args.limit] if args.limit else images
        for i, path in enumerate(images, 1):
            prob, disk = predict_image(
                model, path, device, cfg.get("scale", 0.5),
                tile=cfg.get("tile", 512), overlap=cfg.get("overlap", 128), tta=tta, channels=cfg["channels"],
            )
            counts = labels_to_counts(instances_from_prob(prob, disk, post))
            if not counts:
                empty.append(path.stem)
            # the id is the image file name without its extension, then a running number
            rows.extend((f"{path.stem}_{k}", c) for k, c in enumerate(counts, 1))
            if i % 20 == 0 or i == len(images):
                print(f"  {i}/{len(images)} images, {len(rows)} filaments ({time.time() - t0:.0f}s)", flush=True)
        source = {"checkpoint": str(args.checkpoint or run / "best.pt"), "tta": tta}
        n_images = len(images)
    else:
        from solarseg.cache import read_meta

        meta = read_meta(args.cache)
        shape = tuple(meta["image_size"])
        folds = sorted(d for d in (Path(args.probs) / "test").iterdir() if d.is_dir())
        stems = meta["test"][: args.limit] if args.limit else meta["test"]
        tasks = []
        for stem in stems:
            maps = [str(d / f"{stem}.npy") for d in folds if (d / f"{stem}.npy").exists()]
            if not maps:
                sys.exit(f"no saved test map for {stem}")
            tasks.append((stem, maps, args.cache, shape, post))
        print(f"averaging {len(folds)} fold maps per image; selection: "
              f"{'LightGBM picker' if args.picker else 'low/high threshold'}", flush=True)
        ctx = mp.get_context("spawn")
        with ctx.Pool(max(1, args.workers), initializer=_init_worker, initargs=(args.picker,)) as pool:
            for i, (stem, counts) in enumerate(pool.imap(_from_saved, tasks), 1):
                if not counts:
                    empty.append(stem)
                rows.extend((f"{stem}_{k}", c) for k, c in enumerate(counts, 1))
                if i % 20 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} images, {len(rows)} filaments ({time.time() - t0:.0f}s)", flush=True)
        source = {"folds": [d.name for d in folds], "picker": args.picker}
        n_images = len(tasks)

    if not rows:
        sys.exit("No filaments predicted at all; check the threshold before submitting.")
    n = write_submission(rows, args.out)
    summary = {"images": n_images, "rows": n, "images_without_detections": empty, "postprocess": vars(post), **source}
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}: {n} rows for {n_images} images ({len(empty)} with no detections)")
    print(f"next: python scripts/check_submission.py --csv {args.out}")


if __name__ == "__main__":
    main()
