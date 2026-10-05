#!/usr/bin/env python
"""Pick post-processing settings (threshold, min area, fragment merging) that maximize
Panoptic Quality on the validation images, and save them to <run>/postprocess.json.

    python scripts/tune.py --run runs/baseline
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from solarseg.data import group_by_stem, load_records, record_instance_rles, resolve_paths  # noqa: E402
from solarseg.infer import get_device, load_checkpoint, predict_image  # noqa: E402
from solarseg.metrics import PQStats, match  # noqa: E402
from solarseg.postprocess import PostConfig, filter_components, label_components  # noqa: E402
from solarseg.rle import counts_to_rle, labels_to_counts  # noqa: E402


def floats(s: str) -> list[float]:
    return [float(v) for v in s.split(",") if v.strip()]


def ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default=None, help="defaults to <run>/best.pt")
    p.add_argument("--data-root", default=None)
    p.add_argument("--thresholds", default="0.3,0.4,0.5,0.6")
    p.add_argument("--min-areas", default="50,150,400")
    p.add_argument("--merge-dists", default="0,10,25")
    p.add_argument("--tta", action="store_true", help="average 4 flips (slower, often better)")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run = Path(args.run)
    device = get_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint or run / "best.pt", device)
    split = json.loads((run / "split.json").read_text())
    val_stems = split["val"]
    if not val_stems:
        sys.exit("This run has no validation images (trained with --all-data). Tune on a run with a split.")
    if args.max_images:
        val_stems = val_stems[: args.max_images]

    paths = resolve_paths(args.data_root)
    records, _ = load_records(paths.annotations, paths.train_images)
    groups = group_by_stem(records)

    print(f"predicting {len(val_stems)} validation images...", flush=True)
    t0 = time.time()
    cache = {}
    for i, stem in enumerate(val_stems, 1):
        recs = groups[stem]
        prob, disk = predict_image(
            model, recs[0].image_path, device, cfg.get("scale", 0.5),
            tile=cfg.get("tile", 512), overlap=cfg.get("overlap", 128), tta=args.tta,
        )
        gts = [record_instance_rles(r) for r in recs]
        cache[stem] = (prob.astype(np.float16), disk, gts)
        if i % 20 == 0 or i == len(val_stems):
            print(f"  {i}/{len(val_stems)} ({time.time() - t0:.0f}s)", flush=True)

    thresholds, areas, merges = floats(args.thresholds), ints(args.min_areas), ints(args.merge_dists)
    print(f"scoring {len(thresholds) * len(areas) * len(merges)} settings...", flush=True)
    totals = {key: PQStats() for key in itertools.product(thresholds, areas, merges)}
    for prob, disk, gts in cache.values():
        for thr, merge in itertools.product(thresholds, merges):
            raw = label_components(prob, disk, thr, merge)  # shared by every min_area value
            for area in areas:
                labels = filter_components(raw, area)
                h, w = labels.shape
                preds = [counts_to_rle(c, h, w) for c in labels_to_counts(labels)]
                for gt in gts:  # pooled over every annotator of the image
                    totals[(thr, area, merge)] += match(preds, gt)

    results = []
    for (thr, area, merge), stats in totals.items():
        results.append((PostConfig(threshold=thr, min_area=area, merge_dist=merge), stats))
        print(f"  thr={thr:.2f} min_area={area:4d} merge={merge:3d} -> {stats.as_dict()}", flush=True)

    results.sort(key=lambda r: r[1].pq, reverse=True)
    best_post, best_stats = results[0]
    extra = {**best_stats.as_dict(), "images": len(cache), "tta": args.tta}
    best_post.save(run / "postprocess.json", extra=extra)
    print("\nTop settings:")
    for post, stats in results[:5]:
        print(f"  PQ={stats.pq:.4f} SQ={stats.sq:.4f} RQ={stats.rq:.4f}  "
              f"thr={post.threshold} min_area={post.min_area} merge={post.merge_dist}")
    print(f"\nsaved {run / 'postprocess.json'}")
    print(f"next: python scripts/predict.py --run {run}" + (" --tta" if args.tta else ""))


if __name__ == "__main__":
    main()
