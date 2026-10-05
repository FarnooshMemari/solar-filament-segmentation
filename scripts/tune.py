#!/usr/bin/env python
"""Pick post-processing settings that maximize Panoptic Quality on the validation images and
save them to <run>/postprocess.json.

    python scripts/tune.py --run runs/baseline --tta 8 --workers 4

Searched: low threshold, high threshold (hysteresis), gap closing, hole filling, fragment
merging and minimum area. See solarseg/tuning.py for how a large grid stays fast.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarseg.postprocess import PostConfig  # noqa: E402
from solarseg.tuning import Grid, add_up, config_to_key, key_to_config, score_image  # noqa: E402


def floats(s: str) -> tuple[float, ...]:
    return tuple(float(v) for v in s.split(",") if v.strip())


def ints(s: str) -> tuple[int, ...]:
    return tuple(int(v) for v in s.split(",") if v.strip())


def parse_args() -> argparse.Namespace:
    g = Grid()
    join = lambda xs: ",".join(str(int(x) if isinstance(x, bool) else x) for x in xs)  # noqa: E731
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default=None, help="defaults to <run>/best.pt")
    p.add_argument("--data-root", default=None)
    p.add_argument("--thresholds", default=join(g.thresholds), help="high thresholds")
    p.add_argument("--low-thresholds", default=join(g.low_thresholds))
    p.add_argument("--min-areas", default=join(g.min_areas))
    p.add_argument("--merge-dists", default=join(g.merge_dists))
    p.add_argument("--close-radii", default=join(g.close_radii))
    p.add_argument("--fill-holes", default=join(g.fill_holes), help="0, 1 or 0,1")
    p.add_argument("--max-instances", type=int, default=g.max_instances)
    p.add_argument("--tta", type=int, nargs="?", const=4, default=0,
                   help="average 4 flips (--tta) or 8 flips/rotations (--tta 8)")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                   help="CPU processes for scoring (0 = run in this process)")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def _init_worker() -> None:
    import cv2

    cv2.setNumThreads(1)


def _score(task):
    prob, disk, gts, grid = task
    return score_image(prob, disk, gts, grid)


def main() -> None:
    args = parse_args()
    import numpy as np

    from solarseg.data import group_by_stem, load_records, record_instance_rles, resolve_paths
    from solarseg.infer import get_device, load_checkpoint, predict_image

    run = Path(args.run)
    device = get_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint or run / "best.pt", device)
    split = json.loads((run / "split.json").read_text())
    val_stems = split["val"]
    if not val_stems:
        sys.exit("This run has no validation images (trained with --all-data). Tune on a run with a split.")
    if args.max_images:
        val_stems = val_stems[: args.max_images]

    grid = Grid(
        thresholds=floats(args.thresholds),
        low_thresholds=floats(args.low_thresholds),
        min_areas=ints(args.min_areas),
        merge_dists=ints(args.merge_dists),
        close_radii=ints(args.close_radii),
        fill_holes=tuple(bool(v) for v in ints(args.fill_holes)),
        max_instances=args.max_instances,
    )
    post_path = run / "postprocess.json"
    previous = PostConfig.load(post_path) if post_path.exists() else None
    if previous is not None:
        shutil.copy(post_path, run / "postprocess_previous.json")

    paths = resolve_paths(args.data_root)
    records, _ = load_records(paths.annotations, paths.train_images)
    groups = group_by_stem(records)

    print(f"predicting {len(val_stems)} validation images (tta={args.tta}), "
          f"scoring {grid.size()} settings with {args.workers} worker(s)...", flush=True)
    t0 = time.time()
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) if args.workers > 0 else None
    pending, results = [], []
    for i, stem in enumerate(val_stems, 1):
        recs = groups[stem]
        prob, disk = predict_image(
            model, recs[0].image_path, device, cfg.get("scale", 0.5),
            tile=cfg.get("tile", 512), overlap=cfg.get("overlap", 128), tta=args.tta,
        )
        task = (prob.astype(np.float16), disk, [record_instance_rles(r) for r in recs], grid)
        if pool is None:
            results.append(_score(task))
        else:
            pending.append(pool.apply_async(_score, (task,)))
        if i % 20 == 0 or i == len(val_stems):
            print(f"  predicted {i}/{len(val_stems)} ({time.time() - t0:.0f}s)", flush=True)
    if pool is not None:
        for j, job in enumerate(pending, 1):
            results.append(job.get())
            if j % 20 == 0 or j == len(pending):
                print(f"  scored {j}/{len(pending)} ({time.time() - t0:.0f}s)", flush=True)
        pool.close()
        pool.join()

    totals = add_up(results)
    ranked = sorted(totals.items(), key=lambda kv: kv[1].pq, reverse=True)
    best_key, best_stats = ranked[0]
    best = key_to_config(best_key, grid.max_instances)

    def describe(key) -> str:
        low, close, fill, merge, high, area = key
        lo = f"low={low:.2f} " if low != high else ""
        return f"thr={high:.2f} {lo}min_area={area} merge={merge} close={close} fill={int(fill)}"

    print(f"\nTop {args.top} of {len(ranked)} settings:")
    for key, st in ranked[: args.top]:
        print(f"  PQ={st.pq:.4f} SQ={st.sq:.4f} RQ={st.rq:.4f} TP={st.tp} FP={st.fp} FN={st.fn}  {describe(key)}")
    if previous is not None:
        prev = totals.get(config_to_key(previous))
        if prev is not None:
            print(f"\nprevious settings ({describe(config_to_key(previous))}): PQ={prev.pq:.4f}")

    plain = [st for key, st in totals.items() if key[0] == key[4] and key[1] == 0 and not key[2]]
    if plain:
        print(f"best with a single threshold, no closing or hole filling: PQ={max(s.pq for s in plain):.4f}")

    extra = {**best_stats.as_dict(), "images": len(results), "tta": args.tta, "settings_searched": len(ranked)}
    best.save(post_path, extra=extra)
    with open(run / "tuning_results.json", "w") as f:
        json.dump([{"setting": describe(k), **st.as_dict()} for k, st in ranked], f, indent=1)
    print(f"\nbest: {describe(best_key)}  PQ={best_stats.pq:.4f}")
    print(f"saved {post_path} and {run / 'tuning_results.json'} ({time.time() - t0:.0f}s)")
    print(f"next: python scripts/predict.py --run {run}")


if __name__ == "__main__":
    main()
