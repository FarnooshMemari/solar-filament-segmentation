#!/usr/bin/env python
"""Save probability maps of one trained fold: its held-out (out-of-fold) training images
and, with --test, the test images. Maps are stored at the working resolution as float16.

    python scripts/oof.py --run runs/fold0 --cache /kaggle/temp/cache --out /kaggle/temp/probs --test --tta 8

Writes <out>/oof/<stem>.npy and <out>/test/<run name>/<stem>.npy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from solarseg.cache import load_x, read_meta  # noqa: E402
from solarseg.infer import get_device, load_checkpoint, predict_prob  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default=None, help="defaults to <run>/best.pt")
    p.add_argument("--test", action="store_true", help="also predict the test images")
    p.add_argument("--tta", type=int, default=8)
    p.add_argument("--tile", type=int, default=1024, help="1024 = whole image at once at scale 0.5")
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def run_set(model, device, cache, stems, folder: Path, args, label: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        x = np.asarray(load_x(cache, stem), dtype=np.float32)
        prob = predict_prob(model, x, device, tile=args.tile, overlap=args.overlap, tta=args.tta)
        np.save(folder / f"{stem}.npy", prob.astype(np.float16))
        if i % 50 == 0 or i == len(stems):
            print(f"  {label}: {i}/{len(stems)} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    args = parse_args()
    run = Path(args.run)
    device = get_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint or run / "best.pt", device)
    meta = read_meta(args.cache)
    if tuple(meta["channels"]) != tuple(cfg["channels"]):
        sys.exit(f"cache channels {meta['channels']} differ from the model's {cfg['channels']}")
    split = json.loads((run / "split.json").read_text())
    out = Path(args.out)
    run_set(model, device, args.cache, split["val"], out / "oof", args, f"{run.name} held-out")
    if args.test:
        run_set(model, device, args.cache, meta["test"], out / "test" / run.name, args, f"{run.name} test")
    (out / "settings.json").write_text(json.dumps({"tta": args.tta, "tile": args.tile, "overlap": args.overlap}))
    print(f"saved probability maps to {out}")


if __name__ == "__main__":
    main()
