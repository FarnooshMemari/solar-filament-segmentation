#!/usr/bin/env python
"""Save side-by-side figures for the README and the report:
image | annotator outlines (green) | predicted filaments (one color each).

    python scripts/visualize.py --run runs/baseline --n 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from solarseg.data import group_by_stem, load_records, read_image, record_union_mask, resolve_paths  # noqa: E402
from solarseg.infer import get_device, load_checkpoint, predict_image  # noqa: E402
from solarseg.postprocess import PostConfig, instances_from_prob, tuned_tta  # noqa: E402

PALETTE = [(255, 99, 71), (65, 105, 225), (255, 215, 0), (186, 85, 211), (0, 206, 209), (255, 140, 0)]


def outline(canvas: np.ndarray, mask: np.ndarray, color, thickness: int) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, color, thickness)


def label(panel: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(panel, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--params", default=None, help="post-processing JSON; defaults to <run>/postprocess.json")
    p.add_argument("--data-root", default=None)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--panel", type=int, default=768, help="output size of each panel in pixels")
    p.add_argument("--tta", type=int, nargs="?", const=4, default=None,
                   help="defaults to what the post-processing was tuned with")
    p.add_argument("--out", default=None, help="defaults to <run>/figures")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    run = Path(args.run)
    out = Path(args.out) if args.out else run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint or run / "best.pt", device)
    post_path = Path(args.params) if args.params else run / "postprocess.json"
    post = PostConfig.load(post_path) if post_path.exists() else PostConfig()
    tta = args.tta if args.tta is not None else tuned_tta(post_path)

    split = json.loads((run / "split.json").read_text())
    stems = split["val"] or split["train"]
    paths = resolve_paths(args.data_root)
    records, _ = load_records(paths.annotations, paths.train_images)
    groups = group_by_stem(records)

    for stem in stems[: args.n]:
        recs = groups[stem]
        img = read_image(recs[0].image_path)
        prob, disk = predict_image(model, recs[0].image_path, device, cfg.get("scale", 0.5),
                                   tile=cfg.get("tile", 512), overlap=cfg.get("overlap", 128), tta=tta,
                                   channels=cfg["channels"])
        labels = instances_from_prob(prob, disk, post)
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gt_panel, pred_panel = base.copy(), base.copy()
        thickness = max(2, img.shape[0] // 512)
        outline(gt_panel, record_union_mask(recs[0]), (0, 200, 0), thickness)
        for k in range(1, int(labels.max()) + 1):
            outline(pred_panel, labels == k, PALETTE[(k - 1) % len(PALETTE)], thickness)
        size = (args.panel, args.panel)
        panels = [
            label(cv2.resize(base, size, interpolation=cv2.INTER_AREA), stem),
            label(cv2.resize(gt_panel, size, interpolation=cv2.INTER_AREA), f"annotator {recs[0].annotator}"),
            label(cv2.resize(pred_panel, size, interpolation=cv2.INTER_AREA), f"predicted: {int(labels.max())}"),
        ]
        path = out / f"{stem}.png"
        cv2.imwrite(str(path), np.hstack(panels))
        print("saved", path)


if __name__ == "__main__":
    main()
