#!/usr/bin/env python
"""Summarize the out-of-fold results of the k-fold setup from the files scripts/scorer.py
saves (oof_pieces.npz, picker.json and postprocess.json).

    python scripts/oof_report.py --run runs/run3

For the tuned low/high threshold rule and for the LightGBM picker it prints the pooled PQ on
every out-of-fold image, on each fold, and on the validation images of runs 1 and 2 (the
month split made with --val-frac 0.15 --seed 42), so the runs can be compared on exactly the
same held-out images. It also shows how well the matched filaments are outlined (IoU
quantiles) and how many filaments are predicted per image. Saves <run>/oof_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from solarseg.data import split_stems  # noqa: E402
from solarseg.metrics import PQStats  # noqa: E402
from solarseg.pieces import keep_by_score, selection_pq  # noqa: E402
from solarseg.postprocess import PostConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="folder with oof_pieces.npz, picker.json and postprocess.json")
    p.add_argument("--val-frac", type=float, default=0.15, help="validation share of the run 1/2 split")
    p.add_argument("--seed", type=int, default=42, help="seed of the run 1/2 split")
    p.add_argument("--out", default=None, help="defaults to <run>/oof_report.json")
    return p.parse_args()


def load_images(run: Path) -> list[dict]:
    """One dict per out-of-fold image: piece features, matched IoUs per annotator, scores."""
    z = np.load(run / "oof_pieces.npz")
    bounds = np.concatenate([[0], np.cumsum(z["pieces_per_image"])])
    images = []
    for i, stem in enumerate(z["stems"]):
        a, b = int(bounds[i]), int(bounds[i + 1])
        n_ann = int(np.sum(~np.isnan(z["n_gt"][i])))  # columns past the annotator count are padding
        images.append({
            "stem": str(stem),
            "fold": int(z["fold"][i]),
            "features": z["features"][a:b],
            "hits": np.nan_to_num(z["matched_iou"][a:b, :n_ann]),
            "n_gt": z["n_gt"][i, :n_ann].astype(np.int64),
            "score": z["score"][a:b],
        })
    return images, [str(n) for n in z["feature_names"]]


def describe(st: PQStats) -> str:
    return f"PQ {st.pq:.4f}  SQ {st.sq:.3f}  RQ {st.rq:.3f}  (TP {st.tp}, FP {st.fp}, FN {st.fn})"


def main() -> None:
    args = parse_args()
    run = Path(args.run)
    info = json.loads((run / "picker.json").read_text())
    post = PostConfig.load(run / "postprocess.json")
    images, names = load_images(run)
    i_peak, i_area = names.index("peak"), names.index("log_area")
    cap = info.get("max_instances", post.max_instances)

    def keep(img: dict, method: str) -> np.ndarray:
        t = img["features"]
        if method == "picker":
            return keep_by_score(img["score"], info["tau"], cap)
        area = np.expm1(t[:, i_area].astype(np.float64))
        ok = (t[:, i_peak] >= post.threshold) & (area >= post.min_area - 0.5)
        return keep_by_score(np.where(ok, 1.0 + area / 1e9, 0.0), 0.5, post.max_instances)

    def pooled(subset: list[dict], method: str) -> PQStats:
        return selection_pq([m["hits"] for m in subset], [m["n_gt"] for m in subset], [keep(m, method) for m in subset])

    _, run2_val = split_stems([m["stem"] for m in images], args.val_frac, args.seed)
    run2_val = set(run2_val)
    groups = {
        "all out-of-fold images": images,
        "run 1/2 validation images": [m for m in images if m["stem"] in run2_val],
    }
    for k in sorted({m["fold"] for m in images}):
        groups[f"fold {k}"] = [m for m in images if m["fold"] == k]

    report = {"settings": {"rule": vars(post), "picker_tau": info["tau"]}, "groups": {}}
    for method, label in (("rule", "low/high threshold rule"), ("picker", "LightGBM picker")):
        print(f"\n{label}:")
        for name, subset in groups.items():
            st = pooled(subset, method)
            report["groups"].setdefault(name, {"images": len(subset)})[method] = st.as_dict()
            print(f"  {name:<28} {len(subset):>4} images  {describe(st)}")

        # outline quality of the found filaments and the number of filaments per image
        ious, kept, annotated = [], [], []
        for m in images:
            k = keep(m, method)
            h = m["hits"][k]
            ious.extend(h[h > 0].tolist())
            kept.append(int(k.sum()))
            annotated.append(float(np.mean(m["n_gt"])) if len(m["n_gt"]) else 0.0)
        q = np.quantile(ious, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist() if ious else []
        report[method] = {"matched_iou_quantiles_10_25_50_75_90": q, "mean_predicted_per_image": float(np.mean(kept)),
                          "mean_annotated_per_image": float(np.mean(annotated))}
        print(f"  matched IoU quantiles (10/25/50/75/90%): {q}")
        print(f"  filaments per image: predicted {np.mean(kept):.1f}, "
              f"annotated {np.mean(annotated):.1f} (mean over annotators)")

    out = Path(args.out) if args.out else run / "oof_report.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
