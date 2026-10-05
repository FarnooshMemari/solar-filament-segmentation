#!/usr/bin/env python
"""Train the filament picker: a LightGBM model that predicts, for every candidate piece, its
mean matched IoU over the annotators, from out-of-fold probability maps.

    python scripts/scorer.py --probs /kaggle/temp/probs --cache /kaggle/temp/cache \\
        --runs runs/fold0 runs/fold1 runs/fold2 runs/fold3 --post runs/run3/postprocess.json --out runs/run3

Pieces are formed with the tuned low threshold and fragment merging from --post. The model
is cross-fitted with the same folds as the networks, the keep threshold is chosen on those
out-of-fold scores, and a final model is trained on every piece. Writes picker.txt (the
LightGBM model), picker.json (features, threshold and out-of-fold results) and
oof_pieces.npz (every piece's features, matched IoUs and out-of-fold score).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from solarseg.pieces import best_tau, feature_names, keep_by_score, selection_pq, station_of  # noqa: E402
from solarseg.postprocess import PostConfig  # noqa: E402

LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "num_threads": 4,
}
ROUNDS = 400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probs", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="fold run folders (for each image's fold)")
    p.add_argument("--post", required=True, help="postprocess.json from scripts/tune.py")
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def _init_worker() -> None:
    import cv2

    cv2.setNumThreads(1)


def _work(task):
    from solarseg.cache import load_disk, load_prob, load_x
    from solarseg.pieces import make_pieces, matched_ious
    from solarseg.rle import labels_to_rles

    prob_path, cache, stem, shape, gts, post = task
    prob = load_prob(prob_path, shape)
    disk = load_disk(cache, stem, shape)
    labels, ids, table = make_pieces(prob, disk, load_x(cache, stem), stem, post)
    rle_ids, rles = labels_to_rles(labels)
    assert np.array_equal(rle_ids, ids)
    hits, n_gt = matched_ious(rles, gts)
    return stem, table, hits, n_gt


def main() -> None:
    args = parse_args()
    import lightgbm as lgb

    from solarseg.cache import read_meta
    from solarseg.data import group_by_stem, load_records, record_instance_rles, resolve_paths

    t0 = time.time()
    meta = read_meta(args.cache)
    shape = tuple(meta["image_size"])
    names = feature_names(meta["channels"])  # every column of piece_table
    # the observing site is read from the GONG frame id; leave it out if test names lack one
    site_known = all(station_of(s) >= 0 for s in meta.get("test", []))
    used = [n for n in names if n != "station" or site_known]
    cols = [names.index(n) for n in used]
    post = PostConfig.load(args.post)
    fold_of = {}
    for k, run in enumerate(args.runs):
        for stem in json.loads((Path(run) / "split.json").read_text())["val"]:
            fold_of[stem] = k
    paths = resolve_paths(args.data_root)
    groups = group_by_stem(load_records(paths.annotations, paths.train_images)[0])
    files = sorted(f for f in (Path(args.probs) / "oof").glob("*.npy") if f.stem in fold_of)
    tasks = [(str(f), args.cache, f.stem, shape, [record_instance_rles(r) for r in groups[f.stem]], post)
             for f in files]
    print(f"measuring pieces on {len(tasks)} out-of-fold maps (low={post.low}, merge={post.merge_dist})...",
          flush=True)
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        rows = []
        for j, res in enumerate(pool.imap(_work, tasks, chunksize=2), 1):
            rows.append(res)
            if j % 100 == 0 or j == len(tasks):
                print(f"  {j}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)

    stems = [r[0] for r in rows]
    tables = [r[1] for r in rows]
    hits = [r[2] for r in rows]
    n_gts = [r[3] for r in rows]
    folds = np.array([fold_of[s] for s in stems])
    targets = [h.mean(axis=1) if h.shape[1] else np.zeros(h.shape[0], np.float32) for h in hits]
    n_pieces = sum(len(t) for t in tables)
    print(f"{n_pieces} pieces, {sum(int((t > 0).sum()) for t in targets)} of them match at least one annotator")

    # reference: the tuned hysteresis rule applied to the same pieces (largest first, as in
    # solarseg.postprocess.select_components, when an image has too many)
    i_peak, i_area = names.index("peak"), names.index("log_area")
    rule = []
    for t in tables:
        area = np.expm1(t[:, i_area].astype(np.float64))
        ok = (t[:, i_peak] >= post.threshold) & (area >= post.min_area - 0.5)
        rule.append(keep_by_score(np.where(ok, 1.0 + area / 1e9, 0.0), 0.5, post.max_instances))
    rule_pq = selection_pq(hits, n_gts, rule)
    all_pq = selection_pq(hits, n_gts, [np.ones(len(t), bool) for t in tables])

    # cross-fitted scores: each fold is scored by a model that never saw its images
    scores = [np.zeros(len(t), np.float32) for t in tables]
    for k in sorted(set(folds)):
        tr = [i for i in range(len(rows)) if folds[i] != k]
        te = [i for i in range(len(rows)) if folds[i] == k]
        if not tr:
            sys.exit("the picker needs at least two folds")
        X = np.concatenate([tables[i][:, cols] for i in tr])
        y = np.concatenate([targets[i] for i in tr])
        model = lgb.train(LGB_PARAMS, lgb.Dataset(X, y, feature_name=used), num_boost_round=ROUNDS)
        for i in te:
            if len(tables[i]):
                scores[i] = model.predict(tables[i][:, cols]).astype(np.float32)
    tau, picked, curve = best_tau(scores, hits, n_gts)
    pq_now = picked.pq
    oracle = selection_pq(hits, n_gts, [t > pq_now / 2 for t in targets])

    X_all = np.concatenate([t[:, cols] for t in tables])
    final = lgb.train(LGB_PARAMS, lgb.Dataset(X_all, np.concatenate(targets), feature_name=used),
                      num_boost_round=ROUNDS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final.save_model(str(out / "picker.txt"))
    gain = final.feature_importance("gain")
    importance = {n: round(float(g) / max(float(gain.sum()), 1e-9), 4)
                  for n, g in sorted(zip(used, gain), key=lambda x: -x[1])}
    report = {
        "features": used,
        "table_features": names,
        "tau": tau,
        "piece_setting": {"low": post.low, "merge_dist": post.merge_dist, "close_radius": post.close_radius,
                          "fill_holes": post.fill_holes, "use_disk": post.use_disk},
        "max_instances": post.max_instances,
        "oof": {
            "picker": picked.as_dict(),
            "tuned_hysteresis_same_pieces": rule_pq.as_dict(),
            "keep_every_piece": all_pq.as_dict(),
            "oracle_upper_bound": oracle.as_dict(),
        },
        "tau_curve": [[t, round(st.pq, 5)] for t, st in curve],
        "images": len(rows),
        "pieces": n_pieces,
        "feature_importance": importance,
        "lightgbm": {**LGB_PARAMS, "rounds": ROUNDS},
    }
    (out / "picker.json").write_text(json.dumps(report, indent=1))

    # every piece with its features, matched IoUs and out-of-fold score, for later analysis
    counts = np.array([len(t) for t in tables])
    n_ann = max((h.shape[1] for h in hits), default=0)
    pad = lambda a, n: np.pad(a.astype(np.float32), ((0, 0), (0, n - a.shape[1])), constant_values=np.nan)  # noqa: E731
    np.savez_compressed(
        out / "oof_pieces.npz",
        stems=np.array(stems), pieces_per_image=counts, fold=folds, feature_names=np.array(names),
        features=np.concatenate(tables), matched_iou=np.concatenate([pad(h, n_ann) for h in hits]),
        n_gt=np.stack([np.pad(g.astype(np.float32), (0, n_ann - len(g)), constant_values=np.nan) for g in n_gts]),
        score=np.concatenate(scores), target=np.concatenate(targets),
    )
    print(f"\nout-of-fold PQ on {len(rows)} images:")
    print(f"  keep every piece:          {all_pq.pq:.4f}")
    print(f"  tuned low/high threshold:  {rule_pq.pq:.4f}  (SQ {rule_pq.sq:.3f}, RQ {rule_pq.rq:.3f})")
    print(f"  LightGBM picker (tau={tau:.2f}): {picked.pq:.4f}  (SQ {picked.sq:.3f}, RQ {picked.rq:.3f})")
    print(f"  oracle upper bound:        {oracle.pq:.4f}")
    print("top features:", ", ".join(f"{k} {v:.2f}" for k, v in list(importance.items())[:6]))
    print(f"saved {out / 'picker.txt'} and {out / 'picker.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
