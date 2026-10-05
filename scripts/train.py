#!/usr/bin/env python
"""Train the U-Net baseline.

Typical run on a Kaggle GPU notebook (about 30-45 minutes):

    python scripts/train.py --out runs/baseline

Quick smoke test on CPU:

    python scripts/train.py --out runs/smoke --epochs 1 --steps-per-epoch 5 --max-images 8 --workers 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from solarseg.data import (  # noqa: E402
    group_by_stem,
    load_records,
    preprocess,
    read_image,
    resize_target,
    resolve_paths,
    soft_target,
    split_stems,
)
from solarseg.dataset import CropDataset  # noqa: E402
from solarseg.infer import get_device, predict_prob  # noqa: E402
from solarseg.losses import BCEDiceLoss  # noqa: E402
from solarseg.model import build_model, count_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=None, help="folder with train/ and test/ (auto-detected on Kaggle)")
    p.add_argument("--out", default="runs/baseline", help="run folder for checkpoints and logs")
    p.add_argument("--scale", type=float, default=0.5, help="working resolution = 2048 x scale")
    p.add_argument("--crop", type=int, default=512, help="training crop size at working resolution")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base", type=int, default=32, help="U-Net width")
    p.add_argument("--depth", type=int, default=4, help="U-Net downsampling steps")
    p.add_argument("--pos-weight", type=float, default=3.0, help="BCE weight on filament pixels")
    p.add_argument("--pos-frac", type=float, default=0.7, help="share of crops centered on a filament")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--all-data", action="store_true", help="train on every image, no validation split")
    p.add_argument("--max-images", type=int, default=0, help="debug: use only the first N images")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--tile", type=int, default=512, help="sliding-window tile for validation")
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def load_stems(stems, groups, scale, label="train"):
    images, targets = [], []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        recs = groups[stem]
        img = read_image(recs[0].image_path)
        x, _ = preprocess(img, scale)
        y = resize_target(soft_target(recs), scale)
        images.append(x.astype(np.float16))
        targets.append(y.astype(np.float16))
        if i % 50 == 0 or i == len(stems):
            print(f"  {label}: loaded {i}/{len(stems)} images ({time.time() - t0:.0f}s)", flush=True)
    return images, targets


@torch.no_grad()
def validate(model, images, targets, device, tile, overlap):
    model.eval()
    inter = denom = 0.0
    for x, y in zip(images, targets):
        prob = predict_prob(model, x.astype(np.float32), device, tile=tile, overlap=overlap)
        pred = prob >= 0.5
        gt = y.astype(np.float32) >= 0.5
        inter += float(np.logical_and(pred, gt).sum())
        denom += float(pred.sum() + gt.sum())
    return 2 * inter / denom if denom else 1.0


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    use_amp = device.type == "cuda"
    print(f"device={device}")

    paths = resolve_paths(args.data_root)
    records, report = load_records(paths.annotations, paths.train_images)
    print(json.dumps(report, indent=2))
    groups = group_by_stem(records)
    stems = list(groups)
    if args.max_images:
        stems = stems[: args.max_images]
    if args.all_data:
        train_stems, val_stems = stems, []
    else:
        train_stems, val_stems = split_stems(stems, args.val_frac, args.seed)
    (out / "split.json").write_text(json.dumps({"train": train_stems, "val": val_stems}, indent=1))
    print(f"train images: {len(train_stems)}  validation images: {len(val_stems)}")

    train_x, train_y = load_stems(train_stems, groups, args.scale, "train")
    val_x, val_y = load_stems(val_stems, groups, args.scale, "val") if val_stems else ([], [])

    config = {k: v for k, v in vars(args).items()}
    model = build_model(base=args.base, depth=args.depth).to(device)
    print(f"model parameters: {count_parameters(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = BCEDiceLoss(pos_weight=args.pos_weight).to(device)

    ds = CropDataset(
        train_x,
        train_y,
        crop=args.crop,
        samples_per_epoch=args.steps_per_epoch * args.batch_size,
        pos_frac=args.pos_frac,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=use_amp,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    best = -1.0
    log_path = out / "metrics.jsonl"
    log_path.write_text("")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.item()))

        row = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5), "lr": opt.param_groups[0]["lr"]}
        if val_x and (epoch % args.val_every == 0 or epoch == args.epochs):
            row["val_dice"] = round(validate(model, val_x, val_y, device, args.tile, args.overlap), 5)
        row["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(row), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        ckpt = {"model": model.state_dict(), "config": config, "epoch": epoch, "metrics": row}
        torch.save(ckpt, out / "last.pt")
        if not val_x:
            torch.save(ckpt, out / "best.pt")  # no validation: keep the latest weights
        elif "val_dice" in row and row["val_dice"] > best:
            best = row["val_dice"]
            torch.save(ckpt, out / "best.pt")

    print(f"done. best validation Dice: {best:.4f}" if val_x else "done (trained on all data).")
    print(f"next: python scripts/tune.py --run {out}")


if __name__ == "__main__":
    main()
