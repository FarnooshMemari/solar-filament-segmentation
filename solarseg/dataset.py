"""Random training crops, biased toward crops that contain filaments."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class CropDataset(Dataset):
    def __init__(
        self,
        images: list[np.ndarray],
        targets: list[np.ndarray],
        crop: int = 512,
        samples_per_epoch: int = 1600,
        pos_frac: float = 0.7,
        augment: bool = True,
    ):
        assert len(images) == len(targets) and images
        self.images = images
        self.targets = targets
        self.crop = crop
        self.samples_per_epoch = samples_per_epoch
        self.pos_frac = pos_frac
        self.augment = augment
        # a few thousand filament pixels per image, used to center positive crops
        rng = np.random.default_rng(0)
        self.positives = []
        for t in targets:
            ys, xs = np.nonzero(t >= 0.5)
            if len(ys) > 4000:
                pick = rng.choice(len(ys), 4000, replace=False)
                ys, xs = ys[pick], xs[pick]
            self.positives.append((ys, xs))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, i: int):
        rng = np.random.default_rng((torch.initial_seed() + i) % 2**32)
        k = int(rng.integers(len(self.images)))
        img, tgt = self.images[k], self.targets[k]  # img is (H, W) or (C, H, W)
        h, w = img.shape[-2:]
        ch, cw = min(self.crop, h), min(self.crop, w)
        ys, xs = self.positives[k]
        if len(ys) and rng.random() < self.pos_frac:
            j = int(rng.integers(len(ys)))
            jitter = self.crop // 4
            cy = int(ys[j]) + int(rng.integers(-jitter, jitter + 1))
            cx = int(xs[j]) + int(rng.integers(-jitter, jitter + 1))
            y0 = int(np.clip(cy - ch // 2, 0, h - ch))
            x0 = int(np.clip(cx - cw // 2, 0, w - cw))
        else:
            y0 = int(rng.integers(0, h - ch + 1))
            x0 = int(rng.integers(0, w - cw + 1))
        x = img[..., y0 : y0 + ch, x0 : x0 + cw].astype(np.float32)
        y = tgt[y0 : y0 + ch, x0 : x0 + cw].astype(np.float32)

        if self.augment:
            if rng.random() < 0.5:
                x, y = x[..., :, ::-1], y[:, ::-1]
            if rng.random() < 0.5:
                x, y = x[..., ::-1, :], y[::-1, :]
            if ch == cw:
                k90 = int(rng.integers(4))
                x, y = np.rot90(x, k90, axes=(-2, -1)), np.rot90(y, k90)
            # brightness / contrast jitter on the normalized image
            x = x * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.15, 0.15))
            if rng.random() < 0.3:
                x = x + rng.normal(0, 0.05, x.shape).astype(np.float32)

        x = np.ascontiguousarray(x)
        if x.ndim == 2:
            x = x[None]
        y = np.ascontiguousarray(y)[None]
        return torch.from_numpy(x), torch.from_numpy(y)
