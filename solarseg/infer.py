"""Sliding-window inference and the full image -> probability map path."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .data import preprocess, read_image
from .model import build_model


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_model(base=cfg.get("base", 32), depth=cfg.get("depth", 4)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def _starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def _window(h: int, w: int) -> np.ndarray:
    """Blend weights that fade toward tile edges, so overlapping tiles stitch smoothly."""
    wy = np.hanning(h + 2)[1:-1] if h > 2 else np.ones(h)
    wx = np.hanning(w + 2)[1:-1] if w > 2 else np.ones(w)
    return np.maximum(np.outer(wy, wx), 1e-3).astype(np.float32)


@torch.no_grad()
def predict_prob(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    tile: int = 512,
    overlap: int = 128,
    tta: bool = False,
    batch_size: int = 4,
) -> np.ndarray:
    """Normalized image at working resolution -> filament probability (same size)."""
    h, w = x.shape
    th, tw = min(tile, h), min(tile, w)
    stride_y, stride_x = max(1, th - overlap), max(1, tw - overlap)
    coords = [(y, xx) for y in _starts(h, th, stride_y) for xx in _starts(w, tw, stride_x)]
    acc = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    win = _window(th, tw)
    use_amp = device.type == "cuda"

    for b in range(0, len(coords), batch_size):
        chunk = coords[b : b + batch_size]
        batch = np.stack([x[y : y + th, xx : xx + tw] for y, xx in chunk])[:, None]
        inp = torch.from_numpy(batch).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            prob = torch.sigmoid(model(inp).float())
            if tta:
                prob = prob + torch.sigmoid(model(inp.flip(-1)).float()).flip(-1)
                prob = prob + torch.sigmoid(model(inp.flip(-2)).float()).flip(-2)
                prob = prob + torch.sigmoid(model(inp.flip(-1, -2)).float()).flip(-1, -2)
                prob = prob / 4.0
        prob = prob[:, 0].float().cpu().numpy()
        for (y, xx), p in zip(chunk, prob):
            acc[y : y + th, xx : xx + tw] += p * win
            weight[y : y + th, xx : xx + tw] += win
    return acc / np.maximum(weight, 1e-6)


def predict_image(
    model: torch.nn.Module,
    image_path: str | Path,
    device: torch.device,
    scale: float,
    tile: int = 512,
    overlap: int = 128,
    tta: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Image file -> (full-resolution probability map, full-resolution disk mask)."""
    img = read_image(image_path)
    x, disk = preprocess(img, scale)
    prob = predict_prob(model, x, device, tile=tile, overlap=overlap, tta=tta)
    if prob.shape != img.shape:
        prob = cv2.resize(prob, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return prob, disk
