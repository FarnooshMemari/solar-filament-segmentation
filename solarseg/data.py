"""Reading the competition data: COCO-style annotations, image files, masks and splits.

The loader is deliberately forgiving about the exact annotation layout, because the
same physical image can be labeled by several annotators. It handles:

* annotator ids stored on the image entry, on each annotation, or as a prefix of the
  file name or image id, such as ``A1-20150125172714Mh.jpg`` or ``040301-20140609195854Bh``;
* polygons in standard COCO form (``[[x1, y1, x2, y2, ...], ...]``), as a list of
  ``[x, y]`` pairs, or as COCO RLE.

Run ``python scripts/inspect_data.py`` first on the real data to confirm what it finds.
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage

IMAGE_SIZE = 2048  # every competition frame is 2048 x 2048
# GONG H-alpha frame name: 14 digits (YYYYMMDDhhmmss) + site letter + "h", e.g. 20150125172714Mh
STEM_RE = re.compile(r"(\d{14}[A-Za-z]h)")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
ANNOTATOR_KEYS = ("annotator", "annotator_id", "annotator_name", "labeler", "rater", "user_id")
DEFAULT_SEARCH_ROOTS = (Path("/kaggle/input"), Path("data"))


@dataclass
class DataPaths:
    root: Path
    train_images: Path
    test_images: Path | None
    annotations: Path


def _first_dir(base: Path, name: str) -> Path | None:
    for p in sorted(base.rglob(name)):
        if p.is_dir():
            return p
    return None


def find_data_root(explicit: str | os.PathLike | None = None) -> Path:
    """Locate the folder that contains ``train/`` and ``test/``.

    Order: ``--data-root`` argument, ``SOLARSEG_DATA`` environment variable, then a
    search under ``/kaggle/input`` and ``./data``.
    """
    if explicit:
        return Path(explicit)
    env = os.environ.get("SOLARSEG_DATA")
    if env:
        return Path(env)
    for base in DEFAULT_SEARCH_ROOTS:
        if base.exists():
            hit = _first_dir(base, "train_images")
            if hit is not None:
                return hit.parent.parent
    raise FileNotFoundError(
        "Could not find the competition data. Attach the competition to your Kaggle "
        "notebook, or pass --data-root /path/to/MAGFiLO_1.0_Kaggle_2026"
    )


def find_annotation_file(root: Path) -> Path:
    files = sorted(root.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No annotation JSON found under {root}")

    def rank(p: Path) -> tuple:
        name = p.name.lower()
        return ("annotation" not in name, "train" not in name, len(str(p)))

    return sorted(files, key=rank)[0]


def resolve_paths(data_root: str | os.PathLike | None = None) -> DataPaths:
    root = find_data_root(data_root)
    train_images = _first_dir(root, "train_images")
    if train_images is None:
        raise FileNotFoundError(f"No train_images folder under {root}")
    test_images = _first_dir(root, "test_images")
    return DataPaths(root, train_images, test_images, find_annotation_file(root))


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def image_stem(name: str | os.PathLike) -> str:
    """GONG frame id for a file name, e.g. ``A1-20150125172714Mh.jpg`` -> ``20150125172714Mh``."""
    base = os.path.basename(str(name))
    m = STEM_RE.search(base)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0].split("-")[-1]


def index_images(folder: Path) -> dict[str, Path]:
    """Map frame id -> image path for every image in a folder."""
    return {image_stem(p.name): p for p in list_images(folder)}


@dataclass
class Record:
    """One annotator's filaments for one physical image."""

    stem: str
    annotator: str
    image_path: Path
    height: int
    width: int
    segmentations: list = field(default_factory=list)  # one COCO segmentation per filament


def _annotator(entry: dict) -> str | None:
    for key in ANNOTATOR_KEYS:
        value = entry.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _prefix_annotator(file_name: str, stem: str) -> str | None:
    base = os.path.splitext(os.path.basename(file_name))[0]
    prefix = base.split(stem)[0].strip("-_ ") if stem in base else ""
    return prefix or None


def load_records(annotation_file: Path, images_dir: Path) -> tuple[list[Record], dict]:
    """Read the COCO-style JSON into one Record per (image, annotator).

    Returns the records and a small report (counts, missing files) for logging.
    """
    with open(annotation_file) as f:
        coco = json.load(f)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    index = index_images(images_dir)

    by_image: dict = defaultdict(list)
    for ann in annotations:
        by_image[ann.get("image_id")].append(ann)

    groups: dict[tuple[str, str], Record] = {}
    missing: list[str] = []
    for img in images:
        file_name = str(img.get("file_name") or img.get("filename") or img.get("id"))
        stem = image_stem(file_name)
        path = index.get(stem)
        if path is None:
            missing.append(file_name)
            continue
        default_annotator = (
            _annotator(img)
            or _prefix_annotator(file_name, stem)
            or _prefix_annotator(str(img.get("id", "")), stem)
            or f"img{img.get('id')}"
        )
        height = int(img.get("height") or IMAGE_SIZE)
        width = int(img.get("width") or IMAGE_SIZE)
        used = False
        for ann in by_image.get(img.get("id"), []):
            seg = ann.get("segmentation")
            if not seg:
                continue
            annotator = _annotator(ann) or default_annotator
            key = (stem, annotator)
            if key not in groups:
                groups[key] = Record(stem, annotator, path, height, width)
            groups[key].segmentations.append(seg)
            used = True
        if not used:  # an image entry with no filaments is still a valid (empty) label
            groups.setdefault((stem, default_annotator), Record(stem, default_annotator, path, height, width))

    records = sorted(groups.values(), key=lambda r: (r.stem, r.annotator))
    report = {
        "annotation_file": str(annotation_file),
        "images_in_json": len(images),
        "annotations_in_json": len(annotations),
        "records": len(records),
        "physical_images": len({r.stem for r in records}),
        "missing_image_files": len(missing),
        "missing_examples": missing[:5],
        "top_level_keys": sorted(coco.keys()),
    }
    return records, report


def group_by_stem(records: list[Record]) -> dict[str, list[Record]]:
    out: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        out[r.stem].append(r)
    return dict(sorted(out.items()))


def _normalize_polygons(seg: list) -> list[list[float]]:
    """Accept ``[[x1, y1, x2, y2, ...]]``, a flat list, or a list of ``[x, y]`` pairs."""
    if not seg:
        return []
    if all(isinstance(v, (int, float)) for v in seg):  # one flat polygon
        seg = [seg]
    if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in seg) and all(
        isinstance(v, (int, float)) for p in seg for v in p
    ):  # one polygon as [x, y] pairs
        seg = [[v for xy in seg for v in xy]]
    polys = []
    for poly in seg:
        flat = []
        for v in poly:
            if isinstance(v, (list, tuple)):
                flat.extend(float(u) for u in v)
            else:
                flat.append(float(v))
        if len(flat) >= 6:
            polys.append(flat)
    return polys


def segmentation_to_rle(seg, height: int, width: int) -> dict | None:
    """One COCO segmentation (polygons or RLE) -> one compressed RLE dict."""
    if isinstance(seg, dict):
        if isinstance(seg.get("counts"), list):
            return mask_utils.frPyObjects(seg, height, width)
        rle = dict(seg)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        return rle
    polys = _normalize_polygons(seg)
    if not polys:
        return None
    return mask_utils.merge(mask_utils.frPyObjects(polys, height, width))


def record_instance_rles(record: Record) -> list[dict]:
    rles = []
    for seg in record.segmentations:
        rle = segmentation_to_rle(seg, record.height, record.width)
        if rle is not None and mask_utils.area(rle) > 0:
            rles.append(rle)
    return rles


def record_union_mask(record: Record) -> np.ndarray:
    """Binary mask of all filaments this annotator drew (H x W, uint8)."""
    rles = record_instance_rles(record)
    if not rles:
        return np.zeros((record.height, record.width), np.uint8)
    return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)


def soft_target(records: list[Record]) -> np.ndarray:
    """Per-pixel fraction of annotators who marked a filament (float32 in [0, 1])."""
    acc = None
    for r in records:
        m = record_union_mask(r).astype(np.float32)
        acc = m if acc is None else acc + m
    return acc / max(len(records), 1)


def read_image(path: str | os.PathLike) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"Could not read image {path}")
    return img


def disk_mask(img: np.ndarray) -> np.ndarray:
    """Rough solar-disk mask: the largest bright region, with holes filled."""
    small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), 3)
    top = float(np.percentile(blur, 99))
    if top <= 0:
        return np.ones(img.shape, bool)
    binary = (blur > 0.25 * top).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.ones(img.shape, bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    disk = ndimage.binary_fill_holes(labels == largest).astype(np.uint8)
    disk = cv2.resize(disk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return disk.astype(bool)


def normalize(img: np.ndarray, disk: np.ndarray | None = None) -> np.ndarray:
    """Z-score the image using on-disk pixels; zero outside the disk; clip to [-5, 5]."""
    x = img.astype(np.float32)
    if disk is None or not disk.any():
        disk = np.ones(img.shape, bool)
    vals = x[disk]
    x = (x - float(vals.mean())) / (float(vals.std()) + 1e-6)
    x[~disk] = 0.0
    return np.clip(x, -5.0, 5.0)


def working_size(height: int, width: int, scale: float) -> tuple[int, int]:
    return max(32, int(round(height * scale))), max(32, int(round(width * scale)))


def preprocess(img: np.ndarray, scale: float, channels=("z",)) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution uint8 image -> (network input at working size, full-res disk mask).

    With the default single ``z`` channel the input is a 2-D array (the original pipeline);
    with several channels (see solarseg/features.py) it is (C, H, W).
    """
    from .features import compute_channels, parse_channels

    channels = parse_channels(channels)
    disk = disk_mask(img)
    h, w = working_size(*img.shape, scale)
    if channels == ("z",):
        x = normalize(img, disk)
        if (h, w) != img.shape:
            x = cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
        return x.astype(np.float32), disk
    return compute_channels(img, disk, (h, w), channels), disk


def resize_target(target: np.ndarray, scale: float) -> np.ndarray:
    h, w = working_size(*target.shape, scale)
    if (h, w) == target.shape:
        return target.astype(np.float32)
    return cv2.resize(target.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)


def split_stems(stems: list[str], val_frac: float = 0.15, seed: int = 42) -> tuple[list[str], list[str]]:
    """Hold out whole months, so near-duplicate frames from the same days stay together."""
    months: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        months[s[:6]].append(s)
    keys = sorted(months)
    random.Random(seed).shuffle(keys)
    target = max(1, int(round(val_frac * len(stems))))
    val: list[str] = []
    for k in keys:
        if len(val) >= target:
            break
        val.extend(months[k])
    val_set = set(val)
    train = [s for s in stems if s not in val_set]
    if not train:  # tiny datasets: keep at least one training image
        train, val = val[:1], val[1:]
    return sorted(train), sorted(val)


def month_folds(stems: list[str], n_folds: int, seed: int = 42) -> dict[str, int]:
    """Assign every image to one of ``n_folds`` folds, keeping whole months together.

    Months are shuffled, then each goes to the fold with the fewest images so far, so the
    folds end up about the same size and frames hours apart never straddle two folds.
    """
    months: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        months[s[:6]].append(s)
    keys = sorted(months)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda k: -len(months[k]))  # big months first gives a tighter balance
    sizes = [0] * n_folds
    fold_of: dict[str, int] = {}
    for k in keys:
        f = min(range(n_folds), key=lambda i: (sizes[i], i))
        for s in months[k]:
            fold_of[s] = f
        sizes[f] += len(months[k])
    return fold_of


def fold_split(stems: list[str], n_folds: int, fold: int, seed: int = 42) -> tuple[list[str], list[str]]:
    """(train, validation) stems for one fold of ``month_folds``."""
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in 0..{n_folds - 1}")
    fold_of = month_folds(stems, n_folds, seed)
    val = sorted(s for s in stems if fold_of[s] == fold)
    train = sorted(s for s in stems if fold_of[s] != fold)
    return train, val
