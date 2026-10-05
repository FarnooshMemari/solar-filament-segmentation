import json

import numpy as np
import torch

from solarseg.data import (
    _normalize_polygons,
    image_stem,
    load_records,
    record_instance_rles,
    split_stems,
)
from solarseg.model import build_model
from solarseg.infer import _unview, _view, predict_prob
from solarseg.postprocess import PostConfig, instances_from_prob


def test_instances_split_and_filter():
    prob = np.zeros((100, 100), np.float32)
    prob[10:20, 10:40] = 0.9  # filament 1
    prob[60:70, 10:40] = 0.9  # filament 2
    prob[90:92, 90:92] = 0.9  # speck, removed by min_area
    labels = instances_from_prob(prob, None, PostConfig(threshold=0.5, min_area=20))
    assert labels.max() == 2


def test_merge_joins_close_fragments():
    prob = np.zeros((100, 100), np.float32)
    prob[40:45, 10:30] = 0.9
    prob[40:45, 34:60] = 0.9  # 4 px gap
    assert instances_from_prob(prob, None, PostConfig(min_area=10, merge_dist=0)).max() == 2
    assert instances_from_prob(prob, None, PostConfig(min_area=10, merge_dist=5)).max() == 1


def test_disk_mask_is_respected():
    prob = np.full((50, 50), 0.9, np.float32)
    disk = np.zeros((50, 50), bool)
    labels = instances_from_prob(prob, disk, PostConfig(min_area=1))
    assert labels.max() == 0


def test_image_stem():
    assert image_stem("A2-20150125172714Mh.jpg") == "20150125172714Mh"
    assert image_stem("/x/y/20140609195854Bh.jpg") == "20140609195854Bh"


def test_polygon_formats():
    flat = [0, 0, 10, 0, 10, 10]
    assert _normalize_polygons([flat]) == [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0]]
    assert _normalize_polygons(flat) == [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0]]
    assert _normalize_polygons([[0, 0], [10, 0], [10, 10]]) == [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0]]


def test_load_records_with_annotator_prefix(tmp_path):
    images = tmp_path / "train_images"
    images.mkdir()
    import cv2

    cv2.imwrite(str(images / "20150125172714Mh.jpg"), np.zeros((64, 64), np.uint8))
    square = [[10, 10, 30, 10, 30, 30, 10, 30]]
    coco = {
        "images": [
            {"id": 1, "file_name": "A1-20150125172714Mh.jpg", "height": 64, "width": 64},
            {"id": 2, "file_name": "A2-20150125172714Mh.jpg", "height": 64, "width": 64},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "segmentation": square},
            {"id": 2, "image_id": 2, "segmentation": square},
            {"id": 3, "image_id": 2, "segmentation": [[40, 40, 50, 40, 50, 50]]},
        ],
    }
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(coco))
    records, report = load_records(ann, images)
    assert report["physical_images"] == 1
    assert sorted(r.annotator for r in records) == ["A1", "A2"]
    assert [len(record_instance_rles(r)) for r in records] == [1, 2]


def test_load_records_with_annotator_in_image_id(tmp_path):
    # layout seen in the competition JSON: string ids "<annotator>-<frame>", polygons, .jpeg files
    images = tmp_path / "train_images"
    images.mkdir()
    import cv2

    cv2.imwrite(str(images / "20140609195854Bh.jpeg"), np.zeros((64, 64), np.uint8))
    square = [[10, 10, 30, 10, 30, 30, 10, 30]]
    coco = {
        "images": [
            {"id": "040301-20140609195854Bh", "file_name": "20140609195854Bh.jpeg", "height": 64, "width": 64},
            {"id": "040302-20140609195854Bh", "file_name": "20140609195854Bh.jpeg", "height": 64, "width": 64},
        ],
        "annotations": [
            {"id": "a1", "image_id": "040301-20140609195854Bh", "segmentation": square},
            {"id": "a2", "image_id": "040302-20140609195854Bh", "segmentation": square},
        ],
    }
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(coco))
    records, report = load_records(ann, images)
    assert report["physical_images"] == 1 and report["missing_image_files"] == 0
    assert sorted(r.annotator for r in records) == ["040301", "040302"]


def test_split_keeps_months_together():
    stems = [f"2015{m:02d}{d:02d}120000Bh" for m in range(1, 13) for d in (1, 15)]
    train, val = split_stems(stems, 0.25, seed=1)
    assert set(train).isdisjoint(val) and len(train) + len(val) == len(stems)
    assert {s[:6] for s in train}.isdisjoint({s[:6] for s in val})


def test_unet_shapes():
    model = build_model(base=8, depth=3)
    out = model(torch.zeros(2, 1, 100, 76))
    assert out.shape == (2, 1, 100, 76)


def test_hysteresis_keeps_only_pieces_with_a_strong_pixel():
    prob = np.zeros((60, 60), np.float32)
    prob[5:10, 5:40] = 0.4  # weak piece, never reaches 0.6
    prob[30:35, 5:40] = 0.4
    prob[31:34, 20:24] = 0.9  # strong core: the whole 0.4 piece around it is kept
    cfg = PostConfig(threshold=0.6, low_threshold=0.3, min_area=1)
    labels = instances_from_prob(prob, None, cfg)
    assert labels.max() == 1
    assert (labels > 0).sum() == 5 * 35  # extent comes from the low threshold


def test_closing_bridges_small_gaps():
    prob = np.zeros((40, 80), np.float32)
    prob[18:22, 5:38] = 0.9
    prob[18:22, 40:75] = 0.9  # 2 px gap
    assert instances_from_prob(prob, None, PostConfig(min_area=1)).max() == 2
    closed = instances_from_prob(prob, None, PostConfig(min_area=1, close_radius=2))
    assert closed.max() == 1 and closed[19, 38] == 1


def test_fill_holes():
    prob = np.zeros((40, 40), np.float32)
    prob[10:30, 10:30] = 0.9
    prob[15:25, 15:25] = 0.0  # hole
    assert instances_from_prob(prob, None, PostConfig(min_area=1))[20, 20] == 0
    assert instances_from_prob(prob, None, PostConfig(min_area=1, fill_holes=True))[20, 20] == 1


def test_tta_views_are_undone():
    x = torch.arange(2 * 16, dtype=torch.float32).reshape(2, 1, 4, 4)
    for k in range(8):
        assert torch.equal(_unview(_view(x, k), k), x)
    model = torch.nn.Identity()  # an identity model must give the same map with or without TTA
    img = np.random.default_rng(0).normal(size=(40, 40)).astype(np.float32)
    plain = predict_prob(model, img, torch.device("cpu"), tile=16, overlap=4)
    for views in (4, 8):
        assert np.allclose(predict_prob(model, img, torch.device("cpu"), tile=16, overlap=4, tta=views), plain, atol=1e-6)
