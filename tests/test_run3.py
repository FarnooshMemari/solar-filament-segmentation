"""Run 3 parts: input channels, month folds, EMA weights, the cache, piece features and the
maths behind the filament picker."""
import cv2
import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils

from solarseg.cache import has_item, load_disk, load_prob, load_x, load_y, read_meta, save_item, write_meta
from solarseg.data import disk_mask, fold_split, month_folds, normalize, preprocess
from solarseg.dataset import CropDataset
from solarseg.ema import ModelEMA
from solarseg.features import CHANNELS, parse_channels
from solarseg.infer import predict_prob
from solarseg.metrics import PQStats, match
from solarseg.model import build_model
from solarseg.pieces import (
    best_tau,
    feature_names,
    keep_by_score,
    matched_ious,
    piece_table,
    selection_pq,
    station_of,
)
from solarseg.rle import labels_to_rles


def fake_sun(size=512, filament=True, seed=0):
    """Limb-darkened disk with noise and, optionally, one dark filament (a thin bar)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rr = np.hypot(xx - size / 2, yy - size / 2) / (0.42 * size)
    mu = np.sqrt(np.clip(1 - rr**2, 0, 1))
    img = np.where(rr < 1, 60 + 140 * (0.4 + 0.6 * mu), 5).astype(np.float32)
    fil = np.zeros((size, size), bool)
    if filament:
        fil[int(0.3 * size) : int(0.3 * size) + 6, int(0.35 * size) : int(0.6 * size)] = True
        img[fil] *= 0.7
    img += rng.normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), fil


def rle(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


# ---------------------------------------------------------------- input channels


def test_parse_channels():
    assert parse_channels(None) == ("z",)
    assert parse_channels("z, flat,ridge") == ("z", "flat", "ridge")
    assert parse_channels(["z", "contrast"]) == ("z", "contrast")
    with pytest.raises(ValueError):
        parse_channels("z,banana")


def test_single_channel_input_is_unchanged():
    img, _ = fake_sun()
    x, _ = preprocess(img, 0.5)
    expected = cv2.resize(normalize(img, disk_mask(img)), (256, 256), interpolation=cv2.INTER_AREA)
    assert x.ndim == 2 and np.allclose(x, expected)


def test_channels_shape_and_zero_outside_the_disk():
    img, _ = fake_sun()
    x, disk = preprocess(img, 0.5, CHANNELS)
    assert x.shape == (4, 256, 256) and x.dtype == np.float32 and np.isfinite(x).all()
    small = cv2.resize(disk.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
    assert np.abs(x[:, ~small]).max() == 0


def test_flat_channel_removes_limb_darkening():
    img, _ = fake_sun(filament=False)
    x, _ = preprocess(img, 0.5, ("z", "flat"))
    yy, xx = np.mgrid[0:256, 0:256]
    rr = np.hypot(xx - 128, yy - 128) / (0.42 * 256)
    centre, limb = rr < 0.5, (rr > 0.8) & (rr < 0.95)
    assert x[0][centre].mean() - x[0][limb].mean() > 1.0  # plain z-score: the limb is darker
    assert abs(x[1][centre].mean() - x[1][limb].mean()) < 0.3  # flat: no trend left


def test_contrast_and_ridge_light_up_the_filament():
    img, fil = fake_sun()
    x, disk = preprocess(img, 0.5, CHANNELS)
    on = cv2.resize(fil.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
    near = cv2.dilate(on.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    inside = cv2.resize(disk.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
    background = inside & ~near
    contrast, ridge = x[CHANNELS.index("contrast")], x[CHANNELS.index("ridge")]
    assert contrast[on].mean() < -2
    assert ridge[on].mean() > 2 * np.median(ridge[background])


# ---------------------------------------------------------------- folds


def test_month_folds_keep_months_together_and_balance():
    rng = np.random.default_rng(0)
    stems = []
    for m in range(24):
        year, month = 2012 + m // 12, m % 12 + 1
        stems += [f"{year}{month:02d}{d + 1:02d}120000Bh" for d in range(int(rng.integers(1, 21)))]
    fold_of = month_folds(stems, 4)
    months = {}
    for s, f in fold_of.items():
        months.setdefault(s[:6], set()).add(f)
    assert all(len(v) == 1 for v in months.values())
    sizes = np.bincount(list(fold_of.values()), minlength=4)
    assert sizes.min() > 0 and sizes.max() - sizes.min() <= 20
    assert month_folds(stems, 4) == fold_of  # same split every time
    held_out = []
    for k in range(4):
        train, val = fold_split(stems, 4, k)
        assert not set(train) & set(val) and len(train) + len(val) == len(stems)
        held_out += val
    assert sorted(held_out) == sorted(stems)  # every image is held out exactly once
    with pytest.raises(ValueError):
        fold_split(stems, 4, 4)


# ---------------------------------------------------------------- EMA


def test_ema_follows_the_weights_and_copies_counters():
    torch.manual_seed(0)
    model = build_model(base=4, depth=2)
    ema = ModelEMA(model, decay=0.9)
    before = {k: v.clone() for k, v in ema.model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    for name, buf in model.named_buffers():
        if not buf.dtype.is_floating_point:
            buf.fill_(5)
    ema.update(model)
    d = 2 / 11  # warm-up decay after one update
    state = ema.model.state_dict()
    for name, p in model.named_parameters():
        assert torch.allclose(state[name], before[name] + (1 - d), atol=1e-5), name
    counters = [n for n, b in model.named_buffers() if not b.dtype.is_floating_point]
    assert counters and all(int(state[n]) == 5 for n in counters)
    for _ in range(300):
        ema.update(model)
    for name, p in model.named_parameters():
        assert torch.allclose(state[name], p, atol=1e-4), name


# ---------------------------------------------------------------- cache


def test_cache_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.random((3, 16, 16)).astype(np.float32)
    y = rng.random((16, 16)).astype(np.float32)
    disk = np.zeros((32, 32), bool)
    disk[4:20, 5:30] = True
    save_item(tmp_path, "s1", x, disk, y)
    save_item(tmp_path, "t1", x[0], disk)  # 2-D input and no target, like a test image
    assert has_item(tmp_path, "s1", need_target=True)
    assert has_item(tmp_path, "t1") and not has_item(tmp_path, "t1", need_target=True)
    assert np.allclose(load_x(tmp_path, "s1"), x, atol=1e-3)
    assert load_x(tmp_path, "t1").shape == (1, 16, 16)
    assert np.allclose(load_y(tmp_path, "s1"), y, atol=1e-3)
    assert np.array_equal(load_disk(tmp_path, "s1", (32, 32)), disk)
    write_meta(tmp_path, {"channels": ["z"], "image_size": [32, 32]})
    assert read_meta(tmp_path)["image_size"] == [32, 32]
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(a, np.full((16, 16), 0.2, np.float16))
    np.save(b, np.full((16, 16), 0.6, np.float16))
    prob = load_prob([a, b], (32, 32))
    assert prob.shape == (32, 32) and np.allclose(prob, 0.4, atol=1e-3)


# ---------------------------------------------------------------- multi-channel training and prediction


def test_crops_keep_channels_and_target_aligned():
    rng = np.random.default_rng(0)
    tgt = (rng.random((64, 64)) > 0.7).astype(np.float16)
    img = np.stack([tgt, -tgt, np.zeros_like(tgt)]).astype(np.float32)
    ds = CropDataset([img], [tgt], crop=32, samples_per_epoch=30, pos_frac=0.5)
    checked = 0
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == (3, 32, 32) and y.shape == (1, 32, 32)
        x, y = x.numpy(), y.numpy()[0]
        if y.std() > 0:  # channels 0 and 1 are +/- the target before the shared jitter
            assert np.corrcoef((x[0] - x[1]).ravel(), y.ravel())[0, 1] > 0.95
            checked += 1
    assert checked > 10


class FirstChannel(torch.nn.Module):
    """Pixel-wise 'network': the logit is 4 x the first input channel."""

    def forward(self, x):
        return 4 * x[:, :1]


@pytest.mark.parametrize("tta", [0, 4, 8])
def test_multi_channel_prediction_with_tta(tta):
    x = np.random.default_rng(0).normal(size=(3, 40, 56)).astype(np.float32)
    want = 1 / (1 + np.exp(-4 * x[0]))
    prob = predict_prob(FirstChannel(), x, torch.device("cpu"), tile=32, overlap=8, tta=tta)
    assert prob.shape == (40, 56) and np.allclose(prob, want, atol=1e-4)


# ---------------------------------------------------------------- pieces and the picker


def test_piece_table_measures_each_piece():
    labels = np.zeros((64, 64), np.int32)
    labels[10:14, 10:50] = 1  # a 4 x 40 bar
    labels[40:50, 40:50] = 3  # a 10 x 10 square (id 2 is unused)
    prob = np.where(labels > 0, 0.6, 0.0).astype(np.float32)
    prob[12, 20] = 0.95
    x_small = np.stack([np.where(labels[::2, ::2] == 1, 2.0, 0.0), np.ones((32, 32))]).astype(np.float16)
    ids, table = piece_table(labels, prob, np.ones((64, 64), bool), x_small, "20150125172714Mh")
    names = feature_names(["z", "flat"])
    assert ids.tolist() == [1, 3] and table.shape == (2, len(names))
    bar, square = (dict(zip(names, row)) for row in table)
    assert np.isclose(np.expm1(bar["log_area"]), 160, rtol=1e-4)
    assert np.isclose(np.expm1(square["log_area"]), 100, rtol=1e-4)
    assert np.isclose(bar["peak"], 0.95) and np.isclose(square["peak"], 0.6)
    assert bar["elongation"] > 4 * square["elongation"]
    assert np.isclose(bar["mean_z"], 2.0) and np.isclose(square["mean_z"], 0.0)
    assert np.isclose(bar["mean_flat"], 1.0)
    assert bar["n_pieces"] == 2 and bar["station"] == 3  # 'M' = Mauna Loa


def test_station_of():
    assert station_of("A1-20150125172714Bh.jpg") == 0
    assert station_of("20150125172714Uh") == 5
    assert station_of("test_001") == -1


def test_selection_pq_equals_the_metric_for_every_subset():
    labels = np.zeros((64, 64), np.int32)
    labels[2:10, 2:30] = 1
    labels[20:30, 5:15] = 2
    labels[40:44, 40:60] = 3
    _, rles = labels_to_rles(labels)
    a = np.zeros((64, 64), bool)
    a[2:10, 2:30] = True
    b = np.zeros((64, 64), bool)
    b[20:30, 5:15] = True
    c = np.zeros((64, 64), bool)
    c[50:60, 50:60] = True  # a filament nobody predicted
    a2 = np.zeros((64, 64), bool)
    a2[3:11, 4:30] = True  # a second annotator's outline of the same filament
    gts = [[rle(a), rle(b), rle(c)], [rle(a2)]]
    hits, n_gt = matched_ious(rles, gts)
    assert hits.shape == (3, 2) and n_gt.tolist() == [3, 1]
    for bits in range(8):
        keep = np.array([(bits >> k) & 1 for k in range(3)], bool)
        got = selection_pq([hits], [n_gt], [keep])
        want = PQStats()
        for gt in gts:
            want += match([r for r, k in zip(rles, keep) if k], gt)
        assert (got.tp, got.fp, got.fn) == (want.tp, want.fp, want.fn)
        assert np.isclose(got.iou_sum, want.iou_sum)


def test_a_piece_helps_only_when_its_mean_iou_beats_half_the_pq():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n_ann = int(rng.integers(1, 4))
        tp, fp, fn = int(rng.integers(1, 60)), int(rng.integers(0, 60)), int(rng.integers(3, 60))
        base = PQStats(tp=tp, fp=fp, fn=fn, iou_sum=tp * float(rng.uniform(0.55, 0.95)))
        hits = np.where(rng.random(n_ann) < 0.5, rng.uniform(0.51, 1.0, n_ann), 0.0)
        n_hit = int((hits > 0).sum())
        after = PQStats(tp=tp + n_hit, fp=fp + n_ann - n_hit, fn=fn - n_hit, iou_sum=base.iou_sum + hits.sum())
        assert (after.pq > base.pq) == (hits.mean() > base.pq / 2)


def test_keep_by_score_caps_the_count():
    s = np.array([0.1, 0.9, 0.5, 0.7])
    assert keep_by_score(s, 0.4).tolist() == [False, True, True, True]
    assert keep_by_score(s, 0.4, max_instances=2).tolist() == [False, True, False, True]


def test_best_tau_finds_the_cut():
    hits = np.array([[0.9], [0.8], [0.0], [0.0]], np.float32)  # two good pieces, two false ones
    scores = np.array([0.9, 0.8, 0.3, 0.1], np.float32)
    tau, st, curve = best_tau([scores], [hits], [np.array([3])])
    assert 0.3 < tau < 0.62 and (st.tp, st.fp, st.fn) == (2, 0, 1)
    assert len(curve) == 60
