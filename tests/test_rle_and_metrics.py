import numpy as np
import pytest

from solarseg.metrics import PQStats, match, score_ious
from solarseg.rle import (
    counts_to_rle,
    decode_counts,
    encode_mask,
    labels_to_counts,
    read_submission,
    rows_by_image,
    write_submission,
)


def blob(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_rle_round_trip():
    m = blob(64, 48, 5, 20, 10, 30)
    counts = encode_mask(m)
    assert np.array_equal(decode_counts(counts, 64, 48), m)


def test_csv_format(tmp_path):
    m = blob(32, 32, 1, 5, 1, 5)
    path = tmp_path / "sub.csv"
    write_submission([("20150125172714Mh_1", encode_mask(m))], path)
    text = path.read_text()
    assert text.splitlines()[0] == "filament_id,segmentation_rle"
    assert '"' not in text and "'" not in text
    rows = read_submission(path)
    assert rows[0][0] == "20150125172714Mh_1"
    assert list(rows_by_image(rows)) == ["20150125172714Mh"]


def test_labels_to_counts_are_disjoint():
    labels = np.zeros((40, 40), np.int32)
    labels[2:10, 2:10] = 1
    labels[20:30, 20:35] = 2
    masks = [decode_counts(c, 40, 40) for c in labels_to_counts(labels)]
    assert len(masks) == 2
    assert not (masks[0] & masks[1]).any()


def rle(m):
    return counts_to_rle(encode_mask(m), *m.shape)


def test_pq_perfect_and_empty():
    a, b = blob(50, 50, 0, 10, 0, 10), blob(50, 50, 20, 40, 20, 30)
    assert match([rle(a), rle(b)], [rle(a), rle(b)]).pq == pytest.approx(1.0)
    assert match([], [rle(a)]).fn == 1
    assert match([rle(a)], []).fp == 1
    assert PQStats().pq == 0.0


def test_pq_partial_overlap():
    gt = blob(50, 50, 0, 10, 0, 10)  # 100 px
    pred = blob(50, 50, 0, 10, 0, 8)  # 80 px inside it -> IoU 0.8
    stats = match([rle(pred)], [rle(gt)])
    assert stats.tp == 1 and stats.pq == pytest.approx(0.8)
    far = blob(50, 50, 30, 40, 30, 40)
    stats = match([rle(far)], [rle(gt)])  # no overlap: one FP and one FN
    assert (stats.tp, stats.fp, stats.fn) == (0, 1, 1)


def test_labels_to_counts_matches_one_mask_at_a_time():
    rng = np.random.default_rng(0)
    for _ in range(200):
        h, w = (int(v) for v in rng.integers(1, 30, 2))
        labels = np.zeros((h, w), np.int32)
        for _ in range(int(rng.integers(0, 6))):
            y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
            labels[y : y + int(rng.integers(1, h + 1)), x : x + int(rng.integers(1, w + 1))] = rng.integers(1, 9)
        if rng.random() < 0.3:
            labels[0, 0] = 2  # run starting at the first pixel
        if rng.random() < 0.3:
            labels[-1, -1] = 4  # run ending at the last pixel
        expected = [encode_mask(labels == k) for k in range(1, int(labels.max()) + 1) if (labels == k).any()]
        assert labels_to_counts(labels) == expected


def test_host_counting_rule():
    # every (gt, prediction) pair with IoU > 0.5 is a hit; unmatched rows/columns are FP/FN
    ious = np.array([[0.9, 0.0], [0.0, 0.3], [0.0, 0.0]])  # 3 predictions x 2 filaments
    st = score_ious(ious)
    assert (st.tp, st.fp, st.fn) == (1, 2, 1)
    assert st.pq == pytest.approx(0.9 / (1 + 0.5 * 2 + 0.5 * 1))
    assert (score_ious(np.zeros((0, 3))).fn, score_ious(np.zeros((2, 0))).fp) == (3, 2)
