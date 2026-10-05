import numpy as np

from solarseg.postprocess import PostConfig, instances_from_prob
from solarseg.rle import counts_to_rle, encode_mask
from solarseg.tuning import Grid, add_up, config_to_key, key_to_config, score_image


def rle(mask):
    return counts_to_rle(encode_mask(mask), *mask.shape)


def make_case():
    prob = np.zeros((64, 64), np.float32)
    gt1, gt2 = np.zeros((64, 64), bool), np.zeros((64, 64), bool)
    gt1[10:16, 5:40] = True
    gt2[40:46, 10:50] = True
    prob[gt1] = 0.9
    prob[gt2] = 0.45  # only found with a low enough threshold
    prob[30:32, 55:57] = 0.95  # small false positive, removed by min_area
    return prob, [[rle(gt1), rle(gt2)]]


def test_score_image_covers_the_grid_and_finds_the_right_setting():
    prob, gts = make_case()
    grid = Grid(thresholds=(0.4, 0.8), low_thresholds=(0.4, 0.8), min_areas=(1, 10),
                merge_dists=(0, 3), close_radii=(0,), fill_holes=(False,))
    res = score_image(prob, None, gts, grid)
    assert len(res) == grid.size()
    totals = add_up([res, res])  # two identical images pool their counts
    perfect = totals[(0.4, 0, False, 0, 0.4, 10)]
    assert perfect.pq == 1.0 and perfect.tp == 4
    with_speck = totals[(0.4, 0, False, 0, 0.4, 1)]
    assert with_speck.fp == 2 and with_speck.pq < 1.0


def test_scores_match_the_prediction_pipeline():
    prob, gts = make_case()
    grid = Grid(thresholds=(0.4, 0.8), low_thresholds=(0.4,), min_areas=(1, 10),
                merge_dists=(0,), close_radii=(0, 2), fill_holes=(False, True))
    res = score_image(prob, None, gts, grid)
    for key, counts in res.items():
        cfg = key_to_config(key)
        assert config_to_key(cfg) == key
        labels = instances_from_prob(prob, None, cfg)
        alone = score_image(prob, None, gts, Grid(
            thresholds=(cfg.threshold,), low_thresholds=(cfg.low,), min_areas=(cfg.min_area,),
            merge_dists=(cfg.merge_dist,), close_radii=(cfg.close_radius,), fill_holes=(cfg.fill_holes,)))
        assert alone[key] == counts
        assert counts[1] + counts[2] == labels.max()  # every kept piece is a TP or an FP here


def test_old_settings_map_to_a_plain_threshold():
    cfg = PostConfig(threshold=0.6, min_area=400, merge_dist=10)
    assert config_to_key(cfg) == (0.6, 0, False, 10, 0.6, 400)
    assert key_to_config(config_to_key(cfg)).low_threshold is None
