# Solar Filament Segmentation

[![tests](https://github.com/FarnooshMemari/solar-filament-segmentation/actions/workflows/tests.yml/badge.svg)](https://github.com/FarnooshMemari/solar-filament-segmentation/actions/workflows/tests.yml)

A PyTorch pipeline that finds every solar filament in full-disk H-alpha images of the Sun,
built for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
on Kaggle (IEEE BigData Cup 2026).

Filaments are long, dark clouds of plasma held above the Sun's surface by magnetic fields.
When they erupt they can launch coronal mass ejections, so finding them automatically
helps space-weather forecasting. The task: given a 2048x2048 GONG H-alpha image,
return one mask per filament. Submissions are scored with **Panoptic Quality (PQ)**, which
rewards finding each filament once and outlining it well, and penalizes missed filaments,
false ones, and filaments split into pieces.

## How it works

| Step | What happens |
| --- | --- |
| Preprocess | Find the solar disk, normalize brightness on the disk, downscale to 1024x1024 (`--scale`) |
| Extra inputs (run 3) | Three more channels: the image with limb darkening removed (each pixel divided by the median brightness at the same distance from the disk center), local contrast, and a dark-ridge map (Hessian response at three widths) that highlights thin filaments |
| Labels | Up to three annotators labeled each image and they disagree, so the target is the share of annotators who marked each pixel (a soft label) |
| Model | Compact U-Net (7.8M parameters) trained from scratch, no external weights |
| Training | Random 512x512 crops, 70% centered on a filament; flips, rotations, brightness jitter; BCE + Dice loss; AdamW with a one-cycle schedule; mixed precision on GPU |
| Folds (run 3) | Four models, each holding out a quarter of the months, with averaged (EMA) weights; every training image gets an out-of-fold prediction, and the test set is predicted by averaging the four models |
| Inference | Sliding window with blended tiles, optional test-time augmentation (4 flips, or 8 flips and rotations), upsampled to 2048x2048 |
| Instances | Low threshold on the disk, optional gap closing and hole filling, optional joining of nearby fragments, connected components; a piece is kept only if one of its pixels passes the high threshold and it is big enough |
| Tuning | Hundreds to thousands of combinations of those settings are scored on held-out months with the host's PQ counting rule; pieces are encoded and matched once per image, so the search takes minutes |
| Picker (run 3) | A LightGBM model predicts each piece's matched IoU averaged over the annotators from its size, shape, confidence, position and brightness. A piece raises PQ only when that value is above half the current PQ, so pieces are kept above a cut-off tuned out of fold. Used only if it beats the threshold rule out of fold |
| Submission | One COCO RLE per filament, checked for format and overlaps before upload |

Validation holds out whole months of observations, so frames taken hours apart never end
up on both sides of the split.

## Run it on Kaggle

1. Open a new notebook on the competition page and add the competition data.
2. Turn on a GPU and Internet in the notebook settings.
3. Import `notebooks/kaggle_pipeline.ipynb` (or paste its cells) and run all cells.

To redo only the steps after training with an already trained model, use
`notebooks/kaggle_retune.ipynb` and add the training notebook's output as an input.

Run 3 (extra input channels, four fold models, the picker) is
`notebooks/kaggle_run3.ipynb`. It needs *GPU T4 x2*, trains two folds at a time and takes
about 3 hours. `notebooks/kaggle_run3_report.ipynb` summarizes its output without a GPU.

Or run the steps yourself:

```bash
python scripts/inspect_data.py                           # what the loader sees
python scripts/train.py --out runs/baseline              # ~50 min on one T4 GPU
python scripts/tune.py --run runs/baseline --tta 8       # pick post-processing on validation PQ
python scripts/predict.py --run runs/baseline --out submission.csv
python scripts/check_submission.py --csv submission.csv  # must print submission=ok
python scripts/visualize.py --run runs/baseline --n 4    # figures for the report
```

The run 3 steps:

```bash
python scripts/cache.py --out /tmp/cache --channels z,flat,contrast,ridge   # inputs computed once
for k in 0 1 2 3; do
  python scripts/train.py --out runs/fold$k --folds 4 --fold $k --cache /tmp/cache \
      --channels z,flat,contrast,ridge --epochs 40 --ema 0.999 --tile 1024
  python scripts/oof.py --run runs/fold$k --cache /tmp/cache --out /tmp/probs --test --tta 8
done
python scripts/tune.py --probs /tmp/probs --cache /tmp/cache --out runs/run3   # on all 707 images
python scripts/scorer.py --probs /tmp/probs --cache /tmp/cache --post runs/run3/postprocess.json \
    --runs runs/fold0 runs/fold1 runs/fold2 runs/fold3 --out runs/run3
python scripts/predict.py --probs /tmp/probs --cache /tmp/cache --params runs/run3/postprocess.json \
    --picker runs/run3 --out submission.csv                                      # drop --picker for the rule
python scripts/oof_report.py --run runs/run3   # out-of-fold PQ per fold and on the run 1/2 validation images
```

The data folder is found automatically under `/kaggle/input`. Elsewhere, pass
`--data-root /path/to/MAGFiLO_1.0_Kaggle_2026` or set `SOLARSEG_DATA`.

## Run it locally without the data

A synthetic dataset with the same layout lets you test every step on a laptop:

```bash
pip install -r requirements.txt
python scripts/make_synthetic.py --out data/synthetic --size 512
python scripts/train.py --data-root data/synthetic/MAGFiLO_1.0_Kaggle_2026 --out runs/smoke \
    --epochs 2 --steps-per-epoch 5 --crop 128 --base 8 --workers 0
pytest -q
```

## Results

Runs 1 and 2 use one model trained on Kaggle (one T4 GPU, 30 epochs, about 50 minutes) on
601 images and checked on 106 images from held-out months. Its validation Dice peaked at
0.713 at epoch 12 and stayed between 0.69 and 0.71 after that. Run 3 trains four models
with extra input channels, each holding out a quarter of the months, so every training image
also gets a prediction from a model that never saw it. The PQ column below is measured on
the same 106 held-out images for all three runs.

| Run | What changed | PQ on the 106 held-out images | Public leaderboard |
| --- | --- | --- | --- |
| 1 | Single threshold 0.6, minimum area 400 px, fragments within 10 px joined | 0.389 | 0.32 |
| 2 | 8-view test-time augmentation; low/high threshold 0.4/0.9, minimum area 300 px, fragments within 10 px joined | 0.410 | 0.34 |
| 3 | Extra input channels; four fold models with averaged (EMA) weights; settings tuned on all 707 training images (low/high threshold 0.45/0.97, minimum area 300 px, fragments within 10 px joined); LightGBM picker; test set predicted by averaging the four models | 0.429 | not submitted yet |

Runs 1 and 2 were tuned on those 106 images, which flatters them a little. Run 3's settings
were tuned on all 707 images, so its number is the more honest one.

What each change in run 2 added on validation PQ (same model, about 1,600 settings searched):

| Step | Validation PQ |
| --- | --- |
| Run 1 settings | 0.389 |
| + 8-view test-time augmentation | 0.391 |
| + wider search with a single threshold | 0.401 |
| + low/high threshold (hysteresis) | 0.410 |

Gap closing and hole filling did not help. Run 2 matches more real filaments (746 vs 691)
with slightly fewer false ones (448 vs 468), so RQ rose from 0.583 to 0.620 while SQ
stayed near 0.66. On the test set it found 1,315 filaments in 180 images.

### Run 3 in detail

Out-of-fold PQ, where every image is scored by the model that did not train on it
(`notebooks/kaggle_run3_report.ipynb` prints all of these):

| Which pieces are kept | All 707 images | The 106 run 1/2 validation images |
| --- | --- | --- |
| Every candidate piece | 0.374 | |
| Low/high threshold rule | 0.412 | 0.424 |
| LightGBM picker | 0.415 | 0.429 |
| A picker that knows each piece's true IoU (upper bound) | 0.486 | |

- The four models reach validation Dice 0.712, 0.722, 0.707 and 0.708 (best at epoch 15 or
  20 of 40). Training all four and saving their maps took 125 minutes on two T4 GPUs; the
  whole notebook, 2 h 21 min.
- PQ per fold with the picker: 0.4166, 0.4235, 0.4021 and 0.4182, so the gain holds across months.
- The picker keeps fewer pieces than the rule (6.4 per image vs 7.0; the annotators mark 7.4
  on average). It removes 18% of the false detections (2,506 vs 3,066) and loses 4% of the
  matches (4,815 vs 5,022).
- Its most useful inputs are the mean and peak probability of the piece, the share of
  confident pixels and the dark-ridge channel.
- Matched filaments are outlined with a median IoU of 0.673; half of them fall between 0.602
  and 0.735.
- On the test set it keeps 1,136 filaments in 180 images (the rule keeps 1,158).

## Ideas to try next

- High thresholds above 0.97: the best value was again the largest one tried
- Tune on averaged maps: the test set is predicted by averaging four models, while the
  settings are tuned on single-model maps
- Better piece features: the gap between the picker (0.415) and the upper bound (0.486) is
  still large
- Native resolution (`--scale 1.0`) to keep the thinnest filaments
- Bigger model (`--base 48`)
- A loss that rewards connected shapes (e.g. clDice) to reduce fragmentation
- Splitting by 27-day solar rotations instead of calendar months

## Repository layout

```
solarseg/            library code
  data.py            annotations, images, masks, disk detection, splits and month folds
  features.py        extra input channels (limb darkening removed, local contrast, dark ridges)
  cache.py           inputs, targets and probability maps stored once on disk
  dataset.py         random training crops
  model.py           U-Net
  losses.py          BCE + Dice
  ema.py             averaged (EMA) weights
  infer.py           sliding-window inference
  postprocess.py     probability map -> filament instances
  tuning.py          fast search over post-processing settings
  pieces.py          per-piece features and PQ maths for the picker
  metrics.py         Panoptic Quality
  rle.py             masks <-> submission CSV
scripts/             one command per step (inspect, cache, train, oof, tune, scorer, predict,
                     check, visualize, oof_report)
notebooks/           Kaggle notebooks: full pipeline, re-tune, run 3 and its report
tests/               unit tests and an end-to-end run on synthetic data
```

## Competition notes

- Training uses only the competition data. The full public MAGFiLO release may include
  labels for the test images, which the rules don't allow.
- Rejected uploads still count toward the daily submission limit, so every file goes
  through `check_submission.py` first.

## Credits

The run 3 input channels (radial flat-fielding, local contrast and a scale-normalized
dark-ridge response) and the idea of scoring each piece and keeping it only above about half
the current PQ come from the public notebook
[Solar Filament Segmentation 2026 (0.37)](https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-37)
by tushhaaarrrr (Apache 2.0). The code in this repository is a separate implementation.

## License and data

Code: MIT. The images come from the GONG network (National Solar Observatory) and the
labels from MAGFiLO; both keep their own terms and are not included in this repository.

Author: Farnoosh Memari ([@FarnooshMemari](https://github.com/FarnooshMemari))
