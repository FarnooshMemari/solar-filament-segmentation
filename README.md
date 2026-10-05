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
| Labels | Up to three annotators labeled each image and they disagree, so the target is the share of annotators who marked each pixel (a soft label) |
| Model | Compact U-Net (7.8M parameters) trained from scratch, no external weights |
| Training | Random 512x512 crops, 70% centered on a filament; flips, rotations, brightness jitter; BCE + Dice loss; AdamW with a one-cycle schedule; mixed precision on GPU |
| Inference | Sliding window with blended tiles, optional 4-flip test-time augmentation, upsampled to 2048x2048 |
| Instances | Threshold, keep the disk, optionally join nearby fragments, then connected components; tiny pieces are dropped |
| Tuning | Threshold, minimum area and fragment merging are chosen to maximize PQ on held-out months |
| Submission | One COCO RLE per filament, checked for format and overlaps before upload |

Validation holds out whole months of observations, so frames taken hours apart never end
up on both sides of the split.

## Run it on Kaggle

1. Open a new notebook on the competition page and add the competition data.
2. Turn on a GPU and Internet in the notebook settings.
3. Import `notebooks/kaggle_pipeline.ipynb` (or paste its cells) and run all cells.

Or run the steps yourself:

```bash
python scripts/inspect_data.py                           # what the loader sees
python scripts/train.py --out runs/baseline              # ~30-45 min on one GPU
python scripts/tune.py --run runs/baseline               # pick post-processing on validation PQ
python scripts/predict.py --run runs/baseline --out submission.csv
python scripts/check_submission.py --csv submission.csv  # must print submission=ok
python scripts/visualize.py --run runs/baseline --n 4    # figures for the report
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

| Version | Validation PQ | Public leaderboard |
| --- | --- | --- |
| U-Net baseline (scale 0.5, tuned post-processing) | TBD | TBD |

## Ideas to try next

- Native resolution (`--scale 1.0`) to keep the thinnest filaments
- Bigger model (`--base 48`) or longer training
- A loss that rewards connected shapes (e.g. clDice) to reduce fragmentation
- Ensembling models trained on different splits
- Splitting by 27-day solar rotations instead of calendar months

## Repository layout

```
solarseg/            library code
  data.py            annotations, images, masks, disk detection, splits
  dataset.py         random training crops
  model.py           U-Net
  losses.py          BCE + Dice
  infer.py           sliding-window inference
  postprocess.py     probability map -> filament instances
  metrics.py         Panoptic Quality
  rle.py             masks <-> submission CSV
scripts/             one command per step (inspect, train, tune, predict, check, visualize)
notebooks/           Kaggle notebook that runs the whole pipeline
tests/               unit tests and an end-to-end run on synthetic data
```

## Competition notes

- Training uses only the competition data. The full public MAGFiLO release may include
  labels for the test images, which the rules don't allow.
- Rejected uploads still count toward the daily submission limit, so every file goes
  through `check_submission.py` first.

## License and data

Code: MIT. The images come from the GONG network (National Solar Observatory) and the
labels from MAGFiLO; both keep their own terms and are not included in this repository.

Author: Farnoosh Memari ([@FarnooshMemari](https://github.com/FarnooshMemari))
