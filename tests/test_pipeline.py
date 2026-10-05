"""End to end on a tiny synthetic dataset: make data, train, tune, predict, check."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_synthetic_pipeline(tmp_path):
    data = tmp_path / "synthetic"
    run("scripts/make_synthetic.py", "--out", str(data), "--size", "512", "--n-train", "8", "--n-test", "2")
    root = str(data / "MAGFiLO_1.0_Kaggle_2026")
    run_dir = str(tmp_path / "run")
    run("scripts/inspect_data.py", "--data-root", root)
    run(
        "scripts/train.py", "--data-root", root, "--out", run_dir, "--scale", "0.5", "--crop", "128",
        "--epochs", "2", "--steps-per-epoch", "4", "--batch-size", "4", "--base", "8", "--depth", "3",
        "--workers", "0", "--val-frac", "0.3", "--tile", "128", "--overlap", "32",
    )
    run("scripts/tune.py", "--run", run_dir, "--data-root", root, "--tta", "8", "--workers", "2",
        "--thresholds", "0.3,0.5", "--low-thresholds", "0.2,0.3", "--min-areas", "20,50",
        "--merge-dists", "0,5", "--close-radii", "0,3", "--fill-holes", "0,1")
    out = str(tmp_path / "submission.csv")
    run("scripts/predict.py", "--run", run_dir, "--data-root", root, "--out", out,
        "--params", str(Path(run_dir) / "postprocess.json"))
    run("scripts/check_submission.py", "--csv", out, "--data-root", root, "--height", "512", "--width", "512")


def test_synthetic_kfold_pipeline(tmp_path):
    """Run 3: cache the input channels, train two folds, save their maps, tune on the
    out-of-fold maps, train the picker, then predict with and without it."""
    data = tmp_path / "synthetic"
    run("scripts/make_synthetic.py", "--out", str(data), "--size", "512", "--n-train", "8", "--n-test", "2")
    root = str(data / "MAGFiLO_1.0_Kaggle_2026")
    cache, probs, final = tmp_path / "cache", tmp_path / "probs", tmp_path / "run3"
    run("scripts/cache.py", "--out", str(cache), "--data-root", root, "--workers", "2")
    folds = []
    for k in range(2):
        fold_dir = str(tmp_path / f"fold{k}")
        run(
            "scripts/train.py", "--data-root", root, "--out", fold_dir, "--folds", "2", "--fold", str(k),
            "--cache", str(cache), "--channels", "z,flat,contrast,ridge", "--ema", "0.9", "--scale", "0.5",
            "--crop", "128", "--epochs", "8", "--steps-per-epoch", "10", "--lr", "3e-3", "--batch-size", "4",
            "--base", "8", "--depth", "3", "--workers", "0", "--tile", "256", "--overlap", "32",
        )
        run("scripts/oof.py", "--run", fold_dir, "--cache", str(cache), "--out", str(probs), "--test",
            "--tta", "8", "--tile", "256")
        folds.append(fold_dir)
    assert len(list((probs / "oof").glob("*.npy"))) == 8
    assert all(len(list((probs / "test" / f"fold{k}").glob("*.npy"))) == 2 for k in range(2))
    run("scripts/tune.py", "--probs", str(probs), "--cache", str(cache), "--out", str(final), "--data-root", root,
        "--workers", "2", "--thresholds", "0.3,0.5", "--low-thresholds", "0.1,0.2", "--min-areas", "20,50",
        "--merge-dists", "0,5", "--close-radii", "0", "--fill-holes", "0")
    run("scripts/scorer.py", "--probs", str(probs), "--cache", str(cache), "--runs", *folds,
        "--post", str(final / "postprocess.json"), "--out", str(final), "--data-root", root, "--workers", "2")
    assert (final / "picker.txt").exists() and (final / "picker.json").exists()
    for picker in (False, True):
        out = str(tmp_path / f"submission_{int(picker)}.csv")
        extra = ["--picker", str(final)] if picker else []
        run("scripts/predict.py", "--probs", str(probs), "--cache", str(cache),
            "--params", str(final / "postprocess.json"), "--out", out, "--workers", "2", *extra)
        run("scripts/check_submission.py", "--csv", out, "--data-root", root, "--height", "512", "--width", "512")
