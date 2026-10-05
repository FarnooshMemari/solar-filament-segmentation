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
