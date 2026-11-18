"""One-cell Google Colab runner for Mannu's KNN and MLP experiments.

Usage in Colab:
    %run colab_train.py

Before running, put these four files in Google Drive under:
    MyDrive/PathSnap_processed/
        train_candidates.parquet
        test_candidates.parquet
        point_refs.parquet
        folds.csv

The script clones the current `manu` branch, installs requirements, copies
the processed data, runs both models, and downloads one results ZIP.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/likhith76342/ML-Hackathon.git"
BRANCH = "manu"
DRIVE_DATA_DIR = Path("/content/drive/MyDrive/PathSnap_processed")
WORK_DIR = Path("/content/ML-Hackathon")
RESULT_ZIP = Path("/content/PathSnap_Manu_KNN_MLP_results.zip")
REQUIRED = ["train_candidates.parquet", "test_candidates.parquet", "point_refs.parquet", "folds.csv"]


def run(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    try:
        from google.colab import drive, files  # type: ignore
    except ImportError as exc:
        raise SystemExit("Run this file in Google Colab.") from exc

    drive.mount("/content/drive")
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    run("git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(WORK_DIR))
    run(sys.executable, "-m", "pip", "install", "-q", "-r", str(WORK_DIR / "requirements.txt"))

    processed = WORK_DIR / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED if not (DRIVE_DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing from {DRIVE_DATA_DIR}: {', '.join(missing)}. "
            "Upload the four processed files there first."
        )
    for name in REQUIRED:
        shutil.copy2(DRIVE_DATA_DIR / name, processed / name)

    # The repository root is required for module imports.
    import os
    os.chdir(WORK_DIR)
    run(sys.executable, "-m", "models.knn.train")
    run(sys.executable, "-m", "models.mlp.train")

    if RESULT_ZIP.exists():
        RESULT_ZIP.unlink()
    shutil.make_archive(
        str(RESULT_ZIP.with_suffix("")), "zip", root_dir=WORK_DIR,
        base_dir="results",
    )
    print(f"Results ZIP: {RESULT_ZIP}")
    files.download(str(RESULT_ZIP))


if __name__ == "__main__":
    main()
