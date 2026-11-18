"""Tests for reports/charts/make_charts.py on a tiny fake results tree.  Owner: Likhith."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "reports" / "charts" / "make_charts.py"
spec = importlib.util.spec_from_file_location("make_charts", SCRIPT)
make_charts = importlib.util.module_from_spec(spec)
sys.modules["make_charts"] = make_charts          # dataclasses look the module up while it loads
spec.loader.exec_module(make_charts)

PNGS = ["cv_top1_by_model.png", "cv_top1_per_fold.png", "test_top1_and_macro_f1.png", "cv_vs_test_gap.png",
        "time_vs_accuracy.png", "confusion_matrices.png", "log_loss_by_model.png", "primary_vs_all_trips.png"]


def fake_metrics(name: str, top1: float, dv=2, excluded=(33, 54)) -> dict:
    cfg = {"model__n_estimators": 10}
    if dv is not None:
        cfg["dataset_version"] = dv
    if excluded is not None:
        cfg["excluded_trips"] = list(excluded)
    return {"model": name, "family": "fam-" + name, "config": cfg,
            "cv_macro_f1_mean": 0.90, "cv_macro_f1_std": 0.02, "cv_top1_mean": top1 - 0.01, "cv_top1_std": 0.04,
            "cv_log_loss_mean": 0.12, "train_top1": top1 + 0.003, "train_macro_f1": 0.91,
            "test_accuracy": 0.95, "test_precision": 0.86, "test_recall": 0.84, "test_macro_f1": 0.91,
            "test_log_loss": 0.105, "test_top1": top1, "test_candidate_miss_rate": 0.0027,
            "test_route_continuity_argmax": None, "test_confusion_matrix": [[100, 5], [7, 20]],
            "train_time_s": 6.5, "inference_ms_per_point": 0.02, "seed": 42}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def write_folds(path: Path, values: list[float]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "macro_f1", "top1", "log_loss"])
        for i, v in enumerate(values):
            w.writerow([i, 0.9, v, 0.1])


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """logreg: no cv_folds.csv.  lightgbm: no metrics_all_trips.json."""
    res = tmp_path / "results"
    write_json(res / "logreg" / "metrics.json", fake_metrics("logreg", 0.84))
    write_json(res / "logreg" / "metrics_all_trips.json", fake_metrics("logreg", 0.8419, excluded=()))
    write_json(res / "lightgbm" / "metrics.json", fake_metrics("lightgbm", 0.856))
    write_folds(res / "lightgbm" / "cv_folds.csv", [0.89, 0.85, 0.88, 0.76, 0.86])
    (res / "eda").mkdir()                          # folders without metrics are ignored
    return tmp_path


def run(tmp: Path, *extra: str) -> int:
    return make_charts.main(["--results-dir", str(tmp / "results"), "--out-dir", str(tmp / "out"),
                             "--optuna-dir", str(tmp / "no_optuna"), *extra])


def test_runs_writes_charts_and_table(tree, capsys):
    assert run(tree) == 0
    out = capsys.readouterr().out
    assert "Models found (primary): logreg, lightgbm" in out
    for name in PNGS + ["summary_table.csv", "summary_table.md"]:
        f = tree / "out" / name
        assert f.is_file() and f.stat().st_size > 0, name
    assert (tree / "out" / "cv_top1_per_fold.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_missing_inputs_are_skipped_with_warnings(tree, capsys):
    assert run(tree) == 0
    out = capsys.readouterr().out
    assert "WARNING: logreg: no cv_folds.csv" in out
    assert "WARNING: lightgbm: no metrics_all_trips.json" in out
    assert "WARNING: optuna: no trials CSV found" in out
    assert not (tree / "out" / "optuna_history.png").exists()


def test_table_values_equal_the_metrics_files(tree):
    run(tree)
    with open(tree / "out" / "summary_table.csv", newline="") as f:
        rows = {r["model"]: r for r in csv.DictReader(f)}
    assert set(rows) == {"logreg", "lightgbm"}
    for key in ("logreg", "lightgbm"):
        src = json.loads((tree / "results" / key / "metrics.json").read_text())
        for col in ("cv_top1_mean", "cv_top1_std", "cv_macro_f1_mean", "test_top1", "test_macro_f1",
                    "test_log_loss", "train_top1", "train_time_s", "inference_ms_per_point"):
            assert float(rows[key][col]) == src[col], (key, col)
        assert rows[key]["test_route_continuity_argmax"] == ""          # null stays empty, never invented
        assert json.loads(rows[key]["excluded_trips"]) == [33, 54]
    md = (tree / "out" / "summary_table.md").read_text(encoding="utf-8")
    assert "| LightGBM | fam-lightgbm | 84.60 ± 4.00 |" in md               # mean ± std formatting
    assert "N/A" in md                                                     # null continuity shown as N/A


def test_wrong_dataset_version_warns_and_marks_footnote(tree, capsys):
    write_json(tree / "results" / "knn" / "metrics.json", fake_metrics("knn", 0.846, dv=1))
    assert run(tree) == 0
    out = capsys.readouterr().out
    assert "WARNING: knn: dataset_version is 1, expected 2" in out
    assert "knn: dataset_version is 1" in (tree / "out" / "summary_table.md").read_text()


def test_missing_excluded_trips_and_run_summary_override(tree, capsys):
    write_json(tree / "results" / "mlp" / "metrics.json", fake_metrics("mlp", 0.85, excluded=None))
    write_json(tree / "results" / "lightgbm" / "run_summary.json", {"dataset_version": 3, "excluded_trips": [33, 54]})
    run(tree)
    out = capsys.readouterr().out
    assert "WARNING: mlp: excluded_trips not recorded" in out
    assert "WARNING: lightgbm: dataset_version is 3" in out


def test_all_trips_mode_uses_the_other_file_and_its_own_folder(tree, capsys):
    assert run(tree, "--use", "all_trips") == 0
    out = capsys.readouterr().out
    assert "Models found (all_trips): logreg" in out
    assert "WARNING: lightgbm: no metrics_all_trips.json - model skipped" in out
    with open(tree / "out" / "all_trips" / "summary_table.csv", newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert float(row["test_top1"]) == 0.8419
    assert not (tree / "out" / "summary_table.csv").exists()             # primary outputs untouched


def test_optuna_history_from_trials_csv(tree, capsys):
    opt = tree / "results" / "optuna_lightgbm"
    opt.mkdir()
    with open(opt / "trials.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "value", "state"])
        w.writerows([[0, 0.850, "COMPLETE"], [1, 0.851, "COMPLETE"], [2, "", "FAIL"], [3, 0.8505, "COMPLETE"]])
    assert run(tree) == 0
    assert (tree / "out" / "optuna_history.png").is_file()
    assert make_charts.read_trials(opt / "trials.csv", []) == [
        {"number": 0, "value": 0.850}, {"number": 1, "value": 0.851}, {"number": 3, "value": 0.8505}]


def test_output_is_deterministic(tree):
    run(tree)
    first = {p.name: p.read_bytes() for p in (tree / "out").iterdir() if p.is_file()}
    run(tree)
    second = {p.name: p.read_bytes() for p in (tree / "out").iterdir() if p.is_file()}
    assert first == second


def test_no_models_exits_non_zero(tmp_path, capsys):
    (tmp_path / "results").mkdir()
    assert run(tmp_path) == 1
    assert "no model" in capsys.readouterr().out.lower()


def test_same_colour_per_model_everywhere(tree):
    models = make_charts.discover(tree / "results", "primary", [])
    colors = {m.key: m.color for m in models}
    assert colors["logreg"] == make_charts.PALETTE[0] and colors["lightgbm"] == make_charts.PALETTE[1]
    # adding a model must not repaint the existing ones beyond the fixed order
    write_json(tree / "results" / "knn" / "metrics.json", fake_metrics("knn", 0.846))
    colors2 = {m.key: m.color for m in make_charts.discover(tree / "results", "primary", [])}
    assert colors2["logreg"] == colors["logreg"]
