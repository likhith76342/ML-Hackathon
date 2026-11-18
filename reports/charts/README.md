# Model-comparison charts

`make_charts.py` builds every model-comparison figure and the summary table from the **saved** results. It never retrains anything, and no number in it is typed by hand.

```bash
python reports/charts/make_charts.py                  # primary runs (flagged trips excluded)
python reports/charts/make_charts.py --use all_trips  # all-trips runs -> reports/figures/model_comparison/all_trips/
```

Options: `--results-dir results`, `--out-dir reports/figures/model_comparison`, `--optuna-dir optimisation`.

## What it reads (read-only)

- **Model folders:** every `results/<model>/` that contains `metrics.json` (or `metrics_all_trips.json` with `--use all_trips`). Model folders are discovered automatically.
- **Optional files per model:** `cv_folds.csv` / `cv_folds_all_trips.csv` (per-fold Top-1) and `run_summary.json` (dataset version and exclusions).
- **Optuna history:** any `*trials*.csv` or `optuna*/*.csv` under `results/` or `--optuna-dir`, plus `summary.json` next to it for the tuned-vs-untuned comparison. This is optional.

A missing model or file is skipped with a printed `WARNING`. The script exits non-zero only if no model is found at all.

## Comparability check

For each model, `dataset_version` and `excluded_trips` are read from `run_summary.json` if present, otherwise from `metrics.json` → `config`. The script warns when the version isn't 2, when `excluded_trips` isn't recorded, or when a primary run excludes nothing. The warning is also printed in each chart's footnote and in `summary_table.md`, so models that may not be comparable are visible rather than silently mixed.

## Outputs (`reports/figures/model_comparison/`, PNG at 200 dpi)

| File | Shows |
|---|---|
| `cv_top1_by_model.png` | CV Top-1 mean ± std, sorted high to low |
| `cv_top1_per_fold.png` | Per-fold Top-1 for each model |
| `test_top1_and_macro_f1.png` | Student-test Top-1 (solid) and macro-F1 (hatched) |
| `cv_vs_test_gap.png` | CV vs student-test Top-1 per model (generalisation) |
| `time_vs_accuracy.png` | Training time and inference ms/point (log scale) vs test Top-1 |
| `confusion_matrices.png` | Row-normalised test confusion matrix per model, with counts |
| `log_loss_by_model.png` | Student-test (solid) and CV (hatched) log loss |
| `primary_vs_all_trips.png` | CV and test Top-1 with vs without the flagged trips |
| `optuna_history.png` | Best-so-far CV Top-1 by trial, and untuned vs tuned |
| `summary_table.csv` / `.md` | One row per model, from the same data as the charts |

**Optuna importance is not produced.** Parameter importance needs the study's search distributions, which the trials CSV doesn't store, and re-running Optuna isn't allowed here. See `reports/figures/model_lightgbm_optuna_importance.png` from the tuning run.

## Design

- **Colours:** each model always has the same colour, assigned in a fixed order (logreg, random forest, KNN, MLP, LightGBM, then any new model alphabetically). The colours come from the dataviz skill's colourblind-validated palette, and every mark is named on an axis or labelled directly.
- **Reproducibility:** figure sizes and dpi are fixed, and the PNGs carry no software or timestamp metadata. Re-running gives byte-identical files, which the tests check.
- **Tests:** `tests/test_charts.py` builds a fake results tree and checks the outputs, the warnings, and that the table values equal the input files.
