"""LightGBM (boosting).  Owner: Likhith.  Brief sections 9, 10 and 11.1.

Run from the repo root:
    python -m models.lightgbm.train                    # all 80 training trips -> *_all_trips outputs
    python -m models.lightgbm.train --exclude-flagged  # drop quality-flagged trips -> primary outputs

Mirrors Zubed's models/logreg/training.py so the numbers are directly comparable:
same parquet files, same folds.csv, same shared metric functions (pathsnap.metrics), same
metrics.json keys, placeholders (arc_id == -1) dropped for fitting but scored with
probability 0 so Top-1 counts every GPS point.

Leakage rules:
  * Early stopping uses a trip-grouped validation split carved out of each CV TRAINING set
    (GroupShuffleSplit, seed 42). The held-out CV fold is never used for early stopping.
  * The final refit uses n_estimators = median CV best_iteration, with no early stopping.
    student_test is only scored, never used for stopping or tuning.
  * With --exclude-flagged, flagged trips are dropped from fitting and CV scoring; folds.csv is
    not reshuffled and student_test is never filtered.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lightgbm import LGBMClassifier, early_stopping, log_evaluation  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from pathsnap.config import CFG, repo_path  # noqa: E402
from pathsnap.features import FEATURE_COLUMNS  # noqa: E402
from pathsnap.metrics import candidate_metrics, candidate_miss_rate, top1_accuracy  # noqa: E402

MODEL_NAME = "lightgbm"
FAMILY = "boosting"
SEED = CFG["seed"]
EARLY_STOPPING_ROUNDS = 50
INNER_VAL_FRACTION = 0.15   # fraction of TRIPS in each CV training set held out for early stopping

# Brief 11.1 initial configuration. No class_weight: keeps probabilities calibrated for the HMM.
# deterministic + force_row_wise make multi-threaded runs reproducible (LightGBM docs).
BASE_PARAMS = dict(
    objective="binary", num_leaves=31, learning_rate=0.05, n_estimators=1000,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight=None,
    random_state=SEED, n_jobs=-1, deterministic=True, force_row_wise=True, verbose=-1,
)

# Exactly the brief section 10 keys, in the same order as results/logreg/metrics.json.
METRIC_KEYS = [
    "model", "family", "config", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "cv_top1_std",
    "cv_log_loss_mean", "train_top1", "train_macro_f1", "test_accuracy", "test_precision", "test_recall",
    "test_macro_f1", "test_log_loss", "test_top1", "test_candidate_miss_rate",
    "test_route_continuity_argmax", "test_confusion_matrix", "train_time_s", "inference_ms_per_point", "seed",
]


# ----------------------------------------------------------------------------- data
def load_excluded_trips(exclude_flagged: bool, training_ids: set[int]) -> list[int]:
    """Flagged training trips (dataset v2 policy), or [] when the flag is off."""
    if not exclude_flagged:
        return []
    processed = repo_path(CFG["processed_dir"])
    try:
        from pathsnap.quality import flagged_trip_ids
    except ImportError:
        flagged_trip_ids = None
    try:
        if flagged_trip_ids is not None:
            excluded = flagged_trip_ids()
        else:
            path = processed / "trip_quality.csv"
            if not path.exists():
                raise SystemExit(
                    "--exclude-flagged: neither pathsnap/quality.py nor data/processed/trip_quality.csv exists. "
                    "Merge Zubed's quality change (git merge origin/main) or run without the flag.")
            q = pd.read_csv(path)
            if not {"trip_id", "flagged"}.issubset(q.columns):
                raise SystemExit(f"{path} must have columns trip_id, far_fraction, flagged.")
            excluded = sorted(q.loc[q.flagged.astype(str).str.lower() == "true", "trip_id"].astype(int).tolist())
    except FileNotFoundError as e:
        raise SystemExit(f"--exclude-flagged: trip-quality file not found ({e}).")
    excluded = sorted(int(t) for t in excluded)
    if not set(excluded) <= training_ids:
        raise SystemExit(f"Quality file flags non-training trips: {sorted(set(excluded) - training_ids)}")
    return excluded


def load_data(exclude_flagged: bool):
    base = repo_path(CFG["processed_dir"])
    train = pd.read_parquet(base / "train_candidates.parquet")
    test = pd.read_parquet(base / "test_candidates.parquet")
    refs = pd.read_parquet(base / "point_refs.parquet")
    folds = pd.read_csv(base / "folds.csv")
    if set(train.trip_id.unique()) != set(folds.trip_id):
        raise ValueError("folds.csv must cover the training trips exactly")
    if set(test.trip_id.unique()) & set(folds.trip_id):
        raise ValueError("student_test trips overlap the training folds")
    excluded = load_excluded_trips(exclude_flagged, set(int(t) for t in folds.trip_id))
    train = train[~train.trip_id.isin(excluded)].reset_index(drop=True)   # folds.csv is NOT reshuffled
    if set(train.trip_id.unique()) != set(folds.trip_id) - set(excluded):
        raise ValueError("Quality filtering did not match folds")
    return train, test.reset_index(drop=True), refs, folds, excluded


# ----------------------------------------------------------------------------- model
def make_pipeline(**overrides) -> Pipeline:
    return Pipeline([("model", LGBMClassifier(**{**BASE_PARAMS, **overrides}))])


def fit_with_early_stopping(df: pd.DataFrame) -> tuple[Pipeline, int, dict, int]:
    """Fit on a CV training set; early-stop on a trip-grouped split of that SAME training set."""
    real = df[df.arc_id >= 0]
    gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRACTION, random_state=SEED)
    fit_idx, val_idx = next(gss.split(real, groups=real.trip_id))
    fit_df, val_df = real.iloc[fit_idx], real.iloc[val_idx]
    pipe = make_pipeline()
    pipe.fit(
        fit_df[FEATURE_COLUMNS], fit_df.label,
        model__eval_set=[(fit_df[FEATURE_COLUMNS], fit_df.label), (val_df[FEATURE_COLUMNS], val_df.label)],
        model__eval_names=["train", "valid"], model__eval_metric="binary_logloss",
        model__callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
    )
    model = pipe.named_steps["model"]
    return pipe, int(model.best_iteration_), model.evals_result_, int(val_df.trip_id.nunique())


def score(pipe: Pipeline, df: pd.DataFrame, refs: pd.DataFrame) -> dict:
    """Same scoring as Zubed's training.py, plus Top-1 restricted to reachable points."""
    valid = df.arc_id.ge(0).to_numpy()
    probs = np.zeros(len(df))
    start = time.perf_counter()
    probs[valid] = pipe.predict_proba(df.loc[valid, FEATURE_COLUMNS])[:, 1]
    elapsed = time.perf_counter() - start
    selected = refs[refs.trip_id.isin(df.trip_id.unique())]
    reachable = selected[selected.ref_in_candidates.astype(bool)]
    return {
        "cand": candidate_metrics(df.loc[valid, "label"], probs[valid]),
        "top1": top1_accuracy(df, probs, selected),
        "top1_ref_in_candidates": top1_accuracy(df, probs, reachable),
        "miss": candidate_miss_rate(df, selected),
        "predict_s": elapsed,
        "n_points": len(selected),
    }


# ----------------------------------------------------------------------------- figures
def plot_loss(history: pd.DataFrame, final_n: int, path: Path, title_suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    colors = plt.cm.tab10.colors
    for i, (fold, h) in enumerate(history.groupby("fold")):
        c = colors[i % 10]
        ax.plot(h.iteration, h.train_logloss, "--", color=c, lw=1, alpha=.7)
        ax.plot(h.iteration, h.valid_logloss, "-", color=c, lw=1.6, label=f"fold {fold}")
        best = int(h.best_iteration.iloc[0])
        ax.plot(best, h.loc[h.iteration == best, "valid_logloss"].iloc[0], "o", color=c, ms=5)
    ax.axvline(final_n, color="k", ls=":", lw=1.2, label=f"final n_estimators = {final_n}")
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("Binary log loss")
    ax.set_title(f"LightGBM loss per CV fold{title_suffix}\nsolid = early-stopping validation trips, "
                 "dashed = training trips, dot = best iteration", fontsize=10)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: list[list[int]], path: Path, title_suffix: str) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    ax.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(f"LightGBM student-test confusion (t=0.5){title_suffix}", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(pipe: Pipeline, path: Path, title_suffix: str) -> None:
    booster = pipe.named_steps["model"].booster_
    gain = pd.Series(booster.feature_importance("gain"), index=FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.barh(gain.index, gain.values / gain.sum(), color="#3d7192")
    ax.set_xlabel("Share of total split gain")
    ax.set_title(f"LightGBM feature importance (gain){title_suffix}", fontsize=10)
    ax.grid(axis="x", alpha=.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- notes
def write_notes(out: Path) -> None:
    """NOTES.md is generated from the run summaries so every number comes from a run."""
    runs = {}
    for label, fname in (("Primary run (flagged trips excluded)", "run_summary.json"),
                         ("All-trips run (no exclusion)", "run_summary_all_trips.json")):
        if (out / fname).exists():
            runs[label] = json.loads((out / fname).read_text(encoding="utf-8"))
    if not runs:
        return
    any_run = next(iter(runs.values()))
    L = ["# LightGBM notes (generated by `python -m models.lightgbm.train`; do not hand-edit numbers)", ""]
    L += ["## Configuration",
          f"- `LGBMClassifier` inside an sklearn `Pipeline([('model', ...)])`, so the saved object has `predict_proba`. "
          f"Brief 11.1 settings: `{json.dumps({k: v for k, v in BASE_PARAMS.items() if k != 'n_estimators'})}`.",
          f"- Features: the {len(FEATURE_COLUMNS)} shared `FEATURE_COLUMNS`. No scaler, because trees don't need one.",
          "- `class_weight=None` (no reweighting). Reweighting would distort the probabilities that the HMM uses as "
          "emissions, and the log loss (brief 9.1). Top-1 picks the highest-probability candidate per point, so "
          "it does not depend on the 0.5 threshold.",
          "- `deterministic=True` and `force_row_wise=True` were added so multi-threaded runs are reproducible.",
          f"- Seed {SEED}. Shared `folds.csv` (5 trip-grouped folds). Shared metric functions from `pathsnap.metrics`. "
          f"`dataset_version` = {any_run['dataset_version']}.", ""]
    L += ["## Early-stopping design (no leakage)",
          f"- In each CV fold, the fold's training trips are split by trip with `GroupShuffleSplit(test_size="
          f"{INNER_VAL_FRACTION}, random_state={SEED})`. LightGBM early-stops ({EARLY_STOPPING_ROUNDS} rounds, "
          "binary log loss) on that inner validation split. The held-out CV fold is only scored.",
          "- The final model is refit on all retained training trips with `n_estimators` = the median CV "
          "`best_iteration`, with no early stopping. student_test is never used for stopping or tuning.",
          "- `loss_history*.csv` and the loss-curve figure contain the real per-iteration `evals_result_` of each CV "
          "fold (inner-train vs inner-validation trips).", ""]
    for label, s in runs.items():
        m = s["metrics"]
        L += [f"## {label}",
              f"- Excluded training trips: {s['excluded_trips'] or 'none'}. Training trips used: {s['n_train_trips']}. "
              f"student_test trips: {s['n_test_trips']} (never filtered).",
              f"- CV best iterations per fold: {s['cv_best_iterations']}, so the final `n_estimators` is {s['final_n_estimators']}.",
              f"- CV Top-1 {m['cv_top1_mean']:.4f} ± {m['cv_top1_std']:.4f}; CV macro-F1 {m['cv_macro_f1_mean']:.4f} "
              f"± {m['cv_macro_f1_std']:.4f}; CV log loss {m['cv_log_loss_mean']:.4f}.",
              f"- Train (refit, scored on its own training trips): Top-1 {m['train_top1']:.4f}, macro-F1 {m['train_macro_f1']:.4f}.",
              f"- Train minus CV gap: Top-1 {m['train_top1'] - m['cv_top1_mean']:+.4f}, macro-F1 "
              f"{m['train_macro_f1'] - m['cv_macro_f1_mean']:+.4f}.",
              f"- student_test: Top-1 {m['test_top1']:.4f}, macro-F1 {m['test_macro_f1']:.4f}, log loss "
              f"{m['test_log_loss']:.4f}, accuracy {m['test_accuracy']:.4f}, precision {m['test_precision']:.4f}, "
              f"recall {m['test_recall']:.4f}, candidate miss rate {m['test_candidate_miss_rate']:.4f}.",
              f"- Top-1 restricted to GPS points whose reference arc is among the candidates (isolates the classifier "
              f"from the candidate ceiling; not in metrics.json): CV {s['cv_top1_ref_in_candidates_mean']:.4f}, "
              f"student_test {s['test_top1_ref_in_candidates']:.4f}.",
              f"- Full refit time {m['train_time_s']:.1f} s; inference {m['inference_ms_per_point']:.5f} ms per GPS point "
              "(predict_proba only).",
              "- `test_route_continuity_argmax` is null because `pathsnap/route_metrics.py` is still a stub.", ""]
    L += ["## Caveats",
          "- The point labels come from an approximate monotone alignment to the reference route (see "
          "models/logreg/DATA_NOTE.md). Every Top-1 figure is relative to those labels.",
          "- CV numbers from the all-trips run and the excluded run use different evaluation populations, so they are "
          "not a like-for-like comparison. student_test is identical in both runs.", ""]
    (out / "NOTES.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------- main
def run(exclude_flagged: bool) -> dict:
    suffix = "" if exclude_flagged else "_all_trips"
    title_suffix = " (flagged trips excluded)" if exclude_flagged else " (all training trips)"
    out = repo_path(CFG["results_dir"]) / MODEL_NAME
    figs = repo_path(CFG["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    train, test, refs, folds, excluded = load_data(exclude_flagged)
    print(f"[lightgbm] mode={'exclude-flagged' if exclude_flagged else 'all-trips'} excluded={excluded} "
          f"train_trips={train.trip_id.nunique()} train_rows={len(train):,} test_trips={test.trip_id.nunique()} "
          f"test_rows={len(test):,}", flush=True)

    fold_of = train.trip_id.map(dict(zip(folds.trip_id, folds.fold)))
    rows, histories = [], []
    for fold in sorted(folds.fold.unique()):
        t0 = time.perf_counter()
        pipe, best, evals, n_val_trips = fit_with_early_stopping(train[fold_of != fold])
        s = score(pipe, train[fold_of == fold].reset_index(drop=True), refs)
        c = s["cand"]
        rows.append({"fold": int(fold), "macro_f1": c["macro_f1"], "top1": s["top1"], "log_loss": c["log_loss"],
                     "candidate_miss_rate": s["miss"], "accuracy": c["accuracy"], "precision": c["precision"],
                     "recall": c["recall"], "confusion_matrix": json.dumps(c["confusion_matrix"]),
                     "best_iteration": best, "n_early_stop_val_trips": n_val_trips,
                     "top1_ref_in_candidates": s["top1_ref_in_candidates"]})
        tr_loss, va_loss = evals["train"]["binary_logloss"], evals["valid"]["binary_logloss"]
        histories.append(pd.DataFrame({"fold": int(fold), "iteration": np.arange(1, len(tr_loss) + 1),
                                       "train_logloss": tr_loss, "valid_logloss": va_loss, "best_iteration": best}))
        print(f"[lightgbm] fold {fold}: top1={s['top1']:.4f} macro_f1={c['macro_f1']:.4f} "
              f"log_loss={c['log_loss']:.4f} best_iter={best} ({time.perf_counter() - t0:.0f}s)", flush=True)

    cv = pd.DataFrame(rows)
    cv.to_csv(out / f"cv_folds{suffix}.csv", index=False)
    history = pd.concat(histories, ignore_index=True)
    history.to_csv(out / f"loss_history{suffix}.csv", index=False)

    # Final refit: fixed n_estimators, no early stopping, nothing from student_test.
    best_iters = cv.best_iteration.astype(int).tolist()
    final_n = int(np.median(best_iters))
    model = make_pipeline(n_estimators=final_n)
    real = train[train.arc_id >= 0]
    start = time.perf_counter()
    model.fit(real[FEATURE_COLUMNS], real.label)
    train_time = time.perf_counter() - start

    tr = score(model, train, refs)
    te = score(model, test, refs)
    joblib.dump(model, out / f"model{suffix}.joblib")

    params = {k: v for k, v in model.get_params(deep=True).items()
              if isinstance(v, (str, int, float, bool, type(None)))}
    params.update({"early_stopping_rounds": EARLY_STOPPING_ROUNDS, "early_stopping_val_fraction_of_trips":
                   INNER_VAL_FRACTION, "cv_best_iterations": best_iters, "final_n_estimators": final_n,
                   "excluded_trips": excluded, "dataset_version": CFG["dataset_version"]})
    tc = te["cand"]
    metrics = {
        "model": MODEL_NAME, "family": FAMILY, "config": params,
        "cv_macro_f1_mean": float(cv.macro_f1.mean()), "cv_macro_f1_std": float(np.std(cv.macro_f1)),
        "cv_top1_mean": float(cv.top1.mean()), "cv_top1_std": float(np.std(cv.top1)),
        "cv_log_loss_mean": float(cv.log_loss.mean()),
        "train_top1": tr["top1"], "train_macro_f1": tr["cand"]["macro_f1"],
        "test_accuracy": tc["accuracy"], "test_precision": tc["precision"], "test_recall": tc["recall"],
        "test_macro_f1": tc["macro_f1"], "test_log_loss": tc["log_loss"], "test_top1": te["top1"],
        "test_candidate_miss_rate": te["miss"],
        "test_route_continuity_argmax": None,          # route_metrics.py is still a stub
        "test_confusion_matrix": tc["confusion_matrix"],
        "train_time_s": train_time, "inference_ms_per_point": 1000 * te["predict_s"] / te["n_points"],
        "seed": SEED,
    }
    assert list(metrics) == METRIC_KEYS
    (out / f"metrics{suffix}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary = {"excluded_trips": excluded, "dataset_version": CFG["dataset_version"],
               "n_train_trips": int(train.trip_id.nunique()), "n_test_trips": int(test.trip_id.nunique()),
               "cv_best_iterations": best_iters, "final_n_estimators": final_n,
               "cv_top1_ref_in_candidates_mean": float(cv.top1_ref_in_candidates.mean()),
               "train_top1_ref_in_candidates": tr["top1_ref_in_candidates"],
               "test_top1_ref_in_candidates": te["top1_ref_in_candidates"], "metrics": metrics}
    (out / f"run_summary{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_loss(history, final_n, figs / f"model_lightgbm_loss_curve{suffix}.png", title_suffix)
    plot_confusion(tc["confusion_matrix"], figs / f"model_lightgbm_confusion{suffix}.png", title_suffix)
    plot_importance(model, figs / f"model_lightgbm_feature_importance{suffix}.png", title_suffix)
    write_notes(out)

    m = metrics
    print("\n" + "=" * 72)
    print(f"LightGBM summary: {'flagged trips EXCLUDED ' + str(excluded) if excluded else 'ALL training trips'}")
    print("=" * 72)
    print(f"  CV best iterations   : {best_iters} -> final n_estimators = {final_n}")
    print(f"  CV Top-1             : {m['cv_top1_mean']:.4f} ± {m['cv_top1_std']:.4f}")
    print(f"  CV macro-F1          : {m['cv_macro_f1_mean']:.4f} ± {m['cv_macro_f1_std']:.4f}")
    print(f"  CV log loss          : {m['cv_log_loss_mean']:.4f}")
    print(f"  Train Top-1 / mF1    : {m['train_top1']:.4f} / {m['train_macro_f1']:.4f}")
    print(f"  Test Top-1           : {m['test_top1']:.4f}   (candidate miss rate {m['test_candidate_miss_rate']:.4f})")
    print(f"  Test macro-F1 / loss : {m['test_macro_f1']:.4f} / {m['test_log_loss']:.4f}")
    print(f"  Top-1 | ref in cands : CV {summary['cv_top1_ref_in_candidates_mean']:.4f}, "
          f"test {summary['test_top1_ref_in_candidates']:.4f}   (summary/NOTES only)")
    print(f"  Train time           : {m['train_time_s']:.1f} s   Inference: {m['inference_ms_per_point']:.5f} ms/point")
    print(f"  Wrote                : {out / f'metrics{suffix}.json'}")
    print("=" * 72, flush=True)
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate LightGBM (brief section 10).")
    ap.add_argument("--exclude-flagged", action="store_true",
                    help="drop quality-flagged training trips from fitting and CV scoring (dataset v2 policy); "
                         "writes metrics.json. Default off -> metrics_all_trips.json.")
    run(ap.parse_args(argv).exclude_flagged)


if __name__ == "__main__":
    main()
