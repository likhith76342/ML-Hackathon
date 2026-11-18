"""Out-of-fold (OOF) probabilities of the final model for the 80 training trips (owner: Keshav).

    python -m hmm_experiments.make_oof_probs

Why: the HMM decoder's beta / w_e must be tuned on probabilities that look like the ones it will see at
inference. Probabilities from a model that was fitted on the same trip are over-confident, so each
training trip is scored by a model fitted WITHOUT that trip's fold.

How (mirrors optimisation/finalize_study.py):
  * final model = LightGBM pipeline from models.lightgbm.train.make_pipeline with optimisation/best_params.json
    (fixed n_estimators, no early stopping), features pathsnap.features.FEATURE_COLUMNS
  * shared data/processed/folds.csv (5 trip-grouped folds), not reshuffled
  * dataset v2: flagged trips (pathsnap.quality.flagged_trip_ids) are excluded from FITTING; they are still
    scored (by the model of their fold) so the file covers every candidate row, but tune.py skips them
  * placeholder rows (arc_id = -1) get prob 0
Student-test and evaluator data are not read.

Outputs:
  data/processed/oof_probs_train_lightgbm.parquet   trip_id, point_idx, arc_id, prob (row order of train_candidates)
  results/hmm/oof_probs_lightgbm.json                per-fold OOF Top-1 and run details
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from models.lightgbm.train import make_pipeline
from pathsnap.config import CFG, repo_path
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.metrics import top1_accuracy
from pathsnap.quality import flagged_trip_ids
from hmm_experiments.common import results_file


def main():
    base = repo_path(CFG["processed_dir"])
    train = pd.read_parquet(base / "train_candidates.parquet")
    refs = pd.read_parquet(base / "point_refs.parquet")
    folds = pd.read_csv(base / "folds.csv")
    params = json.loads(repo_path("optimisation/best_params.json").read_text(encoding="utf-8-sig"))
    excluded = set(flagged_trip_ids())
    fold_of = dict(zip(folds.trip_id.astype(int), folds.fold.astype(int)))
    if set(train.trip_id.unique()) != set(fold_of):
        raise ValueError("folds.csv must cover the training trips exactly")

    tfold = train.trip_id.map(fold_of).to_numpy()
    real = (train.arc_id >= 0).to_numpy()
    keep = ~train.trip_id.isin(excluded).to_numpy()
    prob = np.zeros(len(train))
    per_fold = []
    for k in sorted(folds.fold.unique()):
        fit_rows = real & keep & (tfold != k)
        score_rows = real & (tfold == k)
        t0 = time.perf_counter()
        model = make_pipeline(**params)
        model.fit(train.loc[fit_rows, FEATURE_COLUMNS], train.loc[fit_rows, "label"])
        prob[score_rows] = model.predict_proba(train.loc[score_rows, FEATURE_COLUMNS])[:, 1]
        trips_k = sorted(int(t) for t, f in fold_of.items() if f == k)
        scored = [t for t in trips_k if t not in excluded]
        sub = train.trip_id.isin(scored).to_numpy()
        top1 = top1_accuracy(train[sub], prob[sub], refs[refs.trip_id.isin(scored)])
        per_fold.append({"fold": int(k), "n_trips_scored": len(scored), "trips_excluded_from_score":
                         [t for t in trips_k if t in excluded], "oof_top1": float(top1),
                         "fit_rows": int(fit_rows.sum()), "fit_s": time.perf_counter() - t0})
        print(per_fold[-1], flush=True)

    out = pd.DataFrame({"trip_id": train.trip_id.to_numpy(), "point_idx": train.point_idx.to_numpy(),
                        "arc_id": train.arc_id.to_numpy(), "prob": prob})
    path = base / "oof_probs_train_lightgbm.parquet"
    out.to_parquet(path, index=False)
    summary = {"model": "lightgbm (final_model settings)", "params": params, "features": FEATURE_COLUMNS,
               "dataset_version": CFG.get("dataset_version"), "excluded_from_fit": sorted(excluded),
               "oof_top1_mean_over_folds": float(np.mean([f["oof_top1"] for f in per_fold])),
               "folds": per_fold, "output": str(path.relative_to(repo_path("."))).replace("\\", "/")}
    with open(results_file("oof_probs_lightgbm.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("oof_top1_mean_over_folds", "excluded_from_fit")}, indent=2))


if __name__ == "__main__":
    main()
