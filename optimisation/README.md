# LightGBM Bayesian optimisation

The five model results in `results/comparison.csv` use dataset version 2 and exclude training trips 33 and 54. LightGBM has the highest mean grouped-CV observation-level Top-1 in the recorded comparison; its lead over random forest is small. Reference point labels are approximate.

Run from the repository root with the team's pinned Python 3.11 environment and the processed data available:

```bash
python -m optimisation.tune_lightgbm --trials 50
```

The TPE sampler uses seed 42. Each trial uses the shared five trip-grouped folds, excludes the flagged training trips, and scores mean observation-level Top-1. Each outer training fold is split by trip for early stopping; its held-out fold is used only for scoring. Candidate macro F1, log loss, per-fold Top-1 and best iteration are also recorded. The search considers num_leaves 15–127, learning_rate 0.02–0.2, min_child_samples 10–100, colsample_bytree 0.6–1.0, subsample 0.6–1.0, and L1/L2 regularisation 1e-8–10. Learning rate, leaf count, minimum child count and regularisation use log-scale sampling. A maximum of 1,000 boosting rounds and 50-round early stopping are used for each fold.

The script checkpoints `trials.csv`, an SQLite study and sampler state after each completed trial. It fingerprints the data, folds, quality file and tuning code; it stops instead of resuming when those inputs change. The SQLite database and sampler state are local working files, not committed. To resume, keep both and rerun the same command. The 50-trial search can take substantial time.

After at least 50 completed trials, the script chooses the best CV trial, refits using the median of its fold-specific best iterations, and only then loads the student-test candidate table for evaluation. The shared point-reference file is read earlier, but is filtered to retained training-trip IDs for the search. It writes `best_params.json`, `summary.json`, `artifacts/final_model.joblib`, `artifacts/feature_columns.json` and plots of optimisation history and parameter importance. A run with fewer than 50 completed trials is incomplete and does not publish a selected model or student-test result.

The team still needs to address the MLP artifact compatibility issue and missing loss history noted during the model audit. That issue does not change the v2 fold scores, but it remains an integration limitation.


For execution efficiency, from trial 10 onward independent folds run two at a time; trial 9 was interrupted and recorded as failed. The folds, data, scoring, model settings and search space did not change.

## Stopped run: 18 completed trials

At the user's request, the search was stopped after 18 completed trials (three interrupted trials are recorded as failed). This is shorter than the team brief's stated minimum of 50; do not describe it as a 50-trial search. `python -m optimisation.finalize_study` was run to select the highest-CV trial and perform one post-selection student-test evaluation. It requires more than 10 completed trials and checks the recorded dataset and code fingerprint before refitting.

The completed-trial log is `optimisation/trials.csv`; final parameters and measurements are in `best_params.json` and `summary.json`. The CSV includes failed/interrupted trials, which are not part of the 18 completed trials. The search uses no student-test candidate data for parameter choice. The student-test reference rows share the parquet file loaded earlier but are filtered out of CV and scoring until final evaluation.
