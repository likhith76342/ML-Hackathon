# Dataset version 2: training-trip quality policy

Version 2 adds `data/processed/trip_quality.csv` without changing the version 1 parquet files or `folds.csv`. The file is generated from training-trip references by `python -m pathsnap.quality`. A training trip is flagged when more than 50% of its GPS points have aligned reference distance strictly greater than 50 metres. The resulting flagged IDs are 33 and 54. These labels are approximate; the rule identifies poor agreement, not a proven cause.

Model and decoder owners can use:

```python
from pathsnap.quality import flagged_trip_ids
excluded_trips = flagged_trip_ids()
```

Exclude those IDs only from model fitting, training cross-validation scoring and HMM tuning. Keep the original fold assignments, all student-test trips, and inference/API support for every trip. No GPS points were deleted from the shared parquet files. The model scripts use the helper, store `dataset_version` and `excluded_trips` inside `config` in `metrics.json`, and retain the original model metrics as `metrics_all_trips.json` and original fold results as `cv_folds_all_trips.csv`.

For the 78 retained training trips, 3,038 of 183,260 points (1.66%) miss their aligned reference candidate, versus 24,473 of 208,417 (11.74%) across all 80. These rates describe different subsets; they do not demonstrate that the dataset issues were repaired or that excluded-trip performance improved. Student-test data is unchanged.

Checksums of all three parquet files and `folds.csv` matched the original shared version 1 archive after the v2 policy was implemented. Other model owners must use the shared exclusion helper and retrain before metrics are compared. The report should describe the exclusion rule, its limitations, and both all-trip and retained-trip results.

## Completed v2 model results

| Model | Mean CV Top-1, all trips | Mean CV Top-1, 78 retained trips | Student-test Top-1, all-trip model | Student-test Top-1, v2 model |
|---|---:|---:|---:|---:|
| Logistic regression | 0.75635 | 0.83827 | 0.84201 | 0.84191 |
| Random forest | 0.76523 | 0.84888 | 0.85553 | 0.85532 |

The CV results use different evaluation populations, so the increase must not be described as a like-for-like improvement. Student-test trips are identical in both runs. Exact values and all other metrics are in each model's `metrics.json` and `metrics_all_trips.json`. Each file records its dataset version and exclusions under `config`. The rerun uses the pinned team dependencies and original fold assignment.
