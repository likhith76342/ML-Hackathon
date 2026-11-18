# Version 1 evidence (superseded for the main comparison)

The values below are the original all-trip results, preserved for the required v1 comparison. Version 2 excludes trips 33 and 54; use the current metrics.json files for the v2 results.

# Zubed: experiment evidence for the team report

Use these as notes for the team's own analysis, not as a substitute for understanding or independently reviewing the results. All figures below come from `results/logreg/metrics.json`, `results/random_forest/metrics.json`, and their fold CSVs. Point-level labels are approximations derived from the supplied reference route. Dataset and model code use the team-pinned environment and 80 training / 10 student-test trips.

## Preprocessing and features

Each trip is projected into local metres using an equirectangular projection centred on the mean track longitude and latitude (Earth radius 6,371 km). Directed road segments within 50 m are sorted by point-to-segment distance, retaining up to 10; points without candidates have a placeholder. A monotone dynamic program aligns observations to route positions while minimizing summed point-to-segment distance. Reference routes are used for labels and evaluation, not inference features.

The features are distance, distance rank, distance minus the minimum, distance ratio, difference between GPS heading and arc bearing, speed, a low-speed heading-valid flag, previous and next time gaps, number of candidates, arc length and projection position. Direction matters because opposite road arcs may share geometry. The five folds are grouped by trip; the committed `data/processed/folds.csv` must be shared for consistent comparisons.

## Two model experiments

| Metric | Logistic regression | Random forest |
|---|---:|---:|
| Mean grouped-CV top-1 | 0.7564 | 0.7652 |
| Grouped-CV top-1 standard deviation | 0.0977 | 0.0971 |
| Mean grouped-CV candidate macro F1 | 0.8815 | 0.9030 |
| Train candidate macro F1 | 0.8790 | 0.9149 |
| Student-test top-1 | 0.8420 | 0.8555 |
| Student-test candidate macro F1 | 0.8905 | 0.9127 |
| Student-test candidate log loss | 0.1529 | 0.1065 |
| Full-data training time (seconds) | 4.19 | 88.65 |
| Inference time (ms per point) | 0.0027 | 0.0168 |

Logistic regression uses standard scaling and L2-regularised logistic loss (C=1, maximum 500 iterations). The random forest uses 20 trees, maximum depth 15, minimum leaf size 5 and the Gini split criterion. Both use seed 42. In this two-model experiment, random forest scored higher on these aggregate scores but took longer to train and run; inference times are machine-specific. The random forest's train macro F1 exceeds its mean validation-fold macro F1 by about 0.012, which is limited evidence by itself; compare fold-level results and further experiments before diagnosing overfitting. Training and CV top-1 summaries use different averaging, so their raw difference alone does not establish overfitting.

The comparison is preliminary until KNN, MLP and LightGBM results on the same features and folds are available. Candidate macro F1 and log loss are binary candidate-level metrics; top-1 is observation-level. Route continuity is not yet available. The 11.74% overall training miss rate and targeted spatial checks are documented in `INVESTIGATION.md`.

