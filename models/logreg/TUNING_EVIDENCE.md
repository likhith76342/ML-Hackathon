# LightGBM tuning evidence

The team selected LightGBM by mean observation-level Top-1 across the shared v2 trip-grouped folds. The Bayesian search used Optuna TPE (seed 42), with trips 33 and 54 excluded from training and CV. Each completed trial evaluated five held-out folds; its early-stopping validation trips were taken only from the other training folds. Student-test trips were scored only after the winning CV trial was fixed.

The user requested that the search stop after more than 10 trials. It completed 18 trials; three interrupted trials are marked failed and did not count. The earlier team brief specifies at least 50, so this run does not satisfy that original 50-trial requirement without team approval or additional trials. Search ranges, code and the trial record are in `optimisation/`.

| Metric | Untuned LightGBM | Selected 18-trial configuration |
|---|---:|---:|
| Mean grouped-CV Top-1 | 0.850487 | 0.851034 |
| Mean grouped-CV candidate macro F1 | 0.910567 | 0.910951 |
| Student-test Top-1 | 0.856054 | 0.856420 |
| Student-test candidate macro F1 | 0.913764 | 0.913167 |
| Student-test candidate log loss | 0.104007 | 0.104816 |
| Inference ms per point | 0.020894 | 0.123413 |

The selected trial's CV Top-1 was 0.000547 higher (0.055 percentage points). Student-test Top-1 was 0.000365 higher (0.037 percentage points), while student-test macro F1 and log loss were slightly worse. These small differences do not establish a reliable improvement; the point labels are approximate. The two inference timings were recorded on different runs and may have used different hardware, so they are not a controlled speed comparison.

Selected parameters: num_leaves=15, learning_rate=0.0216563, min_child_samples=10, colsample_bytree=0.734739, subsample=0.783470, reg_alpha=0.000007636, reg_lambda=0.0503283. The final model was refit with 743 trees, the median of the winning trial's five early-stopped best iteration counts. It uses the shared pipeline and feature order. The complete measurements and exact parameter values are in `optimisation/summary.json`; `optimisation/trials.csv` contains the real trial record.
