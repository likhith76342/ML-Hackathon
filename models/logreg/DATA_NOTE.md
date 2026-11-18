# Shared feature and training notes

The shared pipeline uses the team config's 50 m candidate radius, up to 10 directed arcs per point, 5 trip-grouped folds and seed 42. `pathsnap.features.FEATURE_COLUMNS` is authoritative. Coordinates are projected around each trip's mean GPS longitude and latitude. Candidate generation indexes complete arc geometry and leaves a placeholder when no arc lies within the radius.

The route-position alignment finds the global nondecreasing sequence of reference positions minimizing the sum of point-to-segment distances. It preserves repeated route positions, but the resulting point labels remain approximations: noise, loops, gaps and ambiguous crossings can still lead to incorrect labels. Ground-truth routes are not read by the candidate generation or inference path.

On the supplied 80 training trips, 24,473 of 208,417 observations (11.74%) lack the aligned reference arc among their candidates. 20,538 observations have aligned reference distance greater than 100 m. The largest problem cases are trip 54 (90.67% miss; 16,453 points >100 m) and trip 33 (67.13% miss; 2,978 points >100 m). Review these cases with the team before changing the candidate radius or treating these labels as definitive. Per-trip statistics are in `results/logreg/data_quality_by_trip.csv`.

Run `python -m pathsnap.build_dataset` after placing raw trips under the configured data_root. Then run `python -m models.logreg.train` and `python -m models.random_forest.train`. Logistic regression uses a fitted StandardScaler; random forest uses 20 trees with max_depth=15, min_samples_leaf=5, and n_jobs=2 to make grouped experiments practical on this computer. Both use the shared candidate and point metrics. Training outputs use only the training and student-test partitions. The point labels and results are approximate, not independently verified ground truth.

`test_route_continuity_argmax` is null pending the sequence metric owned by Keshav. Final model selection, cross-team comparisons and Bayesian optimisation await the other model owners' results. Likhith owns config.yaml and requirements.txt; the repository now provides exact pinned dependencies.

The earlier preliminary run used Python 3.11.9, NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1, scikit-learn 1.9.1, Shapely 2.1.2, pyarrow 26.0.1 and PyYAML 6.0.3. Verify compatibility before loading a joblib artifact in another environment.


The current dataset and grouped folds were regenerated using the pinned versions in requirements.txt. GroupKFold can assign tied trip groups differently across scikit-learn versions; all teammates must use the committed folds.csv rather than independently reassigning folds. The earlier preliminary metrics were produced under a different version and should not be compared with runs using these folds.

Both models have now been refit and evaluated using the team's pinned versions and the regenerated shared folds. Mean grouped-CV Top-1 was 0.7564 for logistic regression and 0.7652 for random forest; student-test Top-1 was 0.8420 and 0.8555, respectively. These are approximate reference-alignment results. The team-wide winner cannot be selected until the other three models report.
