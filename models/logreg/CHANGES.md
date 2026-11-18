# Zubed's changes

- Implemented shared geometry, road-candidate generation, feature extraction, route-based point labeling, dataset preparation, and candidate/observation metrics.
- Generated shared trip-based cross-validation folds and trained logistic regression and random forest using the team's pinned dependencies.
- Saved the training scripts, fitted model artifacts, per-fold metrics, student-test results, and data-quality findings.
- Added checks for geometry, candidate placeholders, label alignment, and metrics.

The point labels are approximate. The training candidate miss rate is 11.7%; investigate affected trips before changing the shared settings. Final model selection and Bayesian optimisation await the other team models; route continuity is assigned to Keshav.
