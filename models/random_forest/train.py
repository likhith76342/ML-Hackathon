"""Random Forest (ensemble (bagging)).  Owner: Zubed.

Follow the common training template (brief section 10):
  1. load data/processed/{train,test}_candidates.parquet + folds.csv; drop arc_id == -1 rows for fitting
  2. X = df[FEATURE_COLUMNS], y = df.label; build an sklearn Pipeline (scaler inside if needed)
  3. 5-fold grouped CV using folds.csv -> results/random_forest/cv_folds.csv
  4. refit on all 80 train trips, evaluate on student_test, time training + inference
  5. write results/random_forest/metrics.json with EXACTLY the keys in brief section 10
  6. save fitted pipeline -> results/random_forest/model.joblib

Run from repo root:  python -m models.random_forest.train
"""
from pathsnap.config import CFG
from pathsnap.features import FEATURE_COLUMNS

MODEL_NAME = "random_forest"
FAMILY = "ensemble (bagging)"


def main():
    raise NotImplementedError("Zubed: implement following brief section 10")


if __name__ == "__main__":
    main()
