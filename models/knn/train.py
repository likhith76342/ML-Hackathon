"""KNN (instance-based).  Owner: Mannu.

Follow the common training template (brief section 10):
  1. load data/processed/{train,test}_candidates.parquet + folds.csv; drop arc_id == -1 rows for fitting
  2. X = df[FEATURE_COLUMNS], y = df.label; build an sklearn Pipeline (scaler inside if needed)
  3. 5-fold grouped CV using folds.csv -> results/knn/cv_folds.csv
  4. refit on all 80 train trips, evaluate on student_test, time training + inference
  5. write results/knn/metrics.json with EXACTLY the keys in brief section 10
  6. save fitted pipeline -> results/knn/model.joblib

Run from repo root:  python -m models.knn.train
"""
from pathsnap.config import CFG
from pathsnap.features import FEATURE_COLUMNS

MODEL_NAME = "knn"
FAMILY = "instance-based"


def main():
    raise NotImplementedError("Mannu: implement following brief section 10")


if __name__ == "__main__":
    main()
