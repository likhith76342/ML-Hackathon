"""MLP (neural network).  Owner: Mannu.
Use scikit-learn MLPClassifier (no PyTorch) so the saved model is a normal sklearn Pipeline
with predict_proba - the API loads every model the same way.

Follow the common training template (brief section 10):
  1. load data/processed/{train,test}_candidates.parquet + folds.csv; drop arc_id == -1 rows for fitting
  2. X = df[FEATURE_COLUMNS], y = df.label; build an sklearn Pipeline (scaler inside if needed)
  3. 5-fold grouped CV using folds.csv -> results/mlp/cv_folds.csv
  4. refit on all 80 train trips, evaluate on student_test, time training + inference
  5. write results/mlp/metrics.json with EXACTLY the keys in brief section 10
  6. save fitted pipeline -> results/mlp/model.joblib

Run from repo root:  python -m models.mlp.train
"""
from pathsnap.config import CFG
from pathsnap.features import FEATURE_COLUMNS
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from models.logreg.training import run

MODEL_NAME = "mlp"
FAMILY = "neural network"


def main():
    model = lambda: make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            learning_rate_init=1e-4, batch_size=512, max_iter=50,
            early_stopping=True, validation_fraction=0.1,
            random_state=CFG["seed"], verbose=False,
        ),
    )
    run(MODEL_NAME, FAMILY, model, score_full_train=False)


if __name__ == "__main__":
    main()
