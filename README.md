# ML-Hackathon — PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University.
**Team:** Likhith, Mannu, Zubed, Keshav

PathSnap matches noisy GPS tracks to a road network. For each trip it works in three stages:

1. **Candidates.** Find the road arcs near each GPS point.
2. **Classifier.** Score each (point, arc) pair with the probability that this arc is the right one.
3. **HMM + Viterbi.** Turn the per-point scores into one connected route, using road connectivity.

The system runs on two laptops. A FastAPI backend holds the model and ground truth, and a Streamlit UI calls it over the network.

> **Team:** read `docs/PathSnap_Team_Brief.pdf` before writing code. It defines the shared formats (columns, function signatures, metrics, file paths) that everyone must follow.

---

## Repository layout

```
ML-Hackathon/
├── config.yaml            shared constants (radius, K, seed, folds, gap threshold, API)
├── requirements.txt
├── docs/                  team brief
├── data/
│   ├── raw/               train/ and student_test/ trip folders (not committed)
│   └── processed/         candidate tables + folds.csv            (Zubed)
├── pathsnap/              shared package — import, never copy
│   ├── config.py          loads config.yaml
│   ├── io.py              load_trip / load_route / list_trips
│   ├── geo.py             projection, distances, bearings          (Zubed)
│   ├── candidates.py      build_candidates                         (Zubed)
│   ├── features.py        FEATURE_COLUMNS                          (Zubed)
│   ├── labels.py          per-point reference labels               (Zubed)
│   ├── metrics.py         candidate + observation metrics          (Zubed)
│   ├── route_metrics.py   sequence metrics                         (Keshav)
│   ├── build_dataset.py   builds data/processed/*                  (Zubed)
│   ├── graph.py           road graph + route distance              (Keshav)
│   ├── hmm.py             viterbi + reconstruct_route              (Keshav)
│   └── pipeline.py        end-to-end predict_trip                  (Likhith)
├── eda/                   EDA + Folium maps                        (Mannu)
├── models/                logreg, random_forest (Zubed) · knn, mlp (Mannu) · lightgbm (Likhith)
├── optimisation/          Optuna study                             (Zubed)
├── hmm_experiments/       decoder tuning + HMM comparison          (Keshav)
├── api/                   FastAPI backend — Laptop 1               (Likhith)
├── ui/                    Streamlit UI — Laptop 2                  (Likhith)
├── artifacts/             final_model.joblib, feature_columns.json, hmm_config.json
├── results/               metrics.json per model, CV folds, comparison.csv
└── reports/figures/       every figure used in the report (+ maps/)
```

---

## Setup

```bash
git clone https://github.com/<owner>/ML-Hackathon.git
cd ML-Hackathon
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Data:** the raw data is not committed. Copy the dataset's `train/` and `student_test/` folders into `data/raw/`, so that you have `data/raw/train/00000010/00000010.track` and so on.

Check that it loads:

```bash
python -c "from pathsnap.io import list_trips; print(list_trips('data/raw/student_test'))"
```

All scripts run **from the repo root** as modules, e.g. `python -m models.lightgbm.train`.

---

## Pipeline (run in order)

| Step | Command | Output |
|---|---|---|
| 1. Build candidates, labels, folds | `python -m pathsnap.build_dataset` *(Zubed)* | `data/processed/*.parquet`, `folds.csv` |
| 2. Train each model | `python -m models.<name>.train` | `results/<name>/metrics.json` |
| 3. Compare models | *(Zubed, TBD)* | `results/comparison.csv`, figures |
| 4. Bayesian optimisation | *(Zubed, TBD)* | `optimisation/trials.csv`, `artifacts/final_model.joblib` |
| 5. Tune HMM | *(Keshav, TBD)* | `artifacts/hmm_config.json` |

---

## Deployment (two laptops)

**Laptop 1: API**

```bash
# point at the folder of trips to serve (student_test for testing, evaluator folder on demo day)
export EVAL_DATA_DIR=data/raw/student_test        # Windows PowerShell: $env:EVAL_DATA_DIR="data/raw/student_test"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Laptop 2: UI**

```bash
export API_URL=http://<laptop1-ip>:8000            # Windows PowerShell: $env:API_URL="http://<ip>:8000"
streamlit run ui/app.py
```

**Checklist**

- Put both laptops on the same network. If college Wi-Fi blocks device-to-device traffic, use a phone hotspot.
- Find Laptop 1's IP with `ipconfig` (Windows) or `ip a` / `ifconfig` (macOS/Linux).
- On Windows, allow Python through the firewall for port 8000.
- Open `http://<laptop1-ip>:8000/health` in Laptop 2's browser before starting the UI.

---

## Team conventions

The full rules are in `docs/PathSnap_Team_Brief.pdf`, section 12 (file ownership and Git workflow). In short:

- **Python 3.11** for everyone. Install from the pinned `requirements.txt`, so that models trained on one laptop load on the API laptop.
- **Each file has one owner.** Only the owner edits it. Likhith owns `requirements.txt` and `config.yaml`, so ask him for changes.
- Each person works on their own branch (`likhith`, `mannu`, `zubed`, `keshav`). Before every session: `git fetch origin && git merge origin/main`. Push small, open a PR into `main` every 1 to 2 hours, and Likhith merges. Nobody commits to `main` directly.
- Use seed `42` everywhere. Use `pathlib` and paths from `config.yaml`; no absolute paths.
- Every model is a scikit-learn Pipeline with `predict_proba`, and saves `results/<name>/metrics.json` with exactly the keys in the brief (section 10).
- The processed data isn't in git. Regenerate it with `python -m pathsnap.build_dataset` or copy it from the team Drive. `folds.csv` is committed.
- The prediction path never reads `.route`. Only labelling and the server-side evaluator do.
- Never commit secrets or the raw data. The AI-assistance disclosure goes in the report.
