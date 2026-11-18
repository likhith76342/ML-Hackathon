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
python3.11 -m venv .venv                         # Windows: py -3.11 -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Use **Python 3.11**. Several pins (`numpy==1.26.4`, `scipy==1.13.1`, `pyarrow==17.0.0`) have no wheels for 3.13. On macOS, LightGBM also needs OpenMP: `brew install libomp` (the API and UI run without it).

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

Laptop 1 runs the API. It holds the data, the model and the ground truth. Laptop 2 runs the UI, which talks to the API over HTTP only. Both need the repo and `pip install -r requirements.txt`. Only Laptop 1 needs `data/raw/...`.

### Laptop 1: API

Run these from the repo root with the venv active. `EVAL_DATA_DIR` is the folder of 8-digit trip folders to serve. Use `student_test` for practice and the evaluator's folder on demo day. If it isn't set, the API uses `api.eval_data_dir` from `config.yaml`. Relative paths resolve against the repo root.

macOS / Linux:

```bash
source .venv/bin/activate
export EVAL_DATA_DIR=data/raw/student_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
$env:EVAL_DATA_DIR = "data/raw/student_test"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

On startup the console prints the mode, the data folder and the trip IDs it found. If the folder is missing or has no trip folders, the API refuses to start and says why. Each request then logs one line, which is what the evaluators will look at:

```
2026-11-18 11:32:02 | INFO    | PREDICT trip=52  client=192.168.1.7     points=262    mode=MOCK | candidates=N/A model=N/A viterbi=N/A total=7.4ms | top1_hmm=N/A top1_argmax=N/A route_f1=N/A
```

Endpoints: `GET /health`, `GET /trips`, `POST /predict` with body `{"trip_id": 52}`. Interactive docs are at `http://<ip>:8000/docs`.

### Find Laptop 1's IP

| OS | Command | Look for |
|---|---|---|
| Windows | `ipconfig` | "IPv4 Address" under the Wi-Fi adapter |
| macOS | `ipconfig getifaddr en0` | the address it prints (or `ifconfig`, `inet` under `en0`) |
| Linux | `ip a` | `inet` under the Wi-Fi interface (e.g. `wlan0`) |

Use the `192.168.x.x` / `10.x.x.x` / `172.16-31.x.x` address, not `127.0.0.1`.

### Laptop 2: UI

macOS / Linux:

```bash
source .venv/bin/activate
export API_URL=http://<laptop1-ip>:8000
streamlit run ui/app.py
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
$env:API_URL = "http://<laptop1-ip>:8000"
streamlit run ui/app.py
```

The API URL can also be changed in the UI sidebar. Click **Reconnect / refresh trips** after changing it. Requests to `/predict` time out after 600 s. Set `PREDICT_TIMEOUT_S` to change that.

### Checklist

- Put both laptops on the **same network**. College Wi-Fi often blocks device-to-device traffic, so keep a phone hotspot as a fallback.
- **Firewall.** On Windows, allow Python through the firewall when prompted (tick *Private networks*). If you missed the prompt, open an admin PowerShell and run `New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`. On macOS, allow incoming connections for Python if asked (System Settings → Network → Firewall).
- From **Laptop 2's browser**, open `http://<laptop1-ip>:8000/health` before starting the UI. It must show `"status":"ok"`.
- Run every trip in the list through the UI once, end to end, including the large ones.

### How to tell if you're in mock mode

Mock mode is used automatically when `artifacts/final_model.joblib` doesn't exist. The API then works end to end, but the "predicted route" is just the raw GPS track joined up and **every metric is null**. Nothing is made up. You can tell by any of these:

- the API console prints `*** MOCK MODE ... ***` at startup and a `MOCK MODE` warning on each request, and every request line says `mode=MOCK`;
- `GET /health` returns `"mode": "mock"`, and every `/predict` response has `"mock": true`;
- the UI shows a red **MOCK MODE: no trained model loaded** banner, and all metrics read N/A.

When the real model lands in `artifacts/`, restart the API. It will report `mode: real` and run `pathsnap.pipeline.predict_trip`. The swap point is `api/service.py`.

### Tests

```bash
pytest tests/
```

These need `data/raw/student_test` and skip the data-dependent tests if it's missing. They check every endpoint, run `/predict` on every student-test trip, check that no ground-truth key ever appears in a response, and run the geometry helper on a real route.

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
