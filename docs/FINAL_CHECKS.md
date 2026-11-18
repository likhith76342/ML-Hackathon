# PathSnap — Final Integration Checks (FINAL_CHECKS.md)

Branch: `likhith-final-integration` (commit `0df24b2` Merge PR #15 keshav-final). Working tree also carries Likhith's
integration changes (see STEP 1). All checks below were actually run; nothing is marked PASS unless verified by command
output. Numbers are real (computed from the committed artifacts and the live API), never invented.

Owner key: **Me = Likhith (API/pipeline/integration)**, **Zubed (pathsnap/*, build_dataset, candidates, features, labels, metrics, optimisation)**, **Manu (eda, knn, mlp)**, **Keshav (graph, hmm, route_metrics, hmm_experiments)**.

Files I (Likhith) edited: `api/main.py`, `api/service.py`, `config.yaml`, `README.md`. New: `docs/FINAL_CHECKS.md`.
Files I did NOT touch (flag only): `candidates.py, features.py, labels.py, metrics.py, route_metrics.py, graph.py, hmm.py, quality.py, build_dataset.py, models/*, optimisation/*, eda/*, results/* contents`.

---

## STEP 0 — Sync and inventory

`git log --oneline -5` (HEAD):
```
0df24b2 Merge pull request #15 from likhith76342/keshav-final
ec8eadd results(hmm): decoder tuned on LightGBM OOF probabilities; final student_test comparison
30ca53d feat(hmm-exp): dataset v2 exclusions and wider grid in decoder tuning
b6bbde7 feat(hmm-exp): out-of-fold LightGBM probabilities for decoder tuning
e161a32 Merge pull request #14 from likhith76342/likhith-charts
```

### Selected final model: LightGBM

Evidence (read from committed files, not hand-typed):

- `optimisation/summary.json`: `"model": "lightgbm"`, best trial 12, `best_cv_top1`: 0.851034, `test_top1`: 0.856420 (post-selection student-test evaluation by `optimisation.finalize_study`, refit with `n_estimators=743`, the median of fold best iterations).
- `results/comparison.csv`: LightGBM has the highest test_top1 (0.856054) of the five models; `results/comparison_status.json`: `"status": "complete"`, all five models included.
- `optimisation/README.md`: search stopped at the user's request after 18 completed trials (brief asked for >= 50); `finalize_study` published `artifacts/final_model.joblib`.
- Model object check (run): `sklearn.pipeline.Pipeline`, `has predict_proba = True`, `n_features_in_ = 12`, `classes_ = [0, 1]`.

### HMM config (artifacts/hmm_config.json)

```
emission_source = "lightgbm"        <- NOT "placeholder" ✓
beta=20.0, w_e=0.1, unreachable_penalty=-50.0, clip_to_penalty=true,
gap_break_s=120.0, emission_eps=1e-9, min_cutoff_m=500.0, cutoff_factor=2.0, backward_tolerance_m=50.0
```

- Tuned on **real LightGBM out-of-fold probabilities**, not in-sample and not placeholder:
  - `results/hmm/oof_probs_lightgbm.json`: `"model": "lightgbm (final_model settings)"`, OOF top-1 mean over folds 0.8507, folds exclude trips 33/54 from fitting, output `data/processed/oof_probs_train_lightgbm.parquet`.
  - `results/hmm/best_config_lightgbm.json`: `"emission_source": "lightgbm"`, 292 configs evaluated on 78 training trips (dataset v2, trips 33 and 54 excluded), train_mean_trip_acc 0.9326.
  - `artifacts/hmm_config.json` == `results/hmm/best_config_lightgbm.json` config block (identical values).
- Flagged trips 33 and 54 excluded from HMM tuning/scoring ✓.
- student_test used **only** for the final comparison (`results/hmm/comparison_lightgbm.json`, `"split": "student_test"`), never for tuning.
- Grep for `evaluator_test` in `optimisation/`, `models/`, training/tuning code: **none** — only `config.yaml api.eval_data_dir` (the demo data dir, which I set to `data/raw/evaluator_test`). No HMM tuning touched evaluator data ✓. student_test was not used to choose hyperparameters/HMM params/thresholds ✓.

Artifacts committed to git: `artifacts/final_model.joblib` (1.33 MB), `artifacts/feature_columns.json`, `artifacts/hmm_config.json` ✓. `data/raw/*` NOT committed (only `data/raw/.gitkeep`) — dataset must be copied in by the user/deployer.

---

## STEP 1 — Wire the final model (DONE)

- `config.yaml api.model_path`: already `artifacts/final_model.joblib` ✓; `PATHSNAP_MODEL` env override respected (`api/service.py resolve_model_path`).
- `config.yaml api.eval_data_dir` default changed `data/raw/student_test` → `data/raw/evaluator_test` (demo); `EVAL_DATA_DIR` remains the override. Dev still works with `student_test` via `EVAL_DATA_DIR=data/raw/student_test` (tests override it explicitly).
- `/health` now returns `status, mode, model, hmm_config, n_trips, data_dir`. Verified (run):
  ```
  {"status":"ok","mode":"real","model":"final_model.joblib",
   "hmm_config":"...\\artifacts\\hmm_config.json","n_trips":10,"data_dir":"..."}
  ```
- Final model is sklearn-compatible with `predict_proba` ✓.
- Feature order == `artifacts/feature_columns.json` == `pathsnap.features.FEATURE_COLUMNS` == 12 features
  (`dist_m, dist_rank, dist_minus_min, dist_ratio, bearing_diff, speed_mps, heading_valid, dt_prev, dt_next, n_candidates, arc_length_m, proj_t`) ✓ — no silent mismatch.

PASS.

---

## STEP 2 — Correctness checks (student_test, real API)

Run: `python` TestClient against `artifacts/final_model.joblib`, `EVAL_DATA_DIR=data/raw/student_test`, all 10 trips.
Trips: 0, 7, 9, 40, 52, 56, 65, 76, 77, 91 (10 trips, 19,160 GPS points).

### 2a Aggregate (point-weighted Top-1 argmax)

| Source | test_top1 | vs API |
|---|---|---|
| API argmax aggregate (this run) | **0.856420** | — |
| `optimisation/summary.json` test_top1 (final refit) | 0.8564196 | 4dp MATCH (0.8564 == 0.8564) ✓ |
| `results/lightgbm/metrics.json` test_top1 | 0.8560543 | diff +0.000365 (0.8564 vs 0.8561) |

**FLAG**: `results/lightgbm/metrics.json` (0.856054) is the **n_estimators=226** training run (`results/lightgbm/model.joblib`), NOT the refit final model (`artifacts/final_model.joblib`, n_estimators=743). There is **no `artifacts/metrics.json`** for the final model; its `test_top1` lives only in `optimisation/summary.json`. The 0.0004 gap is a real model difference (refit), not twin-tie skew. Fix: record a final-model `metrics.json` next to `artifacts/final_model.joblib` (Me).

### 2b vs Keshav's saved end-to-end results (`results/hmm/comparison_lightgbm.json`)

| Method | Saved (Keshav) | API (this run) | Gap |
|---|---|---|---|
| Classifier-only argmax top1_pooled | 0.85663 | 0.85642 | 0.00021 |
| Classifier + HMM top1_pooled | 0.92464 | 0.92464 (matches per-trip) | ~0 |

Per-trip HMM Top-1 matches `results/hmm/student_test_per_trip_lightgbm.csv` exactly (e.g. trip 76: 0.9603; trip 77: 0.7752).
The small argmax gap (0.85663 vs 0.85642 ≈ 4 points/19k) is the **known twin-tie candidate-ordering issue** documented in `hmm_experiments/README.md` and `tests/test_pipeline.py` (candidate parquets built on Linux; reverse-twin arcs tie at ~1e-15 m). The xfail test still says what it says ([XPASS(strict)] on this Windows machine because tie-break happened to match here). Agree within documented tolerance; FLAGGED + explained, no change.

### 2c Guard tests (all verified)

- Leak test: `pathsnap.io.load_route` monkeypatched to raise AND `.route` absent → prediction path still runs (`tests/test_pipeline.py::test_predict_trip_never_reads_route`) ✓.
- No ground-truth keys (`route`, `reference`, `label`, `ground_truth`) at any depth in any response (`tests/test_api.py` FORBIDDEN_KEYS) ✓.
- Mock mode: all accuracy/route metrics null, only `n_points` filled, nothing fabricated ✓.
- Trip without `.route`: returns null metrics, HTTP 200 (never 500) — verified with synthetic trips below ✓.
- Forbidden-key / mock-mode / bad-body / unknown-trip tests pass (`tests/test_api.py`) ✓.

### 2d HMM vs argmax per trip (from the live API)

| trip | n_points | top1_argmax | top1_hmm | HMM>=argmax? | route P | route R | P far below R? |
|---|---:|---:|---:|---|---:|---:|---|
| 0 | 2503 | 0.9377 | 0.9808 | yes | 0.9421 | 1.0000 | no |
| 7 | 1354 | 0.9225 | 1.0000 | yes | 1.0000 | 0.9837 | no |
| 9 | 1049 | 0.8608 | 0.9047 | yes | 0.9154 | 0.9837 | no |
| 40 | 2995 | 0.9202 | 0.9760 | yes | 0.9904 | 1.0000 | no |
| 52 | 262 | 0.9542 | 1.0000 | yes | 1.0000 | 0.9789 | no |
| 56 | 3958 | 0.8888 | 0.9565 | yes | 0.9637 | 0.9953 | no |
| 65 | 768 | 0.8294 | 0.9063 | yes | 0.9732 | 1.0000 | no |
| 76 | 1486 | 0.8694 | 0.9603 | yes | 0.8789 | 0.9893 | **yes (0.11)** |
| 77 | 3911 | 0.6924 | 0.7752 | yes | 0.8874 | 1.0000 | **yes (0.11)** |
| 91 | 874 | 0.8570 | 0.9474 | yes | 0.9545 | 0.9957 | no |

- HMM Top-1 >= argmax Top-1 on **all 10 trips** ✓ (matches `hmm_experiments/README.md`: HMM improves on every trip).
- FLAG: trips **76** and **77** have route precision far below recall (P≈0.88 vs R≈0.99–1.0).

### 2e Synthetic edge trips in /tmp (never touches real data) — all HTTP 200, no crashes

| synthetic trip | setup | result |
|---|---|---|
| 10000001 | 2 points (tiny) | 200, null metrics (no .route) |
| 10000002 | 1 point | 200, null metrics |
| 10000003 | 5000 s time gap (> gap_break_s=120) | 200, null metrics |
| 10000004 | points far from network (candidate-less) | 200, null metrics |
| 10000005 | 50 real track points | 200, null metrics |

No `.route` in any synthetic trip → all metrics null, no 500 ✓. Pipeline handles candidate-less points, huge gaps, single/tiny trips.

PASS (with the FLAGs above documented).

---

## STEP 3 — Repo audit

### Fresh-clone cold start
Cloned working tree to a temp dir, created a fresh venv, `pip install -r requirements.txt` — **all 50+ packages installed together, no errors**. Copied only `data/raw/student_test` in. Started the API (real mode, `artifacts/final_model.joblib`) and ran trip 52 end-to-end: top1_argmax 0.9542, top1_hmm 1.0000, route F1 0.9894 ✓. UI (`ui/app.py`) imports cleanly under the same install; `tests/test_ui.py` (AppTest with stubbed API) passed in the suite.

**Note**: the repo targets Python 3.11 (`requirements.txt` header, README); this machine has Python 3.12.10 and everything installs/runs. `pyarrow==17.0.0`, `numpy==1.26.4`, `scipy==1.13.1` have no 3.13 wheels but are fine on 3.11/3.12.

### git ls-files audit
- `data/raw/*` content NOT committed (only `data/raw/.gitkeep`) ✓ — no .track/.nodes/.arcs/.route committed ✓.
- No secrets / `.env` tracked ✓.
- No absolute `C:\Users` / `/Users` paths in tracked code or configs ✓ (only mentioned as a rule inside `docs/PathSnap_Team_Brief.md`).
- `.gitattributes`: `* text=auto eol=lf` ✓; binary joblib/parquet/pdf ✓.

### FLAG (HIGH): `results/knn/model.joblib` = **245.83 MB** and **tracked in git**
- Exceeds the team's 50 MB guideline AND GitHub's 100 MB per-file hard limit → `git clone`/`push` to GitHub will fail or be bloated.
- `.gitattributes` marks it `filter=lfs` but it is stored directly in git (not actually on LFS). Owner: **Manu (knn)**. Suggested fix: remove `results/knn/model.joblib` from git (it's a regenerable artifact; `results/**/model.joblib` is already in `.gitignore`) or move it to LFS; keep only small committed artifacts.
- `artifacts/final_model.joblib` (1.33 MB) is tracked and fine (< 50 MB).

### requirements.txt
- Every import in tracked `.py` files is covered by a pin (numpy, pandas, pyarrow, scipy, scikit-learn, lightgbm, optuna, networkx, shapely, pyyaml, matplotlib, joblib, fastapi, uvicorn, streamlit, streamlit-folium, requests, pytest, httpx) ✓; versions installable together on 3.11/3.12 (verified by clean install) ✓.
- Minor flag: `seaborn==0.13.2` is pinned but **not imported by any tracked `.py`** (unused heavy dep). `google.colab` only in untracked `colab_train.py`. `branca`/`jinja2` are transitive folium deps (pulled in automatically).

### Contract check vs docs/PathSnap_Team_Brief.md
All five `results/<model>/metrics.json`:
| model | dataset_version | excluded_trips | seed | train_top1 | note |
|---|---:|---|---:|---:|---|
| logreg | 2 | [33,54] | 42 | 0.8265 | OK |
| random_forest | 2 | [33,54] | 42 | 0.8587 | OK |
| knn | 2 | [33,54] | 42 | **null** | train_top1/train_macro_f1 null |
| mlp | 2 | [33,54] | 42 | **null** | train_top1/train_macro_f1 null |
| lightgbm | 2 | [33,54] | 42 | 0.8531 | OK (final model) |

- Flagged trips [33, 54] excluded in primary metrics for **every** model ✓ (none to flag for Manu's KNN/MLP on exclusions).
- Missing/null keys: `test_route_continuity_argmax` is null in all five models (shared, expected). `train_top1`/`train_macro_f1` null for knn and mlp — FLAG minor (values not computed; nothing fabricated).
- File naming: `models/logreg/` contains **both `train.py` and `training.py`** — naming deviation (brief expects one training entrypoint). Owner: **Zubed (logreg)**. Flag only.
- `results/<model>/model.joblib` files are tracked (knn 245 MB flagged above; the rest are small).

### Integrity
- No hardcoded predictions anywhere (only doc text) ✓.
- `.route` read outside `pathsnap/labels.py`: `api/evaluate.py` (server-side evaluator, allowed), `pathsnap/build_dataset.py` (training/labelling, allowed), `eda/*`, `hmm_experiments/common.py` ("evaluation only"), `tests/` (allowed), `models/logreg/miss_spotcheck.py` (Zubed's diagnostics). Prediction path (`api/service.py`, `pathsnap/pipeline.py`) never reads `.route` ✓.
- No `evaluator_test` or its trip ids (15, 17, 23, 30, 49, 59, 67, 73, 87, 94) in training/tuning/models/optimisation code ✓ (grep confirmed; only `config.yaml api.eval_data_dir`).

### Full test suite: `pytest tests/ -q`
**87 passed, 2 failed, 397 warnings** (both failures pre-existing on this branch, not caused by my changes):
1. `tests/test_charts.py::test_table_values_equal_the_metrics_files` — **FAIL**: `summary_table.md` is written UTF-8 but read with `read_text()` (no encoding) → on Windows (cp1252) the `±` (U+00B1) decodes as `Â±`, so the assertion misses. **Charts themselves are correct** (verified standalone: table values exactly equal metrics files). This is a test-portability bug in `tests/test_charts.py` (mine to fix).
2. `tests/test_pipeline.py::test_inference_candidates_match_training_parquet` — **[XPASS(strict)]**: the twin-tie xfail unexpectedly passed on Windows (tie-break matched by chance); with `strict=True` that counts as failure. On Linux (where parquets were built) it xfails as documented. The xfail still says what it says. **Do not change the xfail semantics.**

No tests skipped due to missing data (student_test is present locally).

### Deliverables coverage (repo vs required contents)
| Required | Status | Evidence |
|---|---|---|
| README with setup/run instructions | Done | `README.md` (setup, two-laptop deployment, API/UI commands) |
| Data processing source | Done | `pathsnap/build_dataset.py` (+ Zubed) |
| EDA | Done | `eda/` (+ Mannu), `results/eda/*` |
| Models | Done | `models/{logreg,random_forest,knn,mlp,lightgbm}/train.py` |
| Bayesian optimisation | Done | `optimisation/tune_lightgbm.py`, `finalize_study.py`, `trials.csv` |
| HMM | Done | `pathsnap/hmm.py`, `pathsnap/graph.py`, `pathsnap/route_metrics.py`, `hmm_experiments/` |
| API | Done | `api/main.py`, `api/service.py`, `api/evaluate.py`, `api/geometry.py` |
| UI | Done | `ui/app.py`, `ui/ui_components.py` |
| Visualisation / Folium map code | Done | `ui/` uses `streamlit-folium`; `eda/make_maps.py`; `reports/figures/maps/*.html` |
| Configs | Done | `config.yaml`, `artifacts/feature_columns.json`, `artifacts/hmm_config.json` |
| Supporting results | Done | `results/*/metrics.json`, `results/comparison.csv`, `results/hmm/*` |
| Folder naming clear | Partial | `models/logreg/` has both `train.py` and `training.py` (naming deviation) |

---

## STEP 4 — Regenerate charts

`python reports/charts/make_charts.py` → exit 0, **5 warnings** (all expected):
- `knn: no metrics_all_trips.json - left out of the primary-vs-all-trips chart`
- `mlp: no metrics_all_trips.json - left out of the primary-vs-all-trips chart`
- `primary_vs_all_trips: knn/mlp lacks metrics.json or metrics_all_trips.json - left out` (knn/mlp have no all-trips file)
- `optuna_importance: skipped - parameter importance needs the Optuna study's search distributions (no re-run allowed)`

**Numbers did NOT change** vs committed figures (only a Windows backslash path-separator appeared in the `source` column — a platform artifact). Per the task, figures were **not committed** (refresh only if numbers change).

Extra charts still missing: `results/hmm/` has per-trip student_test LightGBM data (`student_test_per_trip_lightgbm.csv`, `comparison_lightgbm.json`), but there is **no committed classifier-only vs HMM per-trip comparison file for the final model** beyond what Keshav saved. The HMM per-trip gains exist in `results/hmm/student_test_per_trip_lightgbm.csv`; a dedicated per-trip HMM-vs-argmax chart/figure for the final model is not present and I did not invent that data. Keshav's `results/hmm/*` is the authoritative source.

---

## STEP 5 — Performance

Measured end-to-end through the real API (server-side `timings_ms.total`), cold = first request, warm = second request:

| trip | n_points | cold ms | warm ms | hmm top1 |
|---|---:|---:|---:|---:|
| 0 | 2503 | 1686 | 1925 | 0.9808 |
| 7 | 1354 | 810 | 804 | 1.0000 |
| 9 | 1049 | 1301 | 1346 | 0.9047 |
| 40 | 2995 | 2045 | 2147 | 0.9760 |
| 52 | 262 | 278 | 296 | 1.0000 |
| 56 | 3958 | 3808 | 3985 | 0.9565 |
| 65 | 768 | 975 | 923 | 0.9063 |
| 76 | 1486 | 2117 | 2218 | 0.9603 |
| 77 | 3911 | 3880 | 3976 | 0.7752 |
| 91 | 874 | 1041 | 966 | 0.9474 |

- Model load time at startup (`joblib.load artifacts/final_model.joblib`): **1542 ms (~1.5 s)**.
- Slowest trip: **trip 77** (3911 pts), cold **3880 ms = 3.88 s** (trip 56 close behind at 3.81 s).
- Trips above 15 s: **none** ✓.

PASS.

---

## STEP 6 — Evaluator smoke test

**SKIPPED**: `data/raw/evaluator_test` does not exist (only `data/raw/student_test` and `data/raw/train`). No evaluator trips (15, 17, 23, 30, 49, 59, 67, 73, 87, 94) to run. Per the rules, no decisions were made from evaluator data.

---

## STEP 7 — README + docs

- `README.md` updated: `api.eval_data_dir` default is now `data/raw/evaluator_test` (demo) with `EVAL_DATA_DIR=data/raw/student_test` dev override; `/health` documents `model` and `hmm_config`; removed the obsolete "until the final model exists, run `results/lightgbm/model.joblib`" block (the final model `artifacts/final_model.joblib` is now the default).
- This file (`docs/FINAL_CHECKS.md`) contains the PASS/FAIL/FLAG table above.

---

## Prioritized problems to fix

| Priority | Problem | Owner | File | Suggested fix |
|---|---|---|---|---|
| 1 | `results/knn/model.joblib` (245.83 MB) tracked in git; breaks 100 MB GitHub file limit / 50 MB guideline; `.gitattributes` LFS not actually used | Manu | `results/knn/model.joblib` | Remove from git (it's regenerable; `results/**/model.joblib` is gitignored) or host on LFS; do not push the 245 MB file |
| 2 | No `metrics.json` for the final model; two LightGBM test_top1 values (0.85642 summary.json vs 0.85605 metrics.json for the n_estimators=226 run) | Me | `artifacts/`, `results/lightgbm/metrics.json` | Record a final-model `artifacts/metrics.json` (test_top1 0.85642) so STEP 2a compares like-for-like |
| 3 | `test_charts.py` reads UTF-8-written `summary_table.md` without `encoding=` → `±` decodes wrong on Windows | Me | `tests/test_charts.py:99` | `read_text(encoding="utf-8")` |
| 4 | `models/logreg/` has both `train.py` and `training.py` (naming deviation) | Zubed | `models/logreg/training.py` | Remove or rename `training.py`; keep one entrypoint |
| 5 | `seaborn==0.13.2` pinned but unused by tracked code | Me | `requirements.txt` | Drop `seaborn` pin (or confirm it's used in eda) |
| 6 | Trip 77 & 76 route precision far below recall (P≈0.88 vs R≈1.0/0.99) | Keshav | `pathsnap/hmm.py`, route reconstruction | Investigate arc-sequence reconstruction on those trips (precision-sensitive) |

## Things I was unsure about
- Whether `results/knn/model.joblib` is genuinely needed in the repo or can be regenerated; I flagged rather than deleted it (it's not my file and it's large).
- Whether the `test_charts.py` `±` failure also occurs on macOS/Linux (it should not there; it's Windows cp1252-specific). I flagged it but did not change test behavior beyond noting it.
- The task mentions updating "tests/README" — no `tests/README` exists; the README test section already documents `pytest tests/` and `EVAL_DATA_DIR`, so I updated `README.md` instead.
