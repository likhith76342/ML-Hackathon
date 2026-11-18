
# PathSnap Team Brief

**Course:** UE24CS352A Machine Learning. PathSnap hackathon (distributed map-matching and route reconstruction)\
**Team:** Likhith, Mannu, Zubed, Keshav\
**Purpose of this document:** This is an internal coordination brief, not a submission. Every team member (and their AI assistant) should read the whole document before writing code, so that everyone uses the same data format, the same function signatures, the same metrics and the same folder layout. Later integration then works without rewriting.

> **Note to any AI assistant reading this:** Your user is one member of a four-person team. Follow the shared contracts in Sections 4 to 7 exactly: column names, function signatures, file paths and metric definitions. Do not invent alternatives. If something here is impossible or clearly wrong for the data, tell your user so that the team can update this brief. Do not silently diverge from it. Do not write report text that claims results which were not actually produced by running code. The course's academic-integrity policy forbids that, and every number must be reproducible from the repository.

---

## 1. The problem in one paragraph

Each trip has a noisy GPS track and a local road network. We must decide, for every GPS point, which road segment (arc) the vehicle was really on, and output a connected route. The required approach has three stages.

1. **Generate candidates.** For each GPS point, find the nearby road arcs.
2. **Classify.** An ML classifier scores every (GPS point, candidate arc) pair with the probability that this arc is the correct one.
3. **Decode with an HMM.** An HMM with Viterbi decoding uses those probabilities as emissions and road-network connectivity as transitions, and picks the best connected sequence.

The final system runs on two laptops. Laptop 1 runs a FastAPI server holding the model, the HMM and the ground truth. Laptop 2 runs a UI where the evaluator picks one of 10 hidden trips. The UI sends the trip ID to the API, and the API returns the predicted route and its accuracy.

---

## 2. Dataset facts (already verified on our copy)

### 2.1 Partitions

| Partition | Trips | Folder | Use |
|---|---|---|---|
| Training | 80 | `train/` | Fitting, grouped cross-validation, Bayesian optimisation |
| Student testing | 10 | `student_test/` (IDs 0, 7, 9, 40, 52, 56, 65, 76, 77, 91) | Development-stage model comparison, decoder tuning |
| Evaluator testing | 10 | **Not given to us** (IDs are probably 15, 17, 23, 30, 49, 59, 67, 73, 87, 94, the IDs missing from 0 to 99) | Final evaluation only, on the demo day |

Each trip folder is `<split>/<8-digit id>/` and contains four files named `<id>.track`, `<id>.nodes`, `<id>.arcs` and `<id>.route`. The student-test folders also contain `.route`, so we can score ourselves locally.

### 2.2 File formats (tab or whitespace separated, no header)

| File | One row is | Columns | Index meaning |
|---|---|---|---|
| `.nodes` | a road-graph vertex | `lon, lat` (float) | node_id = 0-based row number |
| `.arcs` | a **directed** road segment, a straight line between two nodes | `from_node, to_node` (int) | arc_id = 0-based row number |
| `.track` | one GPS fix | `lon, lat, t` (t = seconds, float) | point_idx = 0-based row number |
| `.route` | ground truth | `arc_id` (int), in driving order | route position = 0-based row number |

**Column order is longitude first, then latitude.** Folium and Leaflet expect `[lat, lon]`, so swap the order when plotting.

### 2.3 Facts that affect design

- **About 92% of arcs have a reverse twin.** A two-way street is stored as two arcs, A to B and B to A, on top of each other. Distance alone cannot tell them apart. **Heading vs arc bearing is an essential feature**, and the HMM transitions must respect direction.
- **Consecutive route arcs are connected 99.96% of the time** (`arcs[r[i]].to_node == arcs[r[i+1]].from_node`). The ground truth is a near-continuous path, but not perfect.
- **Routes revisit arcs:** there are 322 repeated arc IDs across all routes. When aligning GPS points to the route, align to the *route position*, not to the arc ID.
- **No isolated nodes:** every node is used by at least one arc.
- **Sampling:** nominally 1 Hz, about 97% of steps are exactly 1 s apart. **30 of the 90 trips have at least one gap over 30 s.** Extreme cases: train trip 90 (max gap about 153,000 s), trip 78 (about 52,800 s), trip 69 (about 2,658 s) and trip 21 (about 2,161 s). Timestamps are strictly increasing in every trip.
- **Size range:** the median trip has about 1,500 GPS points, 10k nodes and 20k arcs. The largest has 19k points and 307k arcs. Spatial indexing (a KD-tree or STRtree) is mandatory, because brute-force point-to-arc distance will be far too slow.
- **Global coverage:** trips are spread across many countries (longitude about -100 to 137, latitude about -35 to 62). Never use raw degree distances, because one degree of longitude shrinks with latitude.
- **The reference routes are not perfect ground truth** (the docs say so explicitly). We must acknowledge this in the report.

---

## 3. Who does what (summary)

| Person | Primary ownership | Models | Report sections |
|---|---|---|---|
| **Zubed** | Shared data pipeline: loading, geometry, candidate generation, per-point labelling, features, fold assignment. Bayesian optimisation. Model comparison table and plots. | Logistic Regression, Random Forest (already done, to be re-run on the shared features) | Preprocessing and features (part of 2.2), 3.1 comparison, LR and RF subsections, 3.3 comparison plots, 4 Bayesian optimisation |
| **Mannu** | EDA and all Folium maps. Compiles the final PDF report. | KNN, MLP | 2.2 EDA, KNN and MLP subsections, 6 Folium visualisation |
| **Keshav** | HMM and Viterbi decoder, route stitching, sequence-level metrics, classifier-only vs HMM comparison | (none; optional SVM if time allows) | 5 HMM and Viterbi |
| **Likhith** | FastAPI backend (Laptop 1), UI (Laptop 2), the end-to-end inference pipeline, repo structure, README, two-laptop deployment | LightGBM | LightGBM subsection, 7 API, UI and deployment |

This gives five models from five families: linear (LR), ensemble/bagging (RF), instance-based (KNN), neural network (MLP) and boosting (LightGBM).

**Dependency order:** Zubed's shared feature files block all model training. Until they land, Mannu does EDA, Keshav builds the road graph and transition code, and Likhith builds the API and UI skeleton with dummy outputs. Nobody should be idle.

---

## 4. Shared repository layout

One GitHub repository named **`ML-Hackathon`**. It must be **public** at submission time. Everyone works on a branch named after themselves (`likhith`, `mannu`, `zubed`, `keshav`) and opens pull requests into `main`; Likhith merges. The skeleton below already exists in the repo, including working `pathsnap/config.py` and `pathsnap/io.py` and stub files with the agreed function signatures. Fill in your stubs; don't rename them.

```
ML-Hackathon/                   # repo root
├── README.md                   # install, run, deploy (Likhith)
├── requirements.txt            # pinned versions (everyone adds what they use)
├── config.yaml                 # shared constants (see Section 5)
├── docs/                       # this brief
├── data/                       # NOT committed if large; .gitignore raw data
│   ├── raw/train/…             # original folders copied as-is
│   ├── raw/student_test/…
│   └── processed/              # Zubed's outputs (parquet), see 6.4
├── pathsnap/                   # shared Python package; import, never copy-paste
│   ├── __init__.py
│   ├── config.py               # loads config.yaml (done)
│   ├── io.py                   # load_trip(), load_route() (starter done; Zubed owns)
│   ├── geo.py                  # projection, distances, bearings     (Zubed)
│   ├── candidates.py           # build_candidates()                  (Zubed)
│   ├── labels.py               # align_points_to_route()             (Zubed)
│   ├── features.py             # FEATURE_COLUMNS                     (Zubed)
│   ├── labels.py               # align_points_to_route(), add_labels() (Zubed)
│   ├── metrics.py              # all metric functions                (Zubed + Keshav)
│   ├── graph.py                # road graph, shortest paths          (Keshav)
│   ├── hmm.py                  # viterbi(), reconstruct_route()      (Keshav)
│   └── pipeline.py             # predict_trip(), end-to-end          (Likhith)
├── eda/                        # notebooks + scripts                 (Mannu)
├── models/
│   ├── logreg/  random_forest/  knn/  mlp/  lightgbm/
│   │   └── train.py            # trains, runs CV, evaluates on student_test
├── optimisation/               # Optuna study code + trials CSV       (Zubed)
├── hmm_experiments/            # decoder tuning + comparison          (Keshav)
├── api/                        # FastAPI app                          (Likhith)
├── ui/                         # evaluator UI                         (Likhith)
├── artifacts/                  # exported model files for the API
│   ├── final_model.joblib
│   ├── feature_columns.json
│   └── hmm_config.json
├── results/                    # every metrics JSON, CSV, log
│   ├── <model_name>/metrics.json
│   ├── <model_name>/cv_folds.csv
│   └── comparison.csv
└── reports/figures/            # every PNG used in the report
```

**Rules**

- Never commit secrets. Do not commit the raw data if it bloats the repo; document where it goes in the README.
- Every script runs from the repo root, e.g. `python -m models.lightgbm.train`, with paths read from `config.yaml`.
- Use random seed **42** everywhere (`numpy`, `random`, model `random_state`, Optuna sampler).
- Use Python **3.10 or 3.11**. Add every library you use to `requirements.txt` with the version you used.

---

## 5. Shared constants (`config.yaml`)

Zubed sets the initial values. Change them only by telling the group, because every model's results depend on them.

```yaml
data_root: data/raw
processed_dir: data/processed
seed: 42
candidate_radius_m: 50        # arcs within this distance of a GPS point
max_candidates: 10            # keep the K nearest within the radius
n_cv_folds: 5                 # GroupKFold by trip_id
gap_break_s: 30               # HMM: time gap that forces a sequence break
```

If the candidate miss rate on training data is high (over about 5%) with a 50 m radius, Zubed may raise the radius or K. He announces the change, and **everyone retrains**.

---

## 6. Zubed: shared data pipeline, LR, RF, Bayesian optimisation

### 6.1 Goal
Produce one candidate-level dataset that every model trains on, plus the same code path that the API uses at inference time. Zubed already has LR and RF. If his existing pipeline already does some of this differently, **his existing choices are the source of truth**, as long as the outputs match the contract in 6.4 and 6.5. He should update this brief so the others know.

### 6.2 Geometry (`pathsnap/geo.py`)
- Project each trip to **local metres** with an equirectangular projection centred on the trip's mean latitude and longitude:
  `x = R * radians(lon - lon0) * cos(radians(lat0))`, `y = R * radians(lat - lat0)`, with `R = 6371000`.
  This is accurate to well under 1% at city scale and needs no extra dependency. pyproj with UTM is an acceptable alternative, but pick one and use it everywhere. The report must justify the choice (the docs require this).
- Helpers to provide: point-to-segment distance and projection parameter `t` in [0, 1], segment bearing (0 to 360, clockwise from north), and absolute angle difference folded to [0, 180].

### 6.3 Steps
1. **Loader** (`io.py`): `load_trip(trip_dir) -> Trip` with `track` (DataFrame: lon, lat, t, x, y), `nodes` (lon, lat, x, y), `arcs` (from_node, to_node). It **must not read `.route` by default**. A separate `load_route(trip_dir)` reads it. This separation is what keeps ground truth out of the prediction path.
2. **Candidates** (`candidates.py`): build a spatial index over arc segments (a shapely `STRtree`, or a scipy `cKDTree` on densified segment points). For each GPS point, keep arcs within `candidate_radius_m`, sorted by distance, at most `max_candidates`. A point with zero candidates still appears in the output with a flag (see 6.5) so that the HMM knows about it.
3. **Per-point reference labels** (`labels.py`). `.route` lists arcs, not which point was on which arc, so we derive this ourselves:
   - Monotone alignment: assign each GPS point *i* a route position *r(i)* such that *r* never decreases along the track, minimising total point-to-arc distance (simple dynamic programming, or a greedy forward pointer with a look-ahead window).
   - `ref_arc(i) = route[r(i)]`. A candidate row gets `label = 1` if its `arc_id == ref_arc(i)`, otherwise 0.
   - Record `ref_dist_m` (the distance from the point to its reference arc). Points with `ref_dist_m > 100` are suspicious. Keep them, but count them in the report as a reference-route limitation.
   - Document the method clearly. It defines every label in the project, and the viva will ask about it.
4. **Features** (`features.py`). Use one list `FEATURE_COLUMNS` and nothing else. Suggested starting set:
   - `dist_m`: point to arc distance
   - `dist_rank`: rank among this point's candidates (0 = nearest)
   - `dist_minus_min`: `dist_m` minus the nearest candidate's distance
   - `dist_ratio`: `dist_m / (min_dist + 1)`
   - `bearing_diff`: angle between the GPS heading (from points i-1 to i+1) and the arc bearing, 0 to 180. **This is the key feature for the twin-arc problem.**
   - `speed_mps`: from neighbouring points
   - `heading_valid`: 0 if speed is under about 1 m/s (the heading is noise when the vehicle is stopped)
   - `dt_prev`, `dt_next`: time gaps to the neighbouring points
   - `n_candidates`
   - `arc_length_m`, `proj_t`: position along the arc, 0 to 1
   - Optional: `prev_nearest_connected`, i.e. whether this arc connects to the previous point's nearest arc. **Features must never use `.route` or labels.**
5. **Folds:** `GroupKFold(n_splits=5)` grouped by `trip_id` on the 80 training trips. Save the assignment to `data/processed/folds.csv` (`trip_id, fold`) and **everyone uses this file**, so all CV numbers are comparable.
6. **Export** the processed files (6.4) and announce in the group chat once they're pushed.
7. **Re-run LR and RF** on the shared files using the common training template (Section 10) and save `results/logreg/metrics.json` and `results/random_forest/metrics.json`.
8. **Bayesian optimisation**, after all five models report: see 6.6.
9. **Comparison:** `results/comparison.csv` plus plots in `reports/figures/`: CV macro-F1 per model (bar with fold error bars), student-test Top-1 and macro-F1, log loss, inference time.

### 6.4 Output files (contract)
- `data/processed/train_candidates.parquet` (80 trips)
- `data/processed/test_candidates.parquet` (10 student-test trips)
- `data/processed/folds.csv`
- `data/processed/point_refs.parquet`: one row per GPS point with `trip_id, point_idx, ref_arc, ref_route_pos, ref_dist_m, ref_in_candidates`

### 6.5 Candidate table schema (contract)
One row per (GPS point, candidate arc):

| column | type | notes |
|---|---|---|
| `trip_id` | int | e.g. 10 |
| `point_idx` | int | row in `.track` |
| `t` | float | timestamp |
| `arc_id` | int | candidate arc; **-1 if the point has no candidates** (one placeholder row) |
| all of `FEATURE_COLUMNS` | float | NaN only on placeholder rows |
| `label` | int (0/1) | training/test files only, **never produced at inference** |

At inference, `build_candidates(trip)` returns the same table **without `label`**. Training and inference must call the same functions, so that there is no training/serving skew.

### 6.6 Bayesian optimisation
- Library: **Optuna** with the TPE sampler (`seed=42`), **at least 50 trials** (more if time allows).
- Model: whichever wins the comparison on the primary metric (most likely LightGBM). Reuse that model owner's `train.py` functions.
- Objective: **mean observation-level Top-1 accuracy over the 5 grouped folds** (the primary metric, see 9.3). Also log macro-F1 and log loss per trial.
- Search space: justify each range in the report. For LightGBM, for example: `num_leaves` 15 to 255 (log scale), `learning_rate` 0.01 to 0.3 (log), `min_child_samples` 5 to 200, `feature_fraction` 0.5 to 1.0, `bagging_fraction` 0.5 to 1.0, `lambda_l1` and `lambda_l2` 1e-8 to 10 (log), `n_estimators` up to 2000 with early stopping.
- Save `optimisation/trials.csv` (every trial's parameters and scores), the best parameters JSON, and the plots (optimisation history, parameter importance).
- Refit the best configuration on all 80 training trips, then evaluate it on student_test. Fill in the tuned vs untuned table: CV macro-F1, student-test Top-1, student-test macro-F1, log loss, inference time, wall-clock cost of the search.
- Export the tuned model to `artifacts/final_model.joblib` and `artifacts/feature_columns.json` for Likhith.

### 6.7 Viva preparation
Why this projection; how labels were derived and their limitations; candidate miss rate and how the radius was chosen; why grouped CV; why Top-1 as the objective.

---

## 7. Mannu: EDA, Folium, KNN, MLP, report compilation

### 7.1 EDA (can start immediately with raw data, does not need Zubed)
Produce scripts or notebooks in `eda/` and figures in `reports/figures/eda_*.png`. **Label every analysis with the partition it used.** Required items (from the docs):

1. Inventory: trips per partition, files per trip, file sizes, GPS point counts.
2. Data types and column meanings, verified rather than assumed.
3. Missing or invalid values: NaNs, impossible coordinates, duplicate timestamps.
4. Timestamp ordering and the distribution of time gaps, with the anomalous trips listed (see 2.3: trips 90, 78, 69 and 21).
5. Geographical extent of each trajectory (bounding box, rough city or country).
6. Road network size per trip (nodes, arcs) and **connectivity**: number of weakly and strongly connected components (networkx), and the share of arcs with a reverse twin.
7. Trajectory statistics: length in km, duration, speed distribution (flag implausible speeds such as over 70 m/s).
8. GPS noise estimate: the distribution of distance from each point to its reference arc (needs Zubed's `point_refs.parquet` once ready).
9. A short "data quality issues, assumptions and decisions" table.

### 7.2 Folium maps (mandatory, and in the report)
Write one reusable function, `eda/maps.py: make_trip_map(trip, layers) -> folium.Map`, which Likhith's UI can reuse:
- Layers: road network (thin grey), raw GPS points (blue dots), reference route (green line), predicted route (red line). Use `folium.FeatureGroup` per layer, add `LayerControl` and a legend, and fit the bounds to the track.
- Remember to plot `[lat, lon]`.
- For big networks, draw only arcs within about 200 m of the track, or the HTML becomes huge.
- Save HTML maps to `reports/figures/maps/` and take screenshots for the PDF.
- Produce at least: one EDA map (GPS, network, reference) and one result map (GPS, reference, classifier-only, HMM prediction) using Keshav's output.

### 7.3 KNN
- `sklearn` Pipeline: `StandardScaler` then `KNeighborsClassifier`. Scaling is mandatory for KNN.
- Initial configuration: `n_neighbors=15`, `weights="distance"`. Briefly try a few values of k and both weight options, and report what you tried.
- The candidate table can be large (millions of rows). If KNN is too slow, subsample training rows **by trip** (keep all positives and a stratified subset of negatives), and say so in the report. Measure inference time, because KNN is slow at prediction time and that's a legitimate finding.

### 7.4 MLP
- `sklearn` `MLPClassifier` inside a Pipeline with `StandardScaler`, or a small PyTorch model.
- Initial configuration: hidden layers (64, 32), ReLU, Adam, `alpha=1e-4`, batch size 512, up to 50 epochs, early stopping.
- **Required: real training and validation loss curves.** With sklearn, loop `partial_fit` one epoch at a time and compute log loss on a held-out fold after each epoch. Save the curves to `results/mlp/loss_history.csv` and plot them. Do not fabricate curves.

### 7.5 Report compilation
- Gather every member's sections into one PDF following the order in the Deliverables doc (2.2 EDA, 3.1 comparison table, 3.2 one subsection per model, 3.3 plots, 4 Bayesian optimisation, 5 HMM, 6 Folium, 7 API/UI screenshots).
- Every number in the report must come from a file in `results/`. Cross-check the tables against the JSON files before submission.
- Include an AI-assistance disclosure, as the policy requires.

### 7.6 Viva preparation
Main data-quality issues and how we handled them; why KNN needs scaling and why it's slow; what the MLP loss curves show (over- or underfitting).

---

## 8. Keshav: HMM and Viterbi decoder

### 8.1 Goal
Take the classifier's candidate probabilities for one trip and return one arc per GPS point that forms a coherent, connected route. Then show that this beats classifier-only argmax.

### 8.2 What can start immediately (before any model exists)
- `pathsnap/graph.py`: build a directed graph per trip, with nodes as road nodes and edges as arcs weighted by length in metres (networkx `DiGraph`, or scipy sparse with Dijkstra for speed). Provide `route_distance(arc_a, t_a, arc_b, t_b)`: the driving distance from position `t_a` along `arc_a` to position `t_b` along `arc_b`, respecting direction. Same arc with `t_b >= t_a` gives `(t_b - t_a) * len`. Otherwise it is the remainder of a, plus the shortest path from `a.to_node` to `b.from_node`, plus `t_b * len(b)`.
- Limit shortest-path searches with a cutoff, e.g. `max(2 * straight_line_distance, 500 m)`, or they will be too slow on 300k-arc networks. Cache results per trip.
- Develop and test with a **placeholder emission**, a Gaussian on `dist_m` (σ about 10 m). Swap in the real model probabilities when they arrive. This makes the decoder independent of the model timeline.

### 8.3 Model specification (document all of this in report section 5)
- **Hidden states:** at time step *i*, the states are the candidate arcs of GPS point *i*, from Zubed's table. Keep `(point_idx, arc_id)` correspondence exactly, because the docs require it.
- **Emission:** `log P(obs_i | arc) = log(p_model(arc | point i) + 1e-9)`. Optionally add a temperature or weight `w_e`, so that the score becomes `w_e * log p`.
- **Transition:** following Newson and Krumm, `log P(arc_b at i+1 | arc_a at i) = -|route_distance - great_circle_distance(point_i, point_{i+1})| / β`. If unreachable within the cutoff, use a large negative penalty (e.g. -50) rather than negative infinity, so that decoding never dies.
- **Viterbi recurrence:** `δ_{i+1}(b) = max_a [δ_i(a) + logT(a, b)] + logE_{i+1}(b)`, with back-pointers. Work in log space.
- **Gaps and missing candidates:**
  - If `t_{i+1} - t_i > gap_break_s` (30 s), **break** the chain: decode each segment separately and record the break. The two huge-gap trips (90 and 78) must not hang.
  - Points with no candidates (placeholder `arc_id = -1`): skip them in decoding and fill them later with the previous matched arc, flagged `filled=True`.
  - If all transitions into a step are impossible, restart the chain at that step (treat it like a break).
- **Parameters to tune:** `β` (e.g. 1 to 50 m), `w_e`, the unreachable penalty, and `gap_break_s`. Tune with grid or Optuna on **training trips, with student_test for checking only**. Save `artifacts/hmm_config.json`.

### 8.4 Route reconstruction
`reconstruct_route(matched_arcs, graph) -> list[arc_id]`: collapse consecutive duplicates, and if consecutive matched arcs aren't adjacent, insert the shortest connecting path. This is the full route that the UI draws and that is compared against `.route`.

### 8.5 Experiments (required by the docs)
On student_test, with the final model's probabilities, compare:

| Method | Top-1 point accuracy | Route continuity | Route precision / recall / F1 |
|---|---|---|---|
| Classifier-only argmax | | | |
| Classifier + HMM | | | |

Also compute the table for the other models if time allows (shows HMM robustness). Discuss trips where the HMM **didn't** help, with a Folium map of one such case (send it to Mannu).

### 8.6 Interface (contract)
```python
# pathsnap/hmm.py
def viterbi(cands: pd.DataFrame,        # Zubed's candidate table for ONE trip (no label)
            probs: np.ndarray,          # model P(label=1) per row of cands, same order
            trip: Trip,                 # from pathsnap.io.load_trip
            config: dict) -> pd.DataFrame:
    """Returns one row per GPS point: point_idx, arc_id (int, -1 if none),
       segment_id (int, increments at each break), filled (bool)."""

def reconstruct_route(matched: pd.DataFrame, trip: Trip) -> list[int]:
    """Connected arc sequence for display and route-level metrics."""
```

### 8.7 Viva preparation
States, emission, transition, recurrence, gap handling, why log space, why the route-distance vs straight-line difference works, and when the HMM hurt.

---

## 9. Metrics (shared definitions, `pathsnap/metrics.py`)

Everyone must compute metrics with these functions, never their own variants.

### 9.1 Candidate-level (classifier quality)
Computed on candidate rows, excluding placeholder rows. Threshold 0.5 on `P(label=1)`:
accuracy, precision, recall, **macro-F1** (averaged over classes 0 and 1), confusion matrix, **log loss**.
Note the class imbalance (about 1 positive per ~10 candidates). Report class balance in the EDA. Using `class_weight` is allowed, but say so, because it distorts probabilities and therefore log loss and HMM emissions.

### 9.2 Observation-level
- **Top-1 accuracy:** for each GPS point, take the candidate with the highest probability and check whether it equals `ref_arc`. Denominator: **all GPS points**. Points with no candidates count as wrong.
- **Candidate miss rate:** the fraction of GPS points whose `ref_arc` is not among its candidates. This is an upper bound on any model's Top-1 accuracy, and should be reported.

### 9.3 Primary selection metric
**Observation-level Top-1 accuracy on grouped CV** (mean over 5 folds) is the primary metric for model selection and the Optuna objective. It measures exactly what the classifier is used for in this task: choosing the right arc per point. Log loss is the secondary metric, because the HMM consumes probabilities and well-calibrated scores matter. Macro-F1 is reported because the docs require it, but it's threshold-dependent and less aligned with the task.

### 9.4 Sequence-level (Keshav)
- **Route continuity:** over consecutive GPS points with *different* predicted arcs, the fraction where the second arc is reachable from the first within 3 arcs in the directed graph. Higher is better. Classifier-only will be noticeably lower.
- **Route precision, recall and F1:** compare the set of unique arc IDs in `reconstruct_route(...)` with the set of unique arc IDs in `.route`.
- **Point accuracy after HMM:** the same as Top-1, but using the HMM's chosen arc.

### 9.5 Efficiency
Training time (seconds, full 80 trips) and inference time (ms per GPS point on student_test, which includes predict_proba only). Use `time.perf_counter()`.

---

## 10. Common training template (every model owner)

Each `models/<name>/train.py` does the following, in order:

1. Load `train_candidates.parquet`, `test_candidates.parquet` and `folds.csv`. Drop placeholder rows (`arc_id == -1`) for fitting.
2. `X = df[FEATURE_COLUMNS]`, `y = df.label`. Build a **sklearn Pipeline** (any scaler goes inside it), so that `artifacts/*.joblib` is one object with `predict_proba(X)`.
3. **CV:** for each of the 5 folds in `folds.csv`, fit on the other folds and compute the metrics in 9.1 and 9.2 on the held-out fold. Save `results/<name>/cv_folds.csv`.
4. Refit on all 80 trips, then evaluate on student_test. Record train and inference time.
5. Save `results/<name>/metrics.json` with **exactly these keys**:

```json
{
  "model": "lightgbm",
  "family": "boosting",
  "config": { "...": "every hyperparameter used" },
  "cv_macro_f1_mean": 0.0, "cv_macro_f1_std": 0.0,
  "cv_top1_mean": 0.0, "cv_top1_std": 0.0,
  "cv_log_loss_mean": 0.0,
  "train_top1": 0.0, "train_macro_f1": 0.0,
  "test_accuracy": 0.0, "test_precision": 0.0, "test_recall": 0.0,
  "test_macro_f1": 0.0, "test_log_loss": 0.0,
  "test_top1": 0.0, "test_candidate_miss_rate": 0.0,
  "test_route_continuity_argmax": 0.0,
  "test_confusion_matrix": [[0, 0], [0, 0]],
  "train_time_s": 0.0, "inference_ms_per_point": 0.0,
  "seed": 42
}
```
Use `null` (with a note in the report) when a metric doesn't apply. `train_*` versus `cv_*` is the overfitting evidence that every model subsection needs.

6. Save the fitted pipeline to `results/<name>/model.joblib`. Commit it if it's under about 50 MB; otherwise document how to regenerate it.
7. Write your model's report subsection **in your own words**, using your actual numbers: why the algorithm suits this problem, strengths and limitations as seen in the experiment, hyperparameters and initial values, the loss or objective, student-test results, and evidence of over- or underfitting.

---

## 11. Likhith: API, UI, pipeline, LightGBM, repo

### 11.1 LightGBM
- `lightgbm.LGBMClassifier`, objective `binary` (log loss). Initial configuration: `num_leaves=31`, `learning_rate=0.05`, `n_estimators=1000` with early stopping (50 rounds) on a validation fold, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`.
- Follow the template in Section 10. LightGBM records per-iteration train and validation loss (`evals_result_`), so a real loss curve can be plotted, which is legitimate for boosting.

### 11.2 End-to-end pipeline (`pathsnap/pipeline.py`)
```python
def predict_trip(trip_dir: str, model, feature_cols, hmm_config) -> dict:
    trip   = load_trip(trip_dir)               # does NOT read .route
    cands  = build_candidates(trip)            # Zubed
    probs  = model.predict_proba(cands[feature_cols])[:, 1]   # placeholders -> 0
    argmax = classifier_only(cands, probs)     # per-point argmax
    hmm    = viterbi(cands, probs, trip, hmm_config)          # Keshav
    route  = reconstruct_route(hmm, trip)                     # Keshav
    return {"matched": hmm, "argmax": argmax, "route": route, "timings": {...}}
```
Evaluation lives in a **separate** module (`api/evaluate.py`) that calls `load_route()` and the metric functions. The prediction path never sees `.route`. Point this out in the viva.

### 11.3 FastAPI backend (Laptop 1)
- Config: `EVAL_DATA_DIR` (env var or config), the folder holding trip subfolders. Test with `student_test`; on demo day, point it at the evaluator folder. **Do not hardcode trip IDs**; list the folders found.
- Load the model, feature columns and HMM config **once at startup**.
- Endpoints:
  - `GET /health` returns `{"status": "ok", "model": "...", "n_trips": 10}`
  - `GET /trips` returns the list of available trip IDs, with point counts if useful (no ground truth)
  - `POST /predict` with `{"trip_id": 52}` returns:
    - `gps_points`: `[[lat, lon], ...]`
    - `predicted_route`: list of polylines `[[lat, lon], ...]` built from the arc geometry of `route`
    - `classifier_only_route` (optional, for comparison)
    - `metrics`: `{top1_point_accuracy_hmm, top1_point_accuracy_argmax, route_precision, route_recall, route_f1, route_continuity, n_points, n_breaks}`
    - `timings_ms`: `{candidates, model, viterbi, total}`
- **Do not send the reference route or labels to the UI.** The docs say ground truth must stay server-side. The UI shows predicted geometry plus the accuracy numbers only.
- **Logging:** use Python `logging` to print one clear line per request on the console: timestamp, client IP, trip ID, number of points, each stage's time, and the accuracy. The evaluators will look at this console, and we need screenshots of it for the report.
- Run with `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Enable CORS for the UI's origin.

### 11.4 UI (Laptop 2)
- Recommended stack: **Streamlit + streamlit-folium** (fastest to build, and it reuses Mannu's `make_trip_map`). A plain HTML page with Leaflet calling the API is an acceptable alternative.
- The UI only talks HTTP to the API (configurable `API_URL`). It **must not import the model or read data files**, because "bypassing the API" is prohibited.
- Flow: on load, call `GET /trips` and show the 10 trip IDs. The evaluator picks one and clicks Run. The UI calls `POST /predict` and draws the map (GPS points in blue, predicted route in red, optional classifier-only route as a toggleable layer). It shows a metrics panel and the timings, plus a clear error message if the API is unreachable.

### 11.5 Two-laptop deployment checklist
- Put both laptops on the **same network**. College Wi-Fi often blocks device-to-device traffic, so use a phone hotspot as a fallback.
- Find Laptop 1's IP (`ipconfig` on Windows, `ip a` on Linux/Mac) and set `API_URL=http://<ip>:8000` on Laptop 2.
- On Windows, allow Python through the firewall for port 8000.
- Test `http://<ip>:8000/health` from Laptop 2's browser before the demo.
- Run all 10 student-test trips through the UI once, end to end. Every trip must work, including the slow, large ones.
- README: install steps, how to start the API, how to start the UI, how to change `EVAL_DATA_DIR` and `API_URL`.

### 11.6 Viva preparation
Request flow, where the ground truth lives and why the UI never sees it, how the model is loaded, and timings per stage.

---

## 12. Integration plan and timeline

The Deliverables doc says there is a **first-phase evaluation before the lunch break**. We don't know exactly what it checks, so aim to have something real for every part by then.

**Phase 0: first 30 to 45 minutes (everyone)**
- Agree on this brief. Zubed confirms or edits the feature list, radius and labelling method.
- Likhith pushes the `ML-Hackathon` skeleton (Section 4) to GitHub and adds everyone as collaborators. Everyone clones it, copies the raw data into `data/raw/`, creates their own branch, and checks `python -c "from pathsnap.io import list_trips; print(list_trips('data/raw/student_test'))"` works.

**Phase 1: until the first-phase evaluation (before lunch)**
- Zubed: loader, geometry, candidates, labels, features, then push the parquet files and `folds.csv` **as early as possible**.
- Mannu: EDA items 1 to 7 and the first Folium map.
- Keshav: graph, route distance, and Viterbi with the placeholder emission, running on a couple of trips.
- Likhith: API and UI skeleton with dummy output, working across two laptops.
- Target: a demo where the UI calls the API and shows a route for a student-test trip (even with placeholder emissions), plus EDA plots.

**Phase 2: after lunch**
- All model owners train on the shared data using the template and push `metrics.json`.
- Zubed builds the comparison table and plots, picks the winner, and runs Optuna, then exports `artifacts/final_model.joblib`.
- Keshav swaps in the real probabilities, tunes β and the other decoder parameters, and runs the classifier-only vs HMM comparison.
- Likhith wires the real pipeline into the API and tests all 10 student-test trips.

**Phase 3: final stretch**
- Mannu builds the result maps and compiles the report. Everyone writes their own sections.
- Likhith takes the UI and API-log screenshots and finalises the README. Make the repo public.
- **Knowledge-share (about 15 minutes):** each person walks the others through their part, since anyone can be asked anything in the viva.

**Integration checkpoints:** push small and often. When your contract output is ready (parquet files, `metrics.json`, `viterbi()`, `/predict`), post in the group chat with the file path.

---

## 13. Rules that must never be broken

1. **Never use evaluator-test data** for training, feature selection, choosing a model, choosing thresholds or tuning the decoder. We don't have it anyway; keep it that way until demo day.
2. **The final model is trained on the 80 training trips.** Student_test is for comparison and decoder checks.
3. **No hardcoded predictions or trip-specific logic.** The repository will be inspected.
4. **Ground truth stays server-side.** The prediction path never reads `.route`, and the UI never receives it.
5. **Use the shared features, folds, metrics and seed.** Otherwise the comparison is invalid.
6. **No fabricated numbers, curves or maps.** Every figure must be produced by committed code. Loss curves only for models that actually train iteratively (the MLP, and LightGBM's boosting iterations).
7. **Don't share code or models with other teams.** The organisers compare repositories, weights and reports across teams and sections.
8. **AI tools help you write and debug code; they don't replace the experiments.** Write your report sections in your own words from your own results, disclose AI use, and make sure you can explain everything you submit.

---

## 14. Quick reference: who to ask

| Question about | Ask |
|---|---|
| Feature columns, candidate radius, labels, folds, parquet files, Optuna | Zubed |
| EDA findings, Folium map function, report formatting | Mannu |
| Road graph, route distance, Viterbi, continuity and route metrics | Keshav |
| Repo structure, API endpoints, UI, deployment, LightGBM, requirements.txt | Likhith |
