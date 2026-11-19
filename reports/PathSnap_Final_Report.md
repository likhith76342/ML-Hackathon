# PathSnap: GPS Map-Matching and Route Reconstruction

**Team members**

| USN | Name | Section |
|---|---|---|
| PES1UG24CS673 | Manu Kudkundi | L |
| PES1UG24CS674 | Mohammed Al Zubed Hulgeri | L |
| PES1UG24CS719 | Keshav Agrawal | L |
| PES1UG24CS720 | Likhith S | L |

## Abstract

PathSnap maps noisy GPS observations to a road network. For every GPS point, the system finds nearby directed road-segment candidates, scores the point–segment pairs with a classifier, and uses an HMM with Viterbi decoding to select one connected route. The selected final classifier is LightGBM. On the retained `student_test` comparison, the final classifier achieved Top-1 accuracy of 0.8564; the HMM route reconstruction achieved pooled Top-1 accuracy of 0.9246. These scores are relative to approximate reference-route labels and should not be interpreted as perfect geographic ground truth.

## 1. Dataset and protocol

The shared dataset uses dataset version 2: a 50 m candidate radius, at most 10 directed candidates per GPS point, five trip-grouped folds, seed 42, and flagged trips 33 and 54 excluded from the primary model results. Training uses 80 trips and `student_test` contains 10 trips. The raw dataset is intentionally not committed to the repository.

Each candidate-table row represents one GPS-point/road-segment pair. The label is 1 when the candidate agrees with the aligned reference route and 0 otherwise. Reference routes are used for training labels and evaluation, not as inference features.

## 2. Feature extraction

The model uses 12 features: point-to-segment distance, distance rank, distance above the closest candidate, distance ratio, GPS/segment bearing difference, speed, heading-valid flag, previous and next time gaps, number of candidates, segment length, and projected position along the segment. Coordinates are projected locally into metres using an equirectangular approximation centred on the trip.

## 3. Model comparison

| Model | CV Top-1 | Test Top-1 | Test macro-F1 | Test log loss | Inference ms/point |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8383 | 0.8419 | 0.8901 | 0.1524 | 0.0025 |
| Random Forest | 0.8489 | 0.8553 | 0.9127 | 0.1071 | 0.0178 |
| KNN | 0.8339 | 0.8461 | 0.9066 | 0.3179 | 5.2239 |
| MLP | 0.8457 | 0.8504 | 0.9077 | 0.1121 | 0.0042 |
| LightGBM | 0.8505 | 0.8561* | 0.9138 | 0.1040 | 0.0209 |

\* The table above uses the ordinary LightGBM training run in `results/lightgbm/metrics.json`. The selected final refit uses 743 trees; its verified test Top-1 0.8564, macro-F1 0.9132, and log loss 0.1048 are recorded in `optimisation/summary.json`.

### Logistic Regression

Logistic Regression provides a simple linear baseline. It is fast, but its linear decision boundary cannot fully capture interactions between distance, heading, speed, and temporal features.

### Random Forest

Random Forest combines many decision trees and captures nonlinear feature interactions. It improves over Logistic Regression but is slower and less compact than the linear baseline.

### KNN

KNN compares each candidate with similar training candidates. It uses standardised features, 15 neighbours, and distance-weighted voting. It is conceptually simple, but its inference cost is much higher because it searches the stored training examples for every new candidate.

### MLP

The MLP is a neural network with hidden layers of 64 and 32 neurons. It learns nonlinear relationships between the candidate features. It uses feature standardisation, batches of 512, early stopping, and at most 50 training iterations.

### LightGBM

LightGBM builds boosted decision trees sequentially. Bayesian optimisation selected the final configuration, which was refit using 743 trees. It achieved the best classifier-level test Top-1 among the compared models and was therefore selected for the final API pipeline.

## 4. HMM route reconstruction

The classifier scores candidates independently for each GPS point. The HMM then combines those scores with road connectivity and transition constraints. Viterbi decoding chooses the most likely complete connected sequence rather than making every point decision independently. On `student_test`, HMM Top-1 was 0.9246 versus 0.8564 for classifier-only Top-1. HMM improved Top-1 on all 10 evaluated student-test trips. Trips 76 and 77 showed high recall but lower route precision, so their reconstructed route behaviour should be discussed as a limitation.

## 5. EDA and visualisations

The repository contains partition-labelled EDA tables and figures, including trip summaries, geographic coverage, network size, time gaps, and model comparison charts. Folium maps show GPS observations and route layers. The maps use the committed generated HTML under `reports/figures/maps/`.

## 6. API and UI

The FastAPI backend loads `artifacts/final_model.joblib`, the feature-column list, and the HMM configuration. The Streamlit UI communicates with the API over HTTP and displays per-trip metrics, routes, timing breakdowns, and a run-all-trips table. The verified real-mode demo loaded `model.joblib`, processed trips successfully, and showed classifier-only and HMM results.

**Screenshot insertion points:**

1. API startup or `/health` response showing `mode: real` and `final_model.joblib`.
2. Streamlit connection showing `real mode`.
3. Single-trip UI result showing metrics and the route map.
4. Run-all-trips UI table.
5. Optional expanded map-layer control or API timing panel.

### Evaluator smoke test

The final API was started in real mode with `EVAL_DATA_DIR=data/raw/evaluator_test`. All 10 evaluator trips completed successfully with HTTP 200 responses and non-null real-mode metrics. Trip 15, the required large-trip check, completed in 4.53 seconds for 6,854 GPS points.

| Trip | GPS points | Classifier Top-1 | HMM Top-1 | Route F1 | Total time | Status |
|---:|---:|---:|---:|---:|---:|---|
| 15 | 6,854 | 58.83% | 92.53% | 79.29% | 4.53 s | OK |
| 17 | 1,346 | 91.08% | 91.98% | 91.96% | 1.77 s | OK |
| 23 | 458 | 93.89% | 95.20% | 78.38% | 447 ms | OK |
| 30 | 654 | 79.05% | 86.85% | 93.75% | 839 ms | OK |
| 49 | 2,642 | 97.88% | 99.58% | 99.76% | 1.69 s | OK |
| 59 | 1,031 | 94.96% | 97.58% | 95.89% | 1.10 s | OK |
| 67 | 916 | 90.50% | 96.18% | 98.78% | 771 ms | OK |
| 73 | 1,144 | 97.20% | 98.51% | 98.96% | 1.12 s | OK |
| 87 | 1,399 | 93.35% | 99.64% | 99.89% | 1.48 s | OK |
| 94 | 3,230 | 85.73% | 94.21% | 96.33% | 2.71 s | OK |

All 10 rows in the UI run-all table reported `ok`. The API log showed `mode=REAL`, `model=final_model.joblib`, `hmm=yes`, and HTTP 200 for every `/predict` request. The UI also issued repeated `GET /v1/models` requests that returned 404; this endpoint is not used by the current API contract and did not affect prediction success.

## 7. Limitations and reproducibility

Reference routes are approximate and may contain alignment or GPS-noise errors. The candidate miss rate and all accuracy values therefore measure agreement with the supplied reference labels, not absolute physical truth. The Bayesian search stopped after 18 completed trials although the brief requested at least 50; this must be disclosed. The evaluator dataset was smoke-tested under `data/raw/evaluator_test`; all 10 trips completed in real mode with HTTP 200 responses. Trip 15 completed in 4.53 seconds. These evaluator metrics are reported for smoke-test evidence only and were not used to tune or change the model.

All reported numerical results are sourced from committed files in `results/`, `results/hmm/`, and `optimisation/summary.json`.

## 8. AI assistance disclosure

AI assistance was used for code debugging, repository auditing, documentation drafting, and explanation. Experiments, metrics, model outputs, and screenshots must be verified by the team from the repository and live system before submission.

## 9 Guideline compliance audit

| Guideline item | Status | Evidence or limitation |
|---|---|---|
| EDA inventory, network size, time gaps, and geographic coverage | Done | Partition-labelled tables and figures in `results/eda/` and `reports/figures/eda_*.png` |
| Candidate miss rate and suspicious reference distances | Done | Training: 24,473/208,417 misses (11.74%); 20,538 points over 100 m. Student_test: 52/19,160 misses (0.27%); 17 points over 100 m. |
| Full test suite | Flag | Recorded audit result: 87 passed, 2 failures, 397 warnings. One is a Windows UTF-8 chart-test portability issue; the other is the documented strict twin-tie XPASS. |
| API/UI 404 probes | Documented | The UI emitted `GET /v1/models` 404 probes, but all required `/health`, `/trips`, and `/predict` requests succeeded. |

The report distinguishes completed evidence from the live-system verification and does not include unverified experiment outputs.
