# HMM / Viterbi decoder — Keshav's part (brief Sections 8 and 9.4)

Files owned by Keshav (brief 12.1): `pathsnap/graph.py`, `pathsnap/hmm.py`, `pathsnap/route_metrics.py`,
`hmm_experiments/`, `artifacts/hmm_config.json`, `results/hmm*`, figures prefixed `hmm_`, `tests/test_hmm.py`.

## Files

| Path | What it is |
|---|---|
| `pathsnap/graph.py` | Directed road graph per trip, in the shared `geo.add_xy` frame; `route_distance(arc_a, t_a, arc_b, t_b)`; cut-off Dijkstra cached per source node; `arcs_within_hops` for continuity |
| `pathsnap/hmm.py` | `viterbi(cands, probs, trip, config)` and `reconstruct_route(matched, trip)` (contract 8.6), plus `classifier_only()` baseline, and `prepare_trip()` / `decode()` used for tuning |
| `pathsnap/route_metrics.py` | Sequence-level metrics (9.4): `point_accuracy`, `route_continuity`, `route_prf` |
| `hmm_experiments/make_oof_probs.py` | Out-of-fold probabilities of the final LightGBM model for the training trips (each trip scored by the fold model that didn't see it; trips 33 and 54 excluded from fitting) |
| `hmm_experiments/tune.py` | Decoder parameter grid on the training trips (dataset v2: trips 33 and 54 excluded) -> `artifacts/hmm_config.json` |
| `hmm_experiments/compare.py` | Classifier-only argmax vs classifier + HMM on student_test (table 8.5, plots, Folium maps) |
| `hmm_experiments/common.py` | Loads a trip through the shared pipeline (`io`, `geo.add_xy`, `candidates`, `labels`), adds placeholder or model probabilities, caches per trip |
| `tests/test_hmm.py` | Unit tests on a 5-node synthetic network (`python -m pytest tests -q`) |

## Run (from repo root)

```bash
python -m pytest tests -q
python -m hmm_experiments.tune                      # placeholder emissions (Gaussian on dist_m, sigma 10 m)
python -m hmm_experiments.compare                   # student_test table + figures + maps
```

With real model probabilities (Phase 2):

```bash
# tuning needs OUT-OF-FOLD probabilities for the training trips (each trip scored by the fold model that didn't see it)
python -m hmm_experiments.tune --emission model --probs-train data/processed/oof_probs_train_<model>.parquet
# student_test: score test_candidates.parquet with the exported pipeline, by row position (as the API does)
python -m hmm_experiments.compare --model artifacts/final_model.joblib --name <model>
```
Probability parquet schema: `trip_id, point_idx, arc_id, prob`, one row per candidate row, placeholders with prob 0.
All paths come from `config.yaml` and are resolved with pathlib against the repo root (Windows/Mac safe).

## Decoder specification (as implemented)

- **States**: candidate rows of point *i* (arc_id from the candidate table; correspondence kept by row).
- **Emission**: `w_e * log(p + 1e-9)`.
- **Transition**: `-|route_distance - great_circle(point_i, point_{i+1})| / beta`; shortest-path cutoff
  `max(2 * great_circle, 500 m)`; unreachable -> `unreachable_penalty`; reachable scores floored at the
  penalty (`clip_to_penalty`) so an absurd-but-reachable move is never preferred over "unreachable".
- **Same-arc backward jitter** (`backward_tolerance_m`): a backward move along the same arc of at most this
  many metres counts as route distance 0. Without it, a stopped or slow vehicle with noisy fixes looks like
  it drove around the block, and the decoder escapes onto neighbouring arcs.
- **Breaks**: `dt > gap_break_s`, or every transition into a step unreachable -> new segment.
- **No-candidate points**: skipped, then filled from the previous matched arc (next one at a segment start), `filled=True`.
- **Route reconstruction**: collapse consecutive duplicates; insert shortest connecting path between non-adjacent
  arcs (cutoff `max(3 * gap, 1000 m)`); if none, leave the discontinuity.

## Final results (LightGBM final model)

Reproduce:
```bash
python -m hmm_experiments.make_oof_probs
python -m hmm_experiments.tune --emission model --probs-train data/processed/oof_probs_train_lightgbm.parquet --name lightgbm
python -m hmm_experiments.compare --model artifacts/final_model.joblib --name lightgbm
```

- **Emissions:** the final model is `artifacts/final_model.joblib`, LightGBM with `optimisation/best_params.json`.
  Tuning uses its out-of-fold probabilities (mean out-of-fold Top-1 0.851 over 5 folds; `results/hmm/oof_probs_lightgbm.json`).
- **Tuning:** 292 configurations on 78 training trips (dataset v2, trips 33 and 54 excluded).
  - Files: `results/hmm/tuning_lightgbm.csv`, `best_config_lightgbm.json`, `reports/figures/hmm_beta_sensitivity_lightgbm.png`.
  - Selected: beta = 20, w_e = 0.1, unreachable penalty = -50, backward tolerance = 50 m, gap_break_s = 120.
  - Training mean per-trip accuracy after the HMM: 0.9326.
  - The optimum is a broad plateau. Only the ratio of w_e to beta matters (scaling both gives the same Viterbi path); every top
    configuration has w_e x beta of about 2-2.5. Backward tolerance is flat from 30 m (0.932) to 100 m, versus 0.852 with none.
    gap_break_s 30 vs 120 differs by 0.0003.
- **Student test** (`results/hmm/comparison_lightgbm.csv`, per trip in `student_test_per_trip_lightgbm.csv`):

| Method | Top-1 (pooled) | Top-1 (mean per trip) | Route continuity | Route precision | Route recall | Route F1 |
|---|---|---|---|---|---|---|
| Classifier-only argmax (LightGBM) | 0.8566 | 0.8735 | 0.8923 | 0.6310 | 0.9921 | 0.7587 |
| Classifier + HMM | **0.9246** | **0.9409** | **0.9949** | **0.9510** | **0.9927** | **0.9709** |

  - The HMM improves Top-1 on all 10 trips: least on trip 9 (+0.043), most on trip 76 (+0.091).
    Figure: `reports/figures/hmm_vs_argmax_lightgbm.png`; maps: `reports/figures/maps/hmm_lightgbm_trip{9,76}_*_gain.html`.
  - Candidate miss rate 0.27%, so the upper bound on Top-1 is 99.73%. Decoder time is about 0.25 ms per GPS point.
  - The argmax Top-1 here (0.85663) differs from `optimisation/summary.json` (0.85642) by about 10 points out of roughly 48k.
    Cause: the known candidate tie-break issue (twin arcs A->B / B->A at equal distance get a machine-dependent order;
    see the xfail in `tests/test_pipeline.py`). The candidate parquets here were built on Linux.
- Evaluator-test data was not used for any of this.

## Development results (placeholder emissions — before the final model and before dataset v2)

Regenerated on Python 3.11 with the pinned requirements, using the team's shared candidates and labels.

All numbers come from `results/hmm/*.csv|json`, produced by the commands above.

- Tuning: 129 configurations evaluated (two-stage grid: 8 beta x 5 tolerance x 3 w_e, then 3 penalty x 3 gap), objective = mean per-trip point accuracy on the
  80 training trips -> `results/hmm/tuning_placeholder.csv`, `best_config_placeholder.json`.
  Best: beta = 20, w_e = 2.0, penalty = -100, backward tolerance = 30 m, gap_break_s = 30 (accuracy is flat
  across beta 10-50; the backward tolerance is what matters).
  See `reports/figures/hmm_beta_sensitivity_placeholder.png`.
- Student-test comparison: `results/hmm/comparison_placeholder.csv`, per trip in
  `student_test_per_trip_placeholder.csv`, figure `reports/figures/hmm_vs_argmax_placeholder.png`,
  maps `reports/figures/maps/hmm_placeholder_trip*_gain.html` (for Mannu).

The placeholder emission ignores heading, so argmax can't separate two-way twin arcs; that's why argmax is so
low here. With the real classifier (which has `bearing_diff`) argmax will be much higher and the HMM gain smaller.
These placeholder numbers are development evidence only; Section 5 of the report must use the model run.

## Data observations to pass on

- Training trips 54 and 33: GPS sits a median ~140 m / ~120 m from the reference route (trip 54: mean offset
  about 118 m north), so with a 50 m radius the reference arc is missing for ~91% / ~67% of points. That's a
  reference-route limitation (Zubed: miss rate; Mannu: EDA table), not a decoder bug.
- Trips 90 and 78 (huge time gaps) decode fine: 1 and 2 breaks respectively.
