# HMM / Viterbi decoder — Keshav's part (brief Sections 8 and 9.4)

Files owned by Keshav (brief 12.1): `pathsnap/graph.py`, `pathsnap/hmm.py`, `pathsnap/route_metrics.py`,
`hmm_experiments/`, `artifacts/hmm_config.json`, `results/hmm*`, figures prefixed `hmm_`, `tests/test_hmm.py`.

## Files

| Path | What it is |
|---|---|
| `pathsnap/graph.py` | Directed road graph per trip, in the shared `geo.add_xy` frame; `route_distance(arc_a, t_a, arc_b, t_b)`; cut-off Dijkstra cached per source node; `arcs_within_hops` for continuity |
| `pathsnap/hmm.py` | `viterbi(cands, probs, trip, config)` and `reconstruct_route(matched, trip)` (contract 8.6), plus `classifier_only()` baseline, and `prepare_trip()` / `decode()` used for tuning |
| `pathsnap/route_metrics.py` | Sequence-level metrics (9.4): `point_accuracy`, `route_continuity`, `route_prf` |
| `hmm_experiments/tune.py` | Decoder parameter grid on the 80 training trips -> `artifacts/hmm_config.json` |
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

## Results so far (placeholder emissions — NOT the final model)

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
