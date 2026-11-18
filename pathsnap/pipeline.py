"""End-to-end inference for one trip.  Owner: Likhith.  (Brief 11.2)

The prediction path NEVER reads .route. Evaluation lives separately in api/evaluate.py.
"""
from __future__ import annotations

from pathlib import Path


def predict_trip(trip_dir: Path | str, model, feature_cols: list[str], hmm_config: dict) -> dict:
    """load_trip -> build_candidates -> predict_proba -> argmax + viterbi -> reconstruct_route.
    Returns {"matched", "argmax", "route", "timings"}."""
    raise NotImplementedError("Likhith: wire up once candidates/hmm exist")
