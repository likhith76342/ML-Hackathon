"""End-to-end inference for one trip.  Owner: Likhith.  (Brief 11.2)

The prediction path NEVER reads .route. Evaluation lives separately in api/evaluate.py.

    trip   = load_trip(trip_dir)                     # does NOT read .route
    cands  = build_candidates(trip)                  # the SAME function that built the training data
    probs  = model.predict_proba(real rows)[:, 1]    # placeholders (arc_id == -1) -> 0, row order kept
    argmax = per-point highest-probability arc       # -1 when a point has no candidates
    hmm    = viterbi(cands, probs, trip, hmm_config) # Keshav; classifier-only fallback ONLY on NotImplementedError
    route  = reconstruct_route(hmm, trip)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import build_candidates
from .geo import add_xy
from .io import Trip, load_trip

log = logging.getLogger("pathsnap.api")


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 1)


def predict_probs(cands: pd.DataFrame, model, feature_cols: list[str]) -> np.ndarray:
    """P(label=1) per candidate row, same order as cands. Placeholder rows get 0."""
    probs = np.zeros(len(cands), dtype=float)
    real = cands["arc_id"].to_numpy() >= 0
    if real.any():
        probs[real] = model.predict_proba(cands.loc[real, feature_cols])[:, 1]
    return probs


def argmax_arcs(cands: pd.DataFrame, probs: np.ndarray, n_points: int) -> pd.DataFrame:
    """Classifier-only choice: one row per GPS point (point_idx, arc_id), arc_id = -1 if no candidates.
    Ties go to the first row in candidate order (nearest arc), as in pathsnap.metrics.top1_accuracy."""
    df = pd.DataFrame({"point_idx": cands["point_idx"].to_numpy(), "arc_id": cands["arc_id"].to_numpy(),
                       "p": np.asarray(probs, dtype=float)})
    real = df[df["arc_id"] >= 0]
    arc = np.full(n_points, -1, dtype=np.int64)
    if len(real):
        best = real.loc[real.groupby("point_idx", sort=False)["p"].idxmax()]
        arc[best["point_idx"].to_numpy(dtype=np.int64)] = best["arc_id"].to_numpy(dtype=np.int64)
    return pd.DataFrame({"point_idx": np.arange(n_points), "arc_id": arc})


def collapse_route(arcs) -> list[int]:
    """Arc sequence with -1 removed and consecutive duplicates collapsed (for display/route metrics)."""
    a = np.asarray(arcs, dtype=np.int64)
    a = a[a >= 0]
    if not len(a):
        return []
    return [int(x) for x in a[np.r_[True, a[1:] != a[:-1]]]]


def n_breaks(matched: pd.DataFrame | None) -> int | None:
    """Chain breaks reported by the HMM (attrs), else segment count - 1. None without an HMM result."""
    if matched is None:
        return None
    if "n_breaks" in matched.attrs:
        return int(matched.attrs["n_breaks"])
    if "segment_id" in matched.columns and len(matched):
        return int(matched["segment_id"].nunique()) - 1
    return None


def predict_trip(trip_dir: Path | str, model, feature_cols: list[str], hmm_config: dict) -> dict:
    """Returns {"matched", "argmax", "route", "timings", "hmm_used", "trip"}.

    matched : HMM DataFrame (point_idx, arc_id, segment_id, filled) or None if the HMM isn't available
    argmax  : DataFrame (point_idx, arc_id) - classifier-only choice per GPS point
    route   : list[int] - reconstructed HMM route, or the collapsed argmax sequence without HMM
    timings : {"candidates", "model", "viterbi"} in ms (viterbi includes reconstruct_route; None without HMM)
    trip    : the loaded Trip (track/nodes/arcs only - never the reference route), reused for geometry
    """
    timings: dict[str, float | None] = {}

    t0 = time.perf_counter()
    trip: Trip = load_trip(trip_dir)
    add_xy(trip)
    cands = build_candidates(trip)
    timings["candidates"] = _ms(time.perf_counter() - t0)

    t0 = time.perf_counter()
    probs = predict_probs(cands, model, feature_cols)
    argmax = argmax_arcs(cands, probs, len(trip.track))
    timings["model"] = _ms(time.perf_counter() - t0)

    t0 = time.perf_counter()
    try:
        from .hmm import reconstruct_route, viterbi

        matched = viterbi(cands, probs, trip, hmm_config)
        route = reconstruct_route(matched, trip)
        hmm_used = True
        timings["viterbi"] = _ms(time.perf_counter() - t0)
    except NotImplementedError as e:   # ONLY a missing HMM falls back; anything else is a real error
        log.warning("HMM not available (%s): falling back to the classifier-only route.", e)
        matched, route, hmm_used = None, collapse_route(argmax["arc_id"]), False
        timings["viterbi"] = None

    return {"matched": matched, "argmax": argmax, "route": route, "timings": timings,
            "hmm_used": hmm_used, "trip": trip}
