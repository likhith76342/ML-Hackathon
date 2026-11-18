"""Server-side evaluation (the only API code that reads .route).  Owner: Likhith.  (Brief 11.2)

Ground truth stays on this laptop. This module may load the reference route, but it returns
ONLY scalar metric values (float | int | None) for the keys in METRIC_KEYS. The reference
route, per-point labels and anything else derived from .route must never leave this module,
and api/service.py whitelists the returned keys before anything goes into a response.
"""
from __future__ import annotations

from pathlib import Path

# Brief 11.3: the metrics block of the /predict response.
METRIC_KEYS: tuple[str, ...] = (
    "top1_point_accuracy_hmm",
    "top1_point_accuracy_argmax",
    "route_precision",
    "route_recall",
    "route_f1",
    "route_continuity",
    "n_points",
    "n_breaks",
)


def null_metrics() -> dict[str, None]:
    """Every metric unknown. Used in mock mode - we never fabricate numbers."""
    return {k: None for k in METRIC_KEYS}


def evaluate_trip(trip_dir: Path | str, prediction: dict) -> dict[str, float | int | None]:
    """Score one prediction against the reference route, server-side.

    `prediction` is the dict returned by pathsnap.pipeline.predict_trip
    ({"matched", "argmax", "route", "timings"}), computed on FULL (un-thinned) data.

    TODO(Likhith, real mode): once Zubed's labels/metrics and Keshav's route_metrics exist:
        from pathsnap.io import load_route              # allowed here and only here
        reference = load_route(trip_dir)
        # point refs   -> pathsnap.labels (align points to the reference route)
        # top1_*       -> pathsnap.metrics on prediction["matched"] / prediction["argmax"]
        # route_*      -> pathsnap.route_metrics on prediction["route"] vs reference
        # return only the scalar values for METRIC_KEYS
    Until then every metric is null.
    """
    return null_metrics()
