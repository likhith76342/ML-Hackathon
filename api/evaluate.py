"""Server-side evaluation (the only API code that reads .route).  Owner: Likhith.  (Brief 11.2)

Ground truth stays on this laptop. This module loads the reference route, aligns the GPS points to
it with the shared labelling function, and scores the prediction with the SHARED metric functions
(pathsnap.route_metrics; same definitions as pathsnap.metrics). It returns ONLY scalar values for
the keys in METRIC_KEYS. The reference route, per-point labels and anything else derived from
.route never leave this module, and api/service.py whitelists the returned keys.
"""
from __future__ import annotations

import logging
from pathlib import Path

from pathsnap.pipeline import n_breaks

log = logging.getLogger("pathsnap.api")

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
    """Every metric unknown. We never fabricate numbers."""
    return {k: None for k in METRIC_KEYS}


def _try(name: str, fn, *args):
    """Run a shared metric; a not-yet-implemented one stays null."""
    try:
        return fn(*args)
    except NotImplementedError:
        log.warning("Metric %s not implemented yet - reported as null.", name)
        return None


def evaluate_trip(trip_dir: Path | str, prediction: dict) -> dict[str, float | int | None]:
    """Score one pathsnap.pipeline.predict_trip result against the reference route, server-side.
    Everything is computed on FULL (un-thinned) data."""
    from pathsnap import route_metrics
    from pathsnap.io import load_route, load_trip      # load_route: allowed here and ONLY here
    from pathsnap.labels import align_points_to_route

    trip_dir = Path(trip_dir)
    hmm_used = bool(prediction.get("hmm_used"))
    matched = prediction.get("matched") if hmm_used else None
    argmax = prediction["argmax"]

    out = null_metrics()
    out["n_points"] = int(len(argmax))
    out["n_breaks"] = n_breaks(matched) if hmm_used else None

    try:
        reference = load_route(trip_dir)
    except FileNotFoundError:
        log.warning("No reference route for %s - accuracy metrics are null.", trip_dir.name)
        return out

    trip = load_trip(trip_dir)   # fresh copy: the reference never touches the prediction's Trip object
    refs = align_points_to_route(trip, reference).sort_values("point_idx")
    ref_arcs = refs["ref_arc"].to_numpy()

    out["top1_point_accuracy_argmax"] = _try(
        "top1_point_accuracy_argmax", route_metrics.point_accuracy,
        argmax.sort_values("point_idx")["arc_id"].to_numpy(), ref_arcs)
    if hmm_used:
        out["top1_point_accuracy_hmm"] = _try(
            "top1_point_accuracy_hmm", route_metrics.point_accuracy,
            matched.sort_values("point_idx")["arc_id"].to_numpy(), ref_arcs)

    prf = _try("route_prf", route_metrics.route_prf, list(prediction["route"]), reference)
    if prf:
        out.update({k: prf.get(k) for k in ("route_precision", "route_recall", "route_f1")})

    per_point = (matched if hmm_used else argmax).sort_values("point_idx")["arc_id"].tolist()
    out["route_continuity"] = _try("route_continuity", route_metrics.route_continuity, per_point, trip)
    return out

