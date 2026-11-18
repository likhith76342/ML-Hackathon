"""Prediction service - THE SWAP POINT between mock mode and the real pipeline.  Owner: Likhith.

Mode is decided once, at startup:
  * MOCK  - artifacts/final_model.joblib does not exist. The "predicted route" is just the GPS
            track joined as one polyline and every accuracy/route metric is null (only the plain
            GPS point count n_points is filled in). Nothing is fabricated.
  * REAL  - the model, feature columns and HMM config are loaded once, and each request runs
            pathsnap.pipeline.predict_trip (see the TODO in _predict_real).

The prediction path never reads .route. Metrics come from api/evaluate.py (server-side), are
computed on FULL data, and only whitelisted scalar keys are copied into the response.
Geometry sent to the UI is thinned to at most `max_points` points per polyline.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from api.evaluate import METRIC_KEYS, evaluate_trip, null_metrics
from api.geometry import arcs_to_polylines, gps_to_latlon, simplify_polyline
from pathsnap.config import CFG, repo_path
from pathsnap.io import load_trip

log = logging.getLogger("pathsnap.api")

MODEL_FILE = "final_model.joblib"
FEATURES_FILE = "feature_columns.json"
HMM_CONFIG_FILE = "hmm_config.json"
DEFAULT_MAX_POINTS = int(CFG.get("api", {}).get("max_polyline_points", 5000))
TIMING_KEYS = ("candidates", "model", "viterbi", "total")   # Brief 11.3

MOCK_WARNING = (
    "MOCK MODE: %s not found - no trained model loaded. predicted_route is the raw GPS "
    "track and every accuracy/route metric is null."
)


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 1)


def trip_duration_s(trip) -> float:
    """Travel time from the INPUT track's first to last timestamp (not ground truth)."""
    t = trip.track["t"]
    return round(float(t.iloc[-1] - t.iloc[0]), 3) if len(t) else 0.0


class PredictionService:
    def __init__(self, artifacts_dir: Path | str | None = None, max_points: int = DEFAULT_MAX_POINTS):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else repo_path(CFG["artifacts_dir"])
        self.model_path = self.artifacts_dir / MODEL_FILE
        self.max_points = max_points
        self.model = None
        self.feature_cols: list[str] | None = None
        self.hmm_config: dict = {}
        self.mock = not self.model_path.exists()
        if self.mock:
            log.warning(MOCK_WARNING, self.model_path)
        else:
            self._load_artifacts()

    @property
    def mode(self) -> str:
        return "mock" if self.mock else "real"

    @property
    def model_name(self) -> str:
        return "mock (no model)" if self.mock else self.model_path.name

    def _load_artifacts(self) -> None:
        """Load model + feature columns + HMM config once at startup (Brief 11.3)."""
        import joblib

        features_path = self.artifacts_dir / FEATURES_FILE
        if not features_path.exists():
            raise RuntimeError(f"{self.model_path} exists but {features_path} is missing.")
        self.model = joblib.load(self.model_path)
        self.feature_cols = json.loads(features_path.read_text(encoding="utf-8"))
        hmm_path = self.artifacts_dir / HMM_CONFIG_FILE
        if hmm_path.exists():
            self.hmm_config = json.loads(hmm_path.read_text(encoding="utf-8"))
        else:
            log.warning("%s not found - using an empty HMM config (decoder defaults).", hmm_path)
        log.info("Loaded model %s with %d feature columns.", self.model_path, len(self.feature_cols))

    # ------------------------------------------------------------------ public
    def predict(self, trip_dir: Path | str) -> dict[str, Any]:
        """Returns {gps_points, predicted_route, classifier_only_route, metrics, timings_ms, mock}."""
        trip_dir = Path(trip_dir)
        if self.mock:
            log.warning("MOCK MODE: request for %s served with GPS-as-route and null metrics.", trip_dir.name)
            out = self._predict_mock(trip_dir)
        else:
            out = self._predict_real(trip_dir)
        out["mock"] = self.mock
        # Whitelist: only the agreed scalar metric keys ever reach the response.
        out["metrics"] = {k: out["metrics"].get(k) for k in METRIC_KEYS}
        return out

    # ------------------------------------------------------------------ mock
    def _predict_mock(self, trip_dir: Path) -> dict[str, Any]:
        t0 = time.perf_counter()
        trip = load_trip(trip_dir)                         # never reads .route
        gps = gps_to_latlon(trip)
        metrics = null_metrics()
        metrics["n_points"] = len(gps)     # plain count of the FULL track, not a ground-truth metric
        out = {
            "n_gps_points": len(gps),
            "trip_duration_s": trip_duration_s(trip),
            "gps_points": simplify_polyline(gps, self.max_points) if len(gps) >= 2 else gps,
            "predicted_route": [simplify_polyline(gps, self.max_points)] if len(gps) >= 2 else [],
            "classifier_only_route": None,
            "metrics": metrics,
        }
        timings = {k: None for k in TIMING_KEYS}
        timings["total"] = _ms(time.perf_counter() - t0)
        out["timings_ms"] = timings
        return out

    # ------------------------------------------------------------------ real
    def _predict_real(self, trip_dir: Path) -> dict[str, Any]:
        # ================================================================================
        # TODO(Likhith): REAL MODE. Enable once pathsnap.candidates / hmm are implemented.
        # predict_trip currently raises NotImplementedError, which /predict turns into a 500.
        # ================================================================================
        from pathsnap.pipeline import predict_trip

        t0 = time.perf_counter()
        result = predict_trip(trip_dir, self.model, self.feature_cols, self.hmm_config)
        # result = {"matched": DataFrame(point_idx, arc_id, segment_id, filled),
        #           "argmax":  per-point classifier-only choice,
        #           "route":   list[int] connected arc sequence,
        #           "timings": {"candidates", "model", "viterbi"} in ms}

        trip = load_trip(trip_dir)   # TODO: have predict_trip return the Trip to avoid a reload
        gps = gps_to_latlon(trip)

        # Metrics on FULL data, before any thinning. Ground truth stays inside evaluate.py.
        metrics = dict(evaluate_trip(trip_dir, result))
        metrics["n_points"] = len(gps)
        matched = result.get("matched")
        if matched is not None and "segment_id" in getattr(matched, "columns", []):
            metrics["n_breaks"] = int(matched["segment_id"].nunique()) - 1 if len(matched) else 0

        predicted = arcs_to_polylines(trip, result["route"])
        # TODO: classifier-only route = reconstruct_route(argmax frame, trip) once hmm.py exists.
        classifier_only = None

        timings = {k: result.get("timings", {}).get(k) for k in TIMING_KEYS}
        timings["total"] = _ms(time.perf_counter() - t0)
        return {
            "n_gps_points": len(gps),
            "trip_duration_s": trip_duration_s(trip),
            "gps_points": simplify_polyline(gps, self.max_points) if len(gps) >= 2 else gps,
            "predicted_route": [simplify_polyline(p, self.max_points) for p in predicted],
            "classifier_only_route": classifier_only,
            "metrics": metrics,
            "timings_ms": timings,
        }


_service: PredictionService | None = None


def get_service() -> PredictionService:
    """The process-wide service (created on first use, i.e. at API startup)."""
    global _service
    if _service is None:
        _service = PredictionService()
    return _service


def predict(trip_dir: Path | str) -> dict[str, Any]:
    return get_service().predict(trip_dir)
