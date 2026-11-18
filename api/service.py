"""Prediction service - THE SWAP POINT between mock mode and the real pipeline.  Owner: Likhith.

Mode is decided once, at startup, from the model path (env PATHSNAP_MODEL, else config.yaml
api.model_path, default artifacts/final_model.joblib; relative paths resolve against the repo root):
  * MOCK  - the model file does not exist. The "predicted route" is just the GPS track joined as
            one polyline and every accuracy/route metric is null (only the plain GPS point count
            n_points is filled in). Nothing is fabricated.
  * REAL  - the model, feature columns and HMM config are loaded once, and each request runs
            pathsnap.pipeline.predict_trip (candidates -> model -> argmax -> HMM when available).

The prediction path never reads .route. Metrics come from api/evaluate.py (server-side), are
computed on FULL data, and only whitelisted scalar keys are copied into the response.
Geometry sent to the UI is thinned to at most `max_points` points per polyline.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from api.evaluate import METRIC_KEYS, evaluate_trip, null_metrics
from api.geometry import arcs_to_polylines, gps_to_latlon, simplify_polyline
from pathsnap.config import CFG, REPO_ROOT, repo_path
from pathsnap.io import load_trip

log = logging.getLogger("pathsnap.api")

DEFAULT_MODEL_PATH = "artifacts/final_model.joblib"
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


def resolve_model_path() -> Path:
    raw = os.environ.get("PATHSNAP_MODEL") or CFG.get("api", {}).get("model_path") or DEFAULT_MODEL_PATH
    p = Path(raw).expanduser()
    return (p if p.is_absolute() else REPO_ROOT / p).resolve()


class PredictionService:
    def __init__(self, model_path: Path | str | None = None, artifacts_dir: Path | str | None = None,
                 max_points: int = DEFAULT_MAX_POINTS):
        self.model_path = Path(model_path) if model_path else resolve_model_path()
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else repo_path(CFG["artifacts_dir"])
        self.max_points = max_points
        self.model = None
        self.feature_cols: list[str] | None = None
        self.hmm_config: dict = {}
        self.mock = not self.model_path.is_file()
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

        from pathsnap.features import FEATURE_COLUMNS

        self.model = joblib.load(self.model_path)
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError(f"{self.model_path} has no predict_proba - not a PathSnap classifier.")

        features_path = self.artifacts_dir / FEATURES_FILE
        if features_path.exists():
            self.feature_cols = list(json.loads(features_path.read_text(encoding="utf-8")))
            features_src = str(features_path)
        else:
            self.feature_cols, features_src = list(FEATURE_COLUMNS), "pathsnap.features.FEATURE_COLUMNS"
        n_in = getattr(self.model, "n_features_in_", None)
        if n_in is not None and n_in != len(self.feature_cols):
            raise RuntimeError(f"Model expects {n_in} features but {features_src} lists {len(self.feature_cols)}.")

        hmm_path = self.artifacts_dir / HMM_CONFIG_FILE
        if hmm_path.exists():
            self.hmm_config, hmm_src = json.loads(hmm_path.read_text(encoding="utf-8")), str(hmm_path)
        else:
            self.hmm_config, hmm_src = {"gap_break_s": CFG["gap_break_s"]}, "default {gap_break_s from config.yaml}"
        log.info("Loaded model %s (%d features from %s); HMM config from %s.",
                 self.model_path, len(self.feature_cols), features_src, hmm_src)

    # ------------------------------------------------------------------ public
    def predict(self, trip_dir: Path | str) -> dict[str, Any]:
        """Returns {hmm_used, n_gps_points, trip_duration_s, gps_points, predicted_route,
        classifier_only_route, metrics, timings_ms, mock}."""
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
            "hmm_used": False,
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
        from pathsnap.pipeline import collapse_route, predict_trip

        t0 = time.perf_counter()
        result = predict_trip(trip_dir, self.model, self.feature_cols, self.hmm_config)
        trip = result["trip"]
        gps = gps_to_latlon(trip)

        # Metrics on FULL data, before any thinning. Ground truth stays inside evaluate.py.
        metrics = dict(evaluate_trip(trip_dir, result))

        argmax_route = collapse_route(result["argmax"]["arc_id"])
        if result["hmm_used"]:
            predicted = arcs_to_polylines(trip, result["route"])
            classifier_only = arcs_to_polylines(trip, argmax_route)
        else:
            predicted, classifier_only = arcs_to_polylines(trip, argmax_route), None

        def thin(lines):
            return [simplify_polyline(p, self.max_points) for p in lines]

        timings = {k: result["timings"].get(k) for k in TIMING_KEYS}
        timings["total"] = _ms(time.perf_counter() - t0)   # end to end: pipeline + evaluation + geometry
        return {
            "hmm_used": bool(result["hmm_used"]),
            "n_gps_points": len(gps),
            "trip_duration_s": trip_duration_s(trip),
            "gps_points": simplify_polyline(gps, self.max_points) if len(gps) >= 2 else gps,
            "predicted_route": thin(predicted),
            "classifier_only_route": thin(classifier_only) if classifier_only is not None else None,
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
