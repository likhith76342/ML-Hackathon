"""API tests on data/raw/student_test, in mock mode and in real mode.  Owner: Likhith.

Run from the repo root:  pytest tests/
Real-mode tests use results/lightgbm/model.joblib (override with PATHSNAP_TEST_MODEL) and are
skipped if it is missing.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from pathsnap.config import CFG, REPO_ROOT, repo_path
from pathsnap.io import Trip, list_trips, load_route, load_trip, trip_dir

STUDENT_TEST = repo_path(CFG["data_root"]) / "student_test"
TRIPS = list_trips(STUDENT_TEST) if STUDENT_TEST.is_dir() else []
REAL_MODEL = REPO_ROOT / os.environ.get("PATHSNAP_TEST_MODEL", "results/lightgbm/model.joblib")
MISSING_MODEL = "artifacts/__no_such_model__.joblib"
FORBIDDEN_KEYS = {"route", "reference", "label", "ground_truth"}
METRIC_KEYS = {"top1_point_accuracy_hmm", "top1_point_accuracy_argmax", "route_precision", "route_recall",
               "route_f1", "route_continuity", "n_points", "n_breaks"}

needs_data = pytest.mark.skipif(not TRIPS, reason=f"student_test data not found at {STUDENT_TEST}")
needs_model = pytest.mark.skipif(not REAL_MODEL.is_file(), reason=f"model not found at {REAL_MODEL}")


@contextmanager
def started_client(model_path: str):
    """A TestClient on a fresh app. Env vars are read once at startup, then restored."""
    from fastapi.testclient import TestClient

    from api.main import create_app

    keys = ("EVAL_DATA_DIR", "PATHSNAP_MODEL")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["EVAL_DATA_DIR"] = str(STUDENT_TEST.relative_to(REPO_ROOT))
    os.environ["PATHSNAP_MODEL"] = model_path
    client = TestClient(create_app())
    try:
        client.__enter__()                  # runs the startup (lifespan) hook
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    try:
        yield client
    finally:
        client.__exit__(None, None, None)


@pytest.fixture(scope="module")
def mock_client():
    with started_client(MISSING_MODEL) as c:
        yield c


@pytest.fixture(scope="module")
def real_client():
    if not REAL_MODEL.is_file():
        pytest.skip(f"model not found at {REAL_MODEL}")
    with started_client(str(REAL_MODEL.relative_to(REPO_ROOT))) as c:
        yield c


@pytest.fixture(scope="module")
def real_responses(real_client):
    """POST /predict once per student_test trip (shared by the real-mode tests)."""
    out = {}
    for tid in TRIPS:
        r = real_client.post("/predict", json={"trip_id": tid})
        assert r.status_code == 200, r.text
        out[tid] = r.json()
    return out


def all_keys(obj) -> set[str]:
    """Every dict key at any depth."""
    if isinstance(obj, dict):
        return set(obj) | set().union(*(all_keys(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(all_keys(v) for v in obj)) if obj else set()
    return set()


def check_common_shape(body: dict, trip_id: int) -> Trip:
    assert body["trip_id"] == trip_id
    for k in ("gps_points", "predicted_route", "classifier_only_route", "metrics", "timings_ms", "mock",
              "hmm_used", "n_gps_points", "trip_duration_s"):
        assert k in body
    assert set(body["metrics"]) == METRIC_KEYS
    assert set(body["timings_ms"]) == {"candidates", "model", "viterbi", "total"}
    assert not (all_keys(body) & FORBIDDEN_KEYS), "ground-truth-like key leaked into the response"

    gps = body["gps_points"]
    assert gps and all(len(p) == 2 for p in gps) and len(gps) <= 5000
    assert body["predicted_route"] and all(len(line) <= 5000 for line in body["predicted_route"])

    trip = load_trip(trip_dir(STUDENT_TEST, trip_id))
    track = trip.track
    assert gps[0] == pytest.approx([track.iloc[0].lat, track.iloc[0].lon])        # [lat, lon] order
    assert body["n_gps_points"] == len(track)
    assert body["metrics"]["n_points"] == len(track) == body["n_gps_points"]       # FULL count, every mode
    assert body["trip_duration_s"] == pytest.approx(track.t.iloc[-1] - track.t.iloc[0])
    return trip


# ----------------------------------------------------------------------------- mock mode
@needs_data
def test_health_mock(mock_client):
    body = mock_client.get("/health").json()
    assert body["status"] == "ok" and body["mode"] == "mock"
    assert body["n_trips"] == len(TRIPS) and "data_dir" in body and "model" in body


@needs_data
def test_trips_lists_folders(mock_client):
    trips = mock_client.get("/trips").json()["trips"]
    assert [t["trip_id"] for t in trips] == TRIPS
    for t in trips:
        assert set(t) == {"trip_id", "n_points"}
        assert t["n_points"] == len(load_trip(trip_dir(STUDENT_TEST, t["trip_id"])).track)


@needs_data
@pytest.mark.parametrize("trip_id", TRIPS)
def test_predict_mock_every_trip(mock_client, trip_id):
    r = mock_client.post("/predict", json={"trip_id": trip_id})
    assert r.status_code == 200, r.text
    body = r.json()
    check_common_shape(body, trip_id)
    assert body["mock"] is True and body["hmm_used"] is False and body["mode"] == "mock"
    others = {k: v for k, v in body["metrics"].items() if k != "n_points"}
    assert all(v is None for v in others.values()), f"mock mode must not report accuracy metrics: {others}"
    assert body["classifier_only_route"] is None


@needs_data
def test_predict_unknown_trip_404(mock_client):
    assert mock_client.post("/predict", json={"trip_id": max(TRIPS) + 1000}).status_code == 404


@needs_data
@pytest.mark.parametrize("body", [{}, {"trip_id": "abc"}, {"trip": 52}, {"trip_id": None}])
def test_predict_bad_body_422(mock_client, body):
    assert mock_client.post("/predict", json=body).status_code == 422


@needs_data
def test_predict_non_json_422(mock_client):
    r = mock_client.post("/predict", content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# ----------------------------------------------------------------------------- real mode
@needs_data
@needs_model
def test_health_real(real_client):
    body = real_client.get("/health").json()
    assert body["mode"] == "real" and body["model"] == REAL_MODEL.name


@needs_data
@needs_model
@pytest.mark.parametrize("trip_id", TRIPS)
def test_predict_real_every_trip(real_responses, trip_id):
    body = real_responses[trip_id]
    check_common_shape(body, trip_id)
    assert body["mock"] is False and body["mode"] == "real"
    m, t = body["metrics"], body["timings_ms"]
    assert 0.0 <= m["top1_point_accuracy_argmax"] <= 1.0
    for k in ("candidates", "model", "total"):
        assert t[k] is not None and t[k] >= 0
    if body["hmm_used"]:
        assert body["classifier_only_route"] is not None
        assert 0.0 <= m["top1_point_accuracy_hmm"] <= 1.0
        assert isinstance(m["n_breaks"], int) and m["n_breaks"] >= 0
        assert t["viterbi"] is not None
        for k in ("route_precision", "route_recall", "route_f1", "route_continuity"):
            assert m[k] is None or 0.0 <= m[k] <= 1.0
    else:
        assert body["classifier_only_route"] is None
        assert m["top1_point_accuracy_hmm"] is None and m["n_breaks"] is None


@needs_data
@needs_model
def test_aggregate_top1_matches_training_metrics(real_responses):
    """Point-weighted Top-1 over all student_test trips via the API == the trainer's test_top1."""
    name = REAL_MODEL.stem                                   # model / model_all_trips
    metrics_file = REAL_MODEL.parent / (name.replace("model", "metrics", 1) + ".json")
    if not metrics_file.is_file():
        pytest.skip(f"no {metrics_file.name} next to {REAL_MODEL.name}")
    expected = json.loads(metrics_file.read_text(encoding="utf-8"))["test_top1"]
    n = sum(b["metrics"]["n_points"] for b in real_responses.values())
    hits = sum(b["metrics"]["top1_point_accuracy_argmax"] * b["metrics"]["n_points"] for b in real_responses.values())
    assert round(hits / n, 4) == round(expected, 4), f"API {hits / n:.6f} vs {metrics_file.name} {expected:.6f}"


# ----------------------------------------------------------------------------- geometry
@needs_data
def test_arcs_to_polylines_on_real_route():
    """Geometry helper on a real reference route (load_route is allowed in tests only)."""
    from api.geometry import arcs_to_polylines

    tdir = trip_dir(STUDENT_TEST, TRIPS[0])
    trip = load_trip(tdir)
    route = load_route(tdir)
    lines = arcs_to_polylines(trip, route)
    assert lines and all(len(line) >= 2 for line in lines)
    lat_min, lat_max = trip.nodes.lat.min(), trip.nodes.lat.max()
    lon_min, lon_max = trip.nodes.lon.min(), trip.nodes.lon.max()
    for line in lines:
        for lat, lon in line:
            assert lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
    assert len(lines) <= max(1, len(route) // 20)    # ~99.96% connected -> very few polylines


def _toy_trip() -> Trip:
    import pandas as pd

    nodes = pd.DataFrame({"lon": [0.0, 1.0, 2.0, 5.0, 6.0], "lat": [10.0, 11.0, 12.0, 15.0, 16.0]})
    arcs = pd.DataFrame({"from_node": [0, 1, 3, 1], "to_node": [1, 2, 4, 0]})
    track = pd.DataFrame({"lon": [0.0, 2.0], "lat": [10.0, 12.0], "t": [0.0, 1.0]})
    return Trip(0, Path(REPO_ROOT), track, nodes, arcs)


def test_arcs_to_polylines_merge_and_split():
    from api.geometry import arcs_to_polylines

    trip = _toy_trip()
    assert arcs_to_polylines(trip, [0, 1, 2]) == [[[10.0, 0.0], [11.0, 1.0], [12.0, 2.0]],
                                                  [[15.0, 5.0], [16.0, 6.0]]]
    assert arcs_to_polylines(trip, []) == []
    assert arcs_to_polylines(trip, [0, -1, 1]) == [[[10.0, 0.0], [11.0, 1.0]], [[11.0, 1.0], [12.0, 2.0]]]


def test_simplify_polyline():
    from api.geometry import simplify_polyline

    pts = [[float(i), float(-i)] for i in range(20_000)]
    out = simplify_polyline(pts, 5000)
    assert len(out) <= 5000 and out[0] == pts[0] and out[-1] == pts[-1]
    assert simplify_polyline(pts[:10], 5000) == pts[:10]
