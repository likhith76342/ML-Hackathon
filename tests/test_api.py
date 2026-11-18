"""API tests (mock or real mode) on data/raw/student_test.  Owner: Likhith.

Run from the repo root:  pytest tests/
"""
from __future__ import annotations

import os

import pytest

from pathsnap.config import CFG, REPO_ROOT, repo_path
from pathsnap.io import Trip, list_trips, load_route, load_trip, trip_dir

STUDENT_TEST = repo_path(CFG["data_root"]) / "student_test"
FORBIDDEN_KEYS = {"route", "reference", "label", "ground_truth"}
METRIC_KEYS = {"top1_point_accuracy_hmm", "top1_point_accuracy_argmax", "route_precision", "route_recall",
               "route_f1", "route_continuity", "n_points", "n_breaks"}

needs_data = pytest.mark.skipif(not STUDENT_TEST.is_dir() or not list_trips(STUDENT_TEST),
                                reason=f"student_test data not found at {STUDENT_TEST}")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    os.environ["EVAL_DATA_DIR"] = str(STUDENT_TEST.relative_to(REPO_ROOT))
    from api.main import app

    with TestClient(app) as c:      # `with` runs the startup (lifespan) hook
        yield c


def all_keys(obj) -> set[str]:
    """Every dict key at any depth."""
    if isinstance(obj, dict):
        return set(obj) | set().union(*(all_keys(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(all_keys(v) for v in obj)) if obj else set()
    return set()


# ----------------------------------------------------------------------------- endpoints
@needs_data
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mode"] in {"mock", "real"}
    assert body["n_trips"] == len(list_trips(STUDENT_TEST))
    assert "data_dir" in body


@needs_data
def test_trips_lists_folders(client):
    r = client.get("/trips")
    assert r.status_code == 200
    trips = r.json()["trips"]
    ids = [t["trip_id"] for t in trips]
    assert ids == sorted(ids) == list_trips(STUDENT_TEST)
    for t in trips:
        assert set(t) == {"trip_id", "n_points"}
        assert t["n_points"] == len(load_trip(trip_dir(STUDENT_TEST, t["trip_id"])).track)


@needs_data
@pytest.mark.parametrize("trip_id", list_trips(STUDENT_TEST) if STUDENT_TEST.is_dir() else [])
def test_predict_every_trip(client, trip_id):
    r = client.post("/predict", json={"trip_id": trip_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trip_id"] == trip_id
    for k in ("gps_points", "predicted_route", "classifier_only_route", "metrics", "timings_ms", "mock"):
        assert k in body
    assert set(body["metrics"]) == METRIC_KEYS
    assert set(body["timings_ms"]) == {"candidates", "model", "viterbi", "total"}
    assert not (all_keys(body) & FORBIDDEN_KEYS), "ground-truth-like key leaked into the response"

    gps = body["gps_points"]
    assert gps and all(len(p) == 2 for p in gps)
    assert len(gps) <= 5000
    assert body["predicted_route"] and all(len(line) <= 5000 for line in body["predicted_route"])

    # [lat, lon] order: the first point must match the .track file (which stores lon, lat).
    first = load_trip(trip_dir(STUDENT_TEST, trip_id)).track.iloc[0]
    assert gps[0] == pytest.approx([first.lat, first.lon])

    if body["mock"]:
        assert all(v is None for v in body["metrics"].values()), "mock mode must never report metrics"
        assert body["classifier_only_route"] is None


@needs_data
def test_predict_unknown_trip_404(client):
    unknown = max(list_trips(STUDENT_TEST)) + 1000
    r = client.post("/predict", json={"trip_id": unknown})
    assert r.status_code == 404


@needs_data
@pytest.mark.parametrize("body", [{}, {"trip_id": "abc"}, {"trip": 52}, {"trip_id": None}])
def test_predict_bad_body_422(client, body):
    assert client.post("/predict", json=body).status_code == 422


@needs_data
def test_predict_non_json_422(client):
    r = client.post("/predict", content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# ----------------------------------------------------------------------------- geometry
@needs_data
def test_arcs_to_polylines_on_real_route():
    """Geometry helper on a real reference route (load_route is allowed in tests only)."""
    from api.geometry import arcs_to_polylines

    tid = list_trips(STUDENT_TEST)[0]
    tdir = trip_dir(STUDENT_TEST, tid)
    trip = load_trip(tdir)
    route = load_route(tdir)
    lines = arcs_to_polylines(trip, route)

    assert lines and all(len(line) >= 2 for line in lines)
    lat_min, lat_max = trip.nodes.lat.min(), trip.nodes.lat.max()
    lon_min, lon_max = trip.nodes.lon.min(), trip.nodes.lon.max()
    for line in lines:
        for lat, lon in line:
            assert lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
    # The reference route is ~99.96% connected, so it should merge into very few polylines.
    assert len(lines) <= max(1, len(route) // 20)


def _toy_trip() -> Trip:
    import pandas as pd

    nodes = pd.DataFrame({"lon": [0.0, 1.0, 2.0, 5.0, 6.0], "lat": [10.0, 11.0, 12.0, 15.0, 16.0]})
    arcs = pd.DataFrame({"from_node": [0, 1, 3, 1], "to_node": [1, 2, 4, 0]})
    track = pd.DataFrame({"lon": [0.0, 2.0], "lat": [10.0, 12.0], "t": [0.0, 1.0]})
    return Trip(0, REPO_ROOT, track, nodes, arcs)


def test_arcs_to_polylines_merge_and_split():
    from api.geometry import arcs_to_polylines

    trip = _toy_trip()
    # 0->1, 1->2 connected; then jump to 3->4
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
