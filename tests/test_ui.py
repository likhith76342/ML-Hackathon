"""UI tests (Streamlit AppTest + pure helpers). The API is stubbed at `requests.request`, so these need
no running server, no data and no model.  Owner: Likhith.

Run from the repo root:  pytest tests/
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"
APP = str(UI_DIR / "app.py")
sys.path.insert(0, str(UI_DIR))

import ui_components as uc  # noqa: E402


# ----------------------------------------------------------------------------- fake API
class FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code, self._payload = status_code, payload
        self.text = str(payload)

    def json(self):
        return self._payload


def predict_body(trip_id: int, mock=False, hmm=0.95, argmax=0.90, n_points=262, **metric_overrides) -> dict:
    metrics = {"top1_point_accuracy_hmm": hmm, "top1_point_accuracy_argmax": argmax, "route_precision": 0.97,
               "route_recall": 0.99, "route_f1": 0.98, "route_continuity": 1.0, "n_points": n_points, "n_breaks": 0}
    if mock:
        metrics = {k: None for k in metrics} | {"n_points": n_points}
    metrics.update(metric_overrides)
    gps = [[56.79 + i * 1e-4, 35.85 + i * 1e-4] for i in range(5)]
    return {"trip_id": trip_id, "mode": "mock" if mock else "real", "mock": mock, "hmm_used": not mock,
            "n_gps_points": n_points, "trip_duration_s": 265.0, "gps_points": gps, "predicted_route": [gps],
            "classifier_only_route": None if mock else [gps[:3]], "metrics": metrics,
            "timings_ms": {"candidates": 84.0, "model": 14.0, "viterbi": 192.0, "total": 318.0}}


class FakeApi:
    """Routes requests.request(method, url, ...) like the real API would."""

    def __init__(self, mode="real", model="model.joblib", trips=((52, 262), (77, 3911)), predict=None,
                 down=False):
        self.mode, self.model, self.trips, self.down = mode, model, list(trips), down
        self.predict = predict or (lambda tid: FakeResponse(200, predict_body(tid, mock=(mode == "mock"))))
        self.calls = []

    def __call__(self, method, url, timeout=None, **kwargs):
        self.calls.append((method, url, kwargs.get("json")))
        if self.down:
            raise requests.exceptions.ConnectionError("refused")
        if url.endswith("/health"):
            return FakeResponse(200, {"status": "ok", "mode": self.mode, "model": self.model,
                                      "n_trips": len(self.trips), "data_dir": "x"})
        if url.endswith("/trips"):
            return FakeResponse(200, {"trips": [{"trip_id": t, "n_points": n} for t, n in self.trips]})
        if url.endswith("/predict"):
            return self.predict(kwargs["json"]["trip_id"])
        return FakeResponse(404, {"detail": "nope"})


@pytest.fixture
def app_with(monkeypatch):
    """Returns a function: FakeApi -> a fresh AppTest that has been run once."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    def make(api: FakeApi):
        st.cache_data.clear()
        monkeypatch.setattr(requests, "request", api)
        at = AppTest.from_file(APP, default_timeout=60)
        return at.run()

    return make


def texts(elements) -> list[str]:
    return [e.value for e in elements]


def click(at, label: str):
    return next(b for b in at.button if b.label == label).click().run()


# ----------------------------------------------------------------------------- (a) / (b) banners
def test_mock_mode_shows_red_banner(app_with):
    at = app_with(FakeApi(mode="mock", model="mock (no model)"))
    assert not at.exception
    assert any(t.startswith("MOCK MODE: no trained model loaded") for t in texts(at.error))
    assert not any(t.startswith("Model:") for t in texts(at.caption))


def test_real_mode_shows_model_caption_and_no_banner(app_with):
    at = app_with(FakeApi(mode="real", model="model.joblib"))
    assert not at.exception
    assert "Model: model.joblib" in texts(at.caption)
    assert not any("MOCK MODE" in t for t in texts(at.error))
    assert any("Connected" in t and "model.joblib" in t for t in texts(at.markdown))   # status pill


# ----------------------------------------------------------------------------- (c) N/A + KPIs
def test_null_metrics_render_as_na(app_with):
    body = predict_body(52, hmm=None, route_precision=None, route_recall=None, route_f1=None,
                        route_continuity=None, n_breaks=None)
    at = click(app_with(FakeApi(predict=lambda tid: FakeResponse(200, body))), "Run PathSnap")
    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    for label in ("Top-1 (HMM)", "Route precision", "Route recall", "Route F1", "Route continuity", "Route breaks"):
        assert values[label] == "N/A", label
    assert values["Top-1 (classifier only)"] == "90.00%"
    hmm_card = next(m for m in at.metric if m.label == "Top-1 (HMM)")
    assert not hmm_card.delta                                       # no uplift without both numbers


def test_kpis_show_hmm_uplift_and_trip_header(app_with):
    at = click(app_with(FakeApi()), "Run PathSnap")
    assert not at.exception
    hmm_card = next(m for m in at.metric if m.label == "Top-1 (HMM)")
    assert hmm_card.value == "95.00%" and hmm_card.delta == "+5.00 pp vs classifier"
    assert "Trip 52 · 4 min · 262 GPS points" in texts(at.subheader)
    # the duration learned from /predict shows in the picker label from the next rerun on
    assert any("4 min" in o for o in at.run().selectbox[0].options)


def test_selected_trip_survives_label_change(app_with):
    """After a run the label gains a duration; the picker must stay on the trip that was run."""
    at = app_with(FakeApi())
    at.selectbox[0].set_value(77)
    at = click(at, "Run PathSnap").run()
    assert at.selectbox[0].value == 77
    assert "Trip 77 · 4 min · 262 GPS points" in texts(at.subheader)


def test_hmm_unavailable_shows_yellow_notice(app_with):
    body = predict_body(52) | {"hmm_used": False, "classifier_only_route": None}
    at = click(app_with(FakeApi(predict=lambda tid: FakeResponse(200, body))), "Run PathSnap")
    assert "HMM not available yet: showing classifier-only route" in texts(at.warning)


# ----------------------------------------------------------------------------- errors
def test_api_unreachable_is_friendly(app_with):
    at = app_with(FakeApi(down=True))
    assert not at.exception
    assert any("Can't reach the API" in t for t in texts(at.error))
    assert any("Unreachable" in t for t in texts(at.markdown))


@pytest.mark.parametrize("status,expect", [(404, "Not found (404)"), (500, "error while processing")])
def test_predict_http_errors_are_friendly(app_with, status, expect):
    at = click(app_with(FakeApi(predict=lambda tid: FakeResponse(status, {"detail": "boom"}))), "Run PathSnap")
    assert not at.exception
    assert any(expect in t for t in texts(at.error))


def test_predict_timeout_is_friendly(app_with):
    def slow(tid):
        raise requests.exceptions.ReadTimeout()

    at = click(app_with(FakeApi(predict=slow)), "Run PathSnap")
    assert not at.exception
    assert any("didn't answer within" in t for t in texts(at.error))


# ----------------------------------------------------------------------------- (e) run all trips
def test_run_all_table_from_stub_client():
    bodies = {1: predict_body(1, hmm=0.8, argmax=0.9, n_points=100), 2: predict_body(2, hmm=0.6, argmax=0.5, n_points=300)}
    seen = []
    rows = uc.run_all(lambda tid: bodies[tid], [1, 2], lambda i, n, tid: seen.append((i, n, tid)))
    assert seen == [(0, 2, 1), (1, 2, 2), (2, 2, None)]
    table = uc.build_run_all_table(rows)
    assert [r["Trip"] for r in table] == ["1", "2", uc.OVERALL_LABEL]
    assert table[0]["Top-1 (classifier)"] == "90.00%" and table[1]["Top-1 (HMM)"] == "60.00%"
    overall = table[-1]
    assert overall["Points"] == "400"
    assert overall["Top-1 (classifier)"] == "60.00%"     # (0.9*100 + 0.5*300) / 400
    assert overall["Top-1 (HMM)"] == "65.00%"            # (0.8*100 + 0.6*300) / 400


def test_run_all_overall_is_na_when_a_trip_fails_or_is_null():
    def post(tid):
        if tid == 2:
            raise uc.ApiError("Not found (404): x")
        return predict_body(tid, hmm=None)

    table = uc.build_run_all_table(uc.run_all(post, [1, 2]))
    assert table[1]["Status"].startswith("Not found") and table[1]["Top-1 (HMM)"] == "N/A"
    assert table[-1]["Top-1 (classifier)"] == "N/A" and table[-1]["Top-1 (HMM)"] == "N/A"


def test_run_all_tab_end_to_end(app_with):
    api = FakeApi(trips=((52, 262), (77, 3911)))
    at = click(app_with(api), "Run all trips")
    assert not at.exception
    df = at.dataframe[0].value
    assert list(df["Trip"]) == ["52", "77", uc.OVERALL_LABEL]
    assert sum(1 for c in api.calls if c[1].endswith("/predict")) == 2
    at.run()                                                    # cached: no re-run on interaction
    assert sum(1 for c in api.calls if c[1].endswith("/predict")) == 2


# ----------------------------------------------------------------------------- (d) isolation
@pytest.mark.parametrize("path", sorted(UI_DIR.glob("*.py")), ids=lambda p: p.name)
def test_ui_source_never_touches_pathsnap_or_ground_truth(path):
    src = path.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+pathsnap\b", src, re.MULTILINE), "UI must not import pathsnap"
    assert ".route" not in src, "UI must not reference .route files"
    for forbidden in ("read_csv", "read_parquet", "data/raw", "data/processed", "load_route"):
        assert forbidden not in src, forbidden


# ----------------------------------------------------------------------------- helpers
def test_map_layers_and_markers():
    body = predict_body(52)
    html = uc.build_map(body).get_root().render()
    assert 0 < html.find('"fillColor": "' + uc.COLOR_GPS) < html.find('"color": "' + uc.COLOR_PRED)   # GPS under route
    assert "Start (first GPS point)" in html and "End (last GPS point)" in html
    assert "fitBounds" in html and "bringToBack" in html
    # 5 displayed points of a 262-point track: tooltips carry the ORIGINAL point index
    assert "GPS point #0" in html and "GPS point #261" in html
    assert len({uc.COLOR_PRED, uc.COLOR_CLF, uc.COLOR_GPS}) == 3


def _tile_urls(html: str) -> list[str]:
    return re.findall(r'L\.tileLayer\(\s*"([^"]+)"', html)


def test_base_layers_need_no_api_key_and_osm_is_default():
    """Carto tiles now show an 'API KEY REQUIRED' watermark: no Carto URL may appear anywhere in the UI."""
    html = uc.build_map(predict_body(52)).get_root().render()
    urls = _tile_urls(html)
    assert urls, "no base layers rendered"
    for url in urls + [u for _, u, _, _ in uc.BASE_LAYERS]:
        assert "cartocdn" not in url and "carto.com" not in url, url
    assert "cartocdn" not in html and "carto.com" not in html
    for path in UI_DIR.glob("*.py"):
        src = path.read_text(encoding="utf-8").lower()
        assert "cartocdn" not in src and "carto.com" not in src, path.name
    assert "tile.openstreetmap.org" in urls[0]                      # first layer = default base map
    defaults = [name for name, _, _, show in uc.BASE_LAYERS if show]
    assert defaults == ["OpenStreetMap"] and uc.BASE_LAYERS[0][0] == "OpenStreetMap"
    assert len(urls) <= 3                                           # default + at most two alternatives


def test_routes_have_contrasting_casing():
    html = uc.build_map(predict_body(52)).get_root().render()
    casing = html.find('"color": "' + uc.CASING_PRED)
    red = html.find('"color": "' + uc.COLOR_PRED)
    assert 0 < casing < red, "white casing must be drawn under the red route"
    assert 0 < html.find('"color": "' + uc.CASING_CLF) < html.find('"color": "' + uc.COLOR_CLF)


def test_display_indices_match_even_thinning():
    assert uc.display_indices(5, 5) == [0, 1, 2, 3, 4]
    idx = uc.display_indices(5000, 7001)
    assert idx[0] == 0 and idx[-1] == 7000 and len(set(idx)) == 5000


def test_formatting_and_debug_summary():
    assert uc.fmt_pct(None) == "N/A" and uc.fmt_pct(0.856) == "85.60%"
    assert uc.fmt_ms(None) == "N/A" and uc.fmt_ms(318.0) == "318 ms" and uc.fmt_ms(4044.0) == "4.04 s"
    assert uc.uplift_pp(None, 0.9) is None and uc.uplift_pp(0.95, 0.9) == "+5.00 pp vs classifier"
    short = uc.summarize_for_debug({"gps_points": [[0, 0]] * 10, "metrics": {"a": 1}})
    assert len(short["gps_points"]) == 4 and short["gps_points"][-1] == "... 7 more"
    bars = uc.timings_bars_html({"candidates": 10.0, "model": None, "viterbi": 30.0, "total": 60.0})
    assert "N/A" in bars and "width:100.0%" in bars
