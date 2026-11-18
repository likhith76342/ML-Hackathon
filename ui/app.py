"""Evaluator UI - runs on Laptop 2.  Owner: Likhith.  (Brief 11.4)

Run:  API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py
Talks to the API over HTTP only - never imports the model or pathsnap, never reads data files.
"""
from __future__ import annotations

import os

import folium
import requests
import streamlit as st
from branca.element import Element
from streamlit_folium import st_folium

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")
QUICK_TIMEOUT = (5, 15)                                                  # (connect, read) seconds
PREDICT_TIMEOUT = (5, float(os.environ.get("PREDICT_TIMEOUT_S", "600")))  # large trips can be slow

METRIC_LABELS = [
    ("top1_point_accuracy_hmm", "Top-1 point accuracy (HMM)", True),
    ("top1_point_accuracy_argmax", "Top-1 point accuracy (classifier only)", True),
    ("route_precision", "Route precision", True),
    ("route_recall", "Route recall", True),
    ("route_f1", "Route F1", True),
    ("route_continuity", "Route continuity", True),
    ("n_points", "GPS points", False),
    ("n_breaks", "Sequence breaks", False),
]
TIMING_LABELS = [("candidates", "Candidates"), ("model", "Model"), ("viterbi", "Viterbi"), ("total", "Total")]


class ApiError(Exception):
    """A failure we can explain to the user in one friendly sentence."""


# ----------------------------------------------------------------------------- HTTP
def call_api(method: str, base: str, path: str, timeout, **kwargs) -> dict:
    url = base.rstrip("/") + path
    try:
        r = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectTimeout:
        raise ApiError(f"Timed out connecting to the API at {base}. Check the IP, the network and the firewall.")
    except requests.exceptions.ReadTimeout:
        raise ApiError(f"The API at {base} took longer than {timeout[1]:.0f}s to answer. "
                       "Large trips can be slow - try again, or raise PREDICT_TIMEOUT_S.")
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Cannot reach the API at {base}. Is it running, on the same network, "
                       "and allowed through the firewall (port 8000)?")
    except requests.exceptions.RequestException as e:
        raise ApiError(f"Request to {url} failed: {e}")

    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        if r.status_code == 404:
            raise ApiError(f"Not found (404): {detail}")
        if r.status_code == 422:
            raise ApiError(f"The API rejected the request (422): {detail}")
        if r.status_code >= 500:
            raise ApiError(f"The API hit an error while processing ({r.status_code}): {detail}")
        raise ApiError(f"Unexpected response {r.status_code}: {detail}")
    try:
        return r.json()
    except ValueError:
        raise ApiError(f"The API at {base} returned something that isn't JSON. Is API_URL correct?")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_overview(base: str) -> dict:
    """GET /health and /trips (cached briefly so map interactions don't refetch)."""
    return {"health": call_api("GET", base, "/health", QUICK_TIMEOUT),
            "trips": call_api("GET", base, "/trips", QUICK_TIMEOUT)["trips"]}


# ----------------------------------------------------------------------------- map
LEGEND_HTML = """
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
            padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px;
            line-height: 1.6; box-shadow: 0 1px 4px rgba(0,0,0,.3); color: #222;">
  <b>Legend</b><br>
  <span style="color:#1f77b4;">&#9679;</span> GPS points<br>
  <span style="display:inline-block;width:18px;height:3px;background:#d62728;vertical-align:middle;"></span> Predicted route<br>
  {orange}
</div>
"""
ORANGE_ROW = ('<span style="display:inline-block;width:18px;height:3px;background:#ff7f0e;'
              'vertical-align:middle;"></span> Classifier-only route<br>')


def build_map(result: dict) -> folium.Map:
    gps = result.get("gps_points") or []
    pred = result.get("predicted_route") or []
    clf = result.get("classifier_only_route") or []

    center = gps[len(gps) // 2] if gps else [0.0, 0.0]
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap", control_scale=True)

    fg_gps = folium.FeatureGroup(name="GPS points (blue)", show=True)
    for lat, lon in gps:
        folium.CircleMarker([lat, lon], radius=2, color="#1f77b4", weight=1,
                            fill=True, fill_opacity=0.8).add_to(fg_gps)
    fg_gps.add_to(m)

    fg_pred = folium.FeatureGroup(name="Predicted route (red)", show=True)
    for line in pred:
        if len(line) >= 2:
            folium.PolyLine(line, color="#d62728", weight=4, opacity=0.85).add_to(fg_pred)
    fg_pred.add_to(m)

    if clf:
        fg_clf = folium.FeatureGroup(name="Classifier-only route (orange)", show=False)
        for line in clf:
            if len(line) >= 2:
                folium.PolyLine(line, color="#ff7f0e", weight=3, opacity=0.8, dash_array="6 4").add_to(fg_clf)
        fg_clf.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(Element(LEGEND_HTML.format(orange=ORANGE_ROW if clf else "")))

    if gps:
        lats = [p[0] for p in gps]
        lons = [p[1] for p in gps]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))
    return m


# ----------------------------------------------------------------------------- panels
def fmt_metric(v, pct: bool) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%" if pct else f"{v:,}"


def fmt_ms(v) -> str:
    if v is None:
        return "N/A"
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms"


def fmt_duration(s) -> str | None:
    """Travel time of the GPS track (not processing time)."""
    if s is None:
        return None
    if s < 60:
        return f"{s:.0f} s"
    minutes = round(s / 60)
    return f"{minutes} min" if minutes < 120 else f"{minutes // 60} h {minutes % 60:02d} min"


def render_result(result: dict) -> None:
    parts = [f"Trip {result.get('trip_id')}"]
    duration = fmt_duration(result.get("trip_duration_s"))
    if duration:
        parts.append(duration)
    if result.get("n_gps_points") is not None:
        parts.append(f"{result['n_gps_points']:,} GPS points")
    st.subheader(" · ".join(parts))

    map_col, side_col = st.columns([3, 1])
    with map_col:
        st_folium(build_map(result), height=620, use_container_width=True,
                  returned_objects=[], key=f"map_{result.get('trip_id')}")
        n_shown, n_total = len(result.get("gps_points") or []), result.get("n_gps_points")
        if n_total and n_shown < n_total:
            st.caption(f"Showing {n_shown:,} of {n_total:,} GPS points (thinned for display; "
                       "metrics use all points).")
    with side_col:
        st.markdown("**Metrics**")
        metrics = result.get("metrics") or {}
        for key, label, pct in METRIC_LABELS:
            st.metric(label, fmt_metric(metrics.get(key), pct))
        st.markdown("**Processing time**")
        st.caption("Server compute time for this request, not the trip's travel time.")
        timings = result.get("timings_ms") or {}
        for key, label in TIMING_LABELS:
            st.metric(label, fmt_ms(timings.get(key)))


# ----------------------------------------------------------------------------- page
def main() -> None:
    st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")
    st.title("PathSnap: GPS map-matching")

    with st.sidebar:
        st.header("Connection")
        api_url = st.text_input("API URL", value=DEFAULT_API_URL, help="e.g. http://192.168.1.20:8000").strip()
        if st.button("Reconnect / refresh trips"):
            fetch_overview.clear()

    try:
        overview = fetch_overview(api_url)
    except ApiError as e:
        st.error(str(e), icon="🔌")
        st.info("Start the API on Laptop 1 with `uvicorn api.main:app --host 0.0.0.0 --port 8000`, "
                f"then check `{api_url}/health` in a browser and click **Reconnect**.")
        return

    health, trips = overview["health"], overview["trips"]
    with st.sidebar:
        st.success(f"Connected: {health.get('n_trips', len(trips))} trips, mode = {health.get('mode', '?')}")
        st.caption(f"Model: {health.get('model', '?')}")
    result = st.session_state.get("result") or {}
    if health.get("mode") == "mock" or result.get("mock"):
        st.error("MOCK MODE: no trained model loaded. The red line is just the GPS track; accuracy and route metrics are N/A.",
                 icon="🚨")
    if not trips:
        st.warning("The API reports no trips.")
        return

    labels = {t["trip_id"]: f"Trip {t['trip_id']}  ({t['n_points']:,} points)" for t in trips}
    col1, col2 = st.columns([3, 1])
    with col1:
        trip_id = st.selectbox("Trip", list(labels), format_func=labels.get)
    with col2:
        st.write("")
        st.write("")
        run = st.button("Run PathSnap", type="primary", use_container_width=True)

    if run:
        with st.spinner(f"Running PathSnap on trip {trip_id}... (large trips can take a while)"):
            try:
                st.session_state["result"] = call_api("POST", api_url, "/predict", PREDICT_TIMEOUT,
                                                      json={"trip_id": int(trip_id)})
                st.session_state.pop("error", None)
            except ApiError as e:
                st.session_state["error"] = str(e)

    if st.session_state.get("error"):
        st.error(st.session_state["error"], icon="⚠️")
    if st.session_state.get("result"):
        render_result(st.session_state["result"])
    elif not st.session_state.get("error"):
        st.info("Pick a trip and click **Run PathSnap**.")


try:
    main()
except Exception as e:  # last line of defence: never show a traceback to the evaluator
    st.error(f"Something went wrong in the UI: {type(e).__name__}: {e}. Try refreshing the page.")
