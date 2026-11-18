"""Evaluator UI - runs on Laptop 2.  Owner: Likhith.  (Brief 11.4)

Run:  API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py
Talks to the API over HTTP only. It never imports the PathSnap package or reads data files, and every
number on the page comes from an API response (nulls show as N/A).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling helper module, however we're launched

import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

import ui_components as uc  # noqa: E402

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")
PREDICT_TIMEOUT = (5, float(os.environ.get("PREDICT_TIMEOUT_S", "900")))   # big trips can take a while
PIPELINE_LABEL = "Candidates → model → HMM → evaluation"
MOCK_BANNER = ("MOCK MODE: no trained model loaded. The red line is just the GPS track, and accuracy and "
               "route metrics are N/A.")
HMM_NOTICE = "HMM not available yet: showing classifier-only route"
FOOTER = "Predictions are produced live by the API; ground truth stays on the server."

CSS = """
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 1.5rem; }
.ps-title { font-size: 1.9rem; font-weight: 700; letter-spacing: -.01em; margin: 0; line-height: 1.2; }
.ps-subtitle { color: #9aa4b2; margin: .15rem 0 0 0; font-size: .98rem; }
.ps-pill { display: inline-flex; align-items: center; gap: .45rem; float: right; margin-top: .4rem;
           padding: .32rem .8rem; border-radius: 999px; font-size: .86rem; font-weight: 600; }
.ps-pill-detail { font-weight: 400; opacity: .85; margin-left: .3rem; }
.ps-dot { width: .55rem; height: .55rem; border-radius: 50%; display: inline-block; }
.ps-ok  { background: rgba(46,158,68,.16); color: #7ee29a; border: 1px solid rgba(46,158,68,.45); }
.ps-ok .ps-dot { background: #34c759; }
.ps-bad { background: rgba(221,51,51,.16); color: #ff8a8a; border: 1px solid rgba(221,51,51,.45); }
.ps-bad .ps-dot { background: #ff453a; }
.ps-bars { margin: .2rem 0 .4rem 0; }
.ps-bar-row { display: grid; grid-template-columns: 8.5rem 1fr 4.6rem; align-items: center; gap: .6rem;
              margin: .28rem 0; font-size: .9rem; }
.ps-bar-label { color: #c9d1db; }
.ps-bar-track { background: rgba(255,255,255,.07); border-radius: 4px; height: .62rem; overflow: hidden; }
.ps-bar-fill { display: block; height: 100%; background: #2bb3a3; border-radius: 4px; }
.ps-bar-fill.ps-total { background: #6b7a8f; }
.ps-bar-value { text-align: right; font-variant-numeric: tabular-nums; color: #e6e9ee; }
.ps-footer { color: #7d8794; font-size: .82rem; text-align: center; margin-top: 1.6rem;
             padding-top: .8rem; border-top: 1px solid rgba(255,255,255,.08); }
.ps-swatch { display: inline-block; width: 1.3rem; height: .28rem; border-radius: 2px; vertical-align: middle;
             margin-right: .45rem; }
div[data-testid="stMetric"] { background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.07);
                              border-radius: 10px; padding: .55rem .75rem; }
</style>
"""


# ----------------------------------------------------------------------------- data
@st.cache_data(ttl=60, show_spinner=False)
def fetch_overview(base: str) -> dict:
    """GET /health and /trips (cached briefly so map interactions don't refetch)."""
    return {"health": uc.call_api("GET", base, "/health", uc.QUICK_TIMEOUT),
            "trips": uc.call_api("GET", base, "/trips", uc.QUICK_TIMEOUT)["trips"]}


def post_predict(base: str, trip_id: int) -> dict:
    return uc.call_api("POST", base, "/predict", PREDICT_TIMEOUT, json={"trip_id": int(trip_id)})


def remember_duration(resp: dict) -> None:
    if uc.is_num(resp.get("trip_duration_s")):
        st.session_state.setdefault("durations", {})[resp["trip_id"]] = resp["trip_duration_s"]


# ----------------------------------------------------------------------------- layout pieces
def render_header(connected: bool, health: dict | None) -> None:
    left, right = st.columns([3, 2], vertical_alignment="center")
    with left:
        st.markdown('<p class="ps-title">PathSnap</p><p class="ps-subtitle">GPS map-matching: classifier + HMM '
                    'route reconstruction</p>', unsafe_allow_html=True)
    with right:
        h = health or {}
        st.markdown(uc.status_pill_html(connected, h.get("mode"), h.get("model")), unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.subheader("Connection")
        api_url = st.text_input("API URL", value=DEFAULT_API_URL, help="e.g. http://192.168.1.20:8000").strip()
        if st.button("Reconnect", use_container_width=True):
            fetch_overview.clear()
        status_slot = st.empty()
        with st.expander("About this map"):
            st.markdown(
                f'<span class="ps-swatch" style="background:{uc.COLOR_PRED}"></span>**Red**: the predicted route '
                "(the HMM's connected road sequence).<br>"
                f'<span class="ps-swatch" style="background:{uc.COLOR_CLF}"></span>**Orange**: the classifier-only '
                "route (best arc per point, no HMM). It's hidden by default; turn it on in the layer menu.<br>"
                f'<span class="ps-swatch" style="background:{uc.COLOR_GPS};height:.5rem;width:.5rem;'
                'border-radius:50%"></span>**Blue**: the raw GPS points.<br><br>'
                "Metrics are computed on the server against the reference route. The reference itself is never "
                "sent to this page.", unsafe_allow_html=True)
    return api_url, status_slot


def render_metrics(result: dict) -> None:
    m = result.get("metrics") or {}
    hmm, argmax = m.get("top1_point_accuracy_hmm"), m.get("top1_point_accuracy_argmax")
    cards = st.columns(7)
    cards[0].metric("Top-1 (HMM)", uc.fmt_pct(hmm), delta=uc.uplift_pp(hmm, argmax))
    cards[1].metric("Top-1 (classifier only)", uc.fmt_pct(argmax))
    cards[2].metric("Route precision", uc.fmt_pct(m.get("route_precision")))
    cards[3].metric("Route recall", uc.fmt_pct(m.get("route_recall")))
    cards[4].metric("Route F1", uc.fmt_pct(m.get("route_f1")))
    cards[5].metric("Route continuity", uc.fmt_pct(m.get("route_continuity")))
    cards[6].metric("Route breaks", uc.fmt_int(m.get("n_breaks")))
    left, right = st.columns([2, 5])
    left.caption("Top-1: share of all GPS points placed on the same road segment as the reference.")
    right.caption("Precision / recall / F1: overlap between predicted and reference road segments. Continuity: "
                  "share of segment changes that are physically connected. Breaks: places the HMM had to restart "
                  "(e.g. long GPS gaps).")


def render_result(result: dict, run_no: int) -> None:
    parts = [f"Trip {result.get('trip_id')}"]
    d = uc.fmt_duration(result.get("trip_duration_s"))
    if d:
        parts.append(d)
    if uc.is_num(result.get("n_gps_points")):
        parts.append(f"{result['n_gps_points']:,} GPS points")
    st.subheader(" · ".join(parts))
    if not result.get("mock") and not result.get("hmm_used", True):
        st.warning(HMM_NOTICE)

    render_metrics(result)

    st_folium(uc.build_map(result), height=620, use_container_width=True, returned_objects=[],
              key=f"map_{run_no}")
    n_shown, n_total = len(result.get("gps_points") or []), result.get("n_gps_points")
    if uc.is_num(n_total) and n_shown < n_total:
        st.caption(f"Showing {n_shown:,} of {n_total:,} GPS points (thinned for display; metrics use all points).")

    t_col, d_col = st.columns([3, 2], gap="large")
    with t_col:
        st.markdown("**Processing time**")
        st.markdown(uc.timings_bars_html(result.get("timings_ms") or {}), unsafe_allow_html=True)
        st.caption("Server time to process this trip, end to end. This is not the trip's travel time.")
    with d_col:
        st.markdown("**Run details**")
        st.caption(f"Mode: {'mock' if result.get('mock') else 'real'} · HMM used: "
                   f"{'yes' if result.get('hmm_used') else 'no'} · Classifier-only layer: "
                   f"{'available' if result.get('classifier_only_route') else 'not available'}")
        with st.expander("Raw API response (debug)", expanded=False):
            st.json(uc.summarize_for_debug(result), expanded=False)


def single_trip_tab(api_url: str, trips: list[dict]) -> None:
    by_id = {t["trip_id"]: t for t in trips}
    ids = list(by_id)
    durations = st.session_state.get("durations", {})
    # A label gains the trip duration after a run, which Streamlit treats as a new widget; remembering the
    # pick and passing it as the default index keeps the selection instead of jumping back to the first trip.
    picked = st.session_state.get("picked_trip")
    pick, go = st.columns([4, 1], vertical_alignment="bottom")
    with pick:
        trip_id = st.selectbox("Trip", ids, index=ids.index(picked) if picked in by_id else 0,
                               format_func=lambda t: uc.trip_label(by_id[t], durations))
    st.session_state["picked_trip"] = trip_id
    with go:
        run = st.button("Run PathSnap", type="primary", use_container_width=True)

    if run:
        with st.status(PIPELINE_LABEL, expanded=False) as status:
            st.write(f"Trip {trip_id}: {by_id[trip_id]['n_points']:,} GPS points. Large trips can take a while.")
            try:
                resp = post_predict(api_url, trip_id)
                st.session_state["result"] = resp
                st.session_state["run_no"] = st.session_state.get("run_no", 0) + 1
                st.session_state.pop("error", None)
                remember_duration(resp)
                total = (resp.get("timings_ms") or {}).get("total")
                status.update(label=f"Done: trip {trip_id} in {uc.fmt_ms(total)}", state="complete")
            except uc.ApiError as e:
                st.session_state["error"] = str(e)
                status.update(label=f"Trip {trip_id} failed", state="error")

    if st.session_state.get("error"):
        st.error(st.session_state["error"])
    if st.session_state.get("result"):
        render_result(st.session_state["result"], st.session_state.get("run_no", 0))
    elif not st.session_state.get("error"):
        st.info("Pick a trip and click **Run PathSnap**.")


def run_all_tab(api_url: str, trips: list[dict]) -> None:
    st.markdown("Runs `/predict` for every trip the API lists, one at a time. All numbers come from the live API.")
    go, clear, _ = st.columns([1, 1, 3])
    if go.button("Run all trips", type="primary", use_container_width=True):
        bar = st.progress(0.0, text="Starting...")

        def progress(i: int, n: int, tid):
            bar.progress(i / n if n else 1.0, text=f"Trip {tid} ({i + 1}/{n})..." if tid is not None else "Done")

        rows = uc.run_all(lambda tid: post_predict(api_url, tid), [t["trip_id"] for t in trips], progress)
        for r in rows:
            if r["response"]:
                remember_duration(r["response"])
        st.session_state["run_all"] = {"api_url": api_url, "rows": rows}
    if clear.button("Clear table", use_container_width=True):
        st.session_state.pop("run_all", None)

    saved = st.session_state.get("run_all")
    if not saved:
        st.info("Click **Run all trips** to fill the table.")
        return
    st.dataframe(uc.build_run_all_table(saved["rows"]), hide_index=True, use_container_width=True)
    st.caption("The last row is computed in this page from the table: Top-1 weighted by each trip's GPS point "
               "count. It shows N/A if any trip failed or returned no value.")
    if saved["api_url"] != api_url:
        st.caption(f"These results came from {saved['api_url']}.")


# ----------------------------------------------------------------------------- page
def main() -> None:
    st.set_page_config(page_title="PathSnap", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    api_url, slot = render_sidebar()

    try:
        overview, problem = fetch_overview(api_url), None
    except uc.ApiError as e:
        overview, problem = None, str(e)

    health = overview["health"] if overview else None
    render_header(overview is not None, health)
    if overview is None:
        slot.error("API unreachable")
        st.error(problem)
        st.info("Start the API on Laptop 1 with `uvicorn api.main:app --host 0.0.0.0 --port 8000`, open "
                f"`{api_url}/health` in a browser to check it, then click **Reconnect**.")
        st.markdown(f'<div class="ps-footer">{FOOTER}</div>', unsafe_allow_html=True)
        return

    trips = overview["trips"]
    slot.success(f"Connected · {health.get('n_trips', len(trips))} trips · {health.get('mode', '?')} mode")

    result = st.session_state.get("result") or {}
    if health.get("mode") == "mock" or result.get("mock"):
        st.error(MOCK_BANNER)
    else:
        st.caption(f"Model: {health.get('model', '?')}")

    if not trips:
        st.warning("The API reports no trips.")
    else:
        single, many = st.tabs(["Single trip", "Run all trips"])
        with single:
            single_trip_tab(api_url, trips)
        with many:
            run_all_tab(api_url, trips)

    st.markdown(f'<div class="ps-footer">{FOOTER}</div>', unsafe_allow_html=True)


try:
    main()
except Exception as e:  # last line of defence: never show a traceback to the evaluator
    st.error(f"Something went wrong in the UI: {type(e).__name__}: {e}. Try refreshing the page.")
