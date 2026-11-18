"""Pure helpers for the PathSnap UI (no Streamlit calls) so they can be unit-tested.  Owner: Likhith.

HTTP only: this module talks to the API with `requests`. It never imports the PathSnap package and
never reads data files. Every number shown in the UI comes from an API response.
"""
from __future__ import annotations

import html
import math
from typing import Any, Callable, Iterable

import folium
import requests
from branca.element import Element, MacroElement
from jinja2 import Template

QUICK_TIMEOUT = (5, 15)   # (connect, read) seconds for /health and /trips

# Route layer colours - must stay distinguishable from each other and from the base map.
COLOR_PRED = "#e0245e"     # predicted (HMM) route: strong crimson
COLOR_CLF = "#ff9f1c"      # classifier-only route: orange
COLOR_GPS = "#1d6fd8"      # GPS points: blue


class ApiError(Exception):
    """A failure we can explain to the user in one friendly sentence."""


# ----------------------------------------------------------------------------- HTTP
def call_api(method: str, base: str, path: str, timeout, **kwargs) -> dict:
    url = base.rstrip("/") + path
    try:
        r = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectTimeout:
        raise ApiError(f"Timed out connecting to the API at {base}. Check the IP address, the network and the firewall.")
    except requests.exceptions.ReadTimeout:
        raise ApiError(f"The API at {base} didn't answer within {timeout[1]:.0f} s. Very large trips can be slow: "
                       "try again, or raise PREDICT_TIMEOUT_S.")
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Can't reach the API at {base}. Is it running, on the same network, and allowed through "
                       "the firewall (port 8000)?")
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
            raise ApiError(f"The API hit an error while processing this trip ({r.status_code}): {detail}")
        raise ApiError(f"Unexpected response {r.status_code}: {detail}")
    try:
        return r.json()
    except ValueError:
        raise ApiError(f"The API at {base} returned something that isn't JSON. Is the API URL correct?")


# ----------------------------------------------------------------------------- formatting
def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def fmt_pct(v) -> str:
    return f"{v * 100:.2f}%" if is_num(v) else "N/A"


def fmt_int(v) -> str:
    return f"{int(v):,}" if is_num(v) else "N/A"


def fmt_ms(v) -> str:
    if not is_num(v):
        return "N/A"
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms"


def fmt_duration(s) -> str | None:
    """Travel time of the GPS track (not processing time)."""
    if not is_num(s):
        return None
    if s < 60:
        return f"{s:.0f} s"
    minutes = round(s / 60)
    return f"{minutes} min" if minutes < 120 else f"{minutes // 60} h {minutes % 60:02d} min"


def uplift_pp(hmm, argmax) -> str | None:
    """HMM minus classifier-only Top-1, in percentage points (None unless both exist)."""
    if not (is_num(hmm) and is_num(argmax)):
        return None
    return f"{(hmm - argmax) * 100:+.2f} pp vs classifier"


def trip_label(trip: dict, durations: dict[int, float]) -> str:
    """Selectbox label: id, point count and (once known from a /predict response) travel duration."""
    parts = [f"Trip {trip['trip_id']}", f"{trip['n_points']:,} points"]
    d = fmt_duration(durations.get(trip["trip_id"]))
    if d:
        parts.append(d)
    return " · ".join(parts)


def summarize_for_debug(obj: Any, max_items: int = 3) -> Any:
    """The raw response with long arrays shortened, for the collapsed debug expander."""
    if isinstance(obj, dict):
        return {k: summarize_for_debug(v, max_items) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > max_items:
            return [summarize_for_debug(v, max_items) for v in obj[:max_items]] + [f"... {len(obj) - max_items} more"]
        return [summarize_for_debug(v, max_items) for v in obj]
    return obj


# ----------------------------------------------------------------------------- run all trips
RUN_ALL_COLUMNS = ["Trip", "Points", "Top-1 (classifier)", "Top-1 (HMM)", "Route F1", "Total time", "Status"]
OVERALL_LABEL = "Overall (point-weighted, computed from this table)"


def run_all(post: Callable[[int], dict], trip_ids: Iterable[int],
            progress: Callable[[int, int, int], None] | None = None) -> list[dict]:
    """POST /predict for each trip, one by one. Errors are recorded per trip, never raised."""
    trip_ids = list(trip_ids)
    rows = []
    for i, tid in enumerate(trip_ids):
        if progress:
            progress(i, len(trip_ids), tid)
        try:
            rows.append({"trip_id": tid, "response": post(tid), "error": None})
        except ApiError as e:
            rows.append({"trip_id": tid, "response": None, "error": str(e)})
    if progress:
        progress(len(trip_ids), len(trip_ids), None)
    return rows


def point_weighted(rows: list[dict], key: str) -> float | None:
    """sum(metric * n_points) / sum(n_points) over successful rows; None if any value is missing."""
    ok = [r["response"] for r in rows if r["response"]]
    if not ok or len(ok) != len(rows):
        return None
    pairs = [(b["metrics"].get(key), b["metrics"].get("n_points")) for b in ok]
    if any(not is_num(v) or not is_num(n) for v, n in pairs):
        return None
    total = sum(n for _, n in pairs)
    return sum(v * n for v, n in pairs) / total if total else None


def build_run_all_table(rows: list[dict]) -> list[dict]:
    """Display rows (strings, N/A for nulls) plus a final point-weighted overall row."""
    table = []
    for r in rows:
        b = r["response"]
        if b is None:
            table.append({"Trip": str(r["trip_id"]), "Points": "N/A", "Top-1 (classifier)": "N/A",
                          "Top-1 (HMM)": "N/A", "Route F1": "N/A", "Total time": "N/A", "Status": r["error"]})
            continue
        m, t = b.get("metrics") or {}, b.get("timings_ms") or {}
        status = "mock" if b.get("mock") else ("ok" if b.get("hmm_used") else "ok (classifier only)")
        table.append({"Trip": str(r["trip_id"]), "Points": fmt_int(m.get("n_points")),
                      "Top-1 (classifier)": fmt_pct(m.get("top1_point_accuracy_argmax")),
                      "Top-1 (HMM)": fmt_pct(m.get("top1_point_accuracy_hmm")),
                      "Route F1": fmt_pct(m.get("route_f1")), "Total time": fmt_ms(t.get("total")), "Status": status})
    if rows:
        n = sum(r["response"]["metrics"].get("n_points") or 0 for r in rows if r["response"])
        table.append({"Trip": OVERALL_LABEL, "Points": fmt_int(n) if n else "N/A",
                      "Top-1 (classifier)": fmt_pct(point_weighted(rows, "top1_point_accuracy_argmax")),
                      "Top-1 (HMM)": fmt_pct(point_weighted(rows, "top1_point_accuracy_hmm")),
                      "Route F1": "", "Total time": "", "Status": ""})
    return table


# ----------------------------------------------------------------------------- timings
TIMING_STAGES = [("candidates", "Candidates"), ("model", "Model"), ("viterbi", "HMM (Viterbi)"), ("total", "Total")]


def timings_bars_html(timings: dict) -> str:
    """Progress-style horizontal bars of per-stage server time, scaled to the total."""
    vals = {k: timings.get(k) for k, _ in TIMING_STAGES}
    scale = max([v for v in vals.values() if is_num(v)] or [0]) or 1
    rows = []
    for key, label in TIMING_STAGES:
        v = vals[key]
        width = 100 * v / scale if is_num(v) else 0
        rows.append(
            f'<div class="ps-bar-row"><span class="ps-bar-label">{label}</span>'
            f'<span class="ps-bar-track"><span class="ps-bar-fill{" ps-total" if key == "total" else ""}" '
            f'style="width:{width:.1f}%"></span></span><span class="ps-bar-value">{fmt_ms(v)}</span></div>')
    return '<div class="ps-bars">' + "".join(rows) + "</div>"


# ----------------------------------------------------------------------------- map
LEGEND_HTML = """
<div style="position:fixed; bottom:22px; left:22px; z-index:9999; background:rgba(255,255,255,.94);
            padding:8px 12px; border-radius:8px; font:13px/1.65 system-ui,-apple-system,Segoe UI,sans-serif;
            color:#1b1f24; box-shadow:0 1px 6px rgba(0,0,0,.25);">
  <div style="font-weight:600; margin-bottom:2px;">Legend</div>
  <div><span style="display:inline-block;width:22px;height:4px;background:{pred};vertical-align:middle;border-radius:2px;"></span>&nbsp; Predicted route</div>
  {clf_row}
  <div><span style="display:inline-block;width:9px;height:9px;background:{gps};opacity:.6;border-radius:50%;vertical-align:middle;margin:0 6px 0 6px;"></span>&nbsp; GPS points</div>
  <div><span style="display:inline-block;width:9px;height:9px;background:#2e9e44;border-radius:50%;vertical-align:middle;margin:0 6px;"></span>&nbsp; Start &nbsp;&nbsp;<span style="display:inline-block;width:9px;height:9px;background:#d33;border-radius:50%;vertical-align:middle;margin:0 6px;"></span>&nbsp; End</div>
</div>
"""
CLF_ROW = ('<div><span style="display:inline-block;width:22px;height:0;border-top:3px dashed {clf};'
           'vertical-align:middle;"></span>&nbsp; Classifier-only route (toggle in layers)</div>')


class _KeepUnderneath(MacroElement):
    """Leaflet hook: when `layer` is re-added via the layer control, send it to the back."""
    _template = Template("""
        {% macro script(this, kwargs) %}
        {{ this._parent.get_name() }}.on('overlayadd', function (e) {
            if (e.layer === {{ this.layer.get_name() }}) { {{ this.layer.get_name() }}.bringToBack(); }
        });
        {% endmacro %}
    """)

    def __init__(self, layer):
        super().__init__()
        self.layer = layer


def display_indices(n_shown: int, n_total: int | None) -> list[int]:
    """Original GPS point index of each displayed point. The API thins evenly (first and last kept),
    so for a thinned track index i maps to round(i * (n_total - 1) / (n_shown - 1))."""
    if not n_total or n_shown >= n_total or n_shown < 2:
        return list(range(n_shown))
    step = (n_total - 1) / (n_shown - 1)
    return [int(round(i * step)) for i in range(n_shown)]


def build_map(result: dict) -> folium.Map:
    gps = result.get("gps_points") or []
    pred = result.get("predicted_route") or []
    clf = result.get("classifier_only_route") or []

    center = gps[len(gps) // 2] if gps else [0.0, 0.0]
    m = folium.Map(location=center, zoom_start=13, tiles=None, prefer_canvas=True, control_scale=True)
    carto = "&copy; OpenStreetMap contributors &copy; CARTO"
    folium.TileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", attr=carto,
                     name="Light", show=True).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Streets", show=False).add_to(m)
    folium.TileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", attr=carto,
                     name="Dark", show=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Satellite", show=False).add_to(m)

    # GPS points first (drawn underneath), as ONE GeoJSON layer so large trips stay responsive.
    fg_gps = folium.FeatureGroup(name="GPS points", show=True)
    if gps:
        idx = display_indices(len(gps), result.get("n_gps_points"))
        features = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                     "properties": {"label": f"GPS point #{i}"}} for (lat, lon), i in zip(gps, idx)]
        folium.GeoJson(
            {"type": "FeatureCollection", "features": features},
            marker=folium.CircleMarker(radius=2.5, weight=0, fill=True, fill_color=COLOR_GPS, fill_opacity=0.45),
            tooltip=folium.GeoJsonTooltip(fields=["label"], labels=False, sticky=False),
            name="gps",
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    if clf:
        fg_clf = folium.FeatureGroup(name="Classifier-only route", show=False)
        for line in clf:
            if len(line) >= 2:
                folium.PolyLine(line, color=COLOR_CLF, weight=3, opacity=0.9, dash_array="6 5").add_to(fg_clf)
        fg_clf.add_to(m)

    fg_pred = folium.FeatureGroup(name="Predicted route", show=True)
    for line in pred:
        if len(line) >= 2:
            folium.PolyLine(line, color=COLOR_PRED, weight=5, opacity=0.9).add_to(fg_pred)
    fg_pred.add_to(m)

    fg_ends = folium.FeatureGroup(name="Start / end", show=True)
    if gps:
        folium.Marker(gps[0], tooltip="Start (first GPS point)",
                      icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(fg_ends)
        folium.Marker(gps[-1], tooltip="End (last GPS point)",
                      icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")).add_to(fg_ends)
    fg_ends.add_to(m)

    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    m.get_root().html.add_child(Element(LEGEND_HTML.format(
        pred=COLOR_PRED, gps=COLOR_GPS, clf_row=CLF_ROW.format(clf=COLOR_CLF) if clf else "")))
    _KeepUnderneath(fg_gps).add_to(m)   # GPS points stay under the routes when toggled back on

    if gps:
        lats = [p[0] for p in gps]
        lons = [p[1] for p in gps]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(24, 24))
    return m


def status_pill_html(connected: bool, mode: str | None = None, model: str | None = None) -> str:
    if not connected:
        return '<div class="ps-pill ps-bad"><span class="ps-dot"></span>Unreachable</div>'
    detail = " · ".join(html.escape(str(x)) for x in (mode, model) if x)
    return f'<div class="ps-pill ps-ok"><span class="ps-dot"></span>Connected<span class="ps-pill-detail">{detail}</span></div>'
