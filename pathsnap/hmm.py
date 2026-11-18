"""HMM + Viterbi route reconstruction.  Owner: Keshav.  (Brief 8.3 - 8.6)

Model
-----
Hidden states   at step i, the candidate arcs of GPS point i (rows of the candidate table,
                so the (point_idx, arc_id) correspondence is kept exactly).
Emission        log E_i(b) = w_e * log(p_model(b | point i) + eps)
Transition      Newson & Krumm (2009):
                log T(a -> b) = -| route_distance(a, t_a, b, t_b) - great_circle(point_i, point_{i+1}) | / beta
                Unreachable within the search cutoff -> `unreachable_penalty` (finite, so decoding never dies).
                With clip_to_penalty=True a reachable but absurd transition is never scored below an
                unreachable one (log T is floored at the penalty).
Recurrence      delta_{i+1}(b) = max_a [delta_i(a) + log T(a, b)] + log E_{i+1}(b), with back-pointers,
                everything in log space.
Breaks          t_{i+1} - t_i > gap_break_s  -> chain broken, segments decoded independently.
                All transitions into a step unreachable -> chain restarted at that step.
                Same-arc backward GPS jitter of at most backward_tolerance_m metres is treated as route
                distance 0 (otherwise standing still with noisy fixes looks like a drive around the block).
                Points without candidates (arc_id == -1) are skipped and later filled with the previous
                matched arc (next one if there is no previous), flagged filled=True.

The work is split into `prepare_trip` (route distances; independent of beta / w_e / penalty) and
`decode` (cheap), so that decoder parameters can be tuned without recomputing shortest paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .graph import RoadGraph, build_graph
from .io import Trip

DEFAULT_CONFIG = {
    "beta": 10.0,                 # metres; scale of the route-vs-straight-line mismatch
    "w_e": 1.0,                   # emission weight
    "unreachable_penalty": -50.0,
    "clip_to_penalty": True,
    "gap_break_s": 30.0,
    "emission_eps": 1e-9,
    "min_cutoff_m": 500.0,        # shortest-path search cutoff = max(cutoff_factor * gc, min_cutoff_m)
    "cutoff_factor": 2.0,
    "backward_tolerance_m": 0.0,  # same-arc backward GPS jitter up to this many metres -> route distance 0 (decode-time)
}

R_EARTH = 6371000.0


def _cfg(config: dict | None) -> dict:
    c = dict(DEFAULT_CONFIG)
    if config:
        c.update({k: v for k, v in config.items() if v is not None})
    return c


def great_circle_m(lon1, lat1, lon2, lat2):
    """Haversine distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _proj_t(cands: pd.DataFrame, trip, graph: RoadGraph) -> np.ndarray:
    """Position of the GPS point's projection along each candidate arc, in [0, 1]."""
    if "proj_t" in cands.columns:
        return cands["proj_t"].to_numpy(dtype=float)
    arc = cands["arc_id"].to_numpy()
    ok = arc >= 0
    t = np.full(len(arc), np.nan)
    p = trip.track[["x", "y"]].to_numpy()[cands["point_idx"].to_numpy()[ok]]
    a = graph.nodes_xy[graph.from_node[arc[ok]]]
    b = graph.nodes_xy[graph.to_node[arc[ok]]]
    ab = b - a
    den = np.maximum((ab ** 2).sum(1), 1e-12)
    t[ok] = np.clip(((p - a) * ab).sum(1) / den, 0, 1)
    return t


@dataclass
class PreparedTrip:
    n_points: int
    point_idx: list = field(default_factory=list)   # per decodable step
    arcs: list = field(default_factory=list)        # np.ndarray of arc ids per step
    rows: list = field(default_factory=list)        # row positions in cands per step
    logp: list = field(default_factory=list)        # log(p + eps) per step
    dt: list = field(default_factory=list)          # dt between step j-1 and j (index j; dt[0] = inf)
    diff: list = field(default_factory=list)        # |route - gc| matrix (K_{j-1} x K_j), inf = unreachable;
                                                    # None if dt exceeded max_gap_s at prepare time
    gc: list = field(default_factory=list)          # great-circle distance from step j-1 to j (metres)
    back: list = field(default_factory=list)        # (rows, cols, metres) of same-arc backward moves
    max_gap_s: float = 30.0


def prepare_trip(cands: pd.DataFrame, probs: np.ndarray, trip, config: dict | None = None,
                 max_gap_s: float | None = None) -> PreparedTrip:
    """Compute everything that does not depend on beta / w_e / penalty."""
    c = _cfg(config)
    max_gap_s = c["gap_break_s"] if max_gap_s is None else max_gap_s
    graph = build_graph(trip)
    probs = np.asarray(probs, dtype=float)
    if len(probs) != len(cands):
        raise ValueError("probs must have one entry per row of cands")

    df = pd.DataFrame({
        "point_idx": cands["point_idx"].to_numpy(),
        "arc_id": cands["arc_id"].to_numpy(),
        "p": probs,
        "t_on_arc": _proj_t(cands, trip, graph),
        "row": np.arange(len(cands)),
    })
    df = df[df["arc_id"] >= 0]
    df = df.sort_values(["point_idx", "row"], kind="stable")

    track = trip.track
    lon = track["lon"].to_numpy()
    lat = track["lat"].to_numpy()
    tt = track["t"].to_numpy()
    lens = graph.arc_len

    prep = PreparedTrip(n_points=len(track), max_gap_s=max_gap_s)
    prev = None
    for pidx, g in df.groupby("point_idx", sort=True):
        arcs = g["arc_id"].to_numpy(dtype=np.int64)
        tpos = g["t_on_arc"].to_numpy()
        prep.point_idx.append(int(pidx))
        prep.arcs.append(arcs)
        prep.rows.append(g["row"].to_numpy())
        prep.logp.append(np.log(g["p"].to_numpy() + c["emission_eps"]))
        if prev is None:
            prep.dt.append(np.inf)
            prep.diff.append(None)
            prep.gc.append(np.nan)
            prep.back.append(None)
        else:
            ppidx, parcs, ptpos = prev
            dt = float(tt[pidx] - tt[ppidx])
            prep.dt.append(dt)
            gc = float(great_circle_m(lon[ppidx], lat[ppidx], lon[pidx], lat[pidx]))
            prep.gc.append(gc)
            if dt > max_gap_s:
                prep.diff.append(None)
                prep.back.append(None)
            else:
                cutoff = max(c["cutoff_factor"] * gc, c["min_cutoff_m"])
                D = np.empty((len(parcs), len(arcs)))
                bi, bj, bm = [], [], []
                for i, (a, ta) in enumerate(zip(parcs.tolist(), ptpos.tolist())):
                    for j, (b, tb) in enumerate(zip(arcs.tolist(), tpos.tolist())):
                        if a == b and tb < ta:
                            bi.append(i); bj.append(j); bm.append((ta - tb) * lens[a])
                        d = graph.route_distance(a, ta, b, tb, cutoff)
                        D[i, j] = abs(d - gc) if np.isfinite(d) else np.inf
                prep.diff.append(D)
                prep.back.append((np.array(bi, dtype=np.int64), np.array(bj, dtype=np.int64), np.array(bm)))
        prev = (pidx, arcs, tpos)
    return prep


def decode(prep: PreparedTrip, config: dict | None = None) -> pd.DataFrame:
    """Viterbi over a prepared trip. Returns the contract DataFrame (point_idx, arc_id, segment_id, filled)."""
    c = _cfg(config)
    if c["gap_break_s"] > prep.max_gap_s + 1e-9:
        raise ValueError("gap_break_s larger than the max_gap_s used in prepare_trip")
    beta, w_e, pen = float(c["beta"]), float(c["w_e"]), float(c["unreachable_penalty"])
    clip = bool(c["clip_to_penalty"])
    tol = float(c["backward_tolerance_m"])

    n_steps = len(prep.point_idx)
    chosen = np.full(n_steps, -1, dtype=np.int64)    # index into prep.arcs[j]
    seg_of_step = np.zeros(n_steps, dtype=np.int64)
    n_breaks = 0

    def backtrack(end, start, delta, bps):
        k = int(np.argmax(delta))
        for j in range(end, start - 1, -1):
            chosen[j] = k
            if j > start:
                k = int(bps[j - start - 1][k])

    seg = 0
    start = 0
    delta = None
    bps: list = []
    for j in range(n_steps):
        logE = w_e * prep.logp[j]
        if j == 0:
            delta = logE.copy()
            continue
        D = prep.diff[j]
        bk = prep.back[j]
        if D is not None and tol > 0 and bk is not None and len(bk[2]):
            sel = bk[2] <= tol
            if sel.any():   # small same-arc backward GPS jitter: treat the route distance as 0
                D = D.copy()
                D[bk[0][sel], bk[1][sel]] = prep.gc[j]
        brk = D is None or prep.dt[j] > c["gap_break_s"] or not np.isfinite(D).any()
        if brk:
            backtrack(j - 1, start, delta, bps)
            seg_of_step[start:j] = seg
            seg += 1
            n_breaks += 1
            start, bps = j, []
            delta = logE.copy()
            continue
        logT = np.where(np.isfinite(D), -D / beta, pen)
        if clip:
            logT = np.maximum(logT, pen)
        scores = delta[:, None] + logT
        bp = np.argmax(scores, axis=0)
        delta = scores[bp, np.arange(scores.shape[1])] + logE
        bps.append(bp)
    if n_steps:
        backtrack(n_steps - 1, start, delta, bps)
        seg_of_step[start:n_steps] = seg

    # Per GPS point output, filling points that had no candidates.
    arc_out = np.full(prep.n_points, -1, dtype=np.int64)
    seg_out = np.full(prep.n_points, -1, dtype=np.int64)
    filled = np.zeros(prep.n_points, dtype=bool)
    for j, p in enumerate(prep.point_idx):
        arc_out[p] = prep.arcs[j][chosen[j]]
        seg_out[p] = seg_of_step[j]
    has = arc_out >= 0
    if has.any() and not has.all():
        idx = np.where(has, np.arange(prep.n_points), -1)
        prev_idx = np.maximum.accumulate(idx)
        nxt = np.where(has, np.arange(prep.n_points), prep.n_points)
        next_idx = np.minimum.accumulate(nxt[::-1])[::-1]
        miss = ~has
        src = np.where(prev_idx >= 0, prev_idx, next_idx)
        arc_out[miss] = arc_out[src[miss]]
        seg_out[miss] = seg_out[src[miss]]
        filled[miss] = True
    out = pd.DataFrame({"point_idx": np.arange(prep.n_points), "arc_id": arc_out,
                        "segment_id": seg_out, "filled": filled})
    out.attrs["n_breaks"] = n_breaks
    return out


def viterbi(cands: pd.DataFrame, probs: np.ndarray, trip: Trip, config: dict) -> pd.DataFrame:
    """Contract entry point (brief 8.6).

    Returns one row per GPS point: point_idx, arc_id (int, -1 if none),
    segment_id (int, increments at each break), filled (bool).
    `out.attrs["n_breaks"]` holds the number of chain breaks.
    """
    prep = prepare_trip(cands, probs, trip, config)
    return decode(prep, config)


def classifier_only(cands: pd.DataFrame, probs: np.ndarray, n_points: int | None = None) -> pd.DataFrame:
    """Per-point argmax of the classifier (baseline). Same columns as viterbi(); no filling."""
    df = pd.DataFrame({"point_idx": cands["point_idx"].to_numpy(),
                       "arc_id": cands["arc_id"].to_numpy(), "p": np.asarray(probs, float)})
    real = df[df["arc_id"] >= 0]
    best = real.loc[real.groupby("point_idx")["p"].idxmax(), ["point_idx", "arc_id"]]
    n = int(df["point_idx"].max()) + 1 if n_points is None else n_points
    arc = np.full(n, -1, dtype=np.int64)
    arc[best["point_idx"].to_numpy()] = best["arc_id"].to_numpy()
    return pd.DataFrame({"point_idx": np.arange(n), "arc_id": arc,
                         "segment_id": np.zeros(n, dtype=np.int64), "filled": np.zeros(n, bool)})


def reconstruct_route(matched: pd.DataFrame, trip: Trip, min_cutoff_m: float = 1000.0,
                      cutoff_factor: float = 3.0) -> list[int]:
    """Connected arc sequence for display and route-level metrics (brief 8.4).

    Collapses consecutive duplicates; when two consecutive matched arcs are not adjacent the
    shortest connecting path is inserted (searched within max(cutoff_factor * gap, min_cutoff_m),
    where gap is the straight-line distance between a.to_node and b.from_node). If no path is
    found the two arcs are simply concatenated, leaving a visible discontinuity.
    """
    g = build_graph(trip)
    seq = matched.sort_values("point_idx")["arc_id"].to_numpy()
    seq = seq[seq >= 0]
    if len(seq) == 0:
        return []
    keep = np.r_[True, seq[1:] != seq[:-1]]
    seq = seq[keep].tolist()
    route = [seq[0]]
    for b in seq[1:]:
        a = route[-1]
        if a == b:
            continue
        if not g.is_adjacent(a, b):
            u, v = g.to_node[a], g.from_node[b]
            gap = float(np.hypot(*(g.nodes_xy[v] - g.nodes_xy[u])))
            mid = g.connecting_arcs(a, b, max(cutoff_factor * gap, min_cutoff_m))
            if mid:
                route.extend(mid)
        route.append(b)
    return [int(x) for x in route]
