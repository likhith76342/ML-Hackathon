"""Unit tests for pathsnap.graph / pathsnap.hmm / sequence metrics on a tiny synthetic network.

    python -m pytest tests -q
"""
import numpy as np
import pandas as pd

from pathsnap.graph import build_graph
from pathsnap.hmm import classifier_only, reconstruct_route, viterbi
from pathsnap.route_metrics import point_accuracy, route_continuity, route_prf


class T:  # minimal Trip stand-in (same attributes as pathsnap.io.Trip)
    pass


def make_trip(track_xy, times):
    """Straight east-west street of 4 nodes 100 m apart, two-way (arcs 0-2 eastbound, 3-5 westbound),
    plus a side street node 4 north of node 1 (arcs 6: 1->4, 7: 4->1)."""
    tr = T()
    nodes = np.array([[0, 0], [100, 0], [200, 0], [300, 0], [100, 100]], float)
    arcs = [(0, 1), (1, 2), (2, 3), (1, 0), (2, 1), (3, 2), (1, 4), (4, 1)]
    lat0 = 0.0
    tr.nodes = pd.DataFrame({"x": nodes[:, 0], "y": nodes[:, 1],
                             "lon": nodes[:, 0] / 111195.0, "lat": nodes[:, 1] / 111195.0 + lat0})
    tr.arcs = pd.DataFrame(arcs, columns=["from_node", "to_node"])
    xy = np.asarray(track_xy, float)
    tr.track = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "lon": xy[:, 0] / 111195.0,
                             "lat": xy[:, 1] / 111195.0, "t": np.asarray(times, float)})
    return tr


def cands_for(trip, radius=30.0):
    g = build_graph(trip)
    rows = []
    for i, (x, y) in enumerate(trip.track[["x", "y"]].to_numpy()):
        got = False
        for a in range(g.n_arcs):
            A, B = g.nodes_xy[g.from_node[a]], g.nodes_xy[g.to_node[a]]
            ab = B - A
            t = np.clip(np.dot([x, y] - A, ab) / np.dot(ab, ab), 0, 1)
            d = np.hypot(*([x, y] - (A + t * ab)))
            if d <= radius:
                rows.append((i, a, d, t))
                got = True
        if not got:
            rows.append((i, -1, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["point_idx", "arc_id", "dist_m", "proj_t"])


def gauss(c, s=10.0):
    return np.where(c.arc_id >= 0, np.exp(-0.5 * (np.nan_to_num(c.dist_m) / s) ** 2), 0.0)


def test_route_distance_cases():
    tr = make_trip([[0, 0]], [0])
    g = build_graph(tr)
    assert np.isclose(g.route_distance(0, 0.2, 0, 0.7), 50)          # same arc forward
    assert np.isclose(g.route_distance(0, 0.5, 1, 0.5), 100)         # next arc
    assert np.isclose(g.route_distance(0, 0.5, 2, 0.5), 200)         # two arcs ahead
    assert np.isclose(g.route_distance(0, 0.5, 3, 0.5), 100)         # U-turn onto twin at node 1
    assert np.isclose(g.route_distance(3, 0.5, 2, 0.5), 300)         # 1->0, turn at 0, back east
    assert not np.isfinite(g.route_distance(0, 0.5, 2, 0.5, cutoff_m=50))  # path longer than cutoff
    assert np.isclose(g.route_distance(1, 0.7, 1, 0.2), 0.3 * 100 + 100 + 0.2 * 100)  # U-turn at node 2 and back


def test_direction_resolved_by_transitions():
    # vehicle drives east along y = 3; distance-only emissions can't tell eastbound from westbound twins
    xs = np.arange(10, 290, 10)
    tr = make_trip(np.c_[xs, np.full_like(xs, 3)], np.arange(len(xs)))
    c = cands_for(tr)
    out = viterbi(c, gauss(c), tr, {"beta": 5.0})
    eastbound = {0, 1, 2}
    assert set(out.arc_id) <= eastbound
    assert out.segment_id.nunique() == 1 and not out.filled.any()
    route = reconstruct_route(out, tr)
    assert route == [0, 1, 2]
    assert route_continuity(out.arc_id.to_numpy(), tr) == 1.0


def test_gap_break_and_missing_candidates():
    xy = [[10, 0], [20, 0], [30, 0], [150, 500], [60, 0], [70, 0]]   # point 3 has no candidates
    t = [0, 1, 2, 3, 100, 101]                                        # gap of 97 s before point 4
    tr = make_trip(xy, t)
    c = cands_for(tr)
    out = viterbi(c, gauss(c), tr, {"gap_break_s": 30})
    assert out.loc[3, "filled"] and out.loc[3, "arc_id"] == out.loc[2, "arc_id"]
    assert out.segment_id.tolist() == [0, 0, 0, 0, 1, 1]
    assert out.attrs["n_breaks"] == 1
    assert len(out) == len(tr.track)


def test_reconstruct_inserts_connecting_path():
    tr = make_trip([[0, 0]], [0])
    m = pd.DataFrame({"point_idx": [0, 1, 2, 3], "arc_id": [0, 0, 2, 2]})
    assert reconstruct_route(m, tr) == [0, 1, 2]


def test_metrics():
    assert point_accuracy([1, 2, -1, 4], [1, 3, -1, 4]) == 0.5   # -1 counts as wrong even if ref is -1
    assert route_prf([0, 1, 2], [1, 2, 3]) == {"route_precision": 2 / 3, "route_recall": 2 / 3,
                                               "route_f1": 2 / 3}
    tr = make_trip([[0, 0]], [0])
    assert route_continuity([0, 0, 2, 7], tr, 3) == 0.5             # 0->2 in 2 hops; 7 not within 3 of 2
