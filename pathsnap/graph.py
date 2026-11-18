"""Directed road graph + route distances.  Owner: Keshav.  (Brief 8.2)

The graph is directed: road nodes are vertices and every arc is a directed
edge weighted by its length in metres (straight segment between its two
nodes, measured in the trip's projected x/y coordinates from pathsnap.geo).

Shortest paths use a plain Dijkstra with a distance cutoff. scipy's dijkstra
allocates an O(n_nodes) array on every call, which is too slow when we call it
tens of thousands of times on 300k-arc networks; a heap search that stops at
the cutoff only touches the few hundred nodes near the source. Results are
cached per source node, so consecutive GPS points (which share most of their
candidate arcs) reuse the same searches.

Only `trip.nodes` (x, y) and `trip.arcs` (from_node, to_node) are used, so this
module never sees the reference route. Coordinates come from pathsnap.geo.add_xy (brief 6.2: one
frame for everyone, origin = mean lon/lat of the track), called here if x/y are not there yet.
"""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np

from .geo import add_xy
from .io import Trip  # noqa: F401  (type of the `trip` arguments)

DEFAULT_MIN_CUTOFF_M = 500.0


def ensure_xy(trip) -> None:
    """Make sure trip.track and trip.nodes carry x, y in the shared frame (brief 6.2).
    geo.add_xy is idempotent, so calling it on an already-projected trip is free."""
    add_xy(trip)


class RoadGraph:
    """Directed road graph of one trip. Construct with RoadGraph(trip) (skeleton signature);
    build_graph(trip) does the same but caches the graph on the trip object."""

    def __init__(self, trip):
        ensure_xy(trip)
        self.from_node = trip.arcs["from_node"].to_numpy(dtype=np.int64)
        self.to_node = trip.arcs["to_node"].to_numpy(dtype=np.int64)
        self.nodes_xy = trip.nodes[["x", "y"]].to_numpy(dtype=float)
        self.n_nodes = len(self.nodes_xy)
        self.n_arcs = len(self.from_node)
        d = self.nodes_xy[self.to_node] - self.nodes_xy[self.from_node]
        self.arc_len = np.hypot(d[:, 0], d[:, 1])

        # Python adjacency lists are much faster than numpy indexing inside the heap loop.
        out = [[] for _ in range(self.n_nodes)]
        fl, tl, ll = self.from_node.tolist(), self.to_node.tolist(), self.arc_len.tolist()
        for a in range(self.n_arcs):
            out[fl[a]].append((a, tl[a], ll[a]))
        self._out = out
        self._fl, self._tl, self._ll = fl, tl, ll
        self._sp_cache: dict[int, tuple[float, dict, dict]] = {}
        self._hop_cache: dict[int, set] = {}

    # ------------------------------------------------------------------ Dijkstra
    def _search(self, src: int, cutoff: float):
        """Return (dist, pred_arc) dicts for nodes reachable from src within cutoff metres."""
        hit = self._sp_cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1], hit[2]
        dist = {src: 0.0}
        pred = {src: -1}
        heap = [(0.0, src)]
        out = self._out
        done = set()
        while heap:
            du, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for a, v, w in out[u]:
                nd = du + w
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = a
                    heapq.heappush(heap, (nd, v))
        self._sp_cache[src] = (cutoff, dist, pred)
        return dist, pred

    def node_distance(self, u: int, v: int, cutoff: float) -> float:
        """Shortest driving distance node u -> node v, or inf if beyond cutoff."""
        if u == v:
            return 0.0
        dist, _ = self._search(u, cutoff)
        d = dist.get(v)
        return np.inf if d is None or d > cutoff else d

    def node_path_arcs(self, u: int, v: int, cutoff: float) -> list[int] | None:
        """Arc ids of a shortest path u -> v (empty if u == v), None if unreachable."""
        if u == v:
            return []
        dist, pred = self._search(u, cutoff)
        if v not in dist:
            return None
        arcs = []
        x = v
        while x != u:
            a = pred[x]
            arcs.append(a)
            x = self._fl[a]
        return arcs[::-1]

    # ----------------------------------------------------------- arc-level API
    def route_distance(self, arc_a: int, t_a: float, arc_b: int, t_b: float,
                       cutoff_m: float = DEFAULT_MIN_CUTOFF_M) -> float:
        """Driving distance from position t_a along arc_a to t_b along arc_b (brief 8.2).

        Same arc with t_b >= t_a: (t_b - t_a) * len.
        Otherwise: remainder of a + shortest path a.to_node -> b.from_node + t_b * len(b).
        Returns inf when the connecting path is longer than `cutoff_m`.
        """
        la = self._ll[arc_a]
        if arc_a == arc_b and t_b >= t_a:
            return (t_b - t_a) * la
        sp = self.node_distance(self._tl[arc_a], self._fl[arc_b], cutoff_m)
        if not np.isfinite(sp):
            return np.inf
        return (1.0 - t_a) * la + sp + t_b * self._ll[arc_b]

    def connecting_arcs(self, arc_a: int, arc_b: int, cutoff: float) -> list[int] | None:
        """Arcs strictly between arc_a and arc_b on a shortest path (None if unreachable)."""
        return self.node_path_arcs(self._tl[arc_a], self._fl[arc_b], cutoff)

    def arcs_within_hops(self, arc_a: int, max_hops: int = 3) -> set[int]:
        """Arcs reachable from arc_a by following 1..max_hops successor arcs."""
        key = arc_a * 16 + max_hops
        hit = self._hop_cache.get(key)
        if hit is not None:
            return hit
        seen: set[int] = set()
        frontier = deque([(arc_a, 0)])
        while frontier:
            a, h = frontier.popleft()
            if h == max_hops:
                continue
            for b, _, _ in self._out[self._tl[a]]:
                if b not in seen:
                    seen.add(b)
                    frontier.append((b, h + 1))
        self._hop_cache[key] = seen
        return seen

    def is_adjacent(self, arc_a: int, arc_b: int) -> bool:
        return self._tl[arc_a] == self._fl[arc_b]


def build_graph(trip) -> RoadGraph:
    """Build (and cache on the trip object) the directed road graph of one trip."""
    g = getattr(trip, "_road_graph", None)
    if g is None:
        g = RoadGraph(trip)
        try:
            trip._road_graph = g
        except AttributeError:  # frozen dataclass etc.
            pass
    return g


def route_distance(graph: RoadGraph, arc_a: int, t_a: float, arc_b: int, t_b: float,
                   cutoff_m: float = DEFAULT_MIN_CUTOFF_M) -> float:
    """Module-level wrapper of RoadGraph.route_distance."""
    return graph.route_distance(arc_a, t_a, arc_b, t_b, cutoff_m)
