"""Geometry helpers for the API response.  Owner: Likhith.

Our data files store longitude first; Folium/Leaflet expect [lat, lon]. Every function
here returns [lat, lon] so the UI can draw the output directly.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from pathsnap.io import Trip

LatLon = list[float]          # [lat, lon]
Polyline = list[LatLon]


def gps_to_latlon(trip: Trip) -> Polyline:
    """GPS track as [[lat, lon], ...] in track order."""
    return trip.track[["lat", "lon"]].to_numpy(dtype=float).tolist()


def arcs_to_polylines(trip: Trip, arc_ids: Sequence[int]) -> list[Polyline]:
    """Turn an arc sequence into drawable polylines of [lat, lon].

    Consecutive arcs that connect (arcs[a].to_node == arcs[b].from_node) are merged into one
    polyline; a disconnected jump starts a new polyline. Invalid ids (e.g. -1 placeholders)
    are skipped and also break the line.
    """
    arc_ids = [int(a) for a in arc_ids]
    if not arc_ids:
        return []

    from_nodes = trip.arcs["from_node"].to_numpy()
    to_nodes = trip.arcs["to_node"].to_numpy()
    lat = trip.nodes["lat"].to_numpy(dtype=float)
    lon = trip.nodes["lon"].to_numpy(dtype=float)
    n_arcs = len(from_nodes)

    def pt(node: int) -> LatLon:
        return [float(lat[node]), float(lon[node])]

    polylines: list[Polyline] = []
    current: Polyline = []
    prev_to: int | None = None
    for a in arc_ids:
        if not 0 <= a < n_arcs:
            if current:
                polylines.append(current)
            current, prev_to = [], None
            continue
        f, t = int(from_nodes[a]), int(to_nodes[a])
        if current and prev_to == f:
            current.append(pt(t))                 # connected: extend the same line
        else:
            if current:
                polylines.append(current)         # disconnected jump: start a new line
            current = [pt(f), pt(t)]
        prev_to = t
    if current:
        polylines.append(current)
    return polylines


def simplify_polyline(points: Sequence[LatLon], max_points: int) -> Polyline:
    """Thin a polyline for display: at most max_points evenly spaced points,
    always keeping the first and the last. Display only - never compute metrics on this."""
    points = list(points)
    n = len(points)
    if max_points < 2:
        raise ValueError("max_points must be >= 2")
    if n <= max_points:
        return points
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return [points[i] for i in idx]
