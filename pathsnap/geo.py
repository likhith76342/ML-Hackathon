"""Geometry helpers in metres.  Owner: Zubed.  (Brief section 6.2)

Agreed choice: local equirectangular projection, R = 6_371_000 m.
THE PROJECTION ORIGIN IS ALWAYS THE MEAN lon/lat OF THE TRACK. Candidates, features and the
HMM graph must all use add_xy() so every module works in the same coordinate frame.
"""
from __future__ import annotations

import numpy as np

from .io import Trip

EARTH_R = 6_371_000.0


def add_xy(trip: Trip) -> Trip:
    """Add x, y (metres) columns to trip.track and trip.nodes, in place, origin = track mean
    lon/lat. Idempotent (does nothing if x/y already exist). Returns the same Trip.
    Every module calls this first - never project with a different origin."""
    raise NotImplementedError("Zubed: implement (brief 6.2)")


def project(lon, lat, lon0: float, lat0: float):
    """Return (x, y) in metres relative to (lon0, lat0)."""
    raise NotImplementedError("Zubed: implement (brief 6.2)")


def point_segment_distance(px, py, ax, ay, bx, by):
    """Return (distance_m, t in [0,1]) from point(s) P to segment AB. Vectorised."""
    raise NotImplementedError("Zubed: implement (brief 6.2)")


def bearing_deg(ax, ay, bx, by):
    """Bearing of A->B, 0-360 clockwise from north (y axis)."""
    raise NotImplementedError("Zubed: implement (brief 6.2)")


def angle_diff_deg(a, b):
    """Absolute angle difference folded to [0, 180]."""
    raise NotImplementedError("Zubed: implement (brief 6.2)")
