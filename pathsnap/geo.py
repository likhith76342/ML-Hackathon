"""Geometry helpers in metres.  Owner: Zubed.  (Brief section 6.2)

Agreed choice: local equirectangular projection centred on each trip's mean lat/lon,
R = 6_371_000 m. Use these helpers everywhere - never raw degree distances.
"""
from __future__ import annotations

import numpy as np

EARTH_R = 6_371_000.0


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
