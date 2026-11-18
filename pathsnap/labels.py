"""Per-point reference labels from .route.  Owner: Zubed.  (Brief 6.3 step 3)

.route lists arcs, not which GPS point was on which arc. We align points to ROUTE POSITIONS
monotonically (routes revisit arcs, so align to positions, not arc IDs).
"""
from __future__ import annotations

import pandas as pd

from .io import Trip


def align_points_to_route(trip: Trip, route: list[int]) -> pd.DataFrame:
    """One row per GPS point: trip_id, point_idx, ref_arc, ref_route_pos, ref_dist_m."""
    raise NotImplementedError("Zubed: implement (brief 6.3)")


def add_labels(cands: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """label = 1 where cands.arc_id == ref_arc for that point, else 0."""
    raise NotImplementedError("Zubed: implement (brief 6.3)")
