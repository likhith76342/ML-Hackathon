"""Candidate generation + features for one trip.  Owner: Zubed.  (Brief 6.3 / 6.5)

Training and the API MUST both call build_candidates() so there is no training/serving skew.
"""
from __future__ import annotations

import pandas as pd

from .io import Trip


def build_candidates(trip: Trip, radius_m: float | None = None, k: int | None = None) -> pd.DataFrame:
    """One row per (GPS point, candidate arc).

    Columns: trip_id, point_idx, t, arc_id, *FEATURE_COLUMNS
    - A point with no candidates gets ONE placeholder row with arc_id = -1 and NaN features.
    - NO label column (labels are added only by labels.py for training/test files).
    """
    raise NotImplementedError("Zubed: implement (brief 6.3)")
