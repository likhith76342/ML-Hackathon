"""HMM + Viterbi route reconstruction.  Owner: Keshav.  (Brief 8.3 - 8.6)"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import Trip


def viterbi(cands: pd.DataFrame, probs: np.ndarray, trip: Trip, config: dict) -> pd.DataFrame:
    """cands: candidate table for ONE trip (no label); probs: P(label=1) per row, same order.
    Returns one row per GPS point: point_idx, arc_id (-1 if none), segment_id, filled (bool)."""
    raise NotImplementedError("Keshav: implement (brief 8.3)")


def reconstruct_route(matched: pd.DataFrame, trip: Trip) -> list[int]:
    """Connected arc sequence (duplicates collapsed, gaps filled with shortest paths)."""
    raise NotImplementedError("Keshav: implement (brief 8.4)")
