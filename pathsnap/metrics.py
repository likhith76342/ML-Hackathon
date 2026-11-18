"""Candidate- and observation-level metrics (brief 9.1, 9.2).  Owner: Zubed.
Everyone uses these - no private variants. Sequence metrics live in route_metrics.py (Keshav).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---- 9.1 candidate-level (Zubed) -------------------------------------------------
def candidate_metrics(y_true, p_pred, threshold: float = 0.5) -> dict:
    """accuracy, precision, recall, macro_f1, log_loss, confusion_matrix (placeholder rows excluded)."""
    raise NotImplementedError("Zubed: implement (brief 9.1)")


# ---- 9.2 observation-level (Zubed) ----------------------------------------------
def top1_accuracy(cands: pd.DataFrame, probs: np.ndarray, refs: pd.DataFrame) -> float:
    """Per GPS point argmax-candidate == ref_arc. Denominator = ALL GPS points."""
    raise NotImplementedError("Zubed: implement (brief 9.2)")


def candidate_miss_rate(cands: pd.DataFrame, refs: pd.DataFrame) -> float:
    """Fraction of GPS points whose ref_arc is not in its candidate set."""
    raise NotImplementedError("Zubed: implement (brief 9.2)")
