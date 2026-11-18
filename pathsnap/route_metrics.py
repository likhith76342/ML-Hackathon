"""Sequence-level metrics (brief 9.4).  Owner: Keshav.  Everyone uses these - no private variants.

Candidate-level and observation-level metrics (9.1, 9.2) live in pathsnap/metrics.py (Zubed).
"""
from __future__ import annotations

import numpy as np

from .graph import build_graph


def point_accuracy(pred_arcs, ref_arcs) -> float:
    """Fraction of GPS points whose predicted arc equals ref_arc.

    Used both for classifier-only Top-1 (pred = per-point argmax) and for point accuracy after HMM
    (pred = Viterbi arc). Denominator = all GPS points; points predicted -1 count as wrong.
    """
    pred = np.asarray(pred_arcs)
    ref = np.asarray(ref_arcs)
    if pred.shape != ref.shape:
        raise ValueError("pred_arcs and ref_arcs must have one entry per GPS point")
    if len(ref) == 0:
        return float("nan")
    return float(np.mean((pred == ref) & (pred >= 0)))


def route_continuity(pred_arcs: list[int], trip, max_hops: int = 3) -> float:
    """Over consecutive GPS points whose predicted arcs differ (both != -1), the fraction where the
    second arc is reachable from the first within `max_hops` successor arcs in the directed graph."""
    g = build_graph(trip)
    a = np.asarray(pred_arcs)
    prev, nxt = a[:-1], a[1:]
    m = (prev != nxt) & (prev >= 0) & (nxt >= 0)
    if not m.any():
        return 1.0
    ok = 0
    pairs = list(zip(prev[m].tolist(), nxt[m].tolist()))
    for x, y in pairs:
        ok += y in g.arcs_within_hops(x, max_hops)
    return ok / len(pairs)


def route_prf(pred_route: list[int], true_route: list[int]) -> dict:
    """Precision / recall / F1 over the sets of unique arc ids of the two routes."""
    p, r = set(int(x) for x in pred_route), set(int(x) for x in true_route)
    tp = len(p & r)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(r) if r else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}
