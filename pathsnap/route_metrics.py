"""Sequence-level metrics (brief 9.4).  Owner: Keshav.  Everyone uses these - no private variants."""
from __future__ import annotations

def route_continuity(pred_arcs: list[int], trip, max_hops: int = 3) -> float:
    """Over consecutive points with different predicted arcs: share reachable within max_hops."""
    raise NotImplementedError("Keshav: implement (brief 9.4)")


def route_prf(pred_route: list[int], true_route: list[int]) -> dict:
    """precision / recall / f1 over sets of unique arc IDs."""
    raise NotImplementedError("Keshav: implement (brief 9.4)")
