"""Directed road graph + route distances.  Owner: Keshav.  (Brief 8.2)"""
from __future__ import annotations

from .io import Trip


class RoadGraph:
    def __init__(self, trip: Trip):
        raise NotImplementedError("Keshav: implement (brief 8.2)")

    def route_distance(self, arc_a: int, t_a: float, arc_b: int, t_b: float, cutoff_m: float) -> float:
        """Driving distance (m) from position t_a on arc_a to t_b on arc_b, respecting direction.
        Return float('inf') if unreachable within cutoff_m."""
        raise NotImplementedError
