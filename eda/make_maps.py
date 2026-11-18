"""Create report-ready Folium maps from raw PathSnap trips.

Run from the repository root:
    python -m eda.make_maps
"""
from __future__ import annotations

from pathlib import Path

from pathsnap.io import list_trips, load_route, load_trip, trip_dir
from eda.maps import make_trip_map

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "figures" / "maps"


def _network_lines(trip):
    return [[(trip.nodes.iloc[a].lon, trip.nodes.iloc[a].lat),
             (trip.nodes.iloc[b].lon, trip.nodes.iloc[b].lat)]
            for a, b in trip.arcs[["from_node", "to_node"]].itertuples(index=False, name=None)
            if 0 <= a < len(trip.nodes) and 0 <= b < len(trip.nodes)]


def _route_lines(trip, route):
    lines = []
    for arc_id in route:
        if not 0 <= arc_id < len(trip.arcs):
            continue
        a, b = trip.arcs.iloc[arc_id][["from_node", "to_node"]]
        if 0 <= a < len(trip.nodes) and 0 <= b < len(trip.nodes):
            lines.append([(trip.nodes.iloc[int(a)].lon, trip.nodes.iloc[int(a)].lat),
                          (trip.nodes.iloc[int(b)].lon, trip.nodes.iloc[int(b)].lat)])
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Use compact examples so the HTML remains practical to open and submit.
    # The EDA tables still cover every trip.
    for split, trip_id in (("train", 5), ("student_test", 7)):
        tdir = trip_dir(DATA / split, trip_id)
        trip = load_trip(tdir)
        gps = list(zip(trip.track["lat"], trip.track["lon"]))
        network = _network_lines(trip)
        reference = _route_lines(trip, load_route(tdir)) if (tdir / f"{tdir.name}.route").exists() else None
        fmap = make_trip_map(gps, network_lines=network, reference_lines=reference)
        output = OUT / f"{split}_{trip_id:08d}.html"
        fmap.save(output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
