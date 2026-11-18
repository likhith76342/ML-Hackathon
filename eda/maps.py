"""Reusable Folium map builder.  Owner: Mannu.  (Brief 7.2)  Also used by the UI.

Layers: road network (grey), GPS points (blue), reference route (green), predicted route (red).
Remember Folium wants [lat, lon]; our files store lon first.
"""


from __future__ import annotations

from typing import Iterable

import folium
from folium import FeatureGroup, LayerControl, PolyLine, CircleMarker


def _line_points(lines: Iterable[Iterable[tuple[float, float]]]):
    """Convert lon/lat line coordinates to Folium's required lat/lon order."""
    return [[(float(lat), float(lon)) for lon, lat in line] for line in lines]


def make_trip_map(gps_latlon, network_lines=None, reference_lines=None, predicted_lines=None,
                  classifier_only_lines=None, tiles="Esri.WorldStreetMap"):
    """Build a layered Folium map for one trip.

    ``gps_latlon`` is already in Folium order: ``(lat, lon)``.
    All line inputs are sequences of ``(lon, lat)`` points because the raw
    PathSnap files store longitude first.
    """
    gps = [(float(lat), float(lon)) for lat, lon in gps_latlon]
    if not gps:
        raise ValueError("gps_latlon must contain at least one point")
    centre = [sum(p[0] for p in gps) / len(gps), sum(p[1] for p in gps) / len(gps)]
    # Do not use OpenStreetMap's default tile endpoint here.  It is a
    # volunteer-run service with strict usage requirements and can return 403
    # for a locally opened report containing many tiles.  Esri's public street
    # tiles are a suitable report/demo basemap and require no API key.
    tile_options = {
        "Esri.WorldStreetMap": {
            "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            "attr": "Tiles &copy; Esri — Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom, 2023",
        },
        "CartoDB.Positron": {
            "tiles": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            "attr": "&copy; OpenStreetMap contributors &copy; CARTO",
        },
    }
    if tiles not in tile_options:
        raise ValueError(f"Unsupported basemap: {tiles}. Choose one of {sorted(tile_options)}")
    basemap = tile_options[tiles]
    fmap = folium.Map(location=centre, zoom_start=13, control_scale=True,
                      tiles=basemap["tiles"], attr=basemap["attr"],
                      name=tiles)

    gps_group = FeatureGroup(name="GPS points", show=True)
    for lat, lon in gps:
        CircleMarker((lat, lon), radius=2, color="#1565c0", fill=True,
                     fill_color="#1565c0", fill_opacity=0.8).add_to(gps_group)
    gps_group.add_to(fmap)

    layers = [
        ("Road network", network_lines, "#777777", 2, 0.45),
        ("Reference route", reference_lines, "#2e7d32", 4, 0.9),
        ("Predicted route", predicted_lines, "#c62828", 4, 0.9),
        ("Classifier-only route", classifier_only_lines, "#ef6c00", 3, 0.85),
    ]
    for name, lines, colour, weight, opacity in layers:
        if not lines:
            continue
        group = FeatureGroup(name=name, show=name != "Road network")
        for points in _line_points(lines):
            if len(points) >= 2:
                PolyLine(points, color=colour, weight=weight, opacity=opacity).add_to(group)
        group.add_to(fmap)
    LayerControl(collapsed=False).add_to(fmap)
    return fmap
