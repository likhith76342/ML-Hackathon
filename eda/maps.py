"""Reusable Folium map builder.  Owner: Mannu.  (Brief 7.2)  Also used by the UI.

Layers: road network (grey), GPS points (blue), reference route (green), predicted route (red).
Remember Folium wants [lat, lon]; our files store lon first.
"""


def make_trip_map(gps_latlon, network_lines=None, reference_lines=None, predicted_lines=None,
                  classifier_only_lines=None):
    raise NotImplementedError("Mannu: implement (brief 7.2)")
