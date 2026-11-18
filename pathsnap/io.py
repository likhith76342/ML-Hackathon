"""Raw-data loading.  Owner: Zubed (Likhith wrote the starter; extend freely, keep the signatures).

File formats (whitespace separated, no header):
  .nodes  lon lat              node_id  = 0-based row
  .arcs   from_node to_node    arc_id   = 0-based row (directed)
  .track  lon lat t            point_idx = 0-based row
  .route  arc_id               ground truth, driving order

IMPORTANT: load_trip() never reads .route. Ground truth is only available via load_route(),
which must only be called by training/labelling code and the server-side evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Trip:
    trip_id: int
    trip_dir: Path
    track: pd.DataFrame   # columns: lon, lat, t   (+ x, y after geo.add_xy)
    nodes: pd.DataFrame   # columns: lon, lat      (+ x, y after geo.add_xy)
    arcs: pd.DataFrame    # columns: from_node, to_node


def _read(path: Path, names: list[str], dtype) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def list_trips(split_dir: Path | str) -> list[int]:
    """Trip IDs found in a split folder (e.g. data/raw/train). Never hardcode IDs."""
    split_dir = Path(split_dir)
    return sorted(int(p.name) for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit())


def trip_dir(split_dir: Path | str, trip_id: int) -> Path:
    return Path(split_dir) / f"{int(trip_id):08d}"


def load_trip(tdir: Path | str) -> Trip:
    """Load track, nodes and arcs for one trip folder. Does NOT read .route."""
    tdir = Path(tdir)
    stem = tdir / tdir.name
    track = _read(stem.with_suffix(".track"), ["lon", "lat", "t"], float)
    nodes = _read(stem.with_suffix(".nodes"), ["lon", "lat"], float)
    arcs = _read(stem.with_suffix(".arcs"), ["from_node", "to_node"], "int64")
    return Trip(int(tdir.name), tdir, track, nodes, arcs)


def load_route(tdir: Path | str) -> list[int]:
    """Ground-truth arc sequence. Training/labelling and server-side evaluation ONLY."""
    tdir = Path(tdir)
    path = (tdir / tdir.name).with_suffix(".route")
    return _read(path, ["arc_id"], "int64")["arc_id"].tolist()
