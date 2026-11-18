"""Shared helpers for the decoder experiments (owner: Keshav).

Two emission sources:
  placeholder  Gaussian on dist_m (sigma = 10 m), as in brief 8.2. Lets the decoder be built and tuned
               before any classifier exists. Candidates and point references come from the shared
               pipeline (pathsnap.candidates.build_candidates, pathsnap.labels.align_points_to_route).
  model        Real classifier probabilities. Candidates/refs come from Zubed's parquet files and
               probabilities from a parquet with columns trip_id, point_idx, arc_id, prob.
               For TRAINING trips this must be OUT-OF-FOLD probabilities (each trip scored by the model
               fitted without its fold); otherwise beta / w_e are tuned on over-confident scores.

Per-trip results are cached under data/processed/hmm_cache/ so repeated runs are fast.
"""
from __future__ import annotations

import pickle
from pathlib import Path
import time

import numpy as np
import pandas as pd

from pathsnap.candidates import build_candidates
from pathsnap.config import CFG, REPO_ROOT, repo_path  # noqa: F401  (REPO_ROOT used by compare.py)
from pathsnap.geo import add_xy
from pathsnap.hmm import prepare_trip
from pathsnap.io import list_trips, load_route, load_trip, trip_dir
from pathsnap.labels import align_points_to_route

PLACEHOLDER_SIGMA_M = 10.0
PREP_MAX_GAP_S = 120.0          # largest gap_break_s in the tuning grid


def load_config() -> dict:
    """Shared config.yaml via pathsnap.config, with the data folders resolved against the repo root."""
    cfg = dict(CFG)
    for k in ("data_root", "processed_dir"):
        cfg[k] = repo_path(cfg[k])
    return cfg


def _out(base_key: str, *parts) -> Path:
    p = repo_path(CFG[base_key]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def results_file(name: str) -> Path:      # results/hmm/<name>
    return _out("results_dir", "hmm", name)


def figure_file(name: str) -> Path:       # reports/figures/<name>  (names start with hmm_)
    return _out("figures_dir", name)


def map_file(name: str) -> Path:          # reports/figures/maps/<name>
    return _out("figures_dir", "maps", name)


def artifact_file(name: str) -> Path:     # artifacts/<name>
    return _out("artifacts_dir", name)


def placeholder_probs(cands: pd.DataFrame, sigma: float = PLACEHOLDER_SIGMA_M) -> np.ndarray:
    d = cands["dist_m"].to_numpy(dtype=float)
    p = np.exp(-0.5 * (np.nan_to_num(d, nan=0.0) / sigma) ** 2)
    return np.where(cands["arc_id"].to_numpy() >= 0, p, 0.0)


class TripData:
    """Everything the experiments need for one trip (prediction inputs + server-side reference)."""

    def __init__(self, trip, cands, probs, refs, route, prep, timings):
        self.trip, self.cands, self.probs = trip, cands, probs
        self.refs = refs                                   # trip_id, point_idx, ref_arc, ... (one row per point)
        self.ref_arc = refs.sort_values("point_idx")["ref_arc"].to_numpy()
        self.route, self.prep, self.timings = route, prep, timings


def _model_inputs(split, trip, tid, cfg, probs_path):
    proc = Path(cfg["processed_dir"])
    fname = "train_candidates.parquet" if split == "train" else "test_candidates.parquet"
    # kept in file order: rows are already sorted by (point_idx, dist_rank) per brief 6.5
    cands = pd.read_parquet(proc / fname, filters=[("trip_id", "==", tid)]).reset_index(drop=True)
    pr = pd.read_parquet(probs_path, filters=[("trip_id", "==", tid)])
    m = cands[["point_idx", "arc_id"]].merge(pr[["point_idx", "arc_id", "prob"]], how="left",
                                             on=["point_idx", "arc_id"])
    probs = m["prob"].fillna(0.0).to_numpy()
    refs = pd.read_parquet(proc / "point_refs.parquet", filters=[("trip_id", "==", tid)])
    return cands, probs, refs.sort_values("point_idx").reset_index(drop=True)


def get_trip(split: str, tdir, cfg: dict, emission: str = "placeholder",
             probs_path: str | None = None, use_cache: bool = True) -> TripData:
    tid = int(Path(tdir).name)
    tag = emission if emission == "placeholder" else f"model_{Path(probs_path or 'x').stem}"
    cdir = Path(cfg["processed_dir"]) / "hmm_cache" / tag
    cpath = cdir / f"{split}_{tid:08d}.pkl"
    if use_cache and cpath.exists():
        with open(cpath, "rb") as f:
            return pickle.load(f)
    t0 = time.perf_counter()
    trip = add_xy(load_trip(tdir))
    route = np.asarray(load_route(tdir), dtype=np.int64)   # evaluation only; never given to the decoder
    if emission == "placeholder":
        cands = build_candidates(trip)
        probs = placeholder_probs(cands)
        refs = align_points_to_route(trip, route)
    else:
        cands, probs, refs = _model_inputs(split, trip, tid, cfg, probs_path)
    t1 = time.perf_counter()
    prep = prepare_trip(cands, probs, trip, {"gap_break_s": cfg["gap_break_s"]}, max_gap_s=PREP_MAX_GAP_S)
    t2 = time.perf_counter()
    td = TripData(trip, cands, probs, refs, route, prep,
                  {"inputs_s": t1 - t0, "prepare_s": t2 - t1})
    if use_cache:
        trip.__dict__.pop("_road_graph", None)    # rebuilt on demand; keeps the cache small
        cdir.mkdir(parents=True, exist_ok=True)
        with open(cpath, "wb") as f:
            pickle.dump(td, f, protocol=pickle.HIGHEST_PROTOCOL)
    return td


def split_dirs(cfg: dict, split: str) -> list[Path]:
    root = Path(cfg["data_root"]) / split
    return [trip_dir(root, t) for t in list_trips(root)]
