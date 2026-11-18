"""pathsnap.pipeline tests: training/serving skew, no .route on the prediction path, HMM fallback.

Run from the repo root:  pytest tests/
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from pathsnap.config import CFG, repo_path
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.io import list_trips, load_trip, trip_dir

STUDENT_TEST = repo_path(CFG["data_root"]) / "student_test"
TRIPS = list_trips(STUDENT_TEST) if STUDENT_TEST.is_dir() else []
TEST_PARQUET = repo_path(CFG["processed_dir"]) / "test_candidates.parquet"

needs_data = pytest.mark.skipif(not TRIPS, reason=f"student_test data not found at {STUDENT_TEST}")
needs_parquet = pytest.mark.skipif(not TEST_PARQUET.is_file(), reason=f"{TEST_PARQUET} not found")


class DistanceModel:
    """Stand-in classifier (no artifact needed): nearer arcs score higher."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = 1.0 / (1.0 + X["dist_m"].to_numpy(dtype=float) / 10.0)
        return np.column_stack([1 - p, p])


def smallest_trip() -> int:
    return min(TRIPS, key=lambda t: sum(1 for _ in open(trip_dir(STUDENT_TEST, t) / f"{t:08d}.track")))


HMM_CFG = {"gap_break_s": CFG["gap_break_s"]}


# ----------------------------------------------------------------------------- skew
def _live_and_stored(tid: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from pathsnap.candidates import build_candidates

    live = build_candidates(load_trip(trip_dir(STUDENT_TEST, tid))).reset_index(drop=True)
    stored = pd.read_parquet(TEST_PARQUET)
    return live, stored[stored.trip_id == tid].reset_index(drop=True)


TWIN_TIE_REASON = (
    "Known issue in pathsnap/candidates.py (Zubed): reverse-twin arcs (A->B, B->A) are at the same distance, "
    "but build_candidates breaks the tie with lexsort on float distances that differ only by rounding noise "
    "(~1e-15 m), so their order (and dist_rank) can differ between machines. Remove this xfail once ties "
    "break deterministically by arc_id.")


@needs_data
@needs_parquet
@pytest.mark.xfail(strict=True, reason=TWIN_TIE_REASON)
def test_inference_candidates_match_training_parquet():
    """build_candidates at inference == the rows the models were trained/evaluated on (strict)."""
    live, stored = _live_and_stored(smallest_trip())
    assert len(live) == len(stored) > 0
    for col in ("trip_id", "point_idx", "arc_id"):
        np.testing.assert_array_equal(live[col].to_numpy(), stored[col].to_numpy(), err_msg=col)
    for col in ["t"] + FEATURE_COLUMNS:
        np.testing.assert_allclose(live[col].to_numpy(dtype=float), stored[col].to_numpy(dtype=float),
                                   rtol=0, atol=1e-6, equal_nan=True, err_msg=col)


@needs_data
@needs_parquet
@pytest.mark.parametrize("tid", TRIPS)
def test_inference_candidates_match_parquet_up_to_twin_ties(tid):
    """Same candidates and per-pair features as the training parquet, except where reverse-twin arcs tie:
    their order may differ, and at the K-th slot the other twin of the pair may have been kept."""
    live, stored = _live_and_stored(tid)
    assert len(live) == len(stored) > 0
    np.testing.assert_array_equal(live.point_idx.to_numpy(), stored.point_idx.to_numpy())

    arcs = load_trip(trip_dir(STUDENT_TEST, tid)).arcs.to_numpy()
    both = live.merge(stored, on=["trip_id", "point_idx", "arc_id"], how="outer", suffixes=("", "_pq"),
                      indicator=True)
    live_only = both[both._merge == "left_only"].sort_values("point_idx")
    pq_only = both[both._merge == "right_only"].sort_values("point_idx")
    assert len(live_only) == len(pq_only), "inference and training candidate sets differ"
    for (_, a), (_, b) in zip(live_only.iterrows(), pq_only.iterrows()):
        assert a.point_idx == b.point_idx
        assert tuple(arcs[int(a.arc_id)]) == tuple(arcs[int(b.arc_id)])[::-1], "set difference is not a twin pair"
        assert abs(a.dist_m - b.dist_m_pq) < 1e-9 and a.dist_rank == b.dist_rank_pq == CFG["max_candidates"] - 1

    keyed = both[both._merge == "both"]
    for col in ["t"] + [c for c in FEATURE_COLUMNS if c != "dist_rank"]:
        np.testing.assert_allclose(keyed[col].to_numpy(dtype=float), keyed[f"{col}_pq"].to_numpy(dtype=float),
                                   rtol=0, atol=1e-6, equal_nan=True, err_msg=col)

    moved = live.arc_id.to_numpy() != stored.arc_id.to_numpy()
    tied = np.abs(live.dist_m.to_numpy() - stored.dist_m.to_numpy()) < 1e-9
    assert np.all(tied[moved]), "a candidate moved that is not part of a distance tie"
    rank_diff = keyed.dist_rank.to_numpy() != keyed.dist_rank_pq.to_numpy()
    assert rank_diff.sum() <= moved.sum()


# ----------------------------------------------------------------------------- leak
@needs_data
def test_predict_trip_never_reads_route(monkeypatch, tmp_path):
    """With load_route sabotaged AND the .route file absent, the prediction path still runs."""
    import pathsnap.io
    from pathsnap.pipeline import predict_trip

    def boom(*a, **k):
        raise AssertionError("prediction path called load_route")

    monkeypatch.setattr(pathsnap.io, "load_route", boom)
    tid = smallest_trip()
    src = trip_dir(STUDENT_TEST, tid)
    dst = tmp_path / src.name
    dst.mkdir()
    for ext in (".track", ".nodes", ".arcs"):                 # deliberately no .route
        shutil.copy(src / f"{src.name}{ext}", dst / f"{src.name}{ext}")

    out = predict_trip(dst, DistanceModel(), FEATURE_COLUMNS, HMM_CFG)
    n = len(load_trip(dst).track)
    assert set(out) >= {"matched", "argmax", "route", "timings", "hmm_used"}
    assert len(out["argmax"]) == n and out["route"]
    assert set(out["timings"]) == {"candidates", "model", "viterbi"}


# ----------------------------------------------------------------------------- HMM fallback
@needs_data
def test_hmm_not_implemented_falls_back_to_classifier(monkeypatch):
    import pathsnap.hmm
    from pathsnap.pipeline import collapse_route, predict_trip

    def not_yet(*a, **k):
        raise NotImplementedError("hmm stub")

    monkeypatch.setattr(pathsnap.hmm, "viterbi", not_yet)
    out = predict_trip(trip_dir(STUDENT_TEST, smallest_trip()), DistanceModel(), FEATURE_COLUMNS, HMM_CFG)
    assert out["hmm_used"] is False and out["matched"] is None
    assert out["route"] == collapse_route(out["argmax"]["arc_id"])
    assert out["timings"]["viterbi"] is None


@needs_data
def test_other_hmm_errors_propagate(monkeypatch):
    import pathsnap.hmm
    from pathsnap.pipeline import predict_trip

    def broken(*a, **k):
        raise ValueError("real bug")

    monkeypatch.setattr(pathsnap.hmm, "viterbi", broken)
    with pytest.raises(ValueError, match="real bug"):
        predict_trip(trip_dir(STUDENT_TEST, smallest_trip()), DistanceModel(), FEATURE_COLUMNS, HMM_CFG)


@needs_data
def test_hmm_output_contract():
    from pathsnap.pipeline import predict_trip

    out = predict_trip(trip_dir(STUDENT_TEST, smallest_trip()), DistanceModel(), FEATURE_COLUMNS, HMM_CFG)
    if not out["hmm_used"]:
        pytest.skip("pathsnap.hmm not implemented")
    m = out["matched"]
    assert {"point_idx", "arc_id", "segment_id", "filled"} <= set(m.columns)
    assert len(m) == len(out["argmax"])


# ----------------------------------------------------------------------------- helpers
def test_argmax_ties_and_placeholders():
    from pathsnap.pipeline import argmax_arcs

    cands = pd.DataFrame({"point_idx": [0, 0, 1, 2, 2], "arc_id": [5, 6, -1, 7, 8]})
    probs = np.array([0.4, 0.4, 0.0, 0.1, 0.9])
    out = argmax_arcs(cands, probs, 3)
    assert out["arc_id"].tolist() == [5, -1, 8]       # tie -> first row (nearest); no candidates -> -1


def test_collapse_route():
    from pathsnap.pipeline import collapse_route

    assert collapse_route([3, 3, -1, 3, 4, 4, 5, -1]) == [3, 4, 5]
    assert collapse_route([-1, -1]) == []


def test_n_breaks_from_attrs_or_segments():
    from pathsnap.pipeline import n_breaks

    m = pd.DataFrame({"segment_id": [0, 0, 1, 1, 2]})
    assert n_breaks(m) == 2
    m.attrs["n_breaks"] = 5
    assert n_breaks(m) == 5
    assert n_breaks(None) is None
