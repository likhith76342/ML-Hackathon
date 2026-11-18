"""Reproducible exploratory data analysis for PathSnap.

Run from the repository root with:
    python -m eda.run_eda

The script reads raw trip folders, never changes them, and writes compact
tables and PNG figures under reports/figures.  Ground-truth .route files are
used only for descriptive train/test analysis, never as model features.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.io import list_trips, load_route, load_trip, trip_dir


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "figures"
TABLES = ROOT / "results" / "eda"


def _summarise_trip(split: str, trip_id: int) -> dict:
    """Return one row of descriptive statistics for one trip."""
    tdir = trip_dir(DATA_ROOT / split, trip_id)
    trip = load_trip(tdir)
    track = trip.track
    dt = track["t"].diff()
    distances = np.hypot(track["lon"].diff(), track["lat"].diff())
    speed = distances / dt.replace(0, np.nan)
    route = load_route(tdir) if (tdir / f"{tdir.name}.route").exists() else []
    return {
        "split": split,
        "trip_id": trip_id,
        "gps_points": len(track),
        "nodes": len(trip.nodes),
        "arcs": len(trip.arcs),
        "route_arcs": len(route),
        "duration_s": float(track["t"].iloc[-1] - track["t"].iloc[0]) if len(track) else 0.0,
        "min_dt_s": float(dt.min()) if len(dt) > 1 else np.nan,
        "max_dt_s": float(dt.max()) if len(dt) > 1 else np.nan,
        "gap_over_30s": int((dt > 30).sum()),
        "duplicate_timestamps": int(track["t"].duplicated().sum()),
        "invalid_coordinates": int((~track[["lon", "lat"]].apply(np.isfinite).all(axis=1)).sum()),
        "lon_min": float(track["lon"].min()),
        "lon_max": float(track["lon"].max()),
        "lat_min": float(track["lat"].min()),
        "lat_max": float(track["lat"].max()),
        "track_span_deg": float(np.nanmax(distances)) if len(distances) else 0.0,
        "nonpositive_dt": int((dt <= 0).sum()),
        "implausible_degree_step": int((distances > 0.01).sum()),
        "speed_degree_per_s_max": float(speed.replace([np.inf, -np.inf], np.nan).max()) if len(speed) else np.nan,
    }


def _savefig(name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    rows: list[dict] = []
    for split in ("train", "student_test"):
        for trip_id in list_trips(DATA_ROOT / split):
            rows.append(_summarise_trip(split, trip_id))
    summary = pd.DataFrame(rows).sort_values(["split", "trip_id"])
    TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / "trip_summary_partitioned.csv", index=False)

    # 1. How many observations does each trip contribute?
    plt.figure(figsize=(10, 4))
    for split, group in summary.groupby("split"):
        plt.hist(group["gps_points"], bins=20, alpha=0.65, label=split)
    plt.xlabel("GPS points per trip")
    plt.ylabel("Number of trips")
    plt.title("Trip-size distribution by partition")
    plt.legend()
    _savefig("eda_trip_size_distribution.png")

    # 2. Time gaps reveal sampling regularity and HMM break points.
    plt.figure(figsize=(8, 4))
    for split, group in summary.groupby("split"):
        plt.hist(group["max_dt_s"].clip(upper=120), bins=24, alpha=0.65, label=split)
    plt.axvline(30, color="black", linestyle="--", label="30 s HMM threshold")
    plt.xlabel("Largest time gap per trip (seconds, clipped at 120)")
    plt.ylabel("Number of trips")
    plt.title("Maximum GPS sampling gap")
    plt.legend()
    _savefig("eda_max_time_gap.png")

    # 3. Geographic coverage prevents accidental assumptions about one city.
    plt.figure(figsize=(8, 5))
    for split in ("train", "student_test"):
        for trip_id in list_trips(DATA_ROOT / split):
            track = load_trip(trip_dir(DATA_ROOT / split, trip_id)).track
            plt.plot(track["lon"], track["lat"], alpha=0.25, linewidth=0.6,
                     label=split if trip_id == list_trips(DATA_ROOT / split)[0] else None)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("GPS trajectory coverage")
    plt.legend()
    _savefig("eda_geographic_coverage.png")

    # 4. Network size shows the computational range of candidate search.
    plt.figure(figsize=(8, 4))
    for split, group in summary.groupby("split"):
        plt.scatter(group["nodes"], group["arcs"], c=group["gps_points"], cmap="viridis",
                    label=split, edgecolors="black", linewidths=0.25)
    plt.colorbar(label="GPS points")
    plt.xlabel("Road-network nodes")
    plt.ylabel("Directed road arcs")
    plt.title("Road-network size per trip (train and student_test)")
    plt.legend(title="Partition")
    _savefig("eda_network_size.png")

    # 5. Directed reverse-edge check on all available networks.
    connectivity_rows = []
    for split in ("train", "student_test"):
        for trip_id in list_trips(DATA_ROOT / split):
            arcs = load_trip(trip_dir(DATA_ROOT / split, trip_id)).arcs
            edge_set = set(map(tuple, arcs[["from_node", "to_node"]].to_numpy()))
            reverse_count = sum((b, a) in edge_set for a, b in edge_set)
            connectivity_rows.append({
                "split": split,
                "trip_id": trip_id,
                "arcs": len(arcs),
                "reverse_arc_fraction": reverse_count / len(edge_set) if edge_set else np.nan,
            })
    pd.DataFrame(connectivity_rows).to_csv(TABLES / "network_connectivity_partitioned.csv", index=False)

    # A machine-readable data-quality report for the written report.
    checks = []
    for split, group in summary.groupby("split"):
        checks.extend([
            {"partition": split, "check": "trips", "value": len(group)},
            {"partition": split, "check": "trips_with_gap_over_30s", "value": int((group["gap_over_30s"] > 0).sum())},
            {"partition": split, "check": "trips_with_duplicate_timestamps", "value": int((group["duplicate_timestamps"] > 0).sum())},
            {"partition": split, "check": "trips_with_nonpositive_dt", "value": int((group["nonpositive_dt"] > 0).sum())},
            {"partition": split, "check": "rows_with_invalid_coordinates", "value": int(group["invalid_coordinates"].sum())},
        ])
    pd.DataFrame(checks).to_csv(TABLES / "data_quality_checks_partitioned.csv", index=False)
    total_gaps = int((summary["gap_over_30s"] > 0).sum())
    total_duplicates = int((summary["duplicate_timestamps"] > 0).sum())
    total_nonpositive = int((summary["nonpositive_dt"] > 0).sum())
    total_invalid = int(summary["invalid_coordinates"].sum())
    checks_print = {"total_trips": len(summary), "trips_with_gap_over_30s": total_gaps,
                    "trips_with_duplicate_timestamps": total_duplicates,
                    "trips_with_nonpositive_dt": total_nonpositive,
                    "rows_with_invalid_coordinates": total_invalid}
    print(summary.groupby("split")[["gps_points", "nodes", "arcs", "route_arcs"]].agg(["count", "min", "median", "max"]))
    print(pd.Series(checks_print))
    print(f"Wrote tables to {TABLES}")
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
