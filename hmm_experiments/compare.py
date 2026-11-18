"""Classifier-only argmax vs classifier + HMM on the 10 STUDENT-TEST trips (brief 8.5).

    python -m hmm_experiments.compare                                    # placeholder emissions
    python -m hmm_experiments.compare --model artifacts/final_model.joblib --name lightgbm
    python -m hmm_experiments.compare --emission model --probs-test data/processed/probs_test_lightgbm.parquet \
                                      --name lightgbm

Uses the decoder config in artifacts/hmm_config.json (tuned on training trips by tune.py).
Outputs:
  results/hmm/student_test_per_trip_<name>.csv
  results/hmm/comparison_<name>.csv  and  .json
  reports/figures/hmm_vs_argmax_<name>.png
  reports/figures/maps/hmm_<name>_trip<id>_<least|most>_gain.html
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.graph import build_graph
from pathsnap.hmm import classifier_only, decode, reconstruct_route
from pathsnap.metrics import candidate_miss_rate, top1_accuracy   # shared 9.2 metrics (Zubed)
from pathsnap.route_metrics import point_accuracy, route_continuity, route_prf
from hmm_experiments.common import (REPO_ROOT, figure_file, get_trip, load_config, map_file,
                                     results_file, split_dirs)


def score_with_model(cfg, model_path, features_path, name):
    """predict_proba on test_candidates.parquet in row order (positional, brief 6.5); placeholders -> 0.
    Writes a probability parquet that get_trip() reads."""
    import joblib
    cands = pd.read_parquet(Path(cfg["processed_dir"]) / "test_candidates.parquet")
    with open(features_path) as f:
        cols = json.load(f)
    model = joblib.load(model_path)
    real = (cands["arc_id"] >= 0).to_numpy()
    prob = np.zeros(len(cands))
    prob[real] = model.predict_proba(cands.loc[real, cols])[:, 1]
    out = Path(cfg["processed_dir"]) / f"probs_test_{name}.parquet"
    pd.DataFrame({"trip_id": cands["trip_id"], "point_idx": cands["point_idx"],
                  "arc_id": cands["arc_id"], "prob": prob}).to_parquet(out)
    return out


def evaluate_method(name, matched, td):
    arcs = matched.sort_values("point_idx")["arc_id"].to_numpy()
    route = reconstruct_route(matched, td.trip)
    out = {"method": name, "top1": point_accuracy(arcs, td.ref_arc),
           "continuity": route_continuity(arcs, td.trip, 3)}
    out.update(route_prf(route, td.route))
    return out, route


def folium_map(td, layers, path, title):
    import folium
    tr = td.trip
    g = build_graph(tr)
    lonlat = tr.nodes[["lon", "lat"]].to_numpy()
    P = tr.track[["x", "y"]].to_numpy()
    m = folium.Map(location=[tr.track.lat.mean(), tr.track.lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    # road network near the track only (brief 7.2: keep the HTML small)
    from scipy.spatial import cKDTree
    mid = (g.nodes_xy[g.from_node] + g.nodes_xy[g.to_node]) / 2
    hits = cKDTree(mid).query_ball_point(P[::5], r=200)
    near = np.unique(np.concatenate([np.asarray(h, dtype=int) for h in hits] + [np.array([], int)]))
    fg = folium.FeatureGroup(name="Road network (200 m)")
    for a in near:
        folium.PolyLine([lonlat[g.from_node[a]][::-1], lonlat[g.to_node[a]][::-1]], color="#999999",
                        weight=1, opacity=0.6).add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup(name="GPS points")
    for la, lo in tr.track[["lat", "lon"]].to_numpy()[::2]:
        folium.CircleMarker([la, lo], radius=2, color="#1f5fbf", fill=True, weight=0).add_to(fg)
    fg.add_to(m)
    for lname, route, color, w, dash in layers:
        fg = folium.FeatureGroup(name=lname)
        for a in route:
            folium.PolyLine([lonlat[g.from_node[a]][::-1], lonlat[g.to_node[a]][::-1]], color=color,
                            weight=w, opacity=0.85, dash_array=dash).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = ("<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;"
              "border:1px solid #888;font-size:12px'><b>" + title + "</b><br>"
              "<span style='color:#1f5fbf'>&#9679;</span> GPS points<br>"
              + "".join(f"<span style='color:{c}'>&#9644;</span> {n}<br>" for n, _, c, _, _ in layers)
              + "<span style='color:#999'>&#9644;</span> road network</div>")
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[tr.track.lat.min(), tr.track.lon.min()], [tr.track.lat.max(), tr.track.lon.max()]])
    m.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emission", default="placeholder", choices=["placeholder", "model"])
    ap.add_argument("--probs-test", default=None)
    ap.add_argument("--name", default=None, help="label for output files (default: emission)")
    ap.add_argument("--model", default=None,
                    help="joblib pipeline with predict_proba (e.g. artifacts/final_model.joblib); scores "
                         "test_candidates.parquet by position, as the API does")
    ap.add_argument("--features", default="artifacts/feature_columns.json")
    ap.add_argument("--config", default="artifacts/hmm_config.json")
    a = ap.parse_args()
    name = a.name or a.emission
    cfg = load_config()
    with open(REPO_ROOT / a.config) as f:
        hcfg = json.load(f)
    probs_test = a.probs_test
    if a.model:
        a.emission = "model"
        probs_test = score_with_model(cfg, REPO_ROOT / a.model, REPO_ROOT / a.features, name)

    rows, keep = [], {}
    for d in split_dirs(cfg, "student_test"):
        td = get_trip("student_test", d, cfg, a.emission, probs_test)
        tid = td.trip.trip_id
        n = len(td.trip.track)
        argmax = classifier_only(td.cands, td.probs, n)
        t0 = time.perf_counter()
        hmm = decode(td.prep, hcfg)
        t_dec = time.perf_counter() - t0
        t_total = td.timings["prepare_s"] + t_dec
        base = {"trip_id": tid, "n_points": n, "miss_rate": candidate_miss_rate(td.cands, td.refs),
                "n_breaks": hmm.attrs["n_breaks"], "n_filled": int(hmm.filled.sum()),
                "decoder_ms_per_point": 1000 * t_total / n}
        r1, route_a = evaluate_method("Classifier-only argmax", argmax, td)
        r2, route_h = evaluate_method("Classifier + HMM", hmm, td)
        # classifier-only Top-1 must equal the shared definition (brief 9.2) - fail loudly if not
        shared_top1 = top1_accuracy(td.cands, td.probs, td.refs)
        if not np.isclose(shared_top1, r1["top1"]):
            raise AssertionError(f"trip {tid}: argmax Top-1 {r1['top1']} != shared top1_accuracy {shared_top1}")
        rows += [{**base, **r1}, {**base, **r2}]
        keep[tid] = (td, route_a, route_h)
        print(tid, n, f"argmax {r1['top1']:.3f} -> hmm {r2['top1']:.3f} | cont {r1['continuity']:.3f} -> "
                      f"{r2['continuity']:.3f} | routeF1 {r1['route_f1']:.3f} -> {r2['route_f1']:.3f}")
    per = pd.DataFrame(rows)
    per.to_csv(results_file(f"student_test_per_trip_{name}.csv"), index=False)

    summ = []
    for meth, g in per.groupby("method", sort=False):
        summ.append({"method": meth,
                     "top1_pooled": float((g.top1 * g.n_points).sum() / g.n_points.sum()),
                     "top1_mean_trip": float(g.top1.mean()),
                     "continuity_mean_trip": float(g.continuity.mean()),
                     "route_precision": float(g.route_precision.mean()),
                     "route_recall": float(g.route_recall.mean()),
                     "route_f1": float(g.route_f1.mean())})
    summ = pd.DataFrame(summ)
    summ.to_csv(results_file(f"comparison_{name}.csv"), index=False)
    miss = float((per.miss_rate * per.n_points).sum() / per.n_points.sum() / 1)
    info = {"emission_source": name, "split": "student_test", "hmm_config": hcfg,
            "candidate_miss_rate_pooled": miss, "upper_bound_top1": 1 - miss,
            "decoder_ms_per_point_mean": float(per.decoder_ms_per_point.mean()),
            "summary": summ.to_dict(orient="records")}
    with open(results_file(f"comparison_{name}.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(summ.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wide = per.pivot(index="trip_id", columns="method", values="top1")
    widec = per.pivot(index="trip_id", columns="method", values="continuity")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, w, lab in [(axes[0], wide, "Top-1 point accuracy"), (axes[1], widec, "Route continuity (3 hops)")]:
        x = np.arange(len(w))
        ax.bar(x - 0.2, w["Classifier-only argmax"], 0.4, label="Classifier-only argmax", color="#e08a2c")
        ax.bar(x + 0.2, w["Classifier + HMM"], 0.4, label="Classifier + HMM", color="#2c6fbb")
        ax.set_xticks(x, [str(t) for t in w.index])
        ax.set_xlabel("student-test trip id")
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Student-test: classifier-only vs HMM ({name} emissions)")
    fig.tight_layout()
    fig.savefig(figure_file(f"hmm_vs_argmax_{name}.png"), dpi=150)

    gain = wide["Classifier + HMM"] - wide["Classifier-only argmax"]
    for tid, label in [(int(gain.idxmin()), "least"), (int(gain.idxmax()), "most")]:
        td, ra, rh = keep[tid]
        folium_map(td, [("Reference route", list(td.route), "#2ca02c", 6, None),
                        ("Classifier-only argmax", ra, "#ff8c00", 3, "5,5"),
                        ("HMM prediction", rh, "#d62728", 3, None)],
                   map_file(f"hmm_{name}_trip{tid}_{label}_gain.html"),
                   f"Trip {tid}: HMM gain {gain[tid]:+.3f} ({name})")


if __name__ == "__main__":
    main()
