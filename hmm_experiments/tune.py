"""Tune the HMM decoder parameters on the 80 TRAINING trips (brief 8.3).

    python -m hmm_experiments.tune                                  # placeholder emissions
    python -m hmm_experiments.tune --emission model --probs-train data/processed/oof_probs_train.parquet

Search: two-stage grid (coordinate-wise), objective = mean per-trip point accuracy after HMM
  stage 1  beta x backward_tolerance_m x w_e   (penalty = -50, gap_break_s = 30)
  stage 2  unreachable_penalty x gap_break_s   (at the stage-1 best)
Student-test trips are NOT used here; they are only for checking (compare.py).
Outputs: results/hmm/tuning_<tag>.csv, results/hmm/best_config_<tag>.json, artifacts/hmm_config.json,
         reports/figures/hmm_beta_sensitivity_<tag>.png
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from pathsnap.hmm import DEFAULT_CONFIG, decode
from hmm_experiments.common import (PREP_MAX_GAP_S, artifact_file, figure_file, get_trip, load_config,
                                     results_file, split_dirs)

BETAS = [1, 2, 5, 10, 20, 50, 100, 200]
TOLS = [0, 10, 20, 30, 50]
WES = [0.5, 1.0, 2.0]
PENALTIES = [-20, -50, -100]
GAPS = [15, 30, 60]


def _load(args):
    d, cfg, emission, probs = args
    td = get_trip("train", d, cfg, emission, probs)
    return td.trip.trip_id, td.prep, td.ref_arc


def evaluate(preps, params):
    accs, hits, n = [], 0, 0
    for _, prep, ref in preps:
        m = decode(prep, params)
        ok = (m["arc_id"].to_numpy() == ref)
        accs.append(ok.mean())
        hits += ok.sum()
        n += len(ok)
    return float(np.mean(accs)), float(hits / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emission", default="placeholder", choices=["placeholder", "model"])
    ap.add_argument("--probs-train", default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = load_config()
    tag = a.emission if a.emission == "placeholder" else "model"

    t0 = time.perf_counter()
    dirs = split_dirs(cfg, "train")
    with Pool(a.workers) as pool:
        preps = pool.map(_load, [(d, cfg, a.emission, a.probs_train) for d in dirs], chunksize=1)
    t_prep = time.perf_counter() - t0
    print(f"prepared {len(preps)} training trips in {t_prep:.1f}s")

    rows = []
    base = dict(DEFAULT_CONFIG)

    def run(stage, **kw):
        p = dict(base, **kw)
        ts = time.perf_counter()
        mean_acc, pooled = evaluate(preps, p)
        rows.append({"stage": stage, **{k: p[k] for k in ["beta", "w_e", "unreachable_penalty",
                                                            "backward_tolerance_m", "gap_break_s"]},
                     "train_mean_trip_acc": mean_acc, "train_pooled_acc": pooled,
                     "decode_s": time.perf_counter() - ts})
        return mean_acc

    for beta, tol, we in itertools.product(BETAS, TOLS, WES):
        run(1, beta=beta, backward_tolerance_m=tol, w_e=we, unreachable_penalty=-50.0,
            gap_break_s=float(cfg["gap_break_s"]))
    s1 = pd.DataFrame(rows)
    b1 = s1.loc[s1.train_mean_trip_acc.idxmax()]
    base.update(beta=float(b1.beta), backward_tolerance_m=float(b1.backward_tolerance_m), w_e=float(b1.w_e))
    for pen, gap in itertools.product(PENALTIES, GAPS):
        if gap <= PREP_MAX_GAP_S:
            run(2, unreachable_penalty=float(pen), gap_break_s=float(gap))
    df = pd.DataFrame(rows)
    df.to_csv(results_file(f"tuning_{tag}.csv"), index=False)
    best = df.loc[df.train_mean_trip_acc.idxmax()]
    best_cfg = dict(DEFAULT_CONFIG)
    for k in ["beta", "w_e", "unreachable_penalty", "backward_tolerance_m", "gap_break_s"]:
        best_cfg[k] = float(best[k])
    out = {"emission_source": tag, "objective": "mean per-trip point accuracy after HMM (80 training trips)",
           "train_mean_trip_acc": float(best.train_mean_trip_acc), "train_pooled_acc": float(best.train_pooled_acc),
           "n_configs_evaluated": len(df), "load_prepare_wall_s_incl_cache": t_prep,
           "search_wall_s": float(df.decode_s.sum()), "config": best_cfg}
    with open(results_file(f"best_config_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(artifact_file("hmm_config.json"), "w") as f:
        json.dump({**best_cfg, "emission_source": tag}, f, indent=2)
    print(json.dumps(out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s1 = df[df.stage == 1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for tol in TOLS:
        sub = s1[(s1.backward_tolerance_m == tol) & (s1.w_e == best_cfg["w_e"])].sort_values("beta")
        ax.plot(sub.beta, sub.train_mean_trip_acc, marker="o", label=f"backward tol {tol} m")
    ax.set_xscale("log")
    ax.set_xlabel("beta (m, log scale)")
    ax.set_ylabel("mean per-trip point accuracy (train)")
    ax.set_title(f"HMM sensitivity to beta, w_e={best_cfg['w_e']} ({tag} emissions)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_file(f"hmm_beta_sensitivity_{tag}.png"), dpi=150)


if __name__ == "__main__":
    main()
