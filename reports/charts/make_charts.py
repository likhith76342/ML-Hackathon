"""Model-comparison charts from SAVED results only.  Owner: Likhith.

    python reports/charts/make_charts.py [--results-dir results] [--out-dir reports/figures/model_comparison]
                                         [--use primary|all_trips] [--optuna-dir optimisation]

Reads (never writes) results/<model>/{metrics.json, metrics_all_trips.json, cv_folds*.csv, run_summary*.json}
for every folder that has a metrics file, plus any Optuna trials CSV under results/ or the optimisation folder.
No retraining, and no number is typed in: every plotted or tabulated value comes from those files. Missing
models or files are skipped with a printed warning. The exit code is non-zero only if no model is found.

Output (PNG, 200 dpi, deterministic) goes to --out-dir. With --use all_trips it goes to <out-dir>/all_trips/,
so the primary charts are never overwritten.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------- style
# Categorical palette (dataviz reference, light mode), validated with validate_palette.js for 5 slots:
# all hard gates pass; contrast < 3:1 on 3 slots -> every mark is direct-labelled or named on an axis,
# and summary_table.* is the table view.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OVERFLOW_COLOR = "#8a8984"     # a 9th+ model is never given a generated hue
KNOWN_ORDER = ["logreg", "random_forest", "knn", "mlp", "lightgbm"]   # colour follows the model, not its rank
DISPLAY = {"logreg": "Logistic regression", "random_forest": "Random forest", "knn": "KNN", "mlp": "MLP",
           "lightgbm": "LightGBM"}
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DPI = 200

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 9.5, "axes.labelcolor": INK_2, "axes.edgecolor": INK_3,
    "axes.linewidth": 0.8, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False, "legend.fontsize": 9, "svg.hashsalt": "pathsnap",
})

METRIC_KEYS = ["model", "family", "config", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "cv_top1_std",
               "cv_log_loss_mean", "train_top1", "train_macro_f1", "test_accuracy", "test_precision", "test_recall",
               "test_macro_f1", "test_log_loss", "test_top1", "test_candidate_miss_rate",
               "test_route_continuity_argmax", "test_confusion_matrix", "train_time_s", "inference_ms_per_point",
               "seed"]
EXPECTED_DATASET_VERSION = 2


def warn(msg: str, bucket: list[str] | None = None) -> None:
    print(f"WARNING: {msg}")
    if bucket is not None:
        bucket.append(msg)


def num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


# ----------------------------------------------------------------------------- loading
@dataclass
class Model:
    key: str
    name: str
    color: str
    metrics: dict | None = None            # the selected run (primary or all_trips)
    source: str = ""
    primary: dict | None = None
    all_trips: dict | None = None
    folds: list[dict] | None = None
    folds_source: str = ""
    run_summary: dict | None = None
    notes: list[str] = field(default_factory=list)   # comparability warnings for chart footnotes

    def m(self, key):
        return num((self.metrics or {}).get(key))


def _read_json(path: Path, problems: list[str]) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        warn(f"could not read {path}: {e}", problems)
        return None


def _read_folds(path: Path, problems: list[str]) -> list[dict] | None:
    if not path.is_file():
        return None
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        return [{"fold": int(r["fold"]), "top1": float(r["top1"])} for r in rows]
    except (OSError, KeyError, ValueError) as e:
        warn(f"could not read per-fold Top-1 from {path}: {e}", problems)
        return None


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def discover(results_dir: Path, use: str, problems: list[str]) -> list[Model]:
    folders = sorted(p for p in results_dir.iterdir()
                     if p.is_dir() and ((p / "metrics.json").is_file() or (p / "metrics_all_trips.json").is_file()))
    keys = [p.name for p in folders]
    ordered = [k for k in KNOWN_ORDER if k in keys] + sorted(k for k in keys if k not in KNOWN_ORDER)
    colors = {}
    for i, k in enumerate(ordered):
        colors[k] = PALETTE[i] if i < len(PALETTE) else OVERFLOW_COLOR
        if i >= len(PALETTE):
            warn(f"more than {len(PALETTE)} models: {k} is drawn in neutral grey", problems)

    models = []
    for k in ordered:
        d = results_dir / k
        m = Model(key=k, name=DISPLAY.get(k, k), color=colors[k])
        m.primary = _read_json(d / "metrics.json", problems)
        m.all_trips = _read_json(d / "metrics_all_trips.json", problems)
        suffix = "" if use == "primary" else "_all_trips"
        m.metrics = m.primary if use == "primary" else m.all_trips
        m.source = rel(d / f"metrics{suffix}.json")
        if m.metrics is None:
            warn(f"{k}: no metrics{suffix}.json - model skipped", problems)
            continue
        missing = [key for key in METRIC_KEYS if key not in m.metrics]
        if missing:
            warn(f"{k}: {m.source} lacks keys {missing}; those values are left out", problems)
        m.folds = _read_folds(d / f"cv_folds{suffix}.csv", problems)
        m.folds_source = rel(d / f"cv_folds{suffix}.csv")
        if m.folds is None:
            warn(f"{k}: no cv_folds{suffix}.csv - left out of the per-fold chart", problems)
        if m.all_trips is None and use == "primary":
            warn(f"{k}: no metrics_all_trips.json - left out of the primary-vs-all-trips chart", problems)
        m.run_summary = _read_json(d / f"run_summary{suffix}.json", problems)
        check_comparability(m, use, problems)
        models.append(m)
    return models


def check_comparability(m: Model, use: str, problems: list[str]) -> None:
    """dataset_version / excluded_trips from run_summary (if any) and the metrics config."""
    cfg = (m.metrics or {}).get("config") or {}
    rs = m.run_summary or {}
    dv = rs.get("dataset_version", cfg.get("dataset_version"))
    has_excl = "excluded_trips" in rs or "excluded_trips" in cfg
    excl = rs.get("excluded_trips", cfg.get("excluded_trips"))
    if dv != EXPECTED_DATASET_VERSION:
        note = f"{m.key}: dataset_version is {dv!r}, expected {EXPECTED_DATASET_VERSION} - may not be comparable"
        warn(note, problems)
        m.notes.append(note)
    if not has_excl:
        note = f"{m.key}: excluded_trips not recorded - may not be comparable"
        warn(note, problems)
        m.notes.append(note)
    elif use == "primary" and not excl:
        note = f"{m.key}: primary run excludes no trips - may not be comparable"
        warn(note, problems)
        m.notes.append(note)


def excluded_set(models: list[Model]) -> list[int] | None:
    sets = {tuple(sorted(((x.metrics or {}).get("config") or {}).get("excluded_trips") or [])) for x in models}
    return list(next(iter(sets))) if len(sets) == 1 else None


# ----------------------------------------------------------------------------- figure helpers
def source_note(models: list[Model], use: str, what: str) -> str:
    if use == "primary":
        ex = excluded_set(models)
        run = (f"primary runs (flagged training trips {', '.join(map(str, ex))} excluded)" if ex
               else "primary runs (excluded trips differ between models - see warnings)")
        files = "results/<model>/metrics.json"
    else:
        run, files = "all-trips runs (no trips excluded)", "results/<model>/metrics_all_trips.json"
    return f"Source: {files if not what else what}; {run}."


def finish(fig, path: Path, footnote: str, models: list[Model], written: list[Path], top: float = 1.0) -> None:
    """Footnote (source + comparability warnings) under the plot, then one deterministic layout pass."""
    notes = [n for m in models for n in m.notes]
    lines = [footnote] + [f"! {n}" for n in notes]
    height_in = fig.get_size_inches()[1]
    foot = (0.12 + 0.15 * len(lines)) / height_in
    fig.text(0.01, 0.012, "\n".join(lines), ha="left", va="bottom", fontsize=7.5, color=INK_2, wrap=True)
    fig.tight_layout(rect=(0, foot, 1, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    written.append(path)


def pct_axis(ax, axis="x", decimals=1):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def label_points(ax, pts: list[tuple[float, float, str]], min_gap=0.055) -> None:
    """Direct-label scatter points; a label too close to the previous one (in axes space) moves below its point."""
    to_axes = ax.transData + ax.transAxes.inverted()
    placed: list[tuple[float, float]] = []
    for x, y, text in sorted(pts, key=lambda t: -t[1]):
        ax_x, ax_y = to_axes.transform((x, y))
        clash = any(abs(ax_y - py) < min_gap and abs(ax_x - px) < 0.3 for px, py in placed)
        ax.annotate(text, (x, y), xytext=(8, -12 if clash else 5), textcoords="offset points", fontsize=8.5,
                    color=INK, va="center")
        placed.append((ax_x, ax_y - (0.05 if clash else 0)))


def need(models: list[Model], key: str, chart: str, problems: list[str]) -> list[Model]:
    ok = [m for m in models if m.m(key) is not None]
    for m in models:
        if m.m(key) is None:
            warn(f"{chart}: {m.key} has no {key} - left out", problems)
    return ok


def padded_limits(values: list[float], pad_frac=0.25, min_span=0.004):
    lo, hi = min(values), max(values)
    span = max(hi - lo, min_span)
    return lo - span * pad_frac, hi + span * pad_frac


# ----------------------------------------------------------------------------- charts
def chart_cv_top1(models, use, out, written, problems):
    ms = need(models, "cv_top1_mean", "cv_top1_by_model", problems)
    if not ms:
        return
    ms = sorted(ms, key=lambda m: m.m("cv_top1_mean"), reverse=True)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(ms) + 2.2))
    y = np.arange(len(ms))[::-1]
    for yi, m in zip(y, ms):
        mean, sd = m.m("cv_top1_mean"), m.m("cv_top1_std") or 0.0
        ax.errorbar(mean, yi, xerr=sd, fmt="o", ms=8, color=m.color, ecolor=m.color, elinewidth=2, capsize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{mean * 100:.2f}% ± {sd * 100:.2f}", (mean + sd, yi), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [m.name for m in ms])
    lo, hi = padded_limits([m.m("cv_top1_mean") - (m.m("cv_top1_std") or 0) for m in ms] +
                           [m.m("cv_top1_mean") + (m.m("cv_top1_std") or 0) for m in ms], pad_frac=0.35)
    ax.set_xlim(lo, hi)
    pct_axis(ax)
    ax.grid(axis="x")
    ax.set_xlabel("Grouped 5-fold CV Top-1 point accuracy (%) - mean ± 1 std across folds")
    ax.set_title("CV Top-1 by model (sorted high to low)")
    finish(fig, out / "cv_top1_by_model.png", source_note(models, use, ""), ms, written)


def chart_cv_per_fold(models, use, out, written, problems):
    ms = [m for m in models if m.folds]
    if not ms:
        warn("cv_top1_per_fold: no cv_folds files - chart skipped", problems)
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    folds = sorted({r["fold"] for m in ms for r in m.folds})
    width = 0.7 / max(len(ms), 1)
    for i, m in enumerate(ms):
        xs = [folds.index(r["fold"]) + (i - (len(ms) - 1) / 2) * width for r in m.folds]
        ax.plot(xs, [r["top1"] for r in m.folds], "o", ms=7, color=m.color, label=m.name,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_xticks(range(len(folds)), [f"Fold {f}" for f in folds])
    pct_axis(ax, "y")
    ax.grid(axis="y")
    ax.set_ylabel("Top-1 point accuracy on the held-out fold (%)")
    ax.set_title("CV Top-1 per fold: spread within a fold vs between folds", pad=26)
    ax.legend(loc="lower left", ncol=min(len(ms), 5), bbox_to_anchor=(0, 1.0), borderaxespad=0.3,
              handletextpad=0.3, columnspacing=1.2)
    suffix = "" if use == "primary" else "_all_trips"
    finish(fig, out / "cv_top1_per_fold.png", source_note(models, use, f"results/<model>/cv_folds{suffix}.csv"),
           ms, written)


def chart_grouped(models, use, out, written, problems, keys, labels, fname, title, ylabel, as_pct, src=""):
    ms = [m for m in models if all(m.m(k) is not None for k in keys)]
    for m in models:
        if m not in ms:
            warn(f"{fname}: {m.key} lacks one of {keys} - left out", problems)
    if not ms:
        return
    fig, ax = plt.subplots(figsize=(max(7.5, 1.35 * len(ms) + 2.5), 4.8))
    width = 0.36
    for i, m in enumerate(ms):
        for j, (k, lab) in enumerate(zip(keys, labels)):
            v = m.m(k)
            x = i + (j - (len(keys) - 1) / 2) * (width + 0.03)
            ax.bar(x, v, width, color=m.color if j == 0 else "none", edgecolor=m.color, linewidth=1.6,
                   hatch=None if j == 0 else "////", label=lab if i == 0 else None)
            ax.annotate(f"{v * 100:.2f}%" if as_pct else f"{v:.3f}", (x, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(range(len(ms)), [m.name for m in ms])
    if as_pct:
        pct_axis(ax, "y")
    ax.set_ylim(0, max(m.m(k) for m in ms for k in keys) * 1.12)
    ax.grid(axis="y")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=24)
    handles = [matplotlib.patches.Patch(facecolor=INK_3, edgecolor=INK_3, label=labels[0]),
               matplotlib.patches.Patch(facecolor="none", edgecolor=INK_3, hatch="////", label=labels[1])]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.3)
    finish(fig, out / fname, source_note(models, use, src), ms, written)


def chart_cv_vs_test(models, use, out, written, problems):
    ms = [m for m in models if m.m("cv_top1_mean") is not None and m.m("test_top1") is not None]
    if not ms:
        return
    ms = sorted(ms, key=lambda m: m.m("test_top1"), reverse=True)
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(ms) + 2.3))
    y = np.arange(len(ms))[::-1]
    for yi, m in zip(y, ms):
        cv, te = m.m("cv_top1_mean"), m.m("test_top1")
        ax.plot([cv, te], [yi, yi], color=m.color, lw=2, alpha=0.55, solid_capstyle="round")
        ax.plot(cv, yi, "o", ms=8, markerfacecolor=SURFACE, markeredgecolor=m.color, markeredgewidth=2)
        ax.plot(te, yi, "o", ms=8, color=m.color, markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.annotate(f"{(te - cv) * 100:+.2f} pp", (max(cv, te), yi), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [m.name for m in ms])
    lo, hi = padded_limits([m.m("cv_top1_mean") for m in ms] + [m.m("test_top1") for m in ms], pad_frac=0.3)
    ax.set_xlim(lo, hi)
    pct_axis(ax)
    ax.grid(axis="x")
    ax.set_xlabel("Top-1 point accuracy (%)")
    ax.set_title("Generalisation check: CV Top-1 vs student-test Top-1", pad=24)
    handles = [matplotlib.lines.Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK_2,
                                       markeredgewidth=2, ms=8, label="CV mean (training trips)"),
               matplotlib.lines.Line2D([], [], marker="o", ls="", color=INK_2, ms=8, label="Student-test")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.3)
    finish(fig, out / "cv_vs_test_gap.png",
           source_note(models, use, "") + " Label = test minus CV, in percentage points.", ms, written)


def chart_time_vs_accuracy(models, use, out, written, problems):
    panels = [("train_time_s", "Training time, full refit (s, log scale)"),
              ("inference_ms_per_point", "Inference time (ms per GPS point, log scale)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    used = []
    for ax, (key, xlabel) in zip(axes, panels):
        ms = [m for m in models if m.m(key) is not None and m.m(key) > 0 and m.m("test_top1") is not None]
        for m in models:
            if m not in ms:
                warn(f"time_vs_accuracy: {m.key} lacks a positive {key} or test_top1 - left out of that panel",
                     problems)
        for m in ms:
            ax.plot(m.m(key), m.m("test_top1"), "o", ms=9, color=m.color, markeredgecolor=SURFACE,
                    markeredgewidth=1.5)
            if m not in used:
                used.append(m)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="major")
        if ms:
            lo, hi = min(m.m(key) for m in ms), max(m.m(key) for m in ms)
            ax.set_xlim(lo / 2.5, hi * 6)
    if not used:
        plt.close(fig)
        return
    ys = [m.m("test_top1") for m in used]
    axes[0].set_ylim(*padded_limits(ys, pad_frac=0.4))
    for ax, (key, _) in zip(axes, panels):
        label_points(ax, [(m.m(key), m.m("test_top1"), m.name) for m in used
                          if m.m(key) is not None and m.m(key) > 0])
    pct_axis(axes[0], "y")
    axes[0].set_ylabel("Student-test Top-1 point accuracy (%)")
    axes[0].set_title("Accuracy vs training cost")
    axes[1].set_title("Accuracy vs inference cost")
    finish(fig, out / "time_vs_accuracy.png",
           source_note(models, use, "") + " Timings were recorded on the trainers' machines and are not a "
           "controlled benchmark.", used, written)


def chart_confusion(models, use, out, written, problems):
    ms = []
    for m in models:
        cm = (m.metrics or {}).get("test_confusion_matrix")
        try:
            arr = np.asarray(cm, dtype=float)
            if arr.shape == (2, 2) and arr.sum() > 0:
                ms.append((m, arr))
                continue
        except (TypeError, ValueError):
            pass
        warn(f"confusion_matrices: {m.key} has no 2x2 test_confusion_matrix - left out", problems)
    if not ms:
        return
    ncol = min(3, len(ms))
    nrow = math.ceil(len(ms) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow + 0.6), squeeze=False)
    for ax in axes.flat[len(ms):]:
        ax.axis("off")
    for ax, (m, cm) in zip(axes.flat, ms):
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=SEQ_CMAP, vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(norm):
            ax.text(c, r, f"{v * 100:.1f}%\n({int(cm[r, c]):,})", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.6 else INK)
        ax.set_xticks([0, 1], ["Pred. wrong arc", "Pred. right arc"], fontsize=8)
        ax.set_yticks([0, 1], ["True: wrong\n(label 0)", "True: right\n(label 1)"], fontsize=8)
        ax.set_title(m.name, color=INK, fontsize=10.5, pad=10)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.add_patch(matplotlib.patches.Rectangle((-0.5, -0.6), 2, 0.07, color=m.color, clip_on=False))
    fig.suptitle("Student-test confusion matrices (candidate level, threshold 0.5), row-normalised",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    finish(fig, out / "confusion_matrices.png",
           source_note(models, use, "") + " Cells: share of the true class (count in brackets).",
           [m for m, _ in ms], written, top=0.95)


def chart_primary_vs_all(models, out, written, problems):
    ms = [m for m in models if m.primary and m.all_trips]
    for m in models:
        if m not in ms:
            warn(f"primary_vs_all_trips: {m.key} lacks metrics.json or metrics_all_trips.json - left out", problems)
    if not ms:
        warn("primary_vs_all_trips: no model has both files - chart skipped", problems)
        return
    panels = [("cv_top1_mean", "CV Top-1 (training trips)"), ("test_top1", "Student-test Top-1")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.6 * len(ms) + 2.6), sharey=True)
    y = np.arange(len(ms))[::-1]
    for ax, (key, title) in zip(axes, panels):
        vals = []
        for yi, m in zip(y, ms):
            a, p = num(m.all_trips.get(key)), num(m.primary.get(key))
            if a is None or p is None:
                warn(f"primary_vs_all_trips: {m.key} lacks {key} in one file", problems)
                continue
            vals += [a, p]
            ax.plot([a, p], [yi, yi], color=m.color, lw=2, alpha=0.55)
            ax.plot(a, yi, "o", ms=8, markerfacecolor=SURFACE, markeredgecolor=m.color, markeredgewidth=2)
            ax.plot(p, yi, "o", ms=8, color=m.color, markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.annotate(f"{(p - a) * 100:+.2f} pp", (max(a, p), yi), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_2)
        if vals:
            ax.set_xlim(*padded_limits(vals, pad_frac=0.35))
        pct_axis(ax)
        ax.grid(axis="x")
        ax.set_xlabel(f"{title} (%)")
        ax.set_title(title)
    axes[0].set_yticks(y, [m.name for m in ms])
    ex = sorted({t for m in ms for t in ((m.primary.get("config") or {}).get("excluded_trips") or [])})
    handles = [matplotlib.lines.Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK_2,
                                       markeredgewidth=2, ms=8, label="All training trips"),
               matplotlib.lines.Line2D([], [], marker="o", ls="", color=INK_2, ms=8,
                                       label=f"Flagged trips excluded ({', '.join(map(str, ex)) or 'none recorded'})")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=2)
    finish(fig, out / "primary_vs_all_trips.png",
           "Source: results/<model>/metrics.json (excluded) vs metrics_all_trips.json (all). CV scores cover "
           "different trip populations, so the CV shift is not a like-for-like improvement; student-test trips "
           "are identical in both.", ms, written, top=0.9)


# ----------------------------------------------------------------------------- optuna
def find_trials(results_dir: Path, optuna_dir: Path | None) -> list[Path]:
    found = set()
    for base in [results_dir] + ([optuna_dir] if optuna_dir and optuna_dir.is_dir() else []):
        for pat in ("**/*trials*.csv", "**/optuna*/*.csv"):
            found.update(p for p in base.glob(pat) if p.is_file())
    return sorted(found)


def read_trials(path: Path, problems: list[str]) -> list[dict]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        warn(f"could not read {path}: {e}", problems)
        return []
    out = []
    for r in rows:
        if r.get("state", "COMPLETE") != "COMPLETE" or r.get("value") in (None, ""):
            continue
        try:
            out.append({"number": int(r["number"]), "value": float(r["value"])})
        except (KeyError, ValueError):
            continue
    return sorted(out, key=lambda t: t["number"])


def chart_optuna(results_dir, optuna_dir, models, out, written, problems):
    files = find_trials(results_dir, optuna_dir)
    if not files:
        warn("optuna: no trials CSV found under results/ or the optimisation folder - Optuna charts skipped",
             problems)
        return
    trials_path = files[0]
    if len(files) > 1:
        warn(f"optuna: several trials files {[rel(f) for f in files]}; using {rel(trials_path)}", problems)
    trials = read_trials(trials_path, problems)
    if not trials:
        warn(f"optuna: {rel(trials_path)} has no completed trials - Optuna charts skipped", problems)
        return
    n_total = sum(1 for _ in open(trials_path, encoding="utf-8-sig")) - 1
    summary = _read_json(trials_path.parent / "summary.json", problems)
    baseline = (summary or {}).get("baseline") or {}
    pairs = [(k, lab) for k, lab in (("cv_top1_mean", "CV Top-1"), ("test_top1", "Student-test Top-1"))
             if num(baseline.get(k)) is not None]
    tuned = {"cv_top1_mean": num((summary or {}).get("best_cv_top1")), "test_top1": num((summary or {}).get("test_top1"))}
    pairs = [(k, lab) for k, lab in pairs if tuned[k] is not None]

    tuned_model = (summary or {}).get("model")
    color = next((m.color for m in models if m.key == tuned_model), PALETTE[0])
    ncols = 2 if pairs else 1
    fig, axes = plt.subplots(1, ncols, figsize=(11.5 if pairs else 7.5, 4.6), squeeze=False,
                             gridspec_kw={"width_ratios": [1.6, 1] if pairs else [1]})
    ax = axes[0][0]
    xs = [t["number"] for t in trials]
    vs = [t["value"] for t in trials]
    best = np.maximum.accumulate(vs)
    ax.plot(xs, vs, "o", ms=6, color=color, alpha=0.55, label="Completed trial", markeredgecolor=SURFACE)
    ax.step(xs, best, where="post", color=INK, lw=2, label="Best so far")
    pct_axis(ax, "y", decimals=2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(axis="y")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Objective: mean grouped-CV Top-1 (%)")
    ax.set_title(f"Optuna search history ({len(trials)} completed of {n_total} trials)", pad=24)
    ax.legend(loc="upper left")
    if pairs:
        bx = axes[0][1]
        width = 0.36
        for i, (k, lab) in enumerate(pairs):
            for j, (v, name) in enumerate(((num(baseline[k]), "Untuned"), (tuned[k], "Tuned (best trial)"))):
                x = i + (j - 0.5) * (width + 0.03)
                bx.bar(x, v, width, color=color if j else "none", edgecolor=color, linewidth=1.6,
                       hatch=None if j else "////", label=name if i == 0 else None)
                bx.annotate(f"{v * 100:.2f}%", (x, v), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=7.5, color=INK_2)
        bx.set_xticks(range(len(pairs)), [lab for _, lab in pairs])
        bx.set_ylim(0, max(max(num(baseline[k]), tuned[k]) for k, _ in pairs) * 1.15)
        pct_axis(bx, "y")
        bx.grid(axis="y")
        bx.set_ylabel("Top-1 point accuracy (%)")
        bx.set_title(f"{DISPLAY.get(tuned_model, tuned_model)}: untuned vs tuned", pad=24)
        bx.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.3)
    src = f"Source: {rel(trials_path)}" + (f" and {rel(trials_path.parent / 'summary.json')}" if pairs else "")
    finish(fig, out / "optuna_history.png", src + ". Failed or pruned trials are not plotted.", [], written)
    warn("optuna_importance: skipped - parameter importance needs the Optuna study's search distributions, "
         "which the trials CSV does not store (no re-run allowed)", problems)


# ----------------------------------------------------------------------------- table
TABLE_COLUMNS = ["model", "family", "cv_top1_mean", "cv_top1_std", "cv_macro_f1_mean", "cv_macro_f1_std",
                 "cv_log_loss_mean", "train_top1", "train_macro_f1", "test_top1", "test_macro_f1", "test_log_loss",
                 "test_accuracy", "test_precision", "test_recall", "test_candidate_miss_rate",
                 "test_route_continuity_argmax", "train_time_s", "inference_ms_per_point", "dataset_version",
                 "excluded_trips", "source"]


def write_tables(models, use, out, written):
    rows = []
    for m in models:
        d = m.metrics or {}
        cfg = d.get("config") or {}
        rs = m.run_summary or {}
        row = {k: d.get(k) for k in TABLE_COLUMNS if k in METRIC_KEYS}
        row["model"] = m.key
        row["dataset_version"] = rs.get("dataset_version", cfg.get("dataset_version"))
        row["excluded_trips"] = rs.get("excluded_trips", cfg.get("excluded_trips"))
        row["source"] = m.source
        rows.append(row)
    rows.sort(key=lambda r: -(num(r.get("cv_top1_mean")) or -1))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else json.dumps(r[k]) if isinstance(r[k], list) else r[k])
                        for k in TABLE_COLUMNS})
    written.append(out / "summary_table.csv")

    def pct(v):
        return "N/A" if num(v) is None else f"{num(v) * 100:.2f}"

    def pm(mean, sd):
        return "N/A" if num(mean) is None else f"{num(mean) * 100:.2f} ± {(num(sd) or 0) * 100:.2f}"

    def plain(v, fmt):
        return "N/A" if num(v) is None else format(num(v), fmt)

    header = ["Model", "Family", "CV Top-1 (%)", "CV macro-F1 (%)", "CV log loss", "Train Top-1 (%)",
              "Test Top-1 (%)", "Test macro-F1 (%)", "Test log loss", "Test precision (%)", "Test recall (%)",
              "Candidate miss rate (%)", "Route continuity (argmax)", "Train time (s)", "Inference (ms/point)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * 2 + ["---:"] * (len(header) - 2)) + "|"]
    for r in rows:
        name = DISPLAY.get(r["model"], r["model"])
        lines.append("| " + " | ".join([
            name, str(r.get("family") or "N/A"), pm(r.get("cv_top1_mean"), r.get("cv_top1_std")),
            pm(r.get("cv_macro_f1_mean"), r.get("cv_macro_f1_std")), plain(r.get("cv_log_loss_mean"), ".4f"),
            pct(r.get("train_top1")), pct(r.get("test_top1")), pct(r.get("test_macro_f1")),
            plain(r.get("test_log_loss"), ".4f"), pct(r.get("test_precision")), pct(r.get("test_recall")),
            pct(r.get("test_candidate_miss_rate")), pct(r.get("test_route_continuity_argmax")),
            plain(r.get("train_time_s"), ".2f"), plain(r.get("inference_ms_per_point"), ".4f")]) + " |")
    run = "primary runs (flagged trips excluded)" if use == "primary" else "all-trips runs"
    md = [f"# Model comparison ({run})", "",
          f"Generated by `reports/charts/make_charts.py` from `results/<model>/metrics"
          f"{'' if use == 'primary' else '_all_trips'}.json`. Do not edit by hand.", "",
          "Mean ± std across the 5 trip-grouped CV folds. N/A = not recorded in the results file.", ""] + lines
    notes = [n for m in models for n in m.notes]
    if notes:
        md += ["", "**Comparability warnings:**", ""] + [f"- {n}" for n in notes]
    (out / "summary_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    written.append(out / "summary_table.md")


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Model-comparison charts from saved results (no retraining).")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="reports/figures/model_comparison")
    ap.add_argument("--use", choices=["primary", "all_trips"], default="primary",
                    help="which run to chart: metrics.json (primary, flagged trips excluded) or metrics_all_trips.json")
    ap.add_argument("--optuna-dir", default="optimisation",
                    help="extra folder searched for an Optuna trials CSV (read-only); results/ is always searched")
    args = ap.parse_args(argv)

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else REPO_ROOT / p

    results_dir, out = resolve(args.results_dir), resolve(args.out_dir)
    if args.use == "all_trips":
        out = out / "all_trips"
    optuna_dir = resolve(args.optuna_dir) if args.optuna_dir else None
    problems: list[str] = []
    if not results_dir.is_dir():
        print(f"ERROR: results folder not found: {results_dir}")
        return 1

    models = discover(results_dir, args.use, problems)
    print(f"Models found ({args.use}): {', '.join(m.key for m in models) or 'none'}")
    if not models:
        print("ERROR: no model with a metrics file - nothing to chart.")
        return 1

    written: list[Path] = []
    chart_cv_top1(models, args.use, out, written, problems)
    chart_cv_per_fold(models, args.use, out, written, problems)
    chart_grouped(models, args.use, out, written, problems, ["test_top1", "test_macro_f1"],
                  ["Test Top-1 (point level)", "Test macro-F1 (candidate level)"], "test_top1_and_macro_f1.png",
                  "Student-test Top-1 and macro-F1 by model", "Score (%)", True)
    chart_cv_vs_test(models, args.use, out, written, problems)
    chart_time_vs_accuracy(models, args.use, out, written, problems)
    chart_confusion(models, args.use, out, written, problems)
    chart_grouped(models, args.use, out, written, problems, ["test_log_loss", "cv_log_loss_mean"],
                  ["Student-test log loss", "CV log loss (mean of folds)"], "log_loss_by_model.png",
                  "Candidate log loss by model (lower is better)", "Binary log loss (nats)", False)
    if args.use == "primary":
        chart_primary_vs_all(models, out, written, problems)
    if args.use == "primary":                     # tuning used the primary (flagged-excluded) policy
        chart_optuna(results_dir, optuna_dir, models, out, written, problems)
    write_tables(models, args.use, out, written)

    print("Files written:")
    for p in written:
        print(f"  {rel(p)}")
    print(f"{len(problems)} warning(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
