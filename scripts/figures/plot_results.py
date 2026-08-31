#!/usr/bin/env python
"""Paper figures, built from the metric accessors in src/metrics.py.

  1. plots/perf_mae_spearman.pdf   -- MAE (top row) + Spearman (bottom row), one subplot per task,
                                      one bar per model, task labelled with (n = # test samples).
  2. plots/interp_heldout.pdf      -- interpretability on the held-out set: one column per task,
                                      rows = faithfulness / exactness gap / leakage, bars per
                                      attribution method; columns labelled with (n = # pairs).
  3. plots/attr_timing_ms_per_molecule.pdf + plots/ig_steps_tradeoff.pdf -- attribution compute cost.

The main figures cover the 5 largest-test-n tasks; the "_appendix" variants cover the rest.
Error bars are mean +/- std over the init seeds.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json

from src.metrics import (  # noqa: E402
    MODELS,
    SEEDS,
    TASKS,
    acc,
    attr_timing,
    clean,
    faith_r,
    mp_cell,
    mp_count,
    single_gap,
)


# Figure-only relabel: rename the sum-mean arm to "MolLedger (no context)" and place it
# immediately before "Unanchored MolLedger". metrics.MODELS itself stays untouched.
def _perf_models(models):
    relabelled = [
        (k, "MolLedger (no context)" if k == "summean_best" else lab) for k, lab in models
    ]
    summean = [m for m in relabelled if m[0] == "summean_best"]
    rest = [m for m in relabelled if m[0] != "summean_best"]
    keys = [k for k, _ in rest]
    at = keys.index("additive_none")
    return rest[:at] + summean + rest[at:]


MODELS = _perf_models(MODELS)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["axes.grid.axis"] = "y"  # horizontal gridlines only (whitegrid draws both by default)
PLOTS = Path(__file__).resolve().parents[2] / "plots"
TESTN = json.loads(
    (Path(__file__).resolve().parents[2] / "runs/test_sample_counts.json").read_text()
)


def pretty_task(task):
    """Human-readable task name: drop underscores, Title Case (e.g. clint_mouse_liver -> Clint
    Mouse Liver). Fix up chemistry casings that Title Case mangles (LogD, LogP)."""
    name = task.replace("_", " ").title()
    return name.replace("Logd", "LogD").replace("Logp", "LogP")


def _n(v):
    """Sample size formatted with thousands separators (e.g. 34884 -> '34,884'); non-ints pass through."""
    return f"{v:,}" if isinstance(v, int) else str(v)


# color-blind-safe Okabe-Ito hues. Fallback pool for any label not in METHOD_COLORS.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]

# Canonical method -> colour, keyed by legend label, so a method keeps one colour across every
# figure.
METHOD_COLORS = {
    "Anchored MolLedger": "#0072B2",  # blue           -- shared
    "Unanchored MolLedger": "#E69F00",  # orange         -- shared
    "MolLedger (no context)": "#785EF0",  # violet -- shared (accuracy<->leakage tradeoff bar)
    "Anchored GNAN": "#000000",  # black          -- shared
    "Unanchored GNAN": "#CC79A7",  # reddish-purple -- shared
    "LigandFormer": "#F0E442",  # yellow         -- shared
    "Pooled GNN": "#009E73",  # green          -- perf only
    "Pooled + descriptors": "#56B4E9",  # sky            -- perf only
    "GBT": "#D55E00",  # vermillion     -- perf only
    "IG": "#009E73",  # green          -- interp only (disjoint from Pooled GNN)
    "Grad-CAM": "#56B4E9",  # sky            -- interp only (disjoint from Pooled+desc)
    "LIME": "#D55E00",  # vermillion     -- interp only (disjoint from GBT)
    "WISP": "#8B4513",  # brown          -- interp only
}


def _colors(labels):
    """Canonical colour per legend label; positional PALETTE fallback (wraps via i % len) for labels
    not in the map."""
    return [METHOD_COLORS.get(lab, PALETTE[i % len(PALETTE)]) for i, lab in enumerate(labels)]


# interpretability methods, bar order: native per-atom read-outs first (MolLedger pair, GNAN pair,
# LigandFormer), then the post-hoc pooled-head attribution baselines (IG, Grad-CAM, LIME).
INTERP_METHODS = [
    ("ours_best", "Anchored MolLedger"),
    ("summean_best", "MolLedger (no context)"),  # accuracy<->leakage tradeoff ablation
    ("ours_none", "Unanchored MolLedger"),
    ("gnan_best", "Anchored GNAN"),
    ("gnan", "Unanchored GNAN"),
    ("ligandformer", "LigandFormer"),  # self-attention; leakage/exactness-gap are None for this arm
    ("ig_pool_zeros", "IG"),
    ("gradcam_pool", "Grad-CAM"),
    ("lime_pool", "LIME"),
    ("wisp_pool", "WISP"),  # element-substitution occlusion: no completeness axiom
]
# columns: (key, source, pretty label, lower_is_better). Direction shown by an arrow next to the
# y-axis label ($\downarrow$ = lower better, $\uparrow$ = higher better).
INTERP_COLS = [
    ("faith", "faith", "Faithfulness", False),
    # single-molecule exactness gap over the held-out test split (source "single_gap"), not the
    # matched-pair completeness gap.
    ("gap", "single_gap", "Exactness gap", True),
    ("leakage", "mp", "Leakage", True),
]
DIR_ARROW = {True: r"$\downarrow$", False: r"$\uparrow$"}  # lower-better / higher-better

# Each performance model maps to the mp arm carrying the same checkpoint's predictions, for the
# matched-pair delta MAE row. Pooled GNN shares its checkpoint with the other pooled attribution
# arms, so ig_pool_zeros is the canonical read for all of them.
PERF_TO_MP_ARM = {
    "additive_best": "ours_best",
    "additive_none": "ours_none",
    "summean_best": "summean_best",
    "pooled_none": "ig_pool_zeros",
    "gnan_best": "gnan_best",
    "gnan": "gnan",
    "ligandformer": "ligandformer",
    # non-attribution baselines: their own predicted-delta dumps (eval_pair_delta_baselines.py)
    "pooled_desc": "pooled_desc",
    "gbt_227": "gbt_227",
}


LOG_FLOOR = 1e-3  # log-axis bottom for the exactness-gap row: below IG's gap, above exact-arm round-off


def _bar(
    ax,
    labels,
    means,
    stds,
    title=None,
    ylabel=None,
    annotate_zero=False,
    na_mask=None,
    empty_label="n/a",
    ylabel_y=None,
    floor_zero=False,
    logscale=False,
):
    x = np.arange(len(labels))
    ms = [m if m is not None else np.nan for m in means]
    es = [s if s is not None else 0.0 for s in stds]
    if logscale:
        # Suppress error whiskers on sub-floor bars (their lower end goes non-positive on a log axis).
        es = [0.0 if (not np.isnan(m) and m < LOG_FLOOR) else e for m, e in zip(ms, es)]
    ax.bar(x, ms, yerr=es, color=_colors(labels), capsize=2, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([])  # method identity carried by the shared colour legend
    # Structurally not-computable cells get a full-column diagonal hatch instead of an empty (0) bar.
    for xi, na in zip(x, na_mask or [False] * len(x)):
        if na:
            ax.axvspan(
                xi - 0.4,
                xi + 0.4,
                facecolor="none",
                hatch="////",
                edgecolor="0.6",
                linewidth=0.0,
                zorder=0.5,
            )
    if all(np.isnan(m) for m in ms):  # e.g. a task with no faithfulness reference
        ax.set_ylim(0, 1)
        ax.set_yticks([])  # empty cell: drop ticks/gridlines so only the label reads
        ax.grid(False)
        ax.text(
            0.5,
            0.5,
            empty_label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="black",
        )
    else:
        if logscale:
            ax.set_yscale("log")
            finite = [m for m in ms if not np.isnan(m) and m > 0]
            ax.set_ylim(LOG_FLOOR, (max(finite) * 4 if finite else 1.0))
        if annotate_zero:  # write "0" above vanishing bars so they don't read as missing data
            zero_y = LOG_FLOOR if logscale else 0
            for xi, m in zip(x, ms):
                if not np.isnan(m) and abs(m) < 1e-4:
                    ax.annotate(
                        "0",
                        (xi, zero_y),
                        textcoords="offset points",
                        xytext=(0, 2),
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color="black",
                    )
    if title:
        ax.set_title(title, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
        if ylabel_y is not None:  # nudge the label along the axis, keeping x aligned with other rows
            x = ax.yaxis.label.get_position()[0]
            ax.yaxis.label.set_position((x, ylabel_y))
    if floor_zero and not logscale and not all(np.isnan(m) for m in ms):
        # Pin the y-axis floor at 0; a metric dipping below 0 just clips there instead of pushing
        # the whole axis negative.
        ax.set_ylim(bottom=0)
        # A wholly-negative bar clips to zero height and reads as missing; label its signed value.
        for xi, m in zip(np.arange(len(ms)), ms):
            if not np.isnan(m) and m < 0:
                ax.annotate(
                    f"-{abs(m):.2f}".replace("-0.", "-."),
                    (xi - 0.4, 0),
                    textcoords="offset points",
                    xytext=(1, 2),
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    color="black",
                )
    ax.tick_params(axis="y", labelsize=9)
    sns.despine(ax=ax)


def _legend(fig, labels, ncol, y=0.998):
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=col, edgecolor="black", linewidth=0.4, label=lab)
        for lab, col in zip(labels, _colors(labels))
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=ncol,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, y),
    )


def _title_legend(fig, figh, suptitle, labels, ncol):
    """Height-aware title+legend so they don't collide on tall multi-row figures. Returns the
    tight_layout top fraction. The title, legend, and first plot row are packed into a tight
    ~0.50in strip (no dead space between them)."""
    nrow = int(np.ceil(len(labels) / ncol))  # legend rows, so we reserve the right amount of strip
    fig.suptitle(suptitle, fontsize=15, y=1.0 - 0.15 / figh)
    _legend(fig, labels, ncol, y=1.0 - 0.32 / figh)
    # tightened so column titles sit just under the legend; each extra legend row adds ~0.17in.
    return 1.0 - (0.20 + 0.17 * nrow) / figh


TASKS_BY_TESTN = sorted(TASKS, key=lambda t: -(TESTN.get(t) or 0))  # largest test-n first


# ------------------------------------------------------------------- Plot 1: performance
def plot_performance(tasks, fname):
    """MAE (top) + Spearman (middle) + matched-pair delta MAE (bottom), one column per task.
    Delta-MAE is the predicted-vs-measured pair delta, pooled over held-out matched pairs
    (mp arm per PERF_TO_MP_ARM)."""
    metrics = [
        ("mae", "MAE"),
        ("spearman", "Spearman"),
        ("delta_mae", r"Matched-pair $\Delta$ MAE"),
    ]
    nrow, ncol = len(metrics), len(tasks)
    figh = 2.0 * nrow  # matches the interpretability figure's aspect
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, figh), squeeze=False)
    short = [lab for _, lab in MODELS]
    for r, (mkey, mname) in enumerate(metrics):
        for c, task in enumerate(tasks):
            ax = axes[r][c]
            means, stds = [], []
            for mk, _ in MODELS:
                if mkey == "delta_mae":
                    arm = PERF_TO_MP_ARM.get(mk)
                    cell = mp_cell("held_out", "all", task, arm, "mae_delta") if arm else None
                    means.append(cell["mean"] if cell else None)
                    stds.append(cell["sd"] if cell else None)
                else:
                    vals = clean(
                        [acc(s, mk, task).get(mkey) if acc(s, mk, task) else None for s in SEEDS]
                    )
                    means.append(np.mean(vals) if vals else None)
                    stds.append(np.std(vals) if vals else None)
            # n = held-out test molecules; m = pooled held-out matched pairs (Delta-MAE row's sample)
            title = (
                (
                    f"{pretty_task(task)}\n(n={_n(TESTN.get(task, '?'))}, "
                    f"m={_n(mp_count('held_out', 'all', task))})"
                )
                if r == 0
                else None
            )
            _bar(ax, short, means, stds, title=title, ylabel=mname if c == 0 else None)
            if mkey == "spearman":
                ax.set_ylim(0, 1)
    top = _title_legend(fig, figh, "Model performance", short, ncol=int(np.ceil(len(short) / 2)))
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(rect=(0, 0, 1, top))
    out = PLOTS / fname
    fig.savefig(out, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out


# ------------------------------------------------------------------- interp helpers
def _interp_cell(coll, cls, task, arm, field, source):
    """(mean, std) for one method on one property/metric."""
    if source == "faith":
        vals = clean([faith_r(s, arm, task) for s in SEEDS])  # signed r (anchor direction)
        return (np.mean(vals), np.std(vals)) if vals else (None, None)
    if source == "single_gap":
        vals = clean([single_gap(s, arm, task) for s in SEEDS])
        return (np.mean(vals), np.std(vals)) if vals else (None, None)
    c = mp_cell(coll, cls, task, arm, field)
    return (c["mean"], c["sd"]) if c else (None, None)


def _interp_grid(props, coll, fname, suptitle, log_gap=False):
    """props = [(cls, task), ...] -> one column each; the 4 interpretability metrics are the rows.
    Bars per method; column labelled with the property and (n = # matched pairs).
    log_gap=True renders only the exactness-gap row on a log y-axis; every other row stays linear."""
    labels = [lab for _, lab in INTERP_METHODS]
    nrow, ncol = len(INTERP_COLS), len(props)
    figh = 2.0 * nrow  # matches plot_performance's per-row height
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, figh), squeeze=False)
    for c, (cls, task) in enumerate(props):
        m_pairs = mp_count(coll, cls, task)  # matched pairs -> leakage/gap sample size
        n_mol = TESTN.get(task)  # held-out test molecules -> faithfulness sample size
        for r, (field, source, clabel, lower) in enumerate(INTERP_COLS):
            ax = axes[r][c]
            means, stds = [], []
            for arm, _ in INTERP_METHODS:
                m, s = _interp_cell(coll, cls, task, arm, field, source)
                means.append(m)
                stds.append(s)
            # LigandFormer's leakage/exactness-gap are non-commensurable (attention read-out): mark
            # those cells as not-computable (hatch) rather than letting the empty slot read as 0.
            na_mask = [
                arm == "ligandformer" and field in ("leakage", "gap") for arm, _ in INTERP_METHODS
            ]
            # anchor-less tasks have no faithfulness reference or single-molecule gap: label those
            # empty cells "no anchor" rather than the generic "n/a".
            empty_label = (
                "no anchor"
                if (source in ("faith", "single_gap") and not _has_faith(task))
                else "n/a"
            )
            row_log = log_gap and field == "gap"  # only the exactness-gap row goes log
            ylab = None
            if c == 0:
                ylab = f"{clabel}{DIR_ARROW[lower]}" + ("\n(log scale)" if row_log else "")
            _bar(
                ax,
                labels,
                means,
                stds,
                title=(f"{pretty_task(task)}\n(n={_n(n_mol)}, m={_n(m_pairs)})")
                if r == 0
                else None,
                ylabel=ylab,
                ylabel_y=0.55 if (c == 0 and field == "faith") else None,  # lift Faithfulness a touch
                annotate_zero=(field in ("gap", "leakage")),
                na_mask=na_mask,
                empty_label=empty_label,
                floor_zero=True,
                logscale=row_log,
            )
    top = _title_legend(fig, figh, suptitle, labels, ncol=int(np.ceil(len(labels) / 2)))
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(rect=(0, 0, 1, top))
    out = PLOTS / fname
    fig.savefig(out, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out


# ------------------------------------------------------------------- Plot 2 & 3
def _has_faith(task):
    """True if the task has a Crippen/TPSA faithfulness reference."""
    return any(faith_r(s, "ours_best", task) is not None for s in SEEDS)


def plot_interp_heldout_main():
    """Main figure: one column per task over all held-out matched pairs pooled across classes.
    Columns = the 5 largest-test-n tasks, same set/order as the performance main figure."""
    props = [("all", t) for t in TASKS_BY_TESTN[:5]]
    return _interp_grid(props, "held_out", "interp_heldout.pdf", "Interpretability")


def plot_interp_heldout_full():
    """Appendix: the remaining tasks, same layout and order as the performance appendix."""
    props = [("all", t) for t in TASKS_BY_TESTN[5:]]
    return _interp_grid(props, "held_out", "interp_heldout_appendix.pdf", "Interpretability")


# ------------------------------------------------------------------- IG exactness/timing tradeoff
def plot_ig_steps_tradeoff(fname="ig_steps_tradeoff.pdf", src="runs/ig_steps_sweep_val.json"):
    """IG exactness gap vs wall-time, from the validation-split sweep (scripts/score/ig_steps_sweep.py).
    Single panel, log-log: each point is one IG path-step count; the 256-step operating point is
    circled. Skipped (returns None) if the sweep JSON is absent."""
    from matplotlib.ticker import LogLocator

    path = Path(__file__).resolve().parents[2] / src
    if not path.exists():
        print(f"skip ig_steps_tradeoff: {src} not found (run scripts/score/ig_steps_sweep.py)")
        return None
    d = json.loads(path.read_text())
    steps = d["steps"]
    seeds = [str(s) for s in d["seeds"]]
    gap_m = np.array([[d["per_seed"][s][str(k)]["gap_mean"] for s in seeds] for k in steps])
    ms_m = np.array([[d["per_seed"][s][str(k)]["ms_per_mol"] for s in seeds] for k in steps])
    gap, gap_sd = gap_m.mean(1), gap_m.std(1)
    ms = ms_m.mean(1)
    col = METHOD_COLORS["IG"]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.errorbar(ms, gap, yerr=gap_sd, marker="o", color=col, capsize=3, lw=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=tuple(np.arange(0.2, 1.0, 0.1)), numticks=15)
    )
    ax.grid(True, which="major", axis="y", ls="-", lw=0.6, color="0.85", zorder=0)
    ax.grid(True, which="minor", axis="y", ls=":", lw=0.5, color="0.9", zorder=0)
    i256 = steps.index(256) if 256 in steps else None
    for i, k in enumerate(steps):
        if i == i256:
            continue
        ax.annotate(
            f"{k}",
            (ms[i], gap[i]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8.5,
            ha="left",
            va="bottom",
            color="0.25",
        )
    if i256 is not None:
        ax.scatter(
            [ms[i256]],
            [gap[i256]],
            s=110,
            facecolor="none",
            edgecolor="black",
            linewidth=1.6,
            zorder=4,
        )
        ax.annotate(
            f"{steps[i256]} steps\n(selected)",
            (ms[i256], gap[i256]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            ha="left",
            va="bottom",
            fontweight="bold",
        )
    ax.set_xlabel("Wall-time (ms/mol)", fontsize=12)
    ax.set_ylabel(r"Exactness gap $\downarrow$", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title("Integrated gradients # steps trade-off", fontsize=15, pad=14)
    sns.despine(ax=ax)
    fig.tight_layout()
    out = PLOTS / fname
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out


# ------------------------------------------------------------------- Plot 4: attribution timing
TIMING_COLORS = {
    "MolLedger (additive)": METHOD_COLORS["Anchored MolLedger"],
    "IG-zero (pooled)": METHOD_COLORS["IG"],
    "Grad-CAM (pooled)": METHOD_COLORS["Grad-CAM"],
    "LIME (pooled)": METHOD_COLORS["LIME"],
    "WISP (pooled)": METHOD_COLORS["WISP"],
    "GNAN (additive)": METHOD_COLORS["Unanchored GNAN"],
    "LigandFormer (attention)": METHOD_COLORS["LigandFormer"],
}


def plot_attr_timing():
    """Per-molecule attribution wall-time, one bar per method, log scale (costs span ~100x)."""
    from matplotlib.ticker import LogLocator

    rows = sorted(attr_timing(), key=lambda r: r[1])  # cheapest left
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    sds = [r[2] for r in rows]
    base = min(means)
    colors = [TIMING_COLORS.get(lab, PALETTE[i]) for i, lab in enumerate(labels)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sds, color=colors, capsize=3, edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 150)  # WISP is ~48 ms/mol; leave headroom above its bar for the value label
    # log-scale y grid lines (major decades + minor decade subdivisions) -- matches the IG-steps plot and
    # reminds the reader the axis is logarithmic now that the ylabel no longer says so.
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=tuple(np.arange(0.2, 1.0, 0.1)), numticks=15)
    )
    ax.grid(True, which="major", axis="y", ls="-", lw=0.6, color="0.85", zorder=0)
    ax.grid(True, which="minor", axis="y", ls=":", lw=0.5, color="0.9", zorder=0)
    ax.set_xticks(x)
    # strip every method qualifier -- (pooled)/(additive)/(attention) -- IG-zero displays as IG
    disp = [lab.replace("IG-zero", "IG").split(" (")[0] for lab in labels]
    ax.set_xticklabels(disp, fontsize=11)
    ax.set_ylabel("Wall-time (ms/mol)", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title("Attribution compute cost per molecule", fontsize=15)
    for xi, m, s in zip(x, means, sds):
        ax.annotate(
            f"{m:.3g} ms\n({m / base:.1f}x)",
            (xi, m + s),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    sns.despine(ax=ax)
    fig.tight_layout()
    out = PLOTS / "attr_timing_ms_per_molecule.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out


if __name__ == "__main__":
    PLOTS.mkdir(parents=True, exist_ok=True)
    outs = [
        plot_attr_timing(),
        plot_performance(TASKS_BY_TESTN[:5], "perf_mae_spearman.pdf"),  # paper: 5 largest
        plot_performance(TASKS_BY_TESTN[5:], "perf_mae_spearman_appendix.pdf"),  # appendix: rest
        plot_interp_heldout_main(),  # paper: 5 columns
        plot_interp_heldout_full(),  # appendix: all tasks
        plot_ig_steps_tradeoff(),  # IG gap vs steps/time
    ]
    for f in outs:
        print("wrote", f)
