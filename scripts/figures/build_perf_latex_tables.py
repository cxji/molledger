"""Per-metric LaTeX tables for model performance: one row per model, one column per task, entries
mean (std) over 3 seeds, best-in-column bolded. One table per metric, tasks split into two column-
blocks so the table is not too wide.

The three metrics are the rows of the performance figure (perf_mae_spearman.pdf):
  * MAE                    -- test-split MAE in each task's native units (lower better).
  * Spearman rho           -- test-split rank correlation (higher better).
  * Matched-pair Delta MAE -- |predicted pair delta - measured pair delta| (lower better).

Model set + column order match the performance figure (src.metrics.MODELS; columns ordered by
test-set size, largest first). Data helpers (acc / mp_cell / MODELS / SEEDS / TASKS) come from
src.metrics so the numbers match the figure.

Usage:
    python scripts/figures/build_perf_latex_tables.py     # -> runs/perf_tables.tex (+ stdout)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.metrics import MODELS, SEEDS, TASKS, acc, clean, mp_cell  # noqa: E402

OUT_TEX = REPO / "runs/perf_tables.tex"

# each performance model -> the matched-pair arm(s) carrying the same checkpoint's predicted deltas.
# the posthoc attribution arms are applied to the same pooled GNN and have identical delta MAE,
# so it falls back to wherever that metric is stored
PERF_TO_MP_ARM = {
    "additive_best": ["ours_best"],
    "additive_none": ["ours_none"],
    "summean_best": ["summean_best"],
    "pooled_none": ["ig_pool", "gradcam_pool", "lime_pool", "wisp_pool"],
    "gnan_best": ["gnan_best"],
    "gnan": ["gnan"],
    "ligandformer": ["ligandformer"],
    "pooled_desc": ["pooled_desc"],
    "gbt_227": ["gbt_227"],
}

# (display, key, higher_is_better)
METRICS = [
    ("MAE", "mae", False),
    (r"Spearman $\rho$", "spearman", True),
    (r"Matched-pair $\Delta$ MAE", "delta_mae", False),
]

TASK_LABEL = {
    "logd": "LogD",
    "kinetic_solubility": "Kin.\\ sol.",
    "aqueous_solubility": "Aq.\\ sol.",
    "clint_mouse_liver": "CL$_{int}$ mouse",
    "clint_human_liver": "CL$_{int}$ human",
    "caco2_efflux_ratio": "Caco2 efflux",
    "caco2_papp_ab": "Caco2 P$_{app}$",
    "ppb_mouse_plasma": "PPB plasma",
    "ppb_mouse_brain": "PPB brain",
    "ppb_mouse_muscle": "PPB muscle",
    "half_life": "Half-life",
}

# per-task decimals for the MAE table; tasks omitted fall back to the adaptive rule in col_decimals().
MAE_DECIMALS = {
    "logd": 2,
    "caco2_papp_ab": 2,
    "caco2_efflux_ratio": 2,
    "aqueous_solubility": 2,
    "ppb_mouse_brain": 2,
    "ppb_mouse_muscle": 2,
    "clint_human_liver": 1,
}


def cell(mkey, mk, task):
    """(mean, sd) over seeds for one model/metric/task, or None. MAE/Spearman from the test-split
    accuracy dumps; Delta-MAE from the pre-aggregated matched-pair cell."""
    if mkey == "delta_mae":
        c = None  # first arm (preference order) with a cell; siblings carry identical delta MAE
        for arm in PERF_TO_MP_ARM.get(mk, []):
            c = mp_cell("held_out", "all", task, arm, "mae_delta")
            if c and c.get("mean") is not None:
                break
            c = None
        if not c:
            return None
        return float(c["mean"]), float(c.get("sd") or 0.0)
    vals = clean([acc(s, mk, task).get(mkey) if acc(s, mk, task) else None for s in SEEDS])
    return (float(np.mean(vals)), float(np.std(vals))) if vals else None


def col_decimals(means, spearman):
    """Decimals for a task column: Spearman is fixed 2; native-unit columns pick decimals from the
    column's magnitude so each prints ~3 significant figures."""
    if spearman:
        return 2
    mags = [abs(m) for m in means if m is not None]
    if not mags:
        return 2
    top = max(mags)
    return 1 if top >= 100 else (2 if top >= 10 else 3)


def fmt(cellval, dec):
    if cellval is None:
        return "--"
    m, sd = cellval
    return f"{m:.{dec}f} ({sd:.{dec}f})"


def best_labels(table, tasks, higher):
    """Per task, model labels whose mean is best, within a 0.5% display-tie tolerance."""
    best = {}
    for t in tasks:
        vals = [(lab, table[lab][t][0]) for lab in table if table[lab][t] is not None]
        if not vals:
            best[t] = set()
            continue
        target = max(v for _, v in vals) if higher else min(v for _, v in vals)
        tol = 1e-9 + 0.005 * abs(target)
        best[t] = {lab for lab, v in vals if abs(v - target) <= tol}
    return best


def render_metric(name, table, tasks, higher, mkey=None):
    """One LaTeX table for a metric; tasks split into two roughly equal column-blocks."""
    half = (len(tasks) + 1) // 2
    blocks = [tasks[:half], tasks[half:]]
    best = best_labels(table, tasks, higher)
    spearman = name.startswith("Spearman")
    slug = "".join(c for c in name.split()[0].lower() if c.isalnum())
    if mkey == "delta_mae":
        overrides = {t: 2 for t in tasks}
    elif mkey == "mae":
        overrides = MAE_DECIMALS
    else:
        overrides = {}
    dec = {
        t: overrides.get(
            t,
            col_decimals(
                [table[lab][t][0] for lab in table if table[lab][t] is not None], spearman
            ),
        )
        for t in tasks
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{name}.}}",
        rf"\label{{tab:perf-{slug}}}",
    ]
    for bi, block in enumerate(blocks):
        lines.append(r"\begin{tabular}{l" + "c" * len(block) + "}")
        lines.append(r"\toprule")
        lines.append("Model & " + " & ".join(TASK_LABEL.get(t, t) for t in block) + r" \\")
        lines.append(r"\midrule")
        for _, lab in MODELS:
            cells = []
            for t in block:
                s = fmt(table[lab][t], dec[t])
                if lab in best[t] and s != "--":
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            lines.append(lab + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if bi == 0 and len(blocks) > 1:
            lines.append(r"\\[4pt]")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_TEX))
    args = ap.parse_args()
    # order columns by test-set size (largest first), matching the plots + interp LaTeX tables
    testn = json.loads((REPO / "runs/test_sample_counts.json").read_text())
    tasks = sorted(TASKS, key=lambda t: -(testn.get(t) or 0))

    chunks = []
    for name, mkey, higher in METRICS:
        table = {lab: {t: cell(mkey, mk, t) for t in tasks} for mk, lab in MODELS}
        n_filled = sum(1 for lab in table for t in tasks if table[lab][t] is not None)
        chunks.append(render_metric(name, table, tasks, higher, mkey))
        print(f"[{name}] cells filled: {n_filled}/{len(MODELS) * len(tasks)}")
    Path(args.out).write_text("\n\n".join(chunks) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
