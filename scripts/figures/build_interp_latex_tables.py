"""Per-metric LaTeX tables: one row per model/interp method, one column per task, entries mean (std),
best-in-column bolded. One table per interpretability metric, tasks split into two column-blocks so the
table is not too wide.

Model set + task set match the interp_heldout figure.
Sources, all on disk:
  * Exactness gap       -- runs/test_split_tables.json
  * Leakage / Explained Delta-MAE
                          -- runs/matched_pair_tables.json  results[held_out][all][task][arm]
                             (fields: leakage / dsub_mae)
  * Non-circular AUROC  -- runs/noncircular_concordance.json

Usage:
    python scripts/figures/build_interp_latex_tables.py    # -> runs/interp_tables.tex (+ stdout)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MP_JSON = ROOT / "runs/matched_pair_tables.json"
TS_JSON = ROOT / "runs/test_split_tables.json"
NC_JSON = (
    ROOT / "runs/noncircular_concordance.json"
)  # non-circular (physical-property) AUROC panels
OUT_TEX = ROOT / "runs/interp_tables.tex"

# non-circular concordance: task -> column header (task + the held-out descriptor it is scored against)
NC_COL = {
    "logd": "LogD (hydrophobic)",
    "ppb_mouse_plasma": "PPB plasma (aromatic)",
    "caco2_papp_ab": "Caco2 P$_{app}$ (HBD)",
    "caco2_efflux_ratio": "Caco2 efflux (HBA)",
}

# (display label, {source: arm-key}). Order + set match interp_heldout.
MODELS = [
    ("Anchored MolLedger", {"mp": "ours_best"}),
    ("MolLedger (no context)", {"mp": "summean_best"}),
    ("Unanchored MolLedger", {"mp": "ours_none"}),
    ("Anchored GNAN", {"mp": "gnan_best"}),
    ("Unanchored GNAN", {"mp": "gnan"}),
    ("LigandFormer", {"mp": "ligandformer"}),
    ("IG", {"mp": "ig_pool_zeros"}),
    ("Grad-CAM", {"mp": "gradcam_pool"}),
    ("LIME", {"mp": "lime_pool"}),
    ("WISP", {"mp": "wisp_pool"}),
]

# (display, source, mp field or None, higher_is_better, digits)
METRICS = [
    ("Leakage", "mp", "leakage", False, 2),
    ("Exactness gap", "tsgap", None, False, 2),
    ("Explained $\\Delta$ MAE", "mp", "dsub_mae", False, 2),
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


def load():
    mp = json.loads(MP_JSON.read_text())
    ts = json.loads(TS_JSON.read_text())
    return mp, ts


def gap_cell(ts, arm, task):
    """mean/sd over seeds of the single-molecule exactness gap. None if unavailable."""
    vals = []
    for _seed, blk in ts["per_seed"].items():
        v = blk.get("gap", {}).get(arm, {}).get(task)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        vals.append(v)
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


def mp_cell(mp, arm, task, field):
    # the only IG arm is the zeros-baseline pooled IG, stored as 'ig_pool'; the figures address it as
    # 'ig_pool_zeros', so strip the suffix to reach it (matches src.metrics.mp_cell).
    if arm.endswith("_zeros"):
        arm = arm[: -len("_zeros")]
    node = mp["results"]["held_out"]["all"].get(task, {}).get(arm)
    if node is None:
        return None
    st = node.get(field)
    if st is None:
        return None
    m, sd = st.get("mean"), st.get("sd")
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return None
    return float(m), float(sd)


def gather(mp, ts, source, field):
    """{model_label: {task: (mean, sd) or None}} for one metric."""
    out = {}
    for label, arms in MODELS:
        row = {}
        for task in mp["ordered_tasks"]:
            if source == "tsgap":
                row[task] = gap_cell(ts, arms["mp"], task)
            else:
                row[task] = mp_cell(mp, arms["mp"], task, field)
        out[label] = row
    return out


def mean_token(m, digits):
    """Printed mean; ~1e-7 gaps read as 0."""
    if abs(m) < 10 ** (-digits) and abs(m) > 0:
        return f"0{'.' + '0' * digits if digits else ''}"
    return f"{m:.{digits}f}"


def fmt(cell, digits):
    if cell is None:
        return "--"
    m, sd = cell
    return f"{mean_token(m, digits)} ({sd:.{digits}f})"


def best_labels(table, tasks, higher, digits):
    """per task, set of model labels within 0.5% of the best mean or printing the same mean."""
    best = {}
    for t in tasks:
        vals = [(lab, table[lab][t][0]) for lab in table if table[lab][t] is not None]
        if not vals:
            best[t] = set()
            continue
        target = max(v for _, v in vals) if higher else min(v for _, v in vals)
        target_tok = mean_token(target, digits)
        tol = 1e-9 + 0.005 * abs(target)
        best[t] = {
            lab
            for lab, v in vals
            if abs(v - target) <= tol or mean_token(v, digits) == target_tok
        }
    return best


def render_metric(name, table, tasks, higher, digits):
    """One LaTeX table, tasks split into two column-blocks."""
    half = (len(tasks) + 1) // 2
    blocks = [tasks[:half], tasks[half:]]
    best = best_labels(table, tasks, higher, digits)
    slug = "".join(c for c in name.split()[0].lower() if c.isalnum())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{name}.}}",
        rf"\label{{tab:interp-{slug}}}",
    ]
    for bi, block in enumerate(blocks):
        ncol = len(block)
        lines.append(r"\begin{tabular}{l" + "c" * ncol + "}")
        lines.append(r"\toprule")
        lines.append("Method & " + " & ".join(TASK_LABEL[t] for t in block) + r" \\")
        lines.append(r"\midrule")
        for lab, _ in MODELS:
            cells = []
            for t in block:
                s = fmt(table[lab][t], digits)
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


def render_noncircular():
    """Single LaTeX table for the non-circular concordance (AUROC) panels: rows = methods, columns =
    the tasks with their held-out descriptor. Returns None if the json is absent."""
    if not NC_JSON.exists():
        return None
    d = json.loads(NC_JSON.read_text())["tasks"]
    tasks = [t for t in NC_COL if t in d]
    table = {}
    for label, arms in MODELS:
        key = arms["mp"]  # ig_pool_zeros / gradcam_pool / ... match the nc json keys
        row = {}
        for t in tasks:
            c = d[t]["arms"].get(key, {})
            m = c.get("mean")
            row[t] = (
                (float(m), float(c.get("sd") or 0.0))
                if (m is not None and not (isinstance(m, float) and np.isnan(m)))
                else None
            )
        table[label] = row
    best = best_labels(table, tasks, higher=True, digits=2)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Non-circular AUROC.}",
        r"\label{tab:noncircular-concordance}",
        r"\begin{tabular}{l" + "c" * len(tasks) + "}",
        r"\toprule",
        "Method & " + " & ".join(NC_COL[t] for t in tasks) + r" \\",
        r"\midrule",
    ]
    for lab, _ in MODELS:
        cells = []
        for t in tasks:
            s = fmt(table[lab][t], 2)
            if lab in best[t] and s != "--":
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_TEX))
    args = ap.parse_args()
    mp, ts = load()
    # order columns by test-set size (largest first), matching the performance/interp plots
    testn = json.loads((ROOT / "runs/test_sample_counts.json").read_text())
    tasks = sorted(mp["ordered_tasks"], key=lambda t: -(testn.get(t) or 0))

    chunks = []
    for name, source, field, higher, digits in METRICS:
        table = gather(mp, ts, source, field)
        n_filled = sum(1 for lab in table for t in tasks if table[lab][t] is not None)
        chunks.append(render_metric(name, table, tasks, higher, digits))
        print(f"[{name}] cells filled: {n_filled}/{len(MODELS) * len(tasks)}")
    nc = render_noncircular()
    if nc is not None:
        chunks.append(nc)
        print("[Non-circular concordance] table appended")
    Path(args.out).write_text("\n\n".join(chunks) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
