"""Metric loaders + accessors for the paper figures.

Reads the two aggregated result JSONs and exposes the per-model / per-arm numbers that
scripts/figures/plot_results.py plots:
  runs/test_split_tables.json   -- test-split accuracy + faithfulness / single-molecule exactness
  runs/matched_pair_tables.json -- matched-pair localization (leakage), predicted-delta accuracy,
                                   and matched-pair completeness gap

Init seeds are {19, 209, 31}. The GBT descriptor baseline is read from results_descriptor_seeds/*.json
and the pooled+descriptor accuracy from runs/pooled_desc_accuracy.json when present.
"""

import glob
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TS = json.loads((REPO / "runs/test_split_tables.json").read_text())
MP = json.loads((REPO / "runs/matched_pair_tables.json").read_text())

TASKS = MP["ordered_tasks"]
SEEDS = [str(s) for s in TS["seeds"]]

# GBT descriptor baseline (227-descriptor "3d" arm), one JSON per bootstrap seed if present.
GBT_DIR = REPO / "results_descriptor_seeds"
GBT_ARM = "3d"  # 2D(208) + 3D(19) = 227 descriptors


def _gbt_load():
    out = {}
    if not GBT_DIR.exists():
        return out
    for f in sorted(GBT_DIR.glob("results_descriptor_baseline_init*.json")):
        d = json.loads(f.read_text())
        pt = {}
        for t, row in d["per_task"].items():
            arm = row.get(GBT_ARM)
            if not arm:
                continue
            pt[t] = {
                "mae": arm.get("test_mae_native"),
                "rmse": arm.get("test_rmse_native"),
                "pearson": arm.get("test_pearson"),
                "spearman": arm.get("test_spearman"),
                "r2": arm.get("test_r2"),
            }
        out[str(d["init_seed"])] = pt
    return out


GBT = _gbt_load()

# pooled + descriptor-inject-after-fix accuracy (separate 3D+descriptor eval), if present.
_PDESC_F = REPO / "runs/pooled_desc_accuracy.json"
PDESC = json.loads(_PDESC_F.read_text())["per_seed"] if _PDESC_F.exists() else {}

# accuracy models (bars of the performance figure). Order + labels are the paper's fixed presentation
# order; the two GNAN arms mirror the two MolLedger arms (anchored first). plot_results relabels a
# couple of these for display but keys off these exact arm names.
MODELS = [
    ("additive_best", "Anchored MolLedger"),
    ("additive_none", "Unanchored MolLedger"),
    ("summean_best", "Sum-mean MolLedger (no context)"),  # accuracy<->leakage tradeoff ablation:
    #   anchored sum-mean head WITHOUT the global-context vector (fixed anchor 0.3, best by val MAE).
    ("pooled_none", "Pooled GNN"),
    ("gnan_best", "Anchored GNAN"),  # GNAN trained WITH the shape anchor (best per-seed lambda)
    ("gnan", "Unanchored GNAN"),  # GNAN trained property-loss-only (lambda_anchor 0)
    ("ligandformer", "LigandFormer"),  # self-attention arch (accuracy from its own checkpoint)
]
if PDESC:
    MODELS.append(("pooled_desc", "Pooled + descriptors"))
if GBT:
    MODELS.append(("gbt_227", "GBT"))


def clean(vals):
    return [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]


def acc(seed, model, task):
    if model == "gbt_227":
        return GBT.get(seed, {}).get(task)
    if model == "pooled_desc":
        return PDESC.get(seed, {}).get(task)
    return TS["per_seed"][seed]["acc"].get(model, {}).get(task)


@lru_cache(maxsize=1)
def _anchor_signs():
    """task -> anchor_sign from src.data.multitask.ANCHOR_RULE (the 'ruled' anchors)."""
    from src.data.multitask import ANCHOR_RULE  # heavy (torch); import lazily, cache once

    ref = TS.get("anchor_ref")
    if ref not in (None, "ruled"):
        import warnings

        warnings.warn(f"faith_r orients signs for the 'ruled' anchors, but TS anchor_ref={ref!r}")
    return {t: s for t, (_a, s) in ANCHOR_RULE.items()}


def faith_r(seed, arm, task):
    """Per-molecule-mean Pearson r vs the anchor reference, oriented by the task's anchor_sign so
    positive = faithful to the anchor as trained. LigandFormer is the exception: its attention r is
    computed against |anchor| (unsigned), so it is returned unoriented."""
    v = TS["per_seed"][seed]["faith"].get(arm, {}).get(task)
    if not (v and v[0] is not None):
        return None
    sign = 1 if arm == "ligandformer" else _anchor_signs().get(task, 1)
    return sign * v[0]


def single_gap(seed, arm, task):
    """Single-molecule exactness gap: mean |sum_i a_i - dy_hat| over the held-out test split
    (from test_split_tables.json). ~0 for the exact additive arms; None for LigandFormer."""
    return TS["per_seed"][seed]["gap"].get(arm, {}).get(task)


def mp_cell(collection, cls, task, arm, field):
    # The only IG arm is the zero-baseline pooled IG, stored under 'ig_pool'; the interp/perf figures
    # address it as 'ig_pool_zeros', so strip the suffix to reach it.
    if arm.endswith("_zeros"):
        arm = arm[: -len("_zeros")]
    c = MP["results"][collection][cls].get(task, {}).get(arm)
    return c[field] if c else None  # {'mean':..,'sd':..} or None


def mp_count(coll, cls, task):
    return MP["counts"].get(coll, {}).get(cls, {}).get(task, 0)


# ----------------------------------------------------------------------------- attribution timing
# Each matched_pair_attribution run stores _timing.ms_per_molecule_scoring in its --out JSON; meaned
# over the runs per method (label -> glob of those JSONs).
TIMING_SOURCES = [
    ("MolLedger (additive)", "runs/exact_grid_gctx8_constlam/lk*_none.json"),
    ("IG-zero (pooled)", "runs/ig_grid_zeros/ig_pooled_init*.json"),
    ("Grad-CAM (pooled)", "runs/ig_grid/gradcam_signed_pooled_init*.json"),
    ("LIME (pooled)", "runs/ig_grid/lime_pooled_init*.json"),
    ("WISP (pooled)", "runs/ig_grid/wisp_pooled_init*.json"),
    # Both GNAN arms share one timing entry (same architecture, one forward pass).
    ("GNAN (additive)", "runs/gnan_grid/lk*_gnan_init*.json"),
    ("LigandFormer (attention)", "runs/ligandformer_grid/lk*_ligandformer_init*.json"),
]


def _wisp_precompute_ms():
    """WISP mutant-generation cost per molecule (ms), from the sidecar precompute_wisp_mutants.py
    writes next to the cache; 0 if absent."""
    f = REPO / "runs/ig_grid/wisp_mutants_timing.json"
    if f.exists():
        return float(json.loads(f.read_text()).get("ms_per_molecule") or 0.0)
    return 0.0


def attr_timing():
    """[(label, mean_ms, sd_ms, n), ...] -- ms per molecule scored, per method. WISP's bar adds its
    per-molecule mutant-generation cost (_wisp_precompute_ms) to the forward-scoring time."""
    rows = []
    for label, pat in TIMING_SOURCES:
        vals = []
        for f in sorted(glob.glob(str(REPO / pat))):
            t = json.loads(Path(f).read_text()).get("_timing") or {}
            ms = t.get("ms_per_molecule_scoring")
            if ms is not None:
                vals.append(ms)
        if vals:
            add = _wisp_precompute_ms() if label.startswith("WISP") else 0.0
            rows.append((label, float(np.mean(vals)) + add, float(np.std(vals)), len(vals)))
    return rows
