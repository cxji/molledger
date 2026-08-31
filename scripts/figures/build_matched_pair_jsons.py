"""
Aggregate matched-pair attribution dumps (JSONL from scripts/score/matched_pair_attribution.py)
into the paper's report tables. No model, no re-scoring.

Per collection, produces (mean +/- sd over the 3 init seeds):
  * counts                       -- pairs per (class, task).
  * localization (leakage)       -- median |D_core| / (|D_core| + |D_sub|).
  * predicted-delta accuracy     -- Pearson r, Spearman rho, and MAE of predicted vs measured delta.

Collections: `held_out` (pairs with >= 1 test member; classes A, B, D) and `all_pairs` (every mined
pair; classes A and B).

Usage:
    python scripts/figures/build_matched_pair_jsons.py --json_out runs/matched_pair_tables.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = [19, 209, 31]
LAMBDA_LABELS = {0.0: "none", 0.1: "anchor0.1", 0.3: "anchor0.3"}
ANCHOR_DIR_SUFFIX = {0.1: "_anchor0.1-shape-rule", 0.3: "_anchor0.3-shape-rule"}

# report-column key -> human label
COLUMNS = [
    ("ours_none", "Ours (exact, unanchored)"),
    ("ours_best", "Ours (exact, best anchor)"),
    ("summean_best", "Sum-mean (exact, anchored, no ctx)"),
    (
        "ig_pool",
        "IG (pooled, no desc)",
    ),  # zero-baseline IG on the pooled head; the only IG arm plotted
    ("gradcam_pool", "Grad-CAM (pooled)"),
    ("lime_pool", "LIME (pooled)"),
    ("wisp_pool", "WISP (pooled, occlusion)"),
    ("gnan_best", "Anchored GNAN"),
    ("gnan", "Unanchored GNAN"),
    ("ligandformer", "LigandFormer (attention)"),  # attention-received; leakage/gap are NaN
    (
        "gbt_227",
        "GBT (descriptors)",
    ),  # non-attribution baseline: only the Delta-MAE cell is populated
    ("pooled_desc", "Pooled + descriptors"),
]
CLASSES = ["class_a", "class_b", "class_d"]


# --------------------------------------------------------------------------------------------
# dump loading
# --------------------------------------------------------------------------------------------


def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return None
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _exact_files(exact_dir, seed, label):
    # graph-identical pairs: one dump, every pair. fragment swaps: separate held-out-only dump.
    return [
        f"{exact_dir}/pairs_init{seed}_{label}.jsonl",
        f"{exact_dir}/pairsd_init{seed}_{label}.jsonl",
    ]


def _ig_pool_files(ig_zeros_dir, seed):
    # Off-manifold zero-baseline IG (runs/ig_grid_zeros); the only IG arm the figures plot.
    return [
        f"{ig_zeros_dir}/pairs_ig_pooled_init{seed}.jsonl",
        f"{ig_zeros_dir}/pairs_ig_pooled_init{seed}_d.jsonl",
    ]


def _lime_pool_files(ig_dir, seed):
    return [
        f"{ig_dir}/pairs_lime_pooled_init{seed}.jsonl",
        f"{ig_dir}/pairs_lime_pooled_init{seed}_d.jsonl",
    ]


def _wisp_pool_files(ig_dir, seed):
    return [
        f"{ig_dir}/pairs_wisp_pooled_init{seed}.jsonl",
        f"{ig_dir}/pairs_wisp_pooled_init{seed}_d.jsonl",
    ]


def _gradcam_pool_files(ig_dir, seed):
    return [
        f"{ig_dir}/pairs_gradcam_signed_pooled_init{seed}.jsonl",
        f"{ig_dir}/pairs_gradcam_signed_pooled_init{seed}_d.jsonl",
    ]


def _gnan_files(gnan_dir, seed):
    # GNAN, scored with --method additive on its own checkpoint: carries a real leakage/delta-acc/gap.
    return [f"{gnan_dir}/pairs_gnan_init{seed}.jsonl", f"{gnan_dir}/pairs_gnan_init{seed}_d.jsonl"]


def _gnan_best_files(gnan_dir, seed):
    # Anchored GNAN (best per-seed lambda over {0.1,0.3}), same exact-additive scoring as
    # _gnan_files but on the shape-anchored checkpoints. Dumps land in runs/gnan_grid/.
    return [
        f"{gnan_dir}/pairs_gnan_best_init{seed}.jsonl",
        f"{gnan_dir}/pairs_gnan_best_init{seed}_d.jsonl",
    ]


def _ligandformer_files(lf_dir, seed):
    # Scored with --method attention: leakage/gap are NaN (not cross-molecule commensurable).
    return [
        f"{lf_dir}/pairs_ligandformer_init{seed}.jsonl",
        f"{lf_dir}/pairs_ligandformer_init{seed}_d.jsonl",
    ]


def _pairdelta_files(pd_dir, arm, seed):
    # Non-attribution baselines (GBT, pooled+desc): only meas/dy_true, for the Delta-MAE row.
    return [f"{pd_dir}/pairs_{arm}_init{seed}.jsonl", f"{pd_dir}/pairs_{arm}_init{seed}_d.jsonl"]


def load_arm(files, warn):
    """Concatenate the per-class dumps for one (arm, seed); missing files warn and contribute none."""
    rows = []
    any_found = False
    for f in files:
        r = load_jsonl(f)
        if r is None:
            warn.append(f)
        else:
            any_found = True
            rows.extend(r)
    return rows if any_found else None


# --------------------------------------------------------------------------------------------
# best-anchor selection (per seed): argmin validation best_metric over lambda {0.1, 0.3}
# --------------------------------------------------------------------------------------------


def select_best_lambda(repo, seed, forced, warn):
    """Return the anchor lambda to use for `seed`.

    `forced` in {0.1, 0.3} pins it; "auto" reads each candidate checkpoint's `best_metric` and takes
    the smaller. Falls back to 0.1 if torch/the checkpoints are unavailable.
    """
    if forced in (0.1, 0.3):
        return forced, f"forced={forced}"
    try:
        import torch  # local import: only needed for auto-selection
    except Exception:
        warn.append(f"torch unavailable; seed {seed} best-anchor defaulted to 0.1")
        return 0.1, "default(no torch)"
    best_lam, best_val = None, None
    for lam, suffix in ANCHOR_DIR_SUFFIX.items():
        # "ours" = the uniform (constant-lambda) global-context ctx=8 head (MolLedger).
        ck = (
            Path(repo) / f"checkpoints_global_context_constlam/"
            f"ablation_gin_additive_none{suffix}_gctx8_init{seed}/best.pt"
        )
        if not ck.exists():
            warn.append(f"missing {ck} for best-anchor selection")
            continue
        try:
            m = torch.load(ck, map_location="cpu", weights_only=False).get("best_metric")
        except Exception as e:  # noqa: BLE001
            warn.append(f"could not read best_metric from {ck}: {e}")
            continue
        if m is not None and (best_val is None or m < best_val):
            best_val, best_lam = float(m), lam
    if best_lam is None:
        warn.append(f"no readable best_metric for seed {seed}; defaulted to 0.1")
        return 0.1, "default(no metric)"
    return best_lam, f"best_metric={best_val:.4f}"


# --------------------------------------------------------------------------------------------
# metrics (recomputed from dumped per-pair rows; mirror matched_pair_attribution.py)
# --------------------------------------------------------------------------------------------


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return pearson(ra, rb)


def _stat(values):
    """mean +/- sd over seeds, ignoring NaN; returns (mean, sd, n)."""
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std()), int(v.size)


# --------------------------------------------------------------------------------------------
# collection filters
# --------------------------------------------------------------------------------------------


def in_collection(row, collection):
    if collection == "all_pairs":
        return True
    # held_out: >= 1 test member. `split` is "wa/wb", e.g. "train/test".
    return "test" in set(row["split"].split("/"))


# --------------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------------


def build(args):
    warn = []

    # per-seed lambda choice for the "best anchor" (and the IG-additive) columns
    best_lambda = {}
    for s in SEEDS:
        lam, why = select_best_lambda(args.repo, s, "auto", warn)
        best_lambda[s] = (lam, why)

    # Load every (column, seed) arm's rows once.
    # arm_rows[col][seed] = list[row] or None
    arm_rows = defaultdict(dict)
    for s in SEEDS:
        best_label = LAMBDA_LABELS[best_lambda[s][0]]
        arm_rows["ours_none"][s] = load_arm(_exact_files(args.exact_dir, s, "none"), warn)
        arm_rows["ours_best"][s] = load_arm(_exact_files(args.exact_dir, s, best_label), warn)
        # Sum-mean anchored, no context: fixed anchor 0.3 dumps, separate dir.
        arm_rows["summean_best"][s] = load_arm(_exact_files(args.summean_dir, s, "anchor0.3"), warn)
        arm_rows["ig_pool"][s] = load_arm(_ig_pool_files(args.ig_zeros_dir, s), warn)
        arm_rows["gradcam_pool"][s] = load_arm(_gradcam_pool_files(args.ig_dir, s), warn)
        arm_rows["lime_pool"][s] = load_arm(_lime_pool_files(args.ig_dir, s), warn)
        arm_rows["wisp_pool"][s] = load_arm(_wisp_pool_files(args.ig_dir, s), warn)
        arm_rows["gnan"][s] = load_arm(_gnan_files(args.gnan_dir, s), warn)
        arm_rows["gnan_best"][s] = load_arm(_gnan_best_files(args.gnan_dir, s), warn)
        arm_rows["ligandformer"][s] = load_arm(_ligandformer_files(args.ligandformer_dir, s), warn)
        arm_rows["gbt_227"][s] = load_arm(_pairdelta_files(args.pairdelta_dir, "gbt", s), warn)
        arm_rows["pooled_desc"][s] = load_arm(
            _pairdelta_files(args.pairdelta_dir, "pooled_desc", s), warn
        )

    # ---- per (collection, class, task, column, seed) metric bundle, reduced across seeds later ----
    def seed_bundle(rows, collection, cls, task):
        # cls=="all" pools every class in the collection (no class filter); used by the interp figures.
        sel = [
            r
            for r in rows
            if (cls == "all" or r["class"] == cls)
            and r["task"] == task
            and in_collection(r, collection)
        ]
        if not sel:
            return None
        if cls == "all":
            # De-dup pairs mined as both a single-atom swap and a fragment swap (~1152 of 1170), by
            # the unordered {a, b} pair, keeping the first occurrence (graph-identical file loads
            # first). Only "all" needs this; per-class cells are already unique. Rows without a/b
            # identity pass through untouched.
            seen, uniq = set(), []
            for r in sel:
                a, b = r.get("a"), r.get("b")
                if a is not None and b is not None:
                    key = frozenset((a, b))
                    if key in seen:
                        continue
                    seen.add(key)
                uniq.append(r)
            sel = uniq
        dy = [r["dy_true"] for r in sel]
        me = [r["meas"] for r in sel]
        leak = [
            r["leakage"] for r in sel if r["leakage"] is not None and not np.isnan(r["leakage"])
        ]
        return {
            "n": len(sel),
            "pearson": pearson(dy, me),
            "spearman": spearman(dy, me),
            # native-unit mean |predicted delta - measured delta|.
            "mae_delta": float(np.mean(np.abs(np.asarray(dy, float) - np.asarray(me, float)))),
            "leakage": float(np.median(leak)) if leak else float("nan"),
        }

    # task universe from the exact dumps (all 11 tasks)
    tasks_present = set()
    for s in SEEDS:
        for r in arm_rows["ours_none"][s] or []:
            tasks_present.add(r["task"])
    # stable, registry-like order
    TASK_ORDER = [
        "logd",
        "kinetic_solubility",
        "aqueous_solubility",
        "clint_mouse_liver",
        "clint_human_liver",
        "caco2_efflux_ratio",
        "caco2_papp_ab",
        "ppb_mouse_plasma",
        "ppb_mouse_brain",
        "ppb_mouse_muscle",
        "half_life",
    ]
    ordered_tasks = [t for t in TASK_ORDER if t in tasks_present] + sorted(
        tasks_present - set(TASK_ORDER)
    )

    # "all" = pooled pseudo-class (every class in the collection), for the per-task interp figures.
    collections = [
        ("held_out", list(CLASSES) + ["all"]),
        ("all_pairs", ["class_a", "class_b", "all"]),
    ]

    # results[collection][cls][task][col] = {metric: (mean, sd, n_seeds)}  plus counts
    results = {}
    counts = {}  # counts[collection][cls][task] = int (seed-invariant)

    for collection, classes in collections:
        results[collection] = {}
        counts[collection] = {}
        for cls in classes:
            results[collection][cls] = {}
            counts[collection][cls] = {}
            for task in ordered_tasks:
                # counts: from ours_none, first seed with data (split is seed-invariant)
                cnt = None
                for s in SEEDS:
                    b = seed_bundle(arm_rows["ours_none"][s] or [], collection, cls, task)
                    if b:
                        cnt = b["n"]
                        break
                counts[collection][cls][task] = cnt or 0

                results[collection][cls][task] = {}
                for col, _ in COLUMNS:
                    per_seed = []
                    for s in SEEDS:
                        rows = arm_rows[col][s]
                        if rows is None:
                            continue
                        b = seed_bundle(rows, collection, cls, task)
                        if b:
                            per_seed.append(b)
                    if not per_seed:
                        results[collection][cls][task][col] = None
                        continue
                    results[collection][cls][task][col] = {
                        "n": _stat([b["n"] for b in per_seed]),
                        "pearson": _stat([b["pearson"] for b in per_seed]),
                        "spearman": _stat([b["spearman"] for b in per_seed]),
                        "mae_delta": _stat([b["mae_delta"] for b in per_seed]),
                        "leakage": _stat([b["leakage"] for b in per_seed]),
                    }

    return {
        "best_lambda": {s: {"lambda": best_lambda[s][0], "why": best_lambda[s][1]} for s in SEEDS},
        "ordered_tasks": ordered_tasks,
        "collections": collections,
        "counts": counts,
        "results": results,
        "warnings": warn,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument(
        "--exact_dir",
        default="runs/exact_grid_gctx8_constlam",
        help="Per-pair exact-score dumps for the 'ours' model = uniform (constant-lambda) "
        "global-context ctx=8 head; feeds the ours_none/ours_best leakage + matched-pair "
        "delta columns.",
    )
    p.add_argument(
        "--summean_dir",
        default="runs/exact_grid_summean",
        help="Per-pair exact-score dumps for the sum-mean anchored NO-context ablation "
        "(fixed anchor 0.3); feeds the summean_best leakage + matched-pair delta columns "
        "(the accuracy<->leakage tradeoff bar).",
    )
    p.add_argument(
        "--ig_dir",
        default="runs/ig_grid",
        help="Pooled-head post-hoc dumps: Grad-CAM / LIME / WISP.",
    )
    p.add_argument(
        "--ig_zeros_dir",
        default="runs/ig_grid_zeros",
        help="Off-manifold zero-baseline pooled IG dumps (the ig_pool arm).",
    )
    p.add_argument(
        "--gnan_dir",
        default="runs/gnan_grid",
        help="GNAN matched-pair dumps (--method additive).",
    )
    p.add_argument(
        "--ligandformer_dir",
        default="runs/ligandformer_grid",
        help="LigandFormer matched-pair dumps (--method attention).",
    )
    p.add_argument(
        "--pairdelta_dir",
        default="runs/pairdelta_grid",
        help="Predicted-delta dumps for the non-attribution baselines GBT / pooled+desc "
        "(eval_pair_delta_baselines.py); only the Delta-MAE cell is populated.",
    )
    p.add_argument("--json_out", default="runs/matched_pair_tables.json")
    args = p.parse_args()

    # resolve dirs relative to repo if not absolute
    for a in ("exact_dir", "ig_dir", "ig_zeros_dir"):
        v = getattr(args, a)
        if not Path(v).is_absolute():
            setattr(args, a, str(Path(args.repo) / v))

    agg = build(args)

    # JSON: replace the (mean,sd,n) tuples with dicts for portability
    def jsonify(o):
        if isinstance(o, tuple) and len(o) == 3:
            return {"mean": o[0], "sd": o[1], "n_seeds": o[2]}
        if isinstance(o, dict):
            return {k: jsonify(v) for k, v in o.items()}
        if isinstance(o, list):
            return [jsonify(v) for v in o]
        return o

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(jsonify(agg), indent=2, default=str))
    print(f"Wrote {args.json_out}")

    if agg["warnings"]:
        print(f"\n{len(set(agg['warnings']))} warning(s) in the aggregation")


if __name__ == "__main__":
    main()
