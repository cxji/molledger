"""Global-context SIZE sweep figure (plots/val_additive_ablation_appendix.pdf), val split.

Every arm is anchored (uniform, constant-lambda); only context size varies left-to-right:

    1. no-gctx (sum-mean)   -- sum-mean base, ctx=0            [checkpoints_anchor_sweep_summean]
    2. ctx d=2              -- global-context ctx=2            [checkpoints_global_context_constlam]
    3. ctx d=4              -- global-context ctx=4
    4. ctx d=8 (MolLedger)  -- global-context ctx=8
    5. ctx d=16             -- global-context ctx=16
    6. Pooled GNN, LigandFormer -- non-additive references

The two references are read from runs/val_accuracy_appendix.json; the base and gctx arms are scored
fresh from their checkpoints (valid-split MAE + Spearman at best.pt; anchored arm = per-seed argmin
val best_metric over lambda {0.1, 0.3}). Same 5 main tasks + MAE/Spearman panels.

    python scripts/figures/build_val_global_context_ablation.py             # score + plot
    python scripts/figures/build_val_global_context_ablation.py --plot_only # reuse the json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SEEDS = [19, 209, 31]
ANCHOR_SUFFIX = {0.1: "_anchor0.1-shape-rule", 0.3: "_anchor0.3-shape-rule"}
TASKS_MAIN = [
    "logd",
    "kinetic_solubility",
    "clint_mouse_liver",
    "clint_human_liver",
    "caco2_papp_ab",
]

# NEW arm key -> (pretty label, checkpoint-dir template, is_anchored). {suffix}/{seed} as elsewhere.
_GCTX = "checkpoints_global_context_constlam/ablation_gin_additive_none{suffix}_gctx{d}_init{seed}"
NEW_ARMS = [
    # context-size ladder, all uniform (constant-lambda) anchor. Base (ctx=0) is the no-gctx sum-mean head.
    (
        "summean_base",
        "No context",
        "checkpoints_anchor_sweep_summean/ablation_gin_additive_none{suffix}_init{seed}",
        True,
    ),
    ("gctx2_best", "Context dim 2", _GCTX.replace("{d}", "2"), True),
    ("gctx4_best", "Context dim 4", _GCTX.replace("{d}", "4"), True),
    ("gctx8_best", "Context dim 8 (MolLedger)", _GCTX.replace("{d}", "8"), True),
    ("gctx16_best", "Context dim 16", _GCTX.replace("{d}", "16"), True),
]
# the non-additive references come from the shipped appendix json.
BASELINE_LABELS = {
    "pooled": "Pooled GNN",
    "ligandformer": "LigandFormer",
}
BASELINE_KEYS = list(BASELINE_LABELS)

# left-to-right ladder: sum-mean base (ctx=0), then the gctx width sweep, then references.
PLOT_ORDER = [
    "summean_base",
    "gctx2_best",
    "gctx4_best",
    "gctx8_best",
    "gctx16_best",
    "pooled",
    "ligandformer",
]
# base in blue; gctx sweep green->teal deepening with width; references in greys.
COLORS = {
    "summean_base": "#08519C",
    "gctx2_best": "#C7E9C0",
    "gctx4_best": "#74C476",
    "gctx8_best": "#238B45",
    "gctx16_best": "#00441B",
    "pooled": "#999999",
    "ligandformer": "#4D4D4D",
}
# how each arm's faithfulness is obtained: "additive" = exact per-atom scores; "attention" =
# LigandFormer attention (abs-referenced, unsigned); None = Pooled GNN -> hatched n/a bar.
FAITH_METHOD = {
    "summean_base": "additive",
    "gctx2_best": "additive",
    "gctx4_best": "additive",
    "gctx8_best": "additive",
    "gctx16_best": "additive",
    "pooled": None,
    "ligandformer": "attention",
}
LF_CKPT = "checkpoints_ligandformer_seeds/ablation_gin_ligandformer_none_init{seed}/best.pt"

JSON_OUT = ROOT / "runs/val_global_context_ablation.json"
BASELINE_JSON = ROOT / "runs/val_accuracy_appendix.json"
PDF_OUT = ROOT / "plots/val_additive_ablation_appendix.pdf"

# Leakage row: median fragment-pair leakage per seed, from runs/leakage_ctx_sweep/ (written by
# scripts/score/score_leakage_ctx_sweep.py). Maps each figure arm to its dump. Pooled GNN and
# LigandFormer have no entry here -> hatched n/a, same as the faithfulness row.
LEAK_DUMP = ROOT / "runs/leakage_ctx_sweep"
LEAK_ARM = {
    "summean_base": "ctx0_summean",
    "gctx2_best": "gctx2",
    "gctx4_best": "gctx4",
    "gctx8_best": "gctx8",
    "gctx16_best": "gctx16",
}


def select_best_lambda(tmpl, seed, warn):
    import torch

    best_lam, best_val = None, None
    for lam in (0.1, 0.3):
        ck = ROOT / (tmpl.format(suffix=ANCHOR_SUFFIX[lam], seed=seed) + "/best.pt")
        if not ck.exists():
            warn.append(f"missing checkpoint {ck}")
            continue
        m = torch.load(ck, map_location="cpu", weights_only=False).get("best_metric")
        if m is not None and (best_val is None or float(m) < best_val):
            best_val, best_lam = float(m), lam
    return best_lam


def evaluate_new_arms(args):
    import torch
    from torch_geometric.loader import DataLoader

    from scripts.figures.build_test_split_jsons import eval_accuracy
    from scripts.train.train_molledger import NTASK, TASKS, subset_tasks
    from src.attribution import attribute, faithfulness
    from src.data.multitask import GRAPH_TASK_SPECS, apply_anchor_rule, load_multitask_splits
    from src.models.gnn import build_model_from_checkpoint

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    splits = load_multitask_splits(seed=args.split_seed, conformer_cache_path=None)
    splits = {s: subset_tasks(dl) for s, dl in splits.items()}
    data = splits[args.split]
    loader = DataLoader(data, batch_size=args.batch_size)
    print(f"[{args.split}] {len(data)} molecules | device={device}", flush=True)

    task_counts = {s.name: 0 for s in TASKS}
    for d in loader:
        y = d.y.view(-1, NTASK)
        for k, s in enumerate(TASKS):
            task_counts[s.name] += int((~torch.isnan(y[:, k])).sum())

    # anchored tasks for faithfulness
    ref_anchor = {
        s.name: a.anchor for s, a in zip(GRAPH_TASK_SPECS, apply_anchor_rule(GRAPH_TASK_SPECS))
    }
    anchored = [
        (k, ref_anchor[s.name])
        for k, s in enumerate(TASKS)
        if ref_anchor[s.name] in ("crippen", "tpsa")
    ]
    all_idx = list(range(len(TASKS)))

    def score_faith(model, backbone, method):
        attrs, _p, _a = attribute(
            model,
            data,
            method,
            backbone,
            batch_size=args.batch_size,
            device=device,
            task_idx=all_idx,
        )
        out = {}
        for k, anchor_attr in anchored:
            pear, n = faithfulness(attrs, data, k, anchor_attr, abs_ref=(method == "attention"))
            out[TASKS[k].name] = {"pearson": pear, "n": n}
        return out

    warn, best_lam, per_seed, per_seed_faith = [], {}, {}, {}
    for seed in SEEDS:
        per_seed[seed], per_seed_faith[seed] = {}, {}
        for key, _label, tmpl, anchored_arm in NEW_ARMS:
            suffix = ""
            if anchored_arm:
                lam = select_best_lambda(tmpl, seed, warn)
                if lam is None:
                    continue
                best_lam[(key, seed)] = lam
                suffix = ANCHOR_SUFFIX[lam]
            path = ROOT / (tmpl.format(suffix=suffix, seed=seed) + "/best.pt")
            if not path.exists():
                warn.append(f"missing checkpoint {path}")
                continue
            ck = torch.load(path, map_location=device, weights_only=False)
            model, backbone = build_model_from_checkpoint(ck)
            model = model.to(device).eval()
            per_seed[seed][key] = eval_accuracy(model, loader, device, backbone)
            per_seed_faith[seed][key] = score_faith(model, backbone, "additive")
            print(
                f"  seed {seed} | {key:24s}"
                + (f" lam={best_lam[(key, seed)]}" if anchored_arm else "")
                + " done",
                flush=True,
            )
        # LigandFormer faithfulness (attention, abs-ref); its accuracy comes from the baseline json.
        lf = ROOT / LF_CKPT.format(seed=seed)
        if lf.exists():
            ck = torch.load(lf, map_location=device, weights_only=False)
            model, backbone = build_model_from_checkpoint(ck)
            model = model.to(device).eval()
            per_seed_faith[seed]["ligandformer"] = score_faith(model, backbone, "attention")
            print(f"  seed {seed} | ligandformer (attention faith) done", flush=True)
        else:
            warn.append(f"missing checkpoint {lf}")

    payload = {
        "seeds": SEEDS,
        "split": args.split,
        "task_counts": task_counts,
        "arms": [(k, lab) for k, lab, _, _ in NEW_ARMS],
        "best_lambda": {f"{k}|{s}": v for (k, s), v in best_lam.items()},
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "per_seed_faith": {str(s): v for s, v in per_seed_faith.items()},
        "warnings": warn,
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2, default=float))
    print(f"Wrote {JSON_OUT}")
    if warn:
        print(f"{len(set(warn))} warning(s):")
        for w in dict.fromkeys(warn):
            print("  -", w)


def plot():
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.patches import Patch

    from scripts.figures.plot_results import _n, pretty_task

    sns.set_theme(style="whitegrid", context="notebook")
    new = json.loads(JSON_OUT.read_text())
    base = json.loads(BASELINE_JSON.read_text())

    labels = dict(BASELINE_LABELS)
    labels.update({k: lab for k, lab, _t, _a in NEW_ARMS})  # current labels, not the json's
    ps, psf = {}, new.get("per_seed_faith", {})
    for s in [str(x) for x in SEEDS]:
        ps[s] = {}
        ps[s].update({k: v for k, v in base["per_seed"][s].items() if k in BASELINE_KEYS})
        ps[s].update(new["per_seed"][s])
    counts = base["task_counts"]
    seeds = [str(x) for x in SEEDS]

    # sign-orient additive faithfulness per task; LigandFormer attention is unsigned, not flipped.
    from src.data.multitask import ANCHOR_RULE

    faith_sign = {t: s for t, (_a, s) in ANCHOR_RULE.items()}

    def agg(arm, task, metric):
        vals = [ps[s][arm][task][metric] for s in seeds if arm in ps[s] and task in ps[s][arm]]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)

    def agg_faith(arm, task):
        sgn = 1 if arm == "ligandformer" else faith_sign.get(task, 1)
        vals = []
        for s in seeds:
            f = psf.get(s, {}).get(arm, {}).get(task)
            if f and f["pearson"] is not None and not np.isnan(f["pearson"]):
                vals.append(sgn * f["pearson"])
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)

    # ---- leakage row: per-seed median fragment-pair leakage from the exact-score dumps, mean+/-std over seeds
    leak = {}
    for fig_arm, dump_arm in LEAK_ARM.items():
        per_seed = {}
        for s in seeds:
            f = LEAK_DUMP / f"pairsd_{dump_arm}_init{s}.jsonl"
            if not f.exists():
                continue
            buf = {}
            for line in f.open():
                r = json.loads(line)
                if r.get("class") != "class_d":
                    continue
                lk = r.get("leakage")
                if lk is None or (isinstance(lk, float) and np.isnan(lk)):
                    continue
                buf.setdefault(r["task"], []).append(lk)
            per_seed[s] = {t: float(np.median(v)) for t, v in buf.items()}
        leak[fig_arm] = {
            t: (float(np.mean(vs)), float(np.std(vs)))
            for t in TASKS_MAIN
            if (vs := [per_seed[s][t] for s in seeds if t in per_seed.get(s, {})])
        }

    def agg_leak(arm, task):
        cell = leak.get(arm, {}).get(task)
        return cell if cell is not None else (np.nan, 0.0)

    order = [k for k in PLOT_ORDER if any(k in ps[s] for s in seeds)]  # skip arms with no checkpoint
    ncol = len(TASKS_MAIN)
    _figh = 11.5  # 4 rows
    fig, axes = plt.subplots(4, ncol, figsize=(2.9 * ncol, _figh), squeeze=False)
    x = np.arange(len(order))
    rows = [
        ("mae", "MAE ↓", "auto"),
        ("spearman", "Spearman ↑", (0, 1)),
        ("faith", "Faithfulness ↑", "auto"),
        ("leakage", "Leakage ↓", (0, 0.55)),
    ]
    for r, (mkey, mname, ylim) in enumerate(rows):
        for c, task in enumerate(TASKS_MAIN):
            ax = axes[r][c]
            hatch_na = []
            for xi, k in enumerate(order):
                if mkey == "faith":
                    if FAITH_METHOD.get(k) is None:  # hatched n/a
                        hatch_na.append(xi)
                        continue
                    m, sd = agg_faith(k, task)
                elif mkey == "leakage":
                    if k not in LEAK_ARM:  # hatched n/a
                        hatch_na.append(xi)
                        continue
                    m, sd = agg_leak(k, task)
                else:
                    m, sd = agg(k, task, mkey)
                if np.isnan(m):
                    continue
                ax.bar(
                    xi,
                    m,
                    0.8,
                    yerr=sd,
                    color=COLORS[k],
                    capsize=2,
                    edgecolor="black",
                    linewidth=0.4,
                )
            if ylim == (0, 1):
                ax.set_ylim(0, 1)
            elif ylim == (0, 0.55):
                ax.set_ylim(0, 0.55)
            if mkey == "faith":
                ax.set_ylim(bottom=0)
                ax.set_ylim(top=ax.get_ylim()[1] * 1.06)  # small headroom
            if mkey in ("faith", "leakage"):
                for xi in hatch_na:  # full-column diagonal hatch
                    ax.axvspan(
                        xi - 0.4,
                        xi + 0.4,
                        facecolor="none",
                        hatch="////",
                        edgecolor="0.6",
                        linewidth=0.0,
                        zorder=0.5,
                    )
            ax.set_xticks(x)
            ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=9)
            if r == 0:
                ax.set_title(f"{pretty_task(task)}\n(n={_n(counts[task])})", fontsize=11)
            if c == 0:
                ax.set_ylabel(mname, fontsize=12)
            sns.despine(ax=ax)
    handles = [
        Patch(facecolor=COLORS[k], edgecolor="black", linewidth=0.4, label=labels[k]) for k in order
    ]
    # figh-aware inch offsets: title 0.15", legend 0.32", first subplot row 0.20+0.17*legend_rows" from top.
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        fontsize=11,  # 7 entries -> 2 rows
        frameon=False,
        bbox_to_anchor=(0.5, 1 - 0.32 / _figh),
    )
    fig.suptitle("MolLedger with varying global context size", fontsize=15, y=1 - 0.15 / _figh)
    fig.align_ylabels(axes[:, 0])  # align MAE/Spearman/Faithfulness/Leakage labels
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.20 + 0.17 * 2) / _figh))  # 2 legend rows
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_OUT, bbox_inches="tight")
    print(f"Wrote {PDF_OUT}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--plot_only", action="store_true")
    args = p.parse_args()
    if not args.plot_only:
        evaluate_new_arms(args)
    plot()


if __name__ == "__main__":
    main()
