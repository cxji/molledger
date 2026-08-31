"""Anchor faithfulness-vs-accuracy on the VALIDATION split.

Arms = a no-anchor / weak / strong anchor triple at three fixed architectures, varying only the
anchor strength lambda. All anchored arms use the uniform (constant-lambda) scheme; weak = lambda
0.1, strong = lambda 0.3:

  plain sum MolLedger          gctx8 (global-context d=8)
  ------------------           ---------------------------
  sum_none    lambda 0         gctx8_none    lambda 0        (unanchored)
  sum_weak    lambda 0.1       gctx8_weak    lambda 0.1      (--anchor_rule, uniform)
  sum_strong  lambda 0.3       gctx8_strong  lambda 0.3      (--anchor_rule, uniform)

  Exactly one unanchored checkpoint per architecture (lambda 0).

For each arm/seed: exact additive per-atom scores (method="additive") over the val split ->
per-molecule Pearson/Spearman vs the Crippen/TPSA per-atom anchor (src.attribution.faithfulness),
averaged over the anchored tasks; plus val accuracy (eval_accuracy). Each anchored arm is a fixed
lambda (no best-of selection).

    python scripts/figures/build_val_anchor_faithfulness.py             # score + plot
    python scripts/figures/build_val_anchor_faithfulness.py --plot_only # reuse the json

gctx8_weak/strong read checkpoints_global_context_constlam/; the gctx8 unanchored control (lambda 0)
lives in checkpoints_global_context_none/. Arms with no checkpoint yet are skipped and noted.
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
# fixed lambda per arm (no best-of selection). None = unanchored (lambda 0).
LAM_SUFFIX = {None: "", 0.1: "_anchor0.1-shape-rule", 0.3: "_anchor0.3-shape-rule"}

# arm key -> (label, ckpt template {suffix}/{seed}, lam, architecture, scheme, faith_method). lam is
# the fixed anchor strength (None=unanchored, 0.1=weak, 0.3=strong); every anchored arm is uniform.
# faith_method: "additive" = exact per-atom scores; "attention" = LigandFormer attention (abs-referenced,
# unsigned magnitude); None = no interpretability method (drawn as a hatched n/a bar).
_SUM = "checkpoints_anchor_sweep/ablation_gin_additive_none{suffix}_init{seed}"
_SUMMEAN = "checkpoints_anchor_sweep_summean/ablation_gin_additive_none{suffix}_init{seed}"
_GCTX8_NONE = "checkpoints_global_context_none/ablation_gin_additive_none{suffix}_gctx8_init{seed}"
_GCTX8_ANCH = (
    "checkpoints_global_context_constlam/ablation_gin_additive_none{suffix}_gctx8_init{seed}"
)
ARMS = [
    ("sum_none", "no-gctx sum · no anchor", _SUM, None, "sum", "none", "additive"),
    ("sum_weak", "no-gctx sum · weak (λ=0.1)", _SUM, 0.1, "sum", "weak", "additive"),
    ("sum_strong", "no-gctx sum · strong (λ=0.3)", _SUM, 0.3, "sum", "strong", "additive"),
    ("summean_none", "no-gctx sum-mean · no anchor", _SUMMEAN, None, "summean", "none", "additive"),
    (
        "summean_weak",
        "no-gctx sum-mean · weak (λ=0.1)",
        _SUMMEAN,
        0.1,
        "summean",
        "weak",
        "additive",
    ),
    (
        "summean_strong",
        "no-gctx sum-mean · strong (λ=0.3)",
        _SUMMEAN,
        0.3,
        "summean",
        "strong",
        "additive",
    ),
    ("gctx8_none", "MolLedger · no anchor", _GCTX8_NONE, None, "gctx8", "none", "additive"),
    ("gctx8_weak", "MolLedger · weak (λ=0.1)", _GCTX8_ANCH, 0.1, "gctx8", "weak", "additive"),
    ("gctx8_strong", "MolLedger · strong (λ=0.3)", _GCTX8_ANCH, 0.3, "gctx8", "strong", "additive"),
    (
        "pooled",
        "Pooled GNN",
        "checkpoints_pooled_seeds/ablation_gin_pooled_none_init{seed}",
        None,
        "ref",
        "pooled",
        None,
    ),
    (
        "ligandformer",
        "LigandFormer",
        "checkpoints_ligandformer_seeds/ablation_gin_ligandformer_none_init{seed}",
        None,
        "ref",
        "lf",
        "attention",
    ),
]

# Color encodes architecture (hue family) x anchor scheme (shade, deepening with strength). Legend
# is a 3x3 swatch grid (arch columns x scheme rows) + 2 reference boxes. gctx8 = greens, sum-mean =
# blues, sum = purples.
ARCH_SCHEME_COLOR = {
    ("sum", "none"): "#DADAEB",
    ("sum", "weak"): "#9E9AC8",
    ("sum", "strong"): "#54278F",  # Purples
    ("summean", "none"): "#C6DBEF",
    ("summean", "weak"): "#6BAED6",
    ("summean", "strong"): "#08519C",  # Blues
    ("gctx8", "none"): "#C7E9C0",
    ("gctx8", "weak"): "#74C476",
    ("gctx8", "strong"): "#238B45",  # Greens
}
REF_COLOR = {"pooled": "#969696", "lf": "#525252"}
ARCH_HEADER = {
    "sum": "No context\nsum",
    "summean": "No context\nsum + mean",
    "gctx8": "Context\n(MolLedger)",
}
SCHEME_ROW = {"none": "No anchor", "weak": "Weak (λ=0.1)", "strong": "Strong (λ=0.3)"}


def _bar_color(arch, scheme):
    return REF_COLOR[scheme] if arch == "ref" else ARCH_SCHEME_COLOR[(arch, scheme)]


def _grid_legend(fig):
    """3x3 swatch grid (architecture columns x anchor scheme rows) plus two reference boxes, on a
    dedicated top axes."""
    from matplotlib.patches import Rectangle

    axl = fig.add_axes([0.08, 0.845, 0.84, 0.13])
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.axis("off")
    xs = {"sum": 0.40, "summean": 0.52, "gctx8": 0.64}
    ys = {"none": 0.52, "weak": 0.32, "strong": 0.12}
    sw, sh = 0.028, 0.17
    for a in ("sum", "summean", "gctx8"):
        axl.text(xs[a], 0.84, ARCH_HEADER[a], ha="center", va="center", fontsize=11)
        for s in ("none", "weak", "strong"):
            axl.add_patch(
                Rectangle(
                    (xs[a] - sw / 2, ys[s] - sh / 2),
                    sw,
                    sh,
                    facecolor=ARCH_SCHEME_COLOR[(a, s)],
                    edgecolor="black",
                    linewidth=0.4,
                )
            )
    for s in ("none", "weak", "strong"):
        axl.text(0.35, ys[s], SCHEME_ROW[s], ha="right", va="center", fontsize=11)
    for name, col, y in [
        ("Pooled GNN", REF_COLOR["pooled"], 0.44),
        ("LigandFormer", REF_COLOR["lf"], 0.16),
    ]:
        axl.add_patch(
            Rectangle(
                (0.74 - sw / 2, y - sh / 2), sw, sh, facecolor=col, edgecolor="black", linewidth=0.4
            )
        )
        axl.text(0.775, y, name, ha="left", va="center", fontsize=11)


# top-5 tasks, matching the performance appendix (all anchored, so all have faithfulness).
TASKS_MAIN = [
    "logd",
    "kinetic_solubility",
    "clint_mouse_liver",
    "clint_human_liver",
    "caco2_papp_ab",
]
COUNTS_JSON = ROOT / "runs/val_accuracy_appendix.json"  # for the n= panel titles

JSON_OUT = ROOT / "runs/val_anchor_faithfulness.json"
PDF_OUT = ROOT / "plots/val_anchor_faithfulness.pdf"


def evaluate(args):
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader

    from scripts.figures.build_test_split_jsons import eval_accuracy
    from scripts.train.train_molledger import TASKS, subset_tasks
    from src.attribution import attribute, faithfulness
    from src.data.multitask import GRAPH_TASK_SPECS, apply_anchor_rule, load_multitask_splits
    from src.models.gnn import build_model_from_checkpoint

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    splits = load_multitask_splits(seed=args.split_seed, conformer_cache_path=None)
    splits = {s: subset_tasks(dl) for s, dl in splits.items()}
    data = splits[args.split]
    loader = DataLoader(data, batch_size=args.batch_size)
    print(f"[{args.split}] {len(data)} molecules | device={device}", flush=True)

    # per-task Crippen/TPSA anchor identity; anchored tasks only.
    ref_anchor = {
        s.name: a.anchor for s, a in zip(GRAPH_TASK_SPECS, apply_anchor_rule(GRAPH_TASK_SPECS))
    }
    anchored = [
        (k, ref_anchor[s.name])
        for k, s in enumerate(TASKS)
        if ref_anchor[s.name] in ("crippen", "tpsa")
    ]
    anchored_names = [TASKS[k].name for k, _ in anchored]
    all_idx = list(range(len(TASKS)))
    print(
        "faithfulness anchored tasks: " + ", ".join(f"{n}->{ref_anchor[n]}" for n in anchored_names)
    )

    warn, per_seed = [], {}
    for seed in SEEDS:
        per_seed[seed] = {}
        for key, _label, tmpl, lam, _arch, _scheme, faith_method in ARMS:
            suffix = LAM_SUFFIX[lam]
            path = ROOT / (tmpl.format(suffix=suffix, seed=seed) + "/best.pt")
            if not path.exists():
                warn.append(f"missing checkpoint {path}")
                continue
            ck = torch.load(path, map_location=device, weights_only=False)
            model, backbone = build_model_from_checkpoint(ck)
            model = model.to(device).eval()

            acc = eval_accuracy(model, loader, device, backbone)  # {task: {metric: v}}
            faith = {}
            if faith_method is not None:  # None = no interp method (hatched n/a bar)
                abs_ref = faith_method == "attention"  # LigandFormer: correlate against |anchor|
                attrs, _preds, _aux = attribute(
                    model,
                    data,
                    faith_method,
                    backbone,
                    batch_size=args.batch_size,
                    device=device,
                    task_idx=all_idx,
                )
                for k, anchor_attr in anchored:
                    pear, n = faithfulness(attrs, data, k, anchor_attr, abs_ref=abs_ref)
                    faith[TASKS[k].name] = {"pearson": pear, "n": n}
            per_seed[seed][key] = {"acc": acc, "faith": faith, "faith_method": faith_method}
            fr = (
                np.nanmean([faith[n]["pearson"] for n in anchored_names if n in faith])
                if faith
                else float("nan")
            )
            print(
                f"  seed {seed} | {key:22s}"
                + (f" lam={lam}" if lam is not None else " (unanch.)")
                + f" | faith r={fr:.3f}"
                + " done",
                flush=True,
            )

    payload = {
        "seeds": SEEDS,
        "split": args.split,
        "anchored_tasks": anchored_names,
        "arms": [(k, lab, arch, sch, fm) for k, lab, _, _, arch, sch, fm in ARMS],
        "lambda": {k: lam for k, _lab, _t, lam, _a, _s, _fm in ARMS},
        "per_seed": {str(s): v for s, v in per_seed.items()},
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

    from scripts.figures.plot_results import _n, pretty_task

    sns.set_theme(style="whitegrid", context="notebook")
    d = json.loads(JSON_OUT.read_text())
    seeds = [str(s) for s in d["seeds"]]
    arms = d["arms"]  # [(key, label, arch, scheme, faith_method)]
    ps = d["per_seed"]
    counts = (
        json.loads(COUNTS_JSON.read_text()).get("task_counts", {}) if COUNTS_JSON.exists() else {}
    )

    # Sign-orient each additive task's r by its anchor_sign so higher = more faithful (src/metrics.py:
    # faith_r). LigandFormer attention is abs-referenced/unsigned, so it is not sign-flipped.
    from src.data.multitask import ANCHOR_RULE

    faith_sign = {t: s for t, (_a, s) in ANCHOR_RULE.items()}

    def acc_cell(key, task, metric, min_seeds=2):
        vals = [
            ps[s][key]["acc"][task][metric]
            for s in seeds
            if key in ps[s]
            and ps[s][key]["acc"].get(task, {}).get(metric) is not None
            and not np.isnan(ps[s][key]["acc"][task][metric])
        ]
        return (
            (float(np.mean(vals)), float(np.std(vals))) if len(vals) >= min_seeds else (np.nan, 0.0)
        )

    def faith_cell(key, task, is_lf, min_seeds=2):
        """Mean, std over seeds of faithfulness Pearson r (sign-oriented for additive arms; raw for
        LigandFormer attention). Returns (nan, 0) below min_seeds."""
        sgn = 1 if is_lf else faith_sign.get(task, 1)
        vals = []
        for s in seeds:
            f = ps[s].get(key, {}).get("faith", {}).get(task)
            if f and f["pearson"] is not None and not np.isnan(f["pearson"]):
                vals.append(sgn * f["pearson"])
        return (
            (float(np.mean(vals)), float(np.std(vals))) if len(vals) >= min_seeds else (np.nan, 0.0)
        )

    # x layout: 3 additive groups x {no-anchor, weak, strong}, then a references group.
    add_arms = [a for a in arms if a[4] == "additive"]
    ref_arms = [a for a in arms if a[4] != "additive"]
    archs = ["sum", "summean", "gctx8"]
    arch_label = {
        "sum": "no-gctx\n(sum)",
        "summean": "no-gctx\n(sum-mean)",
        "gctx8": "MolLedger\n(gctx8)",
    }
    schemes = ["none", "weak", "strong"]
    arm_by = {(a[2], a[3]): a[0] for a in add_arms}
    w, gap = 0.92, 0.5
    xpos, group_ticks, group_labels = {}, [], []
    cur = 0.0
    for a in archs:
        centers = []
        for s in schemes:
            xpos[arm_by[(a, s)]] = cur
            centers.append(cur)
            cur += w
        group_ticks.append(float(np.mean(centers)))
        group_labels.append(arch_label[a])
        cur += gap
    ref_centers = []
    for k, _l, _a, _s, _fm in ref_arms:
        xpos[k] = cur
        ref_centers.append(cur)
        cur += w
    if ref_centers:
        group_ticks.append(float(np.mean(ref_centers)))
        group_labels.append("reference")

    rows = [
        ("mae", "MAE ↓", "auto"),
        ("spearman", "Spearman ↑", (0, 1)),
        ("faith", "Faithfulness ↑", "auto"),
    ]
    ncol = len(TASKS_MAIN)
    fig, axes = plt.subplots(3, ncol, figsize=(3.0 * ncol, 9.4), squeeze=False)
    pending = set()
    for r, (rkey, rlabel, ylim) in enumerate(rows):
        for c, task in enumerate(TASKS_MAIN):
            ax = axes[r][c]
            hatch_na = []  # x slots to mark n/a
            for k, _l, a, s, fm in arms:
                if rkey == "faith":
                    if fm is None:  # no interp method -> hatched n/a
                        hatch_na.append(xpos[k])
                        continue
                    m, sd = faith_cell(k, task, is_lf=(fm == "attention"))
                else:
                    m, sd = acc_cell(k, task, rkey)
                if np.isnan(m):
                    if rkey == "faith":
                        pending.add(k)
                    continue
                ax.bar(
                    xpos[k],
                    m,
                    w,
                    yerr=sd,
                    color=_bar_color(a, s),
                    edgecolor="black",
                    linewidth=0.4,
                    capsize=2,
                    error_kw={"elinewidth": 0.8},
                )
            if ylim == (0, 1):
                ax.set_ylim(0, 1)
            if rkey == "faith":
                ax.set_ylim(bottom=0)
                ax.set_ylim(top=ax.get_ylim()[1] * 1.06)  # small headroom
                for x in hatch_na:  # full-column diagonal hatch
                    ax.axvspan(
                        x - w / 2,
                        x + w / 2,
                        facecolor="none",
                        hatch="////",
                        edgecolor="0.6",
                        linewidth=0.0,
                        zorder=0.5,
                    )
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=9)
            if r == 0:
                ax.set_title(f"{pretty_task(task)}\n(n={_n(counts.get(task, 0))})", fontsize=11)
            if c == 0:
                ax.set_ylabel(rlabel, fontsize=12)
            sns.despine(ax=ax)

    _grid_legend(fig)
    fig.suptitle("MolLedger head and anchor design", fontsize=15, y=0.995)
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(rect=(0, 0, 1, 0.845))
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
        evaluate(args)
    plot()


if __name__ == "__main__":
    main()
