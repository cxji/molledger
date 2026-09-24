"""Non-circular descriptor concordance: whether anchoring makes per-atom attributions agree with
chemistry the model was never trained against:

  * LogD                -> hydrophobic atom (nonpolar C/halogen, binary)
  * ppb_mouse_plasma    -> aromatic atom (binary)
  * caco2_papp_ab       -> HBD (H-bond donor, binary)
  * caco2_efflux_ratio  -> HBA (H-bond acceptor, binary)

Concordance measure: per-molecule AUROC for binary descriptors -- within a molecule, the probability
that a randomly chosen descriptor atom scores higher than a randomly chosen non-descriptor atom. Each
task is oriented by an expected sign so higher = more concordant; the sign says whether descriptor atoms
should score HIGHER (+1) or LOWER (-1).

Usage:
    python scripts/figures/plot_noncircular_concordance.py              # score + plot (all test mols)
    python scripts/figures/plot_noncircular_concordance.py --plot_only  # reuse the json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.transforms import Bbox
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.figures.build_test_split_jsons import (  # noqa: E402
    model_ckpt_paths,
    select_best_lambda,
    select_best_lambda_gnan,
)
from scripts.train.train_molledger import TASKS  # noqa: E402
from src.attribution import attribute  # noqa: E402
from src.data.multitask import build_registry, load_multitask_splits  # noqa: E402
from src.models.gnn import build_model_from_checkpoint  # noqa: E402

SEEDS = [19, 209, 31]
JSON_OUT = ROOT / "runs/noncircular_concordance.json"
PDF_OUT = ROOT / "plots/noncircular_concordance.pdf"

TASK_INDEX = {s.name: k for k, s in enumerate(TASKS)}

# Interpretability methods, matching the interp_heldout figure set: (key, label, ckpt_key, method, extra).
# ckpt_key indexes model_ckpt_paths(); method + extra are passed to src.attribution.attribute().
# LigandFormer is excluded: a non-negative, directionless per-atom weight has no signed concordance.
ARMS = [
    ("ours_best", "Anchored MolLedger", "additive_best", "additive", {}),
    ("summean_best", "MolLedger (no context)", "summean_best", "additive", {}),
    ("ours_none", "Unanchored MolLedger", "additive_none", "additive", {}),
    ("gnan_best", "Anchored GNAN", "gnan_best", "additive", {}),
    ("gnan", "Unanchored GNAN", "gnan", "additive", {}),
    ("ig_pool_zeros", "IG", "pooled_none", "integrated_gradients", {"baseline": "zeros"}),
    ("gradcam_pool", "Grad-CAM", "pooled_none", "grad_cam", {"gradcam_relu": False}),
    ("lime_pool", "LIME", "pooled_none", "lime", {}),
    ("wisp_pool", "WISP", "pooled_none", "wisp", {}),
]
ARM_LABEL = {k: lab for k, lab, *_ in ARMS}

# Colours + legend order mirror interp_heldout so the two figures read as one set.
METHOD_COLORS = {
    "ours_best": "#0072B2",
    "ours_none": "#E69F00",
    "summean_best": "#785EF0",
    "gnan_best": "#000000",
    "gnan": "#CC79A7",
    "ig_pool_zeros": "#009E73",
    "gradcam_pool": "#56B4E9",
    "lime_pool": "#D55E00",
    "wisp_pool": "#8B4513",
}
DISPLAY_ORDER = [
    "ours_best",
    "summean_best",
    "ours_none",
    "gnan_best",
    "gnan",
    "ig_pool_zeros",
    "gradcam_pool",
    "lime_pool",
    "wisp_pool",
]

# task -> (descriptor kind, expected sign). sign orients concordance so higher = more concordant.
# sign +1 => descriptor atoms should score HIGHER, -1 => LOWER.
SPEC = {
    "logd": ("hydrophobic", +1),
    "ppb_mouse_plasma": ("aromatic", -1),
    "caco2_papp_ab": ("hbd", -1),
    "caco2_efflux_ratio": ("hba", +1),
}

# H-bond donor / acceptor per-atom labels via standard SMARTS.
_DONOR_SMARTS = Chem.MolFromSmarts("[#7,#8;!H0]")  # any N/O bearing at least one H
_ACCEPTOR_SMARTS = Chem.MolFromSmarts(
    "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
)
_POLAR = {"N", "O", "S", "P"}


def _atom_descriptors(smiles, n_atoms):
    """Per-atom descriptor arrays aligned to MolFromSmiles(smiles) order (== featurizer order).
    Returns dict or None if the mol fails / atom count mismatches."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() != n_atoms:
        return None
    hydrophobic = np.array([a.GetSymbol() not in _POLAR for a in mol.GetAtoms()], bool)
    donor = np.zeros(n_atoms, bool)
    for (i,) in mol.GetSubstructMatches(_DONOR_SMARTS):
        donor[i] = True
    acceptor = np.zeros(n_atoms, bool)
    for (i,) in mol.GetSubstructMatches(_ACCEPTOR_SMARTS):
        acceptor[i] = True
    aromatic = np.array([a.GetIsAromatic() for a in mol.GetAtoms()], bool)
    return {
        "hbd": donor,
        "hba": acceptor,
        "aromatic": aromatic,
        "hydrophobic": hydrophobic,
    }


def _auroc(scores, pos):
    """P(random positive scores higher than random negative), via the rank-sum identity."""
    pos = np.asarray(pos, bool)
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    s = np.asarray(scores, float)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    avg = {}
    start = 0
    for j, c in enumerate(counts):  # average ranks for ties
        avg[j] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[j] for j in inv])
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def concordance_for_arm(
    model,
    backbone,
    method,
    extra,
    data,
    descs,
    device,
    smiles=None,
    wisp_cache=None,
    ig_steps=128,
    lime_samples=300,
):
    """{task: (mean per-molecule concordance, n_mol)} for one model/method over the test set.
    The SPEC sign orients each metric so higher = more concordant."""
    all_idx = list(range(len(TASKS)))
    kw = dict(task_idx=all_idx, **extra)
    if method == "integrated_gradients":
        kw.setdefault("steps", ig_steps)
    if method == "lime":
        kw.setdefault("num_samples", lime_samples)
    attr_data = data
    if method == "wisp":
        attr_data = [d.clone() for d in data]
        for d, smi in zip(attr_data, smiles or []):
            d.smiles = smi
        kw["wisp_mutants"] = wisp_cache or {}
    attrs, _preds, _aux = attribute(model, attr_data, method, backbone, device=device, **kw)

    out = {}
    for task, (kind, sign) in SPEC.items():
        k = TASK_INDEX[task]
        vals = []
        for a, dsc in zip(attrs, descs):
            if dsc is None:
                continue
            col = a[:, k].detach().cpu().numpy()
            au = _auroc(col, dsc[kind])  # per-molecule AUROC, oriented by sign
            if not np.isnan(au):
                vals.append(au if sign > 0 else 1.0 - au)
        out[task] = (float(np.mean(vals)), int(len(vals))) if vals else (np.nan, 0)
    return out


def build(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arms = [a for a in ARMS if (not args.arms or a[0] in args.arms)]
    print("loading splits + registry ...")
    splits, keys = load_multitask_splits(return_keys=True)
    test = splits["test"]
    test_keys = keys["test"]
    reg = build_registry()
    smiles = [reg[q].smiles for q in test_keys]
    descs = [_atom_descriptors(s, d.x.shape[0]) for s, d in zip(smiles, test)]
    n_ok = sum(x is not None for x in descs)
    print(
        f"test molecules: {len(test)} ({n_ok} with usable descriptors); arms: {[a[0] for a in arms]}"
    )

    wisp_cache = None
    if any(a[3] == "wisp" for a in arms) and args.wisp_cache and Path(args.wisp_cache).exists():
        wisp_cache = torch.load(args.wisp_cache, map_location="cpu", weights_only=False)
        print(f"loaded WISP cache: {len(wisp_cache)} mols")

    warn = []
    per_seed = {a[0]: [] for a in arms}
    for seed in SEEDS:
        best_lam = select_best_lambda(str(ROOT), seed, None, warn)
        best_lam_gnan = select_best_lambda_gnan(str(ROOT), seed, None, warn)
        paths = model_ckpt_paths(str(ROOT), seed, best_lam, best_lam_gnan)
        loaded = {}  # ckpt_key -> (model, backbone), reused across arms sharing a checkpoint
        for key, label, ckpt_key, method, extra in arms:
            path = Path(paths[ckpt_key])
            if not path.exists():
                warn.append(f"missing checkpoint {path} ({key})")
                continue
            if ckpt_key not in loaded:
                ck = torch.load(path, map_location=device, weights_only=False)
                model, backbone = build_model_from_checkpoint(ck)
                model.load_state_dict(ck["model"])
                loaded[ckpt_key] = (model.to(device).eval(), backbone)
            model, backbone = loaded[ckpt_key]
            print(f"seed {seed}  {key:14s} [{method}]  {path.name}")
            per_seed[key].append(
                concordance_for_arm(
                    model,
                    backbone,
                    method,
                    extra,
                    test,
                    descs,
                    device,
                    smiles=smiles,
                    wisp_cache=wisp_cache,
                    ig_steps=args.ig_steps,
                    lime_samples=args.lime_samples,
                )
            )
    if warn:
        print("WARN:", "; ".join(sorted(set(warn))))

    result = {"seeds": SEEDS, "tasks": {}}
    for task in SPEC:
        kind, sign = SPEC[task]
        result["tasks"][task] = {"kind": kind, "sign": sign, "arms": {}}
        for key in per_seed:
            vs = [d[task][0] for d in per_seed[key] if not np.isnan(d[task][0])]
            ns = [d[task][1] for d in per_seed[key]]
            result["tasks"][task]["arms"][key] = {
                "mean": float(np.mean(vs)) if vs else None,
                "sd": float(np.std(vs)) if vs else None,
                "n_mol": int(np.median(ns)) if ns else 0,
            }
    JSON_OUT.parent.mkdir(exist_ok=True)
    # merge into any existing json so a partial --arms run adds to, not overwrites, prior arms
    if JSON_OUT.exists() and args.arms:
        prev = json.loads(JSON_OUT.read_text())
        for task in result["tasks"]:
            prev.setdefault("tasks", {}).setdefault(task, result["tasks"][task])
            prev["tasks"][task].setdefault("arms", {}).update(result["tasks"][task]["arms"])
        result = prev
    JSON_OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {JSON_OUT}")
    return result


# explicit per-task subplot titles (task: descriptor)
PANEL_TITLE = {
    "logd": "LogD:\nHydrophobic atom",
    "ppb_mouse_plasma": "Ppb Mouse Plasma:\nAromatic atom",
    "caco2_papp_ab": "Caco2 Papp Ab:\nHBD",
    "caco2_efflux_ratio": "Caco2 Efflux Ratio:\nHBA",
}


def plot(result):
    """One panel per task; one bar per interpretability method present in the json. All panels are
    per-molecule AUROC with a 0.5 chance baseline. Method order/colours follow ARMS."""
    sns.set_style("whitegrid")
    tasks = list(SPEC)
    present = [
        k
        for k in DISPLAY_ORDER
        if any(result["tasks"][t]["arms"].get(k, {}).get("mean") is not None for t in tasks)
    ]
    cmap = {k: METHOD_COLORS[k] for k in present}

    # panel geometry matches interp_heldout: same cell size and gap, centred in the same width
    FIG_W = 10.753
    FIG_H = 2.95
    CELL_W, CELL_H = 1.461, 1.104
    GAP = 0.668
    BOT = 0.30
    n = len(tasks)
    left0 = (FIG_W - (n * CELL_W + (n - 1) * GAP)) / 2.0
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axes = [
        fig.add_axes(
            [
                (left0 + i * (CELL_W + GAP)) / FIG_W,
                BOT / FIG_H,
                CELL_W / FIG_W,
                CELL_H / FIG_H,
            ]
        )
        for i in range(n)
    ]
    for ax, t in zip(axes, tasks):
        node = result["tasks"][t]
        x = np.arange(len(present))
        ys = [node["arms"].get(k, {}).get("mean") for k in present]
        es = [node["arms"].get(k, {}).get("sd") or 0 for k in present]
        ax.bar(
            x,
            [y if y is not None else np.nan for y in ys],
            yerr=es,
            color=[cmap[k] for k in present],
            edgecolor="black",
            linewidth=0.4,
            capsize=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([])
        n_mol = max((node["arms"].get(k, {}).get("n_mol") or 0) for k in present)
        title = PANEL_TITLE.get(t, t.replace("_", " ") + ":\n")
        task_line, _, prop_line = title.partition(":\n")
        n_fmt = f"{n_mol:,}".replace(",", "{,}")  # thousands separator inside mathtext
        ax.set_title(f"{task_line}\n" + rf"$(n = {n_fmt})$" + f":\n{prop_line}", fontsize=11)
        ax.set_ylim(0, 1)  # all panels are AUROC -> fixed 0-1 scale
        ax.tick_params(axis="y", labelsize=9)
        ax.xaxis.grid(False)
        sns.despine(ax=ax)
    axes[0].set_ylabel("AUROC " + r"$\uparrow$", fontsize=12)
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor=cmap[k], edgecolor="black", linewidth=0.4, label=ARM_LABEL[k])
            for k in present
        ],
        loc="upper center",
        ncol=int(np.ceil(len(present) / 2)),
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.0 - 0.32 / FIG_H),
    )
    fig.suptitle("Concordance with physical properties", y=1.0 - 0.15 / FIG_H, fontsize=15)
    PDF_OUT.parent.mkdir(exist_ok=True)
    # crop vertically to the content, keep the full FIG_W horizontally
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    pad = 0.15
    fig.savefig(PDF_OUT, bbox_inches=Bbox([[0.0, tb.y0 - pad], [FIG_W, tb.y1 + pad]]))
    plt.close(fig)
    print(f"wrote {PDF_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arms",
        nargs="*",
        default=None,
        help="subset of ARMS keys to score (default: all). Results merge into the json.",
    )
    ap.add_argument("--ig_steps", type=int, default=128)
    ap.add_argument("--lime_samples", type=int, default=300)
    ap.add_argument("--wisp_cache", default="runs/ig_grid/wisp_mutants.pt")
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()
    result = json.loads(JSON_OUT.read_text()) if args.plot_only else build(args)
    # compact table: rows = arms present, cols = tasks
    present = [
        k
        for k, *_ in ARMS
        if any(result["tasks"][t]["arms"].get(k, {}).get("mean") is not None for t in SPEC)
    ]
    hdr = "".join(f"{t.split('_')[0][:9]:>12s}" for t in SPEC)
    print(f"\n{'method':22s}{hdr}")
    for k in present:
        cells = ""
        for t in SPEC:
            m = result["tasks"][t]["arms"].get(k, {})
            cells += f"{m['mean']:>12.3f}" if m.get("mean") is not None else f"{'--':>12s}"
        print(f"{ARM_LABEL[k]:22s}{cells}")
    plot(result)


if __name__ == "__main__":
    main()
