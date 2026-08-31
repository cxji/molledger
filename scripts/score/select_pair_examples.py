"""
Select representative matched pairs for the 5 interpretability columns, using a deterministic,
method-independent rule. For each (class, task) the candidate population is every held-out matched
pair (>=1 test member). Keep only clean, resolvable pairs:
  * --min_atoms <= both molecules <= --max_atoms heavy atoms  (drug-sized; floor keeps a real
                                                                core/fragment split)
  * changed fragment <= --max_site heavy atoms                 (single small change, via MCS complement)
  * |meas| >= --min_sd * task_sd                               (measured effect clears the assay noise)
The displayed pair is the one with the largest |meas|/task_sd, tie-broken by fewest heavy atoms then
lowest pair index. Selection reads only structure and real labels, never a prediction.

Outputs:
  * runs/pair_examples.json          -- the chosen pair per column
  * plots/matched_pair_examples.pdf  -- one page per column: top-ranked survivors (chosen one starred),
                                        each drawn A|B with the changed fragment outlined

    python scripts/score/select_pair_examples.py
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parents[2]

# the 5 interpretability columns (class, task), matching plot_results.plot_interp_heldout_main.
# The graph-identical pairs are inherently tiny (aqueous only, <=9 heavy atoms), so MAIN_COLUMNS is
# built from drug-sized fragment swaps only, one max-|delta| example per task; the rest go to the
# appendix. Aqueous_solubility's floor is waived (small on every class), so its three collections
# (two graph-identical variants plus the fragment swap) form one appendix figure.
MAIN_COLUMNS = [
    ("class_d", "logd"),
    ("class_d", "kinetic_solubility"),  # anchor="none": anchor row is blank for this task
    ("class_d", "clint_mouse_liver"),
]
APPENDIX_FRAGMENT = [
    ("class_d", "clint_human_liver"),
    ("class_d", "caco2_papp_ab"),
    ("class_d", "caco2_efflux_ratio"),
    ("class_d", "ppb_mouse_plasma"),
    ("class_d", "ppb_mouse_brain"),
    ("class_d", "ppb_mouse_muscle"),
]
APPENDIX_AQUEOUS = [
    ("class_a", "aqueous_solubility"),  # graph-identical (both substituted)
    ("class_b", "aqueous_solubility"),  # graph-identical (one all-core)
    ("class_d", "aqueous_solubility"),  # fragment swap (small; aqueous molecules are tiny)
]
COLUMNS = MAIN_COLUMNS + APPENDIX_FRAGMENT + APPENDIX_AQUEOUS
# init-19 MolLedger-unanchored dumps are the pair CATALOG (identity + meas/label/task_sd/split only;
# no attribution is read). The graph-identical collections live in one dump; the fragment-swap
# collection lives in the other.
CATALOG = {
    "class_a": REPO / "runs/exact_grid_gctx8_constlam/pairs_init19_none.jsonl",
    "class_b": REPO / "runs/exact_grid_gctx8_constlam/pairs_init19_none.jsonl",
    "class_d": REPO / "runs/exact_grid_gctx8_constlam/pairsd_init19_none.jsonl",
}
# precomputed changed-fragment sites (the same partition scoring used), keyed by InChIKey pair.
ARTIFACT = {
    "class_a": REPO / "data/raw/multitask/matched_pairs.pt",
    "class_b": REPO / "data/raw/multitask/matched_pairs.pt",
    "class_d": REPO / "data/raw/multitask/fragment_pairs.pt",
}
_ART_CACHE = {}


def load_artifact(path):
    if path not in _ART_CACHE:
        _ART_CACHE[path] = torch.load(path, map_location="cpu", weights_only=False)
    return _ART_CACHE[path]


def _aslist(x):
    """Site fields are sometimes a single atom index (int) and sometimes a list -> normalize to list."""
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple, set)) else [int(x)]


def site_for(cls, art, a, b):
    """(site_atoms_a, site_atoms_b, site_size) from the precomputed pair meta, or None if absent.
    site_atoms_* index into molecule A (=a) / B (=b); site_size is the larger changed fragment."""
    meta = art[cls].get((a, b))
    if meta is None:
        return None
    if cls == "class_a":
        sa, sb = _aslist(meta["site_a"]), _aslist(meta["site_b"])
        return sa, sb, max(len(sa), len(sb))
    if cls == "class_b":  # only the `heavy` member is substituted; the other is all-core (no site)
        s = _aslist(meta["site"])
        return (s, [], len(s)) if meta["heavy"] == a else ([], s, len(s))
    sa, sb = _aslist(meta["site_atoms_a"]), _aslist(meta["site_atoms_b"])  # fragment swap
    return sa, sb, max(meta.get("n_atoms_a", len(sa)), meta.get("n_atoms_b", len(sb)))


CLS_DISPLAY = {
    "class_a": "Class A",
    "class_b": "Class B",
    "class_d": "Class C",
}
# clint exact zeros are LEFT-CENSORED readings (below LOQ) that the loader does NOT substitute (unlike
# ppb -> LOQ/2); see src/data/multitask.py:61-64. A censored endpoint is not a resolvable measurement,
# so a pair touching one cannot be a clean interpretation example -> drop it. (model-space label==0
# <=> native==0 <=> censored, since these tasks are log1p and log1p(0)=0.)
CENSORED_ZERO_TASKS = ("clint_mouse_liver", "clint_human_liver")


def pretty_task(t):
    return t.replace("_", " ").title().replace("Logd", "LogD")


def held_out(rec):
    return "test" in rec["split"].split("/")


def gather(cls, task, args):
    """Filtered, deduped, ranked candidate list for one (class, task). Sites come from the precomputed
    pair artifact (the exact partition scoring used) -- no MCS, so this stays fast on the ~10k-pair
    class-D columns."""
    art = load_artifact(ARTIFACT[cls])
    seen, cands = set(), []
    with open(CATALOG[cls]) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["class"] != cls or rec["task"] != task or not held_out(rec):
                continue
            sd = rec.get("task_sd") or 0.0
            if sd <= 0 or abs(rec["meas"]) < args.min_sd * sd:  # resolvable effect only
                continue
            if task in CENSORED_ZERO_TASKS and (
                abs(rec["label_a"]) < 1e-9 or abs(rec["label_b"]) < 1e-9
            ):
                continue  # a censored (below-LOQ) endpoint is not a resolvable measurement
            key = (rec["a"], rec["b"])
            if key in seen:
                continue
            a, b = Chem.MolFromSmiles(rec["smiles_a"]), Chem.MolFromSmiles(rec["smiles_b"])
            if a is None or b is None:
                continue
            if a.GetNumHeavyAtoms() > args.max_atoms or b.GetNumHeavyAtoms() > args.max_atoms:
                continue
            if (
                cls == "class_d"
                and task != "aqueous_solubility"
                and (a.GetNumHeavyAtoms() < args.min_atoms or b.GetNumHeavyAtoms() < args.min_atoms)
            ):
                continue  # drug-sized floor for drug-sized fragment tasks; aqueous & graph-identical exempt
            site = site_for(cls, art, rec["a"], rec["b"])
            if site is None or site[2] > args.max_site or site[2] == 0:
                continue
            seen.add(key)
            cands.append(
                {
                    **{
                        k: rec[k]
                        for k in (
                            "smiles_a",
                            "smiles_b",
                            "meas",
                            "label_a",
                            "label_b",
                            "task_sd",
                            "a",
                            "b",
                            "split",
                        )
                    },
                    "abs_sd": abs(rec["meas"]) / sd,
                    "site_a": site[0],
                    "site_b": site[1],
                    "n_a": a.GetNumHeavyAtoms(),
                    "n_b": b.GetNumHeavyAtoms(),
                }
            )
    # rank by resolvability (desc); the CHOSEN one is cands[0] = the MAX-|Δ|/sd survivor. Selection uses
    # only structure + REAL labels (never predictions or which method wins), so taking the largest
    # measured effect is not cherry-picking -- it is simply the clearest experimental swap.
    cands.sort(key=lambda c: (-c["abs_sd"], c["n_a"] + c["n_b"], c["a"]))
    if cands:
        cands[0]["_chosen"] = True
    return cands


def mol_png(smiles, highlight, w=260, h=200):
    mol = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    d.drawOptions().highlightBondWidthMultiplier = 12
    hlc = {i: (1.0, 0.55, 0.0) for i in highlight}  # orange = the changed fragment
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=list(highlight), highlightAtomColors=hlc
    )
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def render(all_cands, args):
    out = REPO / "plots/matched_pair_examples.pdf"
    with PdfPages(out) as pdf:
        for (cls, task), cands in all_cands.items():
            # Show a WINDOW of `show` candidates from the top of the |Δ| ranking (the chosen max-|Δ|
            # pair is rank #1), so the displayed pair is visible together with its nearest-|Δ|
            # neighbours -- the "the runners-up look comparable, it's not a lone island" transparency
            # check. Rank labels are the pair's true rank among all survivors.
            ci = next((i for i, c in enumerate(cands) if c.get("_chosen")), 0)
            lo = max(0, min(ci - args.show // 2, len(cands) - args.show))
            window = list(enumerate(cands))[lo : lo + args.show]  # (true_rank, cand)
            n = max(len(window), 1)
            fig, axes = plt.subplots(n, 2, figsize=(5.4, 1.9 * n + 0.7), squeeze=False)
            fig.suptitle(
                f"{CLS_DISPLAY[cls]} · {pretty_task(task)} — "
                f"{len(cands)} held-out pairs pass the filter "
                f"(|Δ|≥{args.min_sd}·SD, {args.min_atoms}–{args.max_atoms} atoms, ≤{args.max_site}-atom site)\n"
                f"★ = max-|Δ| (the displayed pair); #rank is by |Δ|/SD",
                fontsize=8,
            )
            for i, (rank, c) in enumerate(window):
                star = "★ " if c.get("_chosen") else ""
                for j, (sm, site, lab, natoms) in enumerate(
                    [
                        (c["smiles_a"], c["site_a"], c["label_a"], c["n_a"]),
                        (c["smiles_b"], c["site_b"], c["label_b"], c["n_b"]),
                    ]
                ):
                    ax = axes[i][j]
                    ax.imshow(mol_png(sm, site))
                    ax.axis("off")
                    tag = "A" if j == 0 else "B"
                    ax.set_title(f"{tag}: y={lab:.2f} ({natoms} atoms)", fontsize=7)
                axes[i][0].text(
                    -0.06,
                    0.5,
                    f"{star}#{rank + 1}\nΔ={c['meas']:+.2f}\n({c['abs_sd']:.1f} SD)",
                    transform=axes[i][0].transAxes,
                    fontsize=7,
                    rotation=90,
                    va="center",
                    ha="right",
                )
            if not window:
                axes[0][0].text(0.5, 0.5, "no pairs pass the filter", ha="center")
                axes[0][0].axis("off")
                axes[0][1].axis("off")
            fig.tight_layout(rect=(0.02, 0, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--min_sd", type=float, default=1.0, help="resolvable-effect floor, in task SDs")
    p.add_argument(
        "--min_atoms", type=int, default=15, help="min heavy atoms per molecule (drug-sized floor)"
    )
    p.add_argument("--max_atoms", type=int, default=32, help="max heavy atoms per molecule")
    p.add_argument(
        "--max_site", type=int, default=6, help="max heavy atoms in the changed fragment"
    )
    p.add_argument(
        "--show", type=int, default=6, help="candidates to show per column in the contact sheet"
    )
    args = p.parse_args()

    all_cands, chosen = {}, {}
    for cls, task in COLUMNS:
        cands = gather(cls, task, args)
        all_cands[(cls, task)] = cands
        ch = next((c for c in cands if c.get("_chosen")), None)
        chosen[f"{cls}/{task}"] = ch
        top = (
            f"Δ={ch['meas']:+.2f} ({ch['abs_sd']:.1f}SD), site {max(len(ch['site_a']), len(ch['site_b']))}a"
            if ch
            else "NONE"
        )
        print(f"{cls:8s} {task:20s} : {len(cands):4d} pass -> chosen {top}")

    def _emit(k, v):
        if v is None:
            return None
        cls, task = k.split(
            "/", 1
        )  # the key encodes class/task; store them explicitly for downstream
        return {
            "class": cls,
            "task": task,
            **{
                kk: v[kk]
                for kk in (
                    "smiles_a",
                    "smiles_b",
                    "meas",
                    "label_a",
                    "label_b",
                    "task_sd",
                    "abs_sd",
                    "site_a",
                    "site_b",
                    "a",
                    "b",
                    "split",
                )
            },
        }

    (REPO / "runs/pair_examples.json").write_text(
        json.dumps({k: _emit(k, v) for k, v in chosen.items()}, indent=2)
    )
    out = render(all_cands, args)
    print("Wrote", out)
    print("Wrote", REPO / "runs/pair_examples.json")


if __name__ == "__main__":
    main()
