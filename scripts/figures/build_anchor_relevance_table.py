"""LaTeX table of anchor relevance per task.

Anchor relevance is the Pearson correlation between the sum of the anchors
and the label: r = anchor_sign * Pearson(descriptor, label), computed over the
training split.

Usage:
    python scripts/figures/build_anchor_relevance_table.py   # -> runs/anchor_relevance_table.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.score.matched_pair_attribution import to_space  # noqa: E402
from src.data.multitask import (  # noqa: E402
    ANCHOR_RULE,
    TASK_INDEX,
    TASK_REGISTRY,
    build_registry,
    load_multitask_splits,
)

OUT_TEX = ROOT / "runs/anchor_relevance_table.tex"

# same test-set-size ordering as the performance/interp plots (largest first)
TESTN = json.loads((ROOT / "runs/test_sample_counts.json").read_text())

SHORT_LABEL = {
    "logd": "LogD",
    "kinetic_solubility": "Kin. sol.",
    "aqueous_solubility": "Aq. sol.",
    "clint_mouse_liver": "CLint mouse",
    "clint_human_liver": "CLint human",
    "caco2_efflux_ratio": "Caco2 efflux",
    "caco2_papp_ab": "Caco2 Papp",
    "ppb_mouse_plasma": "PPB plasma",
    "ppb_mouse_brain": "PPB brain",
    "ppb_mouse_muscle": "PPB muscle",
    "half_life": "Half-life",
}
ANCHOR_LABEL = {"crippen": r"Crippen", "tpsa": "TPSA"}


def compute_goodness_train():
    """Anchor-direction-signed Pearson(descriptor, native label) per anchored task, over the TRAIN
    split (the molecule set the anchor regulariser was trained against)."""
    reg = build_registry()
    _splits, keys = load_multitask_splits(return_keys=True)
    train_keys = set(keys["train"])

    def desc(smi, which):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return np.nan
        return Crippen.MolLogP(m) if which == "crippen" else rdMolDescriptors.CalcTPSA(m)

    out = {}
    for t, (a, sign) in ANCHOR_RULE.items():
        if a == "none":
            continue
        spec = TASK_REGISTRY[TASK_INDEX[t]]
        xs, ys = [], []
        for q in train_keys:
            r = reg[q]
            if t in r.labels:
                d = desc(r.smiles, a)
                if np.isfinite(d):
                    xs.append(d)
                    ys.append(to_space(r.labels[t], spec.label_transform, "native"))
        r_raw = pearsonr(xs, ys).statistic if len(xs) > 3 else np.nan
        out[t] = {"anchor": a, "sign": sign, "goodness": sign * float(r_raw), "n": len(xs)}
    return out


def render(data):
    rows = sorted(data.items(), key=lambda kv: -(TESTN.get(kv[0]) or 0))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Anchor relevance per task.}",
        r"\label{tab:anchor-relevance}",
        r"\begin{tabular}{lrlcc}",
        r"\toprule",
        r"Task & $n_{\text{train}}$ & Anchor descriptor & Direction & Relevance $r$ \\",
        r"\midrule",
    ]
    for t, d in rows:
        direction = r"$+$" if d["sign"] > 0 else (r"$-$" if d["sign"] < 0 else r"$0$")
        lines.append(
            f"{SHORT_LABEL.get(t, t)} & {d['n']:,} & {ANCHOR_LABEL.get(d['anchor'], d['anchor'])} & "
            f"{direction} & {d['goodness']:.2f} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    tex = render(compute_goodness_train())
    OUT_TEX.parent.mkdir(exist_ok=True)
    OUT_TEX.write_text(tex + "\n")
    print(tex)
    print(f"\nwrote {OUT_TEX}")


if __name__ == "__main__":
    main()
