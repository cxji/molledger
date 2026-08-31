"""
Per-matched-pair atom-interpretation figures. Reads runs/pair_examples_attr.json and renders, per
pair, a grid of
  rows = ruled anchor (top) + the 8 interpretability methods, in plot_results.INTERP_METHODS order
  cols = 2 molecules (left = starting molecule, right = the "swap-to" target)
Each cell = the molecule drawn with per-atom highlights coloured by that method's attribution.

Rendering:
  * Vector molecules: RDKit MolDraw2DSVG -> PDF (PyMuPDF) composited into a matplotlib vector frame.
  * Atom-only colour on a black/white atom palette; one rounded box around the changed fragment,
    always on the RIGHT.
  * Signed methods use RdBu_r normalised per method row over both members; attention is unsigned
    (Purples). The anchor row shows anchor_sign x anchor.
  * Titles show both molecules' measured labels; each cell shows that arm's model prediction.

    python scripts/figures/plot_pair_interpretations.py   # writes the 3 paper matched-pair figures
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.figures.plot_results import INTERP_METHODS  # noqa: E402  (arm -> legend label)
from scripts.score.select_pair_examples import APPENDIX_FRAGMENT, MAIN_COLUMNS  # noqa: E402
from src.data.multitask import TASK_REGISTRY  # noqa: E402
from src.data.transforms import inverse_transform  # noqa: E402

# task -> native<-model label transform, for converting predictions to native units.
_TASK_TRANSFORM = {t.name: t.label_transform for t in TASK_REGISTRY}


def _pred_native(task, val):
    """Model-space prediction -> native units (matching the displayed label). None passes through."""
    if val is None:
        return None
    tf = _TASK_TRANSFORM.get(task, "none")
    if tf == "none":
        return float(val)
    return float(inverse_transform(tf, torch.tensor(float(val))).item())


RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parents[2]
ARM_LABEL = dict(INTERP_METHODS)  # ours_best -> "Anchored MolLedger", etc.

# per-pair right-column override where the optimal direction is not monotonic: "a"/"b" goes RIGHT.
RIGHT_OVERRIDE = {"class_d/logd": "a"}


def _keys(cols):
    return [f"{c}/{t}" for c, t in cols]


_PAIR_TITLE = "Per-atom attribution on matched pairs"
GROUPS = {
    "pair_interp_main.pdf": (_keys(MAIN_COLUMNS), _PAIR_TITLE),
    "pair_interp_appendix_permeability_solubility.pdf": (_keys(APPENDIX_FRAGMENT[:3]), _PAIR_TITLE),
    "pair_interp_plasma_brain_aqueous.pdf": (
        ["class_d/ppb_mouse_plasma", "class_d/ppb_mouse_brain", "class_b/aqueous_solubility"],
        _PAIR_TITLE,
    ),
}

# cell geometry (inches): 3 pairs (=6 mol columns) tile a portrait page.
CELL_W, CELL_H = 1.02, 0.98
LABEL_GUTTER = 1.2
TITLE_STRIP = 0.60  # suptitle + header + per-molecule label above the first molecule row
KEY_STRIP = 0.52
_DIVERGE = cm.get_cmap("RdBu_r")  # + = red, - = blue
_SEQ = cm.get_cmap("Purples")  # unsigned magnitude (attention)


def pretty_task(t):
    return t.replace("_", " ").title().replace("Logd", "LogD")


def anchor_desc(attr):
    """Display name for the anchor descriptor, sign stripped: crippen{,+,-} -> 'Crippen', tpsa{,+,-}
    -> 'TPSA'. None -> None."""
    if not attr:
        return None
    a = attr.lower()
    if a.startswith("crippen"):
        return "Crippen"
    if a.startswith("tpsa"):
        return "TPSA"
    return attr


def right_member(pr):
    """Which member ('a'/'b') is the swap-to (right column)."""
    task, la, lb = pr["task"], pr["label_a"], pr["label_b"]
    key = f"{pr['class']}/{task}"
    if key in RIGHT_OVERRIDE:
        return RIGHT_OVERRIDE[key]
    if task == "aqueous_solubility":
        return "a" if la > lb else "b"  # higher solubility = more optimal
    if task in ("clint_mouse_liver", "clint_human_liver"):
        return "a" if la < lb else "b"  # lower clearance = more optimal
    if task == "caco2_papp_ab":
        return "a" if la > lb else "b"  # higher Papp A->B = more permeable
    if task == "caco2_efflux_ratio":
        return "a" if la < lb else "b"  # lower efflux ratio = less pumped out
    return "b"


# ---------------------------------------------------------------- depiction (fragment on the RIGHT)
def _pts(mol):
    conf = mol.GetConformer()
    return np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())]
    )


def _orient_fragment_right(mol, site):
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pts = _pts(mol)
    core = [i for i in range(mol.GetNumAtoms()) if i not in set(site)]
    if site and core:
        vec = pts[list(site)].mean(0) - pts[core].mean(0)
        ang = np.arctan2(vec[1], vec[0])
        c, s = np.cos(-ang), np.sin(-ang)
        pts = pts @ np.array([[c, -s], [s, c]]).T
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(float(pts[i, 0]), float(pts[i, 1]), 0.0))
    return mol


def _frag_is_right(mol, site):
    """True if the site centroid sits to the right of the core centroid (x)."""
    if not site:
        return True
    pts = _pts(mol)
    core = [i for i in range(mol.GetNumAtoms()) if i not in set(site)]
    if not core:
        return True
    return pts[list(site)].mean(0)[0] >= pts[core].mean(0)[0]


def depict_pair(sl, site_l, sr, site_r):
    """(mol_left, mol_right): right oriented fragment-right, left templated onto the shared core.
    Re-orients left by its own fragment if templating leaves it not on the right."""
    ml, mr = Chem.MolFromSmiles(sl), Chem.MolFromSmiles(sr)
    _orient_fragment_right(mr, site_r)
    try:
        res = rdFMCS.FindMCS(
            [ml, mr],
            timeout=5,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
        )
        patt = Chem.MolFromSmarts(res.smartsString)
        if patt is not None and res.numAtoms >= 2:
            rdDepictor.GenerateDepictionMatching2DStructure(ml, mr, refPatt=patt)
        else:
            _orient_fragment_right(ml, site_l)
    except Exception:
        _orient_fragment_right(ml, site_l)
    if not _frag_is_right(ml, site_l):
        _orient_fragment_right(ml, site_l)
    return ml, mr


# ---------------------------------------------------------------- per-atom colour + vector molecule
def atom_colors(values, scale, signed):
    if scale <= 0:
        scale = 1.0
    out = {}
    for i, v in enumerate(values):
        if signed:
            out[i] = _DIVERGE(0.5 + 0.5 * float(np.clip(v / scale, -1, 1)))[:3]
        else:
            out[i] = _SEQ(float(np.clip(abs(v) / scale, 0, 1)))[:3]
    return out


def mol_svg(mol, values, scale, signed, site, w, h):
    """RDKit SVG (vector) + [(x,y) fraction of each site atom] for the fragment box. Atom-only colour
    (highlightBonds=[]), BW palette so labels stay black under the fill."""
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    opt = d.drawOptions()
    opt.useBWAtomPalette()
    opt.highlightAtomRadii = {i: 0.30 for i in range(mol.GetNumAtoms())}
    colors = atom_colors(values, scale, signed) if values is not None else {}
    d.DrawMolecule(mol, highlightAtoms=list(colors), highlightBonds=[], highlightAtomColors=colors)
    fracs = [
        (d.GetDrawCoords(int(i)).x / w, d.GetDrawCoords(int(i)).y / h)
        for i in (site or [])
        if int(i) < mol.GetNumAtoms()
    ]
    d.FinishDrawing()
    return d.GetDrawingText(), fracs


# ---------------------------------------------------------------- normalisation
def _row_scale(pr, arm):
    va, vb = pr["attr_a"].get(arm), pr["attr_b"].get(arm)
    vals = [abs(x) for v in (va, vb) if v for x in v]
    return max(vals) if vals else 1.0


def _signed_anchor(pr, member):
    """anchor_sign x raw anchor for one member (None if the task has no anchor)."""
    raw = pr.get(f"anchor_{member}")
    if not raw:
        return None
    sign = pr.get("anchor_sign") or 1
    return [sign * x for x in raw]


def _anchor_scale(pr):
    vals = [abs(x) for v in (_signed_anchor(pr, "a"), _signed_anchor(pr, "b")) if v for x in v]
    return max(vals) if vals else 1.0


# ---------------------------------------------------------------- frame + jobs
def _rows(pr, arm_keys):
    return [("anchor", f"Anchor ({pr['anchor_attr'] or 'none'})")] + [
        (a, ARM_LABEL.get(a, a)) for a in arm_keys
    ]


def build_frame(pairs, keys, arm_keys, title):
    """Matplotlib vector frame (row labels, column titles, colour keys) + a list of molecule 'jobs'
    (cell rect + what to draw there); molecules composited later as vector."""
    ncol_pairs = len(keys)
    nrow = 1 + len(arm_keys)
    figw = LABEL_GUTTER + ncol_pairs * 2 * CELL_W
    figh = TITLE_STRIP + nrow * CELL_H + KEY_STRIP
    fig, axes = plt.subplots(nrow, ncol_pairs * 2, figsize=(figw, figh), squeeze=False)
    fig.patch.set_facecolor("white")
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")
    fig.subplots_adjust(
        left=LABEL_GUTTER / figw,
        right=1 - 0.006,
        top=1 - TITLE_STRIP / figh,
        bottom=KEY_STRIP / figh,
        wspace=0.03,
        hspace=0.03,
    )

    # neutral left label; anchors are not used at inference, so the descriptor is tagged per-column
    # in each anchor cell instead of named here.
    anchor_label = "Reference"

    jobs = []
    for pi, key in enumerate(keys):
        pr = pairs[key]
        side = "left_is_a" if right_member(pr) == "b" else "left_is_b"
        ml, mr = (
            depict_pair(pr["smiles_a"], pr["site_a"], pr["smiles_b"], pr["site_b"])
            if side == "left_is_a"
            else depict_pair(pr["smiles_b"], pr["site_b"], pr["smiles_a"], pr["site_a"])
        )
        left_is_a = side == "left_is_a"
        start_lab, swap_lab = (
            (pr["label_a"], pr["label_b"]) if left_is_a else (pr["label_b"], pr["label_a"])
        )
        # header centered over the pair (task name); each molecule shows its native-scale measured
        # label underneath (no "y=", no swap-to text, no delta).
        posL = axes[0][2 * pi].get_position()
        posR = axes[0][2 * pi + 1].get_position()
        header = pretty_task(pr["task"])
        fig.text(
            (posL.x0 + posR.x1) / 2, 1 - 0.38 / figh, header, ha="center", va="center", fontsize=7.5
        )
        fig.text(
            (posL.x0 + posL.x1) / 2,
            1 - 0.53 / figh,
            f"label={start_lab:.2f}",
            ha="center",
            va="center",
            fontsize=6.5,
        )
        fig.text(
            (posR.x0 + posR.x1) / 2,
            1 - 0.53 / figh,
            f"label={swap_lab:.2f}",
            ha="center",
            va="center",
            fontsize=6.5,
        )

        for r, (arm, label) in enumerate(_rows(pr, arm_keys)):
            if arm == "anchor":
                scale, signed = _anchor_scale(pr), True
                va, vb = _signed_anchor(pr, "a"), _signed_anchor(pr, "b")
                pa = pb = None
            else:
                scale, signed = _row_scale(pr, arm), pr["signed"].get(arm, True)
                va, vb = pr["attr_a"].get(arm), pr["attr_b"].get(arm)
                pa = _pred_native(
                    pr["task"], pr["pred_a"].get(arm)
                )  # -> native, matches the labels
                pb = _pred_native(pr["task"], pr["pred_b"].get(arm))
            # map member a/b -> left/right per the swap-to convention
            (lv, rv) = (va, vb) if left_is_a else (vb, va)
            (lp, rp) = (pa, pb) if left_is_a else (pb, pa)
            lsite = pr["site_a"] if left_is_a else pr["site_b"]
            rsite = pr["site_b"] if left_is_a else pr["site_a"]
            if pi == 0:
                axes[r][0].text(
                    -0.05,
                    0.5,
                    anchor_label if arm == "anchor" else label,
                    transform=axes[r][0].transAxes,
                    fontsize=6.5,
                    va="center",
                    ha="right",
                )
            # tag each anchor cell with this column's descriptor, drawn where pred sits on model rows.
            tag = anchor_desc(pr["anchor_attr"]) if arm == "anchor" else None
            for c, (mol, vals, site, pred) in enumerate([(ml, lv, lsite, lp), (mr, rv, rsite, rp)]):
                if arm == "anchor" and vals is None:
                    continue  # task has no anchor: leave the anchor cell empty
                ax = axes[r][2 * pi + c]
                jobs.append(
                    dict(
                        pos=ax.get_position(),
                        mol=mol,
                        values=vals,
                        scale=scale,
                        signed=signed,
                        site=site,
                        pred=pred,
                        tag=tag,
                    )
                )
    _add_color_keys(fig, figw, figh)
    return fig, jobs, figw, figh, title


def _add_color_keys(fig, figw, figh):
    strip = KEY_STRIP / figh
    bar_h = min(0.011, strip * 0.22)
    bar_y = strip * 0.62
    bar_w = min(0.16, 0.9 / figw)
    for x0, cmap, norm, ticks, ticklabels, label in [
        (
            0.5 - bar_w - 0.06,
            _DIVERGE,
            Normalize(-1, 1),
            [-1, 0, 1],
            ["−", "0", "+"],
            "signed attribution (per-method norm)",
        ),
        (0.5 + 0.06, _SEQ, Normalize(0, 1), [0, 1], ["0", "max"], "attention (unsigned)"),
    ]:
        cax = fig.add_axes([x0, bar_y, bar_w, bar_h])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklabels)
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cb.outline.set_linewidth(0.4)
        cb.set_label(label, fontsize=6, labelpad=2)


# ---------------------------------------------------------------- composite (vector molecules)
def _svg_to_pdf_page(svg):
    doc = pymupdf.open(stream=svg.encode(), filetype="svg")
    return pymupdf.open(stream=doc.convert_to_pdf(), filetype="pdf")


def render_figure(pairs, keys, arm_keys, title, out_path):
    fig, jobs, figw, figh, title = build_frame(pairs, keys, arm_keys, title)
    if title:
        fig.suptitle(title, fontsize=9, y=1 - 0.13 / figh)
    W, H = figw * 72.0, figh * 72.0
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        fig.savefig(tmp.name, format="pdf", facecolor="white")
        plt.close(fig)
        doc = pymupdf.open(tmp.name)
        page = doc[0]
        for job in jobs:
            p = job["pos"]
            rect = pymupdf.Rect(p.x0 * W, (1 - p.y1) * H, p.x1 * W, (1 - p.y0) * H)
            # inset below the top-left prediction/tag label so it's never covered by the molecule
            has_top = job["pred"] is not None or job.get("tag")
            mrect = pymupdf.Rect(rect.x0, rect.y0 + (12 if has_top else 0), rect.x1, rect.y1)
            pw = 320
            ph = max(1, round(pw * mrect.height / mrect.width))
            svg, fracs = mol_svg(
                job["mol"], job["values"], job["scale"], job["signed"], job["site"], pw, ph
            )
            src = _svg_to_pdf_page(svg)
            page.show_pdf_page(mrect, src, 0)
            src.close()
            if fracs:  # one rounded box around the whole changed fragment
                xs = [f[0] for f in fracs]
                ys = [f[1] for f in fracs]
                cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                pad, min_half = 0.06, 0.085  # min half-extent so a single-atom fragment gets a box
                hw = max((max(xs) - min(xs)) / 2 + pad, min_half)
                hh = max((max(ys) - min(ys)) / 2 + pad, min_half)
                box = (
                    pymupdf.Rect(
                        mrect.x0 + (cx - hw) * mrect.width,
                        mrect.y0 + (cy - hh) * mrect.height,
                        mrect.x0 + (cx + hw) * mrect.width,
                        mrect.y0 + (cy + hh) * mrect.height,
                    )
                    & mrect
                )
                page.draw_rect(box, color=(0.07, 0.07, 0.07), width=0.8, radius=0.18)
            if job["pred"] is not None:  # this arm's model prediction for the molecule
                page.insert_text(
                    (rect.x0 + 2.5, rect.y0 + 9),
                    f"pred={job['pred']:+.2f}",
                    fontsize=6.5,
                    color=(0.30, 0.30, 0.30),
                )
            elif job.get("tag"):  # mixed-anchor figure: name this column's anchor where pred sits
                page.insert_text(
                    (rect.x0 + 2.5, rect.y0 + 9), job["tag"], fontsize=6.5, color=(0.30, 0.30, 0.30)
                )
        doc.save(str(out_path))
        doc.close()
    print("wrote", out_path)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--attr_json",
        default=str(REPO / "runs/pair_examples_attr.json"),
        help="attribution cache from scripts/score/score_pair_examples.py",
    )
    args = p.parse_args()

    cache = json.loads(Path(args.attr_json).read_text())
    pairs, arm_keys = cache["pairs"], cache["arm_keys"]
    PLOTS = REPO / "plots"
    PLOTS.mkdir(parents=True, exist_ok=True)

    for fname, (keys, title) in GROUPS.items():
        keys = [k for k in keys if k in pairs]
        if keys:
            render_figure(pairs, keys, arm_keys, title, PLOTS / fname)


if __name__ == "__main__":
    main()
