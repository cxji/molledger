"""
Cache per-atom attributions for the chosen example pairs. For each pair in runs/pair_examples.json,
at a single seed (default init 19), run every interpretability-figure arm's own checkpoint + method
on both members and store the per-atom score for the pair's task column, plus the task's ruled
Crippen/TPSA anchor. The arm set + (checkpoint, method, kwargs) come from
build_test_split_jsons.COLUMNS, so the visualization uses the same attributions as the faithfulness
tables.

Output: runs/pair_examples_attr.json, consumed by scripts/figures/plot_pair_interpretations.py.
Needs a GPU (checkpoints + IG/LIME).

    python scripts/score/score_pair_examples.py            # all 5 pairs, seed 19
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.figures.build_test_split_jsons import (  # noqa: E402
    COLUMNS,
    model_ckpt_paths,
    select_best_lambda,
    select_best_lambda_gnan,
)
from scripts.train.train_molledger import TASKS  # noqa: E402
from src.attribution import attribute  # noqa: E402
from src.data.multitask import (  # noqa: E402
    GRAPH_TASK_SPECS,
    apply_anchor_rule,
    build_registry,
    featurize_registry,
)
from src.models.gnn import build_model_from_checkpoint  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# the arms shown in the interpretability figure (plot_results.INTERP_METHODS), in that order.
ARM_KEYS = [
    "ours_best",
    "ours_none",
    "gnan_best",
    "gnan",
    "ligandformer",
    "ig_pool_zeros",
    "gradcam_pool",
    "lime_pool",
    "wisp_pool",
]
# (model_key, method, extra_kwargs) per arm, taken verbatim from build_test_split_jsons.COLUMNS.
_COL = {c[0]: (c[2], c[3], c[4]) for c in COLUMNS}
TASK_IDX = {s.name: k for k, s in enumerate(TASKS)}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--seed", type=int, default=19, help="single representative init seed")
    p.add_argument(
        "--wisp_cache",
        default=str(REPO / "runs/ig_grid/wisp_mutants.pt"),
        help="Precomputed WISP mutant cache for the wisp_pool arm.",
    )
    p.add_argument("--pairs_json", default=str(REPO / "runs/pair_examples.json"))
    p.add_argument("--out", default=str(REPO / "runs/pair_examples_attr.json"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = args.seed
    pairs = {
        k: v for k, v in json.loads(Path(args.pairs_json).read_text()).items() if v is not None
    }
    print(f"[pairs] {len(pairs)} example pairs; seed={seed}; device={device}")

    # ruled Crippen/TPSA anchor per task (the reference faithfulness is scored against)
    ref_specs = apply_anchor_rule(GRAPH_TASK_SPECS)
    ref_anchor = {s.name: a.anchor for s, a in zip(GRAPH_TASK_SPECS, ref_specs)}
    # anchor_sign is the direction the anchor supervises; the figure displays the anchor with this sign.
    ref_sign = {s.name: a.anchor_sign for s, a in zip(GRAPH_TASK_SPECS, ref_specs)}

    # featurize the whole registry once; pull each pair member by InChIKey (== the dump's a / b)
    print("[featurize] building registry (this is the slow one-time step)...", flush=True)
    registry = build_registry()
    feat = featurize_registry(registry)

    warn = []
    # WISP needs each Data tagged with its canonical SMILES + (optionally) the precomputed mutant cache.
    wisp_cache = {}
    if "wisp_pool" in ARM_KEYS:
        cp = Path(args.wisp_cache)
        if cp.exists():
            wisp_cache = torch.load(cp, map_location="cpu", weights_only=False)
            print(f"[wisp] loaded mutant cache: {len(wisp_cache)} molecules", flush=True)
        else:
            warn.append(f"WISP cache {cp} missing; mutants built on the fly")

    best_lam = select_best_lambda(args.repo, seed, "auto", warn)
    best_lam_gnan = select_best_lambda_gnan(args.repo, seed, "auto", warn)
    paths = model_ckpt_paths(args.repo, seed, best_lam, best_lam_gnan)
    print(f"[lambda] additive={best_lam} gnan={best_lam_gnan}")

    # keys (InChIKey) we actually need to score
    need = {}
    for name, pr in pairs.items():
        for m in ("a", "b"):
            if pr[m] not in feat:
                warn.append(f"{name}: member {pr[m]} not in featurized registry")
        need[name] = (pr["a"], pr["b"])

    out = {
        "seed": seed,
        "anchor_ref": "ruled",
        "best_lambda": best_lam,
        "best_lambda_gnan": best_lam_gnan,
        "arm_keys": ARM_KEYS,
        "arm_labels": {c[0]: c[1] for c in COLUMNS if c[0] in ARM_KEYS},
        "pairs": {},
    }

    # per pair, carry the identity + ruled anchor; attributions filled per arm below
    for name, pr in pairs.items():
        task = pr["task"]
        anchor_attr = ref_anchor.get(task)
        rec = {
            k: pr[k]
            for k in (
                "class",
                "task",
                "smiles_a",
                "smiles_b",
                "site_a",
                "site_b",
                "label_a",
                "label_b",
                "meas",
                "abs_sd",
                "a",
                "b",
            )
        }
        rec["anchor_attr"] = anchor_attr if anchor_attr in ("crippen", "tpsa") else None
        rec["anchor_sign"] = int(ref_sign.get(task, 1))
        for m in ("a", "b"):
            d = feat.get(pr[m])
            rec[f"anchor_{m}"] = (
                getattr(d, rec["anchor_attr"]).tolist()
                if d is not None and rec["anchor_attr"]
                else None
            )
        rec["attr_a"], rec["attr_b"], rec["signed"] = {}, {}, {}
        rec["pred_a"], rec["pred_b"] = (
            {},
            {},
        )  # each arm's model prediction for the pair's task column
        out["pairs"][name] = rec

    # one model load per arm; attribute both members of every pair that uses this arm's task column
    for arm in ARM_KEYS:
        mkey, method, extra = _COL[arm]
        path = paths.get(mkey)
        if path is None or not Path(path).exists():
            warn.append(f"missing checkpoint for arm {arm} ({mkey}): {path}")
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        model, backbone = build_model_from_checkpoint(ck)
        model = model.to(device).eval()
        signed = method != "attention"  # attention is unsigned magnitude
        print(f"[arm] {arm:14s} model={mkey:13s} method={method}", flush=True)
        for name, pr in pairs.items():
            k = TASK_IDX[pr["task"]]
            da, db = feat.get(pr["a"]), feat.get(pr["b"])
            if da is None or db is None:
                continue
            kw = dict(task_idx=[k], **extra)
            if method == "integrated_gradients":
                kw.setdefault("steps", 256)
            if method == "lime":
                kw.setdefault("num_samples", 300)
            members = [da, db]
            if method == "wisp":
                # WISP reads batch.smiles to align its mutant atoms to the featurised rows. Clone so the
                # .smiles tag never leaks onto the shared featurised registry Data; atom order is
                # preserved, so attrs still line up with the drawing.
                members = [da.clone(), db.clone()]
                members[0].smiles = registry[pr["a"]].smiles
                members[1].smiles = registry[pr["b"]].smiles
                kw["wisp_mutants"] = wisp_cache
            attrs, _preds, _aux = attribute(model, members, method, backbone, device=device, **kw)
            out["pairs"][name]["attr_a"][arm] = attrs[0][:, k].tolist()
            out["pairs"][name]["attr_b"][arm] = attrs[1][:, k].tolist()
            out["pairs"][name]["signed"][arm] = signed
            # this arm's model prediction for the pair's task column (both members)
            pnp = np.asarray(
                _preds.detach().cpu().numpy() if hasattr(_preds, "detach") else _preds
            ).reshape(2, -1)
            out["pairs"][name]["pred_a"][arm] = float(pnp[0, k])
            out["pairs"][name]["pred_b"][arm] = float(pnp[1, k])
        del model, ck
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out["warnings"] = list(dict.fromkeys(warn))
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"Wrote {args.out}")
    if warn:
        print(f"{len(out['warnings'])} warning(s):")
        for w in out["warnings"]:
            print("  -", w)


if __name__ == "__main__":
    main()
