"""
Test-split JSON tables: model accuracy, attribution faithfulness, and exactness.

Runs inference once per (model, seed) over the held-out test split and reports, aggregated as
mean [min, max] over the init seeds:

  * Accuracy (native units)  -- per task, for each of the 3 trained models (method-independent):
                                MAE (native units) and Spearman rho (scale-free).
  * Faithfulness             -- per-molecule Pearson of the per-atom attribution against the
                                per-atom Crippen/TPSA reference, averaged over molecules
                                (src.attribution.faithfulness). One column per method arm.
  * Exactness                -- mean |sum_i a_i - dy_hat| completeness gap
                                (src.attribution.completeness). One column per arm.

MODELS (3 checkpoints per seed):
  * additive, unanchored     checkpoints_anchor_sweep/ablation_gin_additive_none_init{seed}
  * additive, best anchor    checkpoints_anchor_sweep/ablation_gin_additive_none_anchor{lam}-shape-rule_init{seed}
                             lam = per-seed argmin validation best_metric over {0.1, 0.3}
  * pooled, no descriptors   checkpoints_pooled_seeds/ablation_gin_pooled_none_init{seed}

METHOD ARMS (faithfulness + exactness columns, matching the matched-pair table columns):
  * ours_none      additive scores on the unanchored additive checkpoint
  * ours_best      additive scores on the best-anchor additive checkpoint
  * ig_pool_zeros  zero-baseline integrated gradients on the pooled/none checkpoint, plus the
                   post-hoc gradcam_pool / lime_pool / wisp_pool arms on the same checkpoint

FAITHFULNESS REFERENCE:
  Per-atom reference is the ruled Crippen/TPSA anchor each model was trained toward, applied across
  all method columns. Tasks whose reference anchor is `none` get no faithfulness row.

Usage:
    python scripts/figures/build_test_split_jsons.py --json_out runs/test_split_tables.json
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

from torch_geometric.loader import DataLoader  # noqa: E402

from scripts.train.train_molledger import (  # noqa: E402
    NTASK,
    TASKS,
    forward_model,
    subset_tasks,
)
from src.attribution import attribute, completeness, faithfulness  # noqa: E402
from src.data.multitask import (  # noqa: E402
    GRAPH_TASK_SPECS,
    apply_anchor_rule,
    load_multitask_splits,
)
from src.data.transforms import inverse_transform  # noqa: E402
from src.models.gnn import build_model_from_checkpoint  # noqa: E402

SEEDS = [19, 209, 31]
ANCHOR_DIR_SUFFIX = {0.1: "_anchor0.1-shape-rule", 0.3: "_anchor0.3-shape-rule"}

# report-column key -> (human label, model key, attribution method, extra method kwargs).
# Keys match the arms in src/metrics.py so faithfulness lines up with the matched-pair metrics.
COLUMNS = [
    ("ours_none", "Ours (exact, unanchored)", "additive_none", "additive", {}),
    ("ours_best", "Ours (exact, best anchor)", "additive_best", "additive", {}),
    (
        "ig_pool_zeros",
        "IG-zero (pooled)",
        "pooled_none",
        "integrated_gradients",
        {"baseline": "zeros"},
    ),
    ("gradcam_pool", "Grad-CAM (pooled)", "pooled_none", "grad_cam", {}),
    ("lime_pool", "LIME (pooled)", "pooled_none", "lime", {}),
    # WISP element-substitution occlusion on the pooled/none checkpoint. Needs batch.smiles (attached
    # in run_seed) and the precomputed mutant cache (--wisp_cache).
    ("wisp_pool", "WISP (pooled, occlusion)", "pooled_none", "wisp", {}),
    # Sum-mean anchored, no global context: exact-additive summean head, fixed anchor 0.3, uniform
    # const-lambda scheme.
    ("summean_best", "Sum-mean (exact, anchored, no ctx)", "summean_best", "additive", {}),
    ("gnan", "GNAN", "gnan", "additive", {}),
    # GNAN trained with the shape anchor (best per-seed lambda over {0.1, 0.3}).
    ("gnan_best", "GNAN (best anchor)", "gnan_best", "additive", {}),
    # LigandFormer's native attention-received attribution (unsigned, magnitude-only); no completeness gap.
    ("ligandformer", "LigandFormer (attention)", "ligandformer", "attention", {}),
]
MODEL_COLUMNS = [  # for the accuracy tables (method-independent)
    ("additive_none", "Additive (unanchored)"),
    ("additive_best", "Additive (best anchor)"),
    ("summean_best", "Sum-mean (anchored, no ctx)"),
    ("pooled_none", "Pooled (no desc)"),
    ("gnan", "GNAN"),
    ("gnan_best", "GNAN (best anchor)"),
    ("ligandformer", "LigandFormer"),
]
ACC_METRICS = ["mae", "spearman"]


# --------------------------------------------------------------------------------------------
# accuracy metrics (native units, per task, over non-NaN entries)
# --------------------------------------------------------------------------------------------


def _pearson(a, b):
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return _pearson(ra, rb)


def _metrics(p, y):
    err = p - y
    return {
        "mae": float(np.abs(err).mean()),
        "spearman": _spearman(p, y),
    }


def eval_accuracy(model, loader, device, backbone):
    """Per-task {mae, spearman} in native units over the non-NaN test entries."""
    model.eval()
    preds = [[] for _ in range(NTASK)]
    labels = [[] for _ in range(NTASK)]
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _ = forward_model(model, batch, backbone)
            for k in range(NTASK):
                mask = ~torch.isnan(batch.y[:, k])
                if mask.any():
                    preds[k].append(pred[mask, k].cpu())
                    labels[k].append(batch.y[mask, k].cpu())
    out = {}
    nan = {m: float("nan") for m in ACC_METRICS}
    for k, spec in enumerate(TASKS):
        if not preds[k]:
            out[spec.name] = dict(nan)
            continue
        p = inverse_transform(spec.label_transform, torch.cat(preds[k])).numpy()
        y = inverse_transform(spec.label_transform, torch.cat(labels[k])).numpy()
        out[spec.name] = _metrics(p, y)
    return out


# --------------------------------------------------------------------------------------------
# checkpoint paths + per-seed best-anchor selection (mirrors build_matched_pair_jsons)
# --------------------------------------------------------------------------------------------

# family -> checkpoint-dir template for the anchor sweep; {suffix} is ANCHOR_DIR_SUFFIX[lam].
# "additive" = MolLedger: the uniform (constant-lambda) global-context head at context_dim=8,
# reconstructed from checkpoint metadata via build_model_from_checkpoint. Trained with --anchor_rule;
# best-lambda per seed over {0.1, 0.3}.
_ANCHOR_CKPT = {
    "additive": "checkpoints_global_context_constlam/"
    "ablation_gin_additive_none{suffix}_gctx8_init{seed}/best.pt",
    "gnan": "checkpoints_gnan_seeds/ablation_gin_gnan_none{suffix}-shape-rule_init{seed}/best.pt",
}


def _select_best_lambda(repo, seed, forced, warn, family):
    """Per-seed argmin validation best_metric over lambda {0.1, 0.3} for the given checkpoint family
    (the additive anchor sweep, or the GNAN anchor sweep)."""
    if forced in (0.1, 0.3):
        return forced
    tmpl = _ANCHOR_CKPT[family]
    best_lam, best_val = None, None
    for lam in (0.1, 0.3):
        suffix = f"_anchor{lam}" if family == "gnan" else ANCHOR_DIR_SUFFIX[lam]
        ck = Path(repo) / tmpl.format(suffix=suffix, seed=seed)
        if not ck.exists():
            warn.append(f"missing {ck} for {family} best-anchor selection")
            continue
        m = torch.load(ck, map_location="cpu", weights_only=False).get("best_metric")
        if m is not None and (best_val is None or m < best_val):
            best_val, best_lam = float(m), lam
    if best_lam is None:
        warn.append(f"no readable best_metric for {family} seed {seed}; defaulted to 0.1")
        return 0.1
    return best_lam


def select_best_lambda(repo, seed, forced, warn):
    return _select_best_lambda(repo, seed, forced, warn, "additive")


def select_best_lambda_gnan(repo, seed, forced, warn):
    return _select_best_lambda(repo, seed, forced, warn, "gnan")


def model_ckpt_paths(repo, seed, best_lam, best_lam_gnan):
    r = Path(repo)
    # "ours" = the uniform (constant-lambda) global-context ctx=8 head. Unanchored (lambda-0) run
    # lives in checkpoints_global_context_none/, separate from the constlam sweep dir.
    gctx8_none = "checkpoints_global_context_none/ablation_gin_additive_none"
    gctx8_anch = "checkpoints_global_context_constlam/ablation_gin_additive_none"
    return {
        "additive_none": r / f"{gctx8_none}_gctx8_init{seed}/best.pt",
        "additive_best": r / f"{gctx8_anch}{ANCHOR_DIR_SUFFIX[best_lam]}_gctx8_init{seed}/best.pt",
        # Sum-mean anchored, no global context. Fixed anchor 0.3, uniform const-lambda summean sweep.
        "summean_best": r / f"checkpoints_anchor_sweep_summean/ablation_gin_additive_none"
        f"_anchor0.3-shape-rule_init{seed}/best.pt",
        "pooled_none": r / f"checkpoints_pooled_seeds/ablation_gin_pooled_none_init{seed}/best.pt",
        "gnan": r / f"checkpoints_gnan_seeds/ablation_gin_gnan_none_init{seed}/best.pt",
        "gnan_best": r / f"checkpoints_gnan_seeds/ablation_gin_gnan_none"
        f"_anchor{best_lam_gnan}-shape-rule_init{seed}/best.pt",
        "ligandformer": r / f"checkpoints_ligandformer_seeds/"
        f"ablation_gin_ligandformer_none_init{seed}/best.pt",
    }


# --------------------------------------------------------------------------------------------
# per-seed inference: accuracy for each model, faithfulness/exactness for each method arm
# --------------------------------------------------------------------------------------------


def run_seed(
    repo,
    seed,
    best_lam,
    best_lam_gnan,
    splits,
    ref_anchor,
    args,
    device,
    warn,
    only=None,
    test_smiles=None,
    wisp_cache=None,
):
    """Return {'acc': {model: {task: {metric: v}}}, 'faith': {col: {task: (pear,n)}},
              'gap': {col: {task: gap}}}  for one init seed.

    `only` (a set of arm/model keys, e.g. {"gnan"}) restricts computation to those keys.

    `test_smiles` is the canonical SMILES aligned to splits["test"] (tags each Data for the WISP
    arm); `wisp_cache` is the precomputed {smiles: packed_mutants} dict."""
    data = splits["test"][: args.limit] if args.limit else splits["test"]
    smiles = (test_smiles[: args.limit] if args.limit else test_smiles) if test_smiles else None
    test_loader = DataLoader(splits["test"], batch_size=args.batch_size)
    paths = model_ckpt_paths(repo, seed, best_lam, best_lam_gnan)

    anchored = [
        (k, ref_anchor[s.name])
        for k, s in enumerate(TASKS)
        if ref_anchor[s.name] in ("crippen", "tpsa")
    ]
    # Faithfulness scores only the anchored tasks; the completeness gap needs no anchor, so it scores
    # every task.
    all_idx = list(range(len(TASKS)))

    cols = [c for c in COLUMNS if only is None or c[0] in only]
    acc_models = [m[0] for m in MODEL_COLUMNS if only is None or m[0] in only]
    needed = set(acc_models) | {c[2] for c in cols}  # models needed for accuracy and/or attribution

    out = {"acc": {}, "faith": {c[0]: {} for c in cols}, "gap": {c[0]: {} for c in cols}}

    # ---- load each needed model once; accuracy for the accuracy models, cache the handle ----
    loaded = {}
    for mkey, path in paths.items():
        if mkey not in needed:
            continue
        if not path.exists():
            warn.append(f"missing checkpoint {path}")
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        model, backbone = build_model_from_checkpoint(ck)
        model = model.to(device).eval()
        loaded[mkey] = (model, backbone)
        if mkey in acc_models:
            out["acc"][mkey] = eval_accuracy(model, test_loader, device, backbone)

    # ---- attribution per method arm ----
    for col, _label, mkey, method, extra in cols:
        if mkey not in loaded:
            continue
        model, backbone = loaded[mkey]
        # Attribute over every task: faithfulness reads the anchored subset below, the completeness
        # gap reads all of them.
        kw = dict(task_idx=all_idx, **extra)  # `extra` carries the IG baseline (mean/zeros) etc.
        if method == "integrated_gradients":
            kw.setdefault("steps", args.ig_steps)
        if method == "lime":
            kw.setdefault("num_samples", args.lime_samples)
        attr_data = data
        if method == "wisp":
            if not smiles:
                warn.append("wisp_pool needs --wisp_cache/registry SMILES; skipped")
                continue
            # Clone so the .smiles tag doesn't leak onto the shared test Data (used by other arms).
            attr_data = [d.clone() for d in data]
            for d, smi in zip(attr_data, smiles):
                d.smiles = smi
            kw["wisp_mutants"] = wisp_cache or {}
        attrs, preds, aux = attribute(model, attr_data, method, backbone, device=device, **kw)
        abs_ref = method == "attention"  # unsigned/magnitude-only: correlate against |anchor|
        for k, anchor_attr in anchored:
            pear, n = faithfulness(attrs, data, k, anchor_attr, abs_ref=abs_ref)
            out["faith"][col][TASKS[k].name] = (pear, n)
        # Attention has no completeness axiom, so leave its gap empty ("--") instead of reporting
        # a meaningless |sum a_i - dy_hat|.
        if method != "attention":
            base = aux.get("pred_base")
            for k in all_idx:
                out["gap"][col][TASKS[k].name] = completeness(attrs, preds, base, task_idx=[k])
    return out


# --------------------------------------------------------------------------------------------
# aggregation (mean [min, max] over seeds) + rendering
# --------------------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Molecules to attribute (0 = all test). Accuracy always uses the full split.",
    )
    p.add_argument("--ig_steps", type=int, default=256)
    p.add_argument(
        "--lime_samples",
        type=int,
        default=300,
        help="LIME perturbation masks per molecule (the lime_pool faithfulness arm).",
    )
    p.add_argument(
        "--wisp_cache",
        default="runs/ig_grid/wisp_mutants.pt",
        help="Precomputed WISP mutant cache ({smiles: packed}) for the wisp_pool arm.",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json_out", default="runs/test_split_tables.json")
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="ARM",
        help="Compute only these arm/model keys and MERGE them into the existing "
        "--json_out, leaving every other arm's numbers untouched.",
    )
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    warn = []

    # Faithfulness reference: ruled Crippen/TPSA anchors.
    ref_specs = apply_anchor_rule(GRAPH_TASK_SPECS)
    ref_anchor = {s.name: a.anchor for s, a in zip(GRAPH_TASK_SPECS, ref_specs)}

    best_lambda = {s: select_best_lambda(args.repo, s, "auto", warn) for s in SEEDS}
    best_lambda_gnan = {s: select_best_lambda_gnan(args.repo, s, "auto", warn) for s in SEEDS}

    # One split load (no conformers) serves every seed. return_keys gives per-split InChIKeys
    # aligned to the Data order, for the WISP arm's SMILES.
    splits, split_keys = load_multitask_splits(seed=42, conformer_cache_path=None, return_keys=True)
    splits = {s: subset_tasks(dl) for s, dl in splits.items()}

    only = set(args.only) if args.only else None

    # WISP setup (only when the wisp_pool arm runs): SMILES aligned to the test Data, plus the
    # mutant cache reused across all three seeds.
    test_smiles, wisp_cache = None, None
    if only is None or "wisp_pool" in only:
        from src.data.multitask import build_registry

        registry = build_registry()
        test_smiles = [registry[k].smiles for k in split_keys["test"]]
        cache_path = Path(args.wisp_cache)
        if cache_path.exists():
            wisp_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(f"loaded WISP mutant cache: {len(wisp_cache)} molecules", flush=True)
        else:
            warn.append(f"WISP cache {cache_path} missing; mutants built on the fly (slow)")
            wisp_cache = {}

    per_seed = {}
    for s in SEEDS:
        print(
            f"[seed {s}] best_lambda={best_lambda[s]} gnan={best_lambda_gnan[s]} "
            + (f"(only {sorted(only)}) " if only else "")
            + "...",
            flush=True,
        )
        per_seed[s] = run_seed(
            args.repo,
            s,
            best_lambda[s],
            best_lambda_gnan[s],
            splits,
            ref_anchor,
            args,
            device,
            warn,
            only=only,
            test_smiles=test_smiles,
            wisp_cache=wisp_cache,
        )

    if only:
        # Merge the freshly computed arms into the existing full run, preserving the untouched ones.
        base = json.loads(Path(args.json_out).read_text())
        merged = {int(k): v for k, v in base["per_seed"].items()}
        for s in SEEDS:
            for grp in ("acc", "faith", "gap"):
                merged.setdefault(s, {}).setdefault(grp, {}).update(per_seed[s][grp])
        warn = base.get("warnings", []) + warn
        per_seed = merged

    payload = {
        "seeds": SEEDS,
        "best_lambda": best_lambda,
        "best_lambda_gnan": best_lambda_gnan,
        "anchor_ref": "ruled",
        "reference_anchor": ref_anchor,
        "ig_steps": args.ig_steps,
        "per_seed": per_seed,
        "warnings": warn,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"Wrote {args.json_out}")
    if warn:
        print(f"\n{len(set(warn))} warning(s) in the run")


if __name__ == "__main__":
    main()
