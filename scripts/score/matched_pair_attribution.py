"""
Score both members of each mined matched pair and decompose the prediction delta. Two ordinary
forward passes per pair -- no baselines, path integrals, or guidance. For an additive head
`y_hat = sum_i s_i`, the analog-to-analog delta splits exactly into the substituted position and
the shared core:

    dy = d_sub + d_core        (src.attribution.matched_pair_decomposition)

Because that needs only a PARTITION of each molecule, it covers Class B and every other N-changing
transform, not just the bijection-friendly Class A swaps.

Two things are reported per task:
  * correlation of predicted dy against measured dy;
  * the leakage distribution |d_core| / (|d_core| + |d_sub|) -- how much of the delta the model
    produced by re-scoring atoms it did not change.

Predictions and measured deltas are both taken in the model's training (transformed) space.

Usage:
    python scripts/score/matched_pair_attribution.py \
        --checkpoint checkpoints/multitask_baseline/best.pt --tasks logd
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.attribution import (
    attribute,
    graph_predictions,
    matched_pair_decomposition,
    score_molecules,
)
from src.data.multitask import TASK_INDEX, TASK_REGISTRY, build_registry, featurize_registry
from src.models.gnn import AdditiveGNN, GNANModel, PooledGNN, build_model_from_checkpoint
from src.models.ligandformer import LigandFormerGNN

# Which attribution path each architecture rides: "additive" = exact per-atom scores
# (AdditiveGNN/GNANModel both return (pred, scores)); "pooled" = no per-atom scores; "ligandformer"
# = attention-received (scored with --method attention).
_ATTRIBUTION_KIND = {
    AdditiveGNN: "additive",
    GNANModel: "additive",
    PooledGNN: "pooled",
    LigandFormerGNN: "ligandformer",
}


def load_model(ckpt_path, device):
    """Load a checkpoint, reconstructing whichever architecture it was trained with."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, _backbone = build_model_from_checkpoint(ckpt)
    kind = _ATTRIBUTION_KIND[type(model)]
    print(
        f"  loaded {ckpt_path}: {kind} ({type(model).__name__}) tasks={model.num_tasks} "
        f"epoch={ckpt.get('epoch')} best={ckpt.get('best_metric')}"
    )
    return model.to(device).eval(), kind, ckpt


GRAPH_TASK_NAMES = [t.name for t in TASK_REGISTRY if t.label_kind != "binary"]


def task_column(task, num_model_tasks):
    """Which output column of THIS checkpoint holds `task`.

    Two task spaces coexist in this repo and the checkpoints do not say which they used:
      12 outputs -> full TASK_REGISTRY, index = TASK_INDEX.
      11 outputs -> binary HIA dropped (train_molledger.py).
    HIA sits at registry index 10, so the two spaces AGREE for every task before it and differ
    only for `half_life` (11 -> 10). Reading the wrong column would therefore be silent for most
    tasks and wrong for exactly one, which is the worst possible failure mode -- hence resolving
    it explicitly from the head width rather than assuming.
    """
    if num_model_tasks == len(TASK_REGISTRY):
        return TASK_INDEX[task]
    if num_model_tasks == len(GRAPH_TASK_NAMES):
        if task not in GRAPH_TASK_NAMES:
            raise ValueError(
                f"'{task}' is binary and absent from this {num_model_tasks}-task "
                f"checkpoint; it only exists in the full {len(TASK_REGISTRY)}-task "
                f"space."
            )
        return GRAPH_TASK_NAMES.index(task)
    raise ValueError(
        f"checkpoint has a {num_model_tasks}-task head; expected "
        f"{len(TASK_REGISTRY)} or {len(GRAPH_TASK_NAMES)}"
    )


def to_space(v, transform, space):
    if space == "native":
        return float(v)
    if transform == "log":
        return float(np.log(max(float(v), 1e-9)))
    if transform == "log1p":
        return float(np.log1p(max(float(v), 0.0)))
    return float(v)


def collect(pairs_art, cls, registry, task, space, split_filter, graph_identical_only=False):
    """Pairs carrying `task` on both members, with the (site_a, site_b) partition resolved.

    Class A stores both sites and a bijection. Class B stores only which member is substituted
    (`heavy`) and that atom's index; the other member is all core, i.e. site=None.
    """
    spec = TASK_REGISTRY[TASK_INDEX[task]]
    where = pairs_art["split"]
    out = []
    for (i, j), meta in pairs_art[cls].items():
        vi, vj = registry[i].labels.get(task), registry[j].labels.get(task)
        if vi is None or vj is None:
            continue
        wi, wj = where.get(i), where.get(j)
        if split_filter == "cross" and wi == wj:
            continue
        # "testany": every pair with at least one held-out (test) member.
        if split_filter == "testany" and "test" not in (wi, wj):
            continue
        # "train_test" / "train_valid" / "test_valid": one member in each named split.
        if "_" in split_filter and {wi, wj} != set(split_filter.split("_")):
            continue
        if split_filter in ("train", "test", "valid") and not (wi == wj == split_filter):
            continue
        if cls == "class_d":
            # Multi-atom fragment swap: both members carry a set of substituent atoms, no bijection.
            site_a, site_b, amap = meta["site_atoms_a"], meta["site_atoms_b"], None
            linker_a, linker_b = meta.get("attach_a"), meta.get("attach_b")
        elif cls == "class_a":
            if graph_identical_only and not meta.get("graph_identical", True):
                continue
            site_a, site_b, amap = meta["site_a"], meta["site_b"], meta["atom_map"]
            linker_a, linker_b = meta.get("linker_a"), meta.get("linker_b")
        else:
            heavy = meta["heavy"]
            site_a = meta["site"] if heavy == i else None
            site_b = meta["site"] if heavy == j else None
            amap = None
            linker_a = meta.get("linker") if heavy == i else None
            linker_b = meta.get("linker") if heavy == j else None
        out.append(
            {
                "a": i,
                "b": j,
                "site_a": site_a,
                "site_b": site_b,
                "atom_map": amap,
                "linker_a": linker_a,
                "linker_b": linker_b,
                "transform": f"{meta['from']}->{meta['to']}",
                "meas": to_space(vi, spec.label_transform, space)
                - to_space(vj, spec.label_transform, space),
                "split": f"{where.get(i)}/{where.get(j)}",
            }
        )
    return out


def _sync(device):
    """CUDA is asynchronous: without a synchronize the wall clock reads kernel-launch time, not
    compute time, so a GPU attribution timing would be meaningless. No-op on CPU."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _pair_record(task, cls, args, it, sd, dy_true, dy_attr, leakage, registry):
    """One flat JSONL row per scored pair, keeping `split` so a --split_filter all run stays
    re-sliceable by split. SMILES and native-unit labels are included so examples render without a
    registry re-run."""
    return {
        "task": task,
        "class": cls,
        "method": args.method,
        "split_filter": args.split_filter,
        "split": it["split"],
        "a": it["a"],
        "b": it["b"],
        "smiles_a": registry[it["a"]].smiles,
        "smiles_b": registry[it["b"]].smiles,
        "label_a": registry[it["a"]].labels.get(task),
        "label_b": registry[it["b"]].labels.get(task),
        "transform": it["transform"],
        "task_sd": sd,
        "meas": it["meas"],
        "dy_true": dy_true,
        "dy_attr": dy_attr,
        "leakage": leakage,
    }


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return pearson(ra, rb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/ablation_gin_additive_none/best.pt")
    p.add_argument("--pairs", default="data/raw/multitask/matched_pairs.pt")
    p.add_argument("--tasks", nargs="+", default=["logd"])
    p.add_argument("--classes", nargs="+", default=["class_a", "class_b"])
    p.add_argument(
        "--method",
        choices=["additive", "ig_fixed", "lime", "grad_cam", "wisp", "attention"],
        default="additive",
        help="additive: per-atom scores, exact, additive heads only. "
        "ig_fixed: IG against ONE shared (zero) baseline, run on both members and "
        "subtracted -- the only route that yields a comparable per-atom/fragment "
        "profile for a POOLED head. "
        "attention: LigandFormer attention-received; NO leakage/completeness "
        "(attention is not cross-molecule commensurable), only dy vs measured "
        "delta-accuracy.",
    )
    p.add_argument(
        "--ig_steps",
        type=int,
        default=128,
        help="IG path steps. Completeness gap falls as O(1/steps) with no floor "
        "(0.049 at 128, 0.012 at 512 on gin/pooled/none).",
    )
    p.add_argument(
        "--lime_samples",
        type=int,
        default=500,
        help="LIME (--method lime): perturbation masks per molecule.",
    )
    p.add_argument(
        "--wisp_cache",
        default=None,
        help="WISP (--method wisp): precomputed {smiles: mutants} cache from "
        "scripts/score/precompute_wisp_mutants.py; built on the fly if omitted.",
    )
    p.add_argument(
        "--split_filter",
        choices=[
            "all",
            "cross",
            "testany",
            "train_test",
            "train_valid",
            "test_valid",
            "train",
            "valid",
            "test",
        ],
        default="all",
    )
    p.add_argument(
        "--graph_identical_only",
        action="store_true",
        help="Class A only: keep pairs whose edge_attr also matches, making R exactly 0 "
        "for the fixed-baseline route.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--dump_pairs", default=None, help="Optional JSONL path: one row per scored pair."
    )
    args = p.parse_args()

    pairs_art = torch.load(args.pairs, map_location="cpu", weights_only=False)
    registry = build_registry()
    featurized = featurize_registry(registry)
    model, kind, _ = load_model(args.checkpoint, args.device)
    where = pairs_art["split"]

    if args.method == "additive" and kind != "additive":
        raise SystemExit(
            "--method additive needs an additive head; this checkpoint is pooled. "
            "Use --method ig_fixed instead (comparable leakage)."
        )

    # ONE baseline for every molecule in the run, from TRAIN only. The per-batch mean that
    # _integrated_gradients uses by default would differ between batches, so the two members'
    # completeness targets would not cancel and the subtracted attributions would not decompose
    # the pair delta at all.
    wisp_cache = None
    if args.method == "wisp" and args.wisp_cache:
        _t0 = time.perf_counter()
        wisp_cache = torch.load(args.wisp_cache, map_location="cpu", weights_only=False)
        print(
            f"  loaded WISP mutant cache: {len(wisp_cache)} molecules "
            f"in {time.perf_counter() - _t0:.1f}s"
        )

    base_h = None
    # timing: separate the ATTRIBUTION-method cost (the fair additive-vs-IG comparison) from the
    # shared one-time setup (registry, featurize). All GPU-synced via _sync.
    timing = {
        "method": args.method,
        "device": args.device,
        "ig_steps": args.ig_steps,
        "attr_seconds": 0.0,
        "n_molecule_scorings": 0,
    }
    if args.method == "ig_fixed":
        base_h = torch.zeros(1, model.node_emb.out_features)  # off-manifold zero-embedding baseline

    report = {}
    pair_dump = []
    for task in args.tasks:
        k = task_column(task, model.num_tasks)
        spec = TASK_REGISTRY[TASK_INDEX[task]]
        allv = [
            to_space(r.labels[task], spec.label_transform, "model")
            for r in registry.values()
            if task in r.labels
        ]
        sd = float(np.std(allv))

        for cls in args.classes:
            items = [
                it
                for it in collect(
                    pairs_art,
                    cls,
                    registry,
                    task,
                    "model",
                    args.split_filter,
                    args.graph_identical_only,
                )
                if it["a"] in featurized and it["b"] in featurized
            ]
            if not items:
                print(f"\n[{task} / {cls} / {args.split_filter} / {args.method}] no usable pairs")
                continue

            keys = sorted({it["a"] for it in items} | {it["b"] for it in items})
            datas = [featurized[q] for q in keys]

            # y_hat is always needed: it defines the model's TRUE pair delta, against which the
            # predicted-vs-measured accuracy is scored.
            yhat = dict(
                zip(keys, graph_predictions(model, datas, backbone="gin", device=args.device))
            )
            attrs = None
            if args.method == "additive":
                _sync(args.device)
                _t0 = time.perf_counter()
                a = score_molecules(model, datas, backbone="gin", device=args.device)
                _sync(args.device)
                timing["attr_seconds"] += time.perf_counter() - _t0
                timing["n_molecule_scorings"] += len(datas)
                attrs = dict(zip(keys, a))
            elif args.method == "ig_fixed":
                _sync(args.device)
                _t0 = time.perf_counter()
                a, _, _ = attribute(
                    model,
                    datas,
                    "integrated_gradients",
                    backbone="gin",
                    device=args.device,
                    task_idx=[k],
                    baseline=base_h,
                    steps=args.ig_steps,
                )
                _sync(args.device)
                timing["attr_seconds"] += time.perf_counter() - _t0
                timing["n_molecule_scorings"] += len(datas)
                attrs = dict(zip(keys, a))
            elif args.method in ("lime", "grad_cam", "wisp"):
                # Post-hoc, no baseline. wisp needs the canonical SMILES on each Data; clone so the
                # tag never leaks onto the shared featurised cache.
                if args.method == "wisp":
                    datas = [d.clone() for d in datas]
                    for q, d in zip(keys, datas):
                        d.smiles = registry[q].smiles
                mkw = (
                    dict(num_samples=args.lime_samples)
                    if args.method == "lime"
                    else dict(wisp_mutants=wisp_cache)
                    if args.method == "wisp"
                    else {}
                )
                _sync(args.device)
                _t0 = time.perf_counter()
                a, _, _ = attribute(
                    model,
                    datas,
                    args.method,
                    backbone="gin",
                    device=args.device,
                    task_idx=[k],
                    **mkw,
                )
                _sync(args.device)
                timing["attr_seconds"] += time.perf_counter() - _t0
                timing["n_molecule_scorings"] += len(datas)
                attrs = dict(zip(keys, a))
            elif args.method == "attention":
                # LigandFormer attention-received. No baseline (R=0); unsigned and not cross-molecule
                # commensurable, so no leakage/completeness decomposition.
                _sync(args.device)
                _t0 = time.perf_counter()
                a, _, _ = attribute(
                    model, datas, "attention", backbone="gin", device=args.device, task_idx=[k]
                )
                _sync(args.device)
                timing["attr_seconds"] += time.perf_counter() - _t0
                timing["n_molecule_scorings"] += len(datas)
                attrs = dict(zip(keys, a))

            dy_true, dy_attr, meas, leak = [], [], [], []
            for it in items:
                true_d = float(yhat[it["a"]][k] - yhat[it["b"]][k])
                meas.append(it["meas"])
                dy_true.append(true_d)

                if args.method == "attention":
                    # Unsigned, not cross-molecule commensurable: no dy_attr/leakage.
                    dy_attr.append(float("nan"))
                    leak.append(float("nan"))
                    if args.dump_pairs:
                        pair_dump.append(
                            _pair_record(
                                task,
                                cls,
                                args,
                                it,
                                sd,
                                true_d,
                                float("nan"),
                                float("nan"),
                                registry,
                            )
                        )
                    continue

                d = matched_pair_decomposition(
                    attrs[it["a"]],
                    attrs[it["b"]],
                    site_a=it["site_a"],
                    site_b=it["site_b"],
                    atom_map=it["atom_map"],
                    linker_a=it["linker_a"],
                    linker_b=it["linker_b"],
                )

                attributed = float(d["dy"][k])
                dy_attr.append(attributed)
                leak.append(float(d["leakage"][k]))
                if args.dump_pairs:
                    pair_dump.append(
                        _pair_record(
                            task,
                            cls,
                            args,
                            it,
                            sd,
                            true_d,
                            attributed,
                            float(d["leakage"][k]),
                            registry,
                        )
                    )

            dy_true, dy_attr = np.array(dy_true), np.array(dy_attr)
            meas, leak = np.array(meas), np.array(leak)

            row = {
                "method": args.method,
                "head": kind,
                "n_pairs": len(items),
                "task_sd": sd,
                "pearson": pearson(dy_true, meas),
                "spearman": spearman(dy_true, meas),
                "median_abs_pred": float(np.median(np.abs(dy_true))),
                "median_abs_meas": float(np.median(np.abs(meas))),
                "leakage_median": float(np.nanmedian(leak)) if np.isfinite(leak).any() else None,
                "leakage_p25": float(np.nanpercentile(leak, 25))
                if np.isfinite(leak).any()
                else None,
                "leakage_p75": float(np.nanpercentile(leak, 75))
                if np.isfinite(leak).any()
                else None,
            }
            report[f"{task}/{cls}/{args.split_filter}/{args.method}"] = row

            print(
                f"\n=== {task} / {cls} / split={args.split_filter} / method={args.method} "
                f"({len(items)} pairs, model units, task SD {sd:.3f}) ==="
            )
            print(
                f"  predicted vs measured dy : pearson {row['pearson']:+.3f}   "
                f"spearman {row['spearman']:+.3f}"
            )
            print(
                f"  median |dy|              : predicted {row['median_abs_pred']:.3f}   "
                f"measured {row['median_abs_meas']:.3f}"
            )
            if row["leakage_median"] is not None:
                print(
                    f"  leakage                  : p25 {row['leakage_p25']:.3f}  "
                    f"median {row['leakage_median']:.3f}  p75 {row['leakage_p75']:.3f}"
                )

    n = timing["n_molecule_scorings"]
    timing["ms_per_molecule_scoring"] = (1000.0 * timing["attr_seconds"] / n) if n else None
    report["_timing"] = timing
    if timing["method"] in ("additive", "ig_fixed"):
        print(f"\n=== timing ({timing['method']}, {timing['device']}) ===")
        print(
            f"  attribution : {timing['attr_seconds']:.2f}s over {n} molecule-scorings"
            + (f"  ({timing['ms_per_molecule_scoring']:.2f} ms each)" if n else "")
        )
        print(
            "  NOTE: this is method-compute only; registry/featurize/IO are excluded. Compare "
            "ms_per_molecule_scoring across a matched additive vs ig_fixed run for the speedup."
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\nWrote {args.out}")

    if args.dump_pairs:
        Path(args.dump_pairs).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_pairs, "w") as f:
            for rec in pair_dump:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(pair_dump)} per-pair records to {args.dump_pairs}")


if __name__ == "__main__":
    main()
