#!/usr/bin/env python
"""
Matched-pair predicted-delta dumps for the GBT (227 descriptors) and pooled+descriptor GNN
baselines: per pair, dy = yhat(A) - yhat(B) vs. the measured delta. No per-atom attribution, so
leakage is null. Rows are written in the schema build_matched_pair_jsons.py reads (task, class,
split, meas, dy_true, task_sd, leakage). Units are model space (the per-task label_transform
applied), matching the exact/IG/GNAN dumps.

    python scripts/score/eval_pair_delta_baselines.py --arm gbt         --seed 19
    python scripts/score/eval_pair_delta_baselines.py --arm pooled_desc --seed 19 --device cuda

Then rebuild the tables and figures:
    python scripts/figures/build_matched_pair_jsons.py --json_out runs/matched_pair_tables.json
    python -m scripts.figures.plot_results
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from scripts.train.descriptor_baseline import fit_select
from scripts.train.train_molledger import (
    NTASK,
    TASKS,
    forward_model,
    subset_tasks,
)
from src.data.descriptors import attach_descriptors, descriptor_matrix
from src.data.multitask import build_registry, compute_global_scaffold_split, featurize_registry
from src.models.gnn import build_model_from_checkpoint

REPO = Path(__file__).resolve().parents[2]
TASK_NAMES = [t.name for t in TASKS]


# --------------------------------------------------------------------------- shared featurization
def load_feats(conformer_cache):
    """inchikey -> Data (model-space y, 3D pos), subset to the 11 continuous tasks, plus the fixed
    scaffold split membership. Molecules without a usable conformer are dropped."""
    registry = build_registry()
    split_keys = compute_global_scaffold_split(registry, seed=42)
    where = {ik: name for name, keys in split_keys.items() for ik in keys}
    raw = featurize_registry(
        registry, conformer_cache_path=conformer_cache
    )  # ik -> Data (full 12 tasks)
    feats = {ik: d for ik, d in zip(raw, subset_tasks(list(raw.values())))}  # slice y -> 11 tasks
    iks = list(feats)
    return feats, iks, where, split_keys


def label_matrix(feats, iks):
    """[n, NTASK] model-space measured labels (NaN where a molecule lacks a task)."""
    return np.stack([feats[ik].y.view(-1).numpy() for ik in iks]).astype(np.float64)


# --------------------------------------------------------------------------- GBT predictions
def predict_gbt(feats, iks, where, Y, seed):
    """Per-task model-space predictions for every featurized molecule, from a 227-descriptor GBT
    refit per scripts/train/descriptor_baseline.py --init_seed {seed}: grid-select on the
    scaffold-valid split, then a per-(seed, task) bootstrap draw over the training rows."""
    X = descriptor_matrix([feats[ik] for ik in iks])  # [n, DIM_ALL] (2D+3D), NaN-tolerant for trees
    is_tr = np.array([where.get(ik) == "train" for ik in iks])
    is_va = np.array([where.get(ik) == "valid" for ik in iks])
    preds = np.full((len(iks), NTASK), np.nan)
    for k in range(NTASK):
        ok = ~np.isnan(Y[:, k])
        tr, va = is_tr & ok, is_va & ok
        Xtr, ytr = X[tr], Y[tr, k]
        if len(ytr) < 50 or va.sum() < 5:
            continue
        # resample train rows with replacement, seeded (seed + task index).
        boot = np.random.default_rng(seed + k).integers(0, len(ytr), len(ytr))
        model, _, _ = fit_select(Xtr[boot], ytr[boot], X[va], Y[va, k], seed)
        preds[:, k] = model.predict(X)  # model-space, every molecule
        print(f"  [gbt seed {seed}] {TASK_NAMES[k]:22s} n_train={len(ytr):5d} fitted", flush=True)
    return preds


# --------------------------------------------------------------------------- pooled+desc predictions
def run_pooled_desc(feats, iks, split_keys, ckpt_dir, name, descriptor_cache, seed, device, bs):
    """Per-task model-space predictions from the pooled+descriptor GNN checkpoint for {seed}.
    attach_descriptors fits standardization on train, so it needs the full train/valid/test split."""
    splits = {
        s: [feats[k] for k in split_keys[s] if k in feats] for s in ("train", "valid", "test")
    }
    desc_dim = attach_descriptors(splits, cache_path=descriptor_cache)  # mutates .desc
    ck = torch.load(
        Path(ckpt_dir) / f"{name}_init{seed}" / "best.pt", map_location=device, weights_only=False
    )
    assert ck["desc_dim"] == desc_dim, (
        f"desc_dim mismatch: checkpoint {ck['desc_dim']} vs attached {desc_dim} -- the descriptor "
        f"cache does not match the one the model was trained with."
    )
    model, backbone = build_model_from_checkpoint(ck)
    model = model.to(device).eval()

    datas = [feats[ik] for ik in iks]  # fixed order == iks
    preds = np.full((len(iks), NTASK), np.nan)
    row = 0
    with torch.no_grad():
        for batch in DataLoader(datas, batch_size=bs, shuffle=False):
            batch = batch.to(device)
            pred, _ = forward_model(model, batch, backbone)  # model-space [B, NTASK]
            b = pred.shape[0]
            preds[row : row + b] = pred.cpu().numpy()
            row += b
    print(f"  [pooled_desc seed {seed}] scored {row} molecules", flush=True)
    return preds


# --------------------------------------------------------------------------- emit dumps
def emit(pairs, cls, feats, iks, where, Y, preds, task_sd, out_f):
    """Append one jsonl row per (pair, task) with both members featurized and both measured labels
    present. leakage is null (no per-atom decomposition for this arm)."""
    idx = {ik: r for r, ik in enumerate(iks)}
    n_rows = 0
    with open(out_f, "a") as fh:
        for i, j in pairs:
            ri, rj = idx.get(i), idx.get(j)
            if ri is None or rj is None:  # not featurized (no usable conformer)
                continue
            split = f"{where.get(i)}/{where.get(j)}"
            for k in range(NTASK):
                yi, yj, pi, pj = Y[ri, k], Y[rj, k], preds[ri, k], preds[rj, k]
                if np.isnan(yi) or np.isnan(yj) or np.isnan(pi) or np.isnan(pj):
                    continue
                fh.write(
                    json.dumps(
                        {
                            "task": TASK_NAMES[k],
                            "class": cls,
                            "split": split,
                            "a": i,
                            "b": j,  # member InChIKeys
                            "meas": float(yi - yj),
                            "dy_true": float(pi - pj),
                            "task_sd": float(task_sd[k]),
                            "leakage": None,
                        }
                    )
                    + "\n"
                )
                n_rows += 1
    return n_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["gbt", "pooled_desc"], required=True)
    p.add_argument("--seed", type=int, required=True, help="init seed in {19, 209, 31}")
    p.add_argument(
        "--pairs", default="data/raw/multitask/matched_pairs.pt", help="class_a + class_b artifact"
    )
    p.add_argument(
        "--frag_pairs", default="data/raw/multitask/fragment_pairs.pt", help="class_d artifact"
    )
    p.add_argument("--conformer_cache", default="data/raw/multitask/conformers.pt")
    p.add_argument("--descriptor_cache", default="data/raw/multitask/descriptor_cache.pt")
    p.add_argument("--ckpt_dir", default="checkpoints_pooled_desc_seeds")
    p.add_argument("--name", default="ablation_gin_pooled_none_desc2d3d-pooled")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="runs/pairdelta_grid")
    args = p.parse_args()

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{args.arm} / seed {args.seed}] featurizing registry ({args.conformer_cache}) ...",
        flush=True,
    )
    feats, iks, where, split_keys = load_feats(args.conformer_cache)
    Y = label_matrix(feats, iks)
    task_sd = np.nanstd(Y, axis=0)  # per-task model-space spread (only the sign metric uses it)

    if args.arm == "gbt":
        preds = predict_gbt(feats, iks, where, Y, args.seed)
    else:
        preds = run_pooled_desc(
            feats,
            iks,
            split_keys,
            args.ckpt_dir,
            args.name,
            args.descriptor_cache,
            args.seed,
            device,
            args.batch_size,
        )

    # graph-identical collections -> one file (matches _exact_files' pairs_*.jsonl); fragment swaps -> *_d.jsonl.
    mp = torch.load(args.pairs, map_location="cpu", weights_only=False)
    fp = torch.load(args.frag_pairs, map_location="cpu", weights_only=False)
    ab_f = out_dir / f"pairs_{args.arm}_init{args.seed}.jsonl"
    d_f = out_dir / f"pairs_{args.arm}_init{args.seed}_d.jsonl"
    ab_f.unlink(missing_ok=True)
    d_f.unlink(missing_ok=True)

    n_a = emit(mp["class_a"], "class_a", feats, iks, where, Y, preds, task_sd, ab_f)
    n_b = emit(mp["class_b"], "class_b", feats, iks, where, Y, preds, task_sd, ab_f)
    n_d = emit(fp["class_d"], "class_d", feats, iks, where, Y, preds, task_sd, d_f)
    print(f"\nWrote {ab_f}  (class_a={n_a}, class_b={n_b} rows)")
    print(f"Wrote {d_f}  (class_d={n_d} rows)")


if __name__ == "__main__":
    main()
