"""
Validation-split accuracy for the two non-additive reference arms (Pooled GNN, LigandFormer),
shared by both val ablation figures: build_val_global_context_ablation.py (MAE/Spearman reference
bars) and build_val_anchor_faithfulness.py (task_counts, for the n= panel titles).

Runs the same per-task accuracy (native units, via
scripts.figures.build_test_split_jsons.eval_accuracy) as the main held-out test-split figure
(plots/perf_mae_spearman.pdf), but on the VALIDATION split.

Arms (backbone=gin, descriptors none, 3 init seeds): Pooled GNN and LigandFormer.

Usage:
    python scripts/figures/build_val_accuracy_jsons.py --json_out runs/val_accuracy_appendix.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torch_geometric.loader import DataLoader  # noqa: E402

from scripts.figures.build_test_split_jsons import eval_accuracy  # noqa: E402
from scripts.train.train_molledger import (  # noqa: E402
    NTASK,
    TASKS,
    subset_tasks,
)
from src.data.multitask import load_multitask_splits  # noqa: E402
from src.models.gnn import build_model_from_checkpoint  # noqa: E402

SEEDS = [19, 209, 31]
ANCHOR_SUFFIX = {0.1: "_anchor0.1-shape-rule", 0.3: "_anchor0.3-shape-rule"}

# arm key -> (pretty label, checkpoint-dir template, is_anchored).
# {suffix} is "" for the unanchored arm or ANCHOR_SUFFIX[lam] for the best-anchor arm; {seed} the init.
ARMS = [
    (
        "pooled",
        "Pooled GNN",
        "checkpoints_pooled_seeds/ablation_gin_pooled_none{suffix}_init{seed}",
        False,
    ),
    (
        "ligandformer",
        "LigandFormer",
        "checkpoints_ligandformer_seeds/ablation_gin_ligandformer_none{suffix}_init{seed}",
        False,
    ),
]


def select_best_lambda(repo, tmpl, seed, warn):
    """Per-seed argmin validation best_metric over lambda {0.1, 0.3} for one anchored arm family."""
    best_lam, best_val = None, None
    for lam in (0.1, 0.3):
        ck = Path(repo) / (tmpl.format(suffix=ANCHOR_SUFFIX[lam], seed=seed) + "/best.pt")
        if not ck.exists():
            warn.append(f"missing {ck} for best-anchor selection")
            continue
        m = torch.load(ck, map_location="cpu", weights_only=False).get("best_metric")
        if m is not None and (best_val is None or m < best_val):
            best_val, best_lam = float(m), lam
    if best_lam is None:
        warn.append(f"no readable best_metric for {tmpl} seed {seed}; defaulted to 0.1")
        return 0.1
    return best_lam


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument(
        "--split",
        default="valid",
        choices=["valid", "test"],
        help="Which split to score (default valid; this is a model-selection figure).",
    )
    p.add_argument(
        "--split_seed", type=int, default=42, help="Scaffold split seed; must match training."
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json_out", default="runs/val_accuracy_appendix.json")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    warn = []

    # One split load (no conformers) serves every arm and seed.
    splits = load_multitask_splits(seed=args.split_seed, conformer_cache_path=None)
    splits = {s: subset_tasks(dl) for s, dl in splits.items()}
    loader = DataLoader(splits[args.split], batch_size=args.batch_size)
    print(f"[{args.split}] {len(splits[args.split])} molecules | device={device}", flush=True)

    # per-task labelled (non-NaN) count on this split; feeds the n= panel titles.
    task_counts = {s.name: 0 for s in TASKS}
    for d in splits[args.split]:
        y = d.y.view(-1, NTASK)
        for k, s in enumerate(TASKS):
            task_counts[s.name] += int((~torch.isnan(y[:, k])).sum())

    best_lam = {}
    per_seed = {}
    for seed in SEEDS:
        per_seed[seed] = {}
        for key, _label, tmpl, anchored in ARMS:
            suffix = ""
            if anchored:
                lam = select_best_lambda(args.repo, tmpl, seed, warn)
                best_lam[(key, seed)] = lam
                suffix = ANCHOR_SUFFIX[lam]
            path = Path(args.repo) / (tmpl.format(suffix=suffix, seed=seed) + "/best.pt")
            if not path.exists():
                warn.append(f"missing checkpoint {path}")
                continue
            ck = torch.load(path, map_location=device, weights_only=False)
            model, backbone = build_model_from_checkpoint(ck)
            model = model.to(device).eval()
            per_seed[seed][key] = eval_accuracy(model, loader, device, backbone)
            print(
                f"  seed {seed} | {key:20s}"
                + (f" lam={best_lam[(key, seed)]}" if anchored else "")
                + " done",
                flush=True,
            )

    payload = {
        "seeds": SEEDS,
        "split": args.split,
        "task_counts": task_counts,
        "arms": [(k, lab) for k, lab, _, _ in ARMS],
        "best_lambda": {f"{k}|{s}": v for (k, s), v in best_lam.items()},
        "per_seed": per_seed,
        "warnings": warn,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"Wrote {args.json_out}")
    if warn:
        print(f"{len(set(warn))} warning(s):")
        for w in dict.fromkeys(warn):
            print("  -", w)


if __name__ == "__main__":
    main()
