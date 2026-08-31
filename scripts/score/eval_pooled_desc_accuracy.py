#!/usr/bin/env python
"""
Test-split accuracy for the pooled + descriptor-inject-after-fix arm (checkpoints_pooled_desc_seeds).

That model reads data.pos and carries 219 whole-molecule 2d3d descriptors concatenated to the pooled
graph vector, so it lives on the 3D split with descriptors attached -- which is why it cannot ride the
single 2D-split load in build_test_split_jsons.py. This computes its per-task {mae,rmse,pearson,
spearman,r2} the same way (reusing eval_accuracy), and writes runs/pooled_desc_accuracy.json for
src/metrics.py to fold in (same pattern as the GBT results).

Run:  python scripts/score/eval_pooled_desc_accuracy.py
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import torch
from torch_geometric.loader import DataLoader

from scripts.figures.build_test_split_jsons import eval_accuracy
from scripts.train.train_molledger import subset_tasks
from src.data.descriptors import attach_descriptors
from src.data.multitask import load_multitask_splits
from src.models.gnn import build_model_from_checkpoint

SEEDS = [19, 209, 31]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--ckpt_dir", default="checkpoints_pooled_desc_seeds")
    p.add_argument("--name", default="ablation_gin_pooled_none_desc2d3d-pooled")
    p.add_argument("--conformer_cache", default="data/raw/multitask/conformers.pt")
    p.add_argument("--descriptor_cache", default="data/raw/multitask/descriptor_cache.pt")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="runs/pooled_desc_accuracy.json")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 3D split + descriptors, exactly as the arm was trained (attach_descriptors is the same helper
    # the trainer used, so desc_dim comes back matching the checkpoint's 219).
    splits = load_multitask_splits(seed=42, conformer_cache_path=args.conformer_cache)
    splits = {s: subset_tasks(dl) for s, dl in splits.items()}
    desc_dim = attach_descriptors(splits, cache_path=args.descriptor_cache)
    test_loader = DataLoader(splits["test"], batch_size=args.batch_size)

    per_seed = {}
    for s in SEEDS:
        ck_path = Path(args.repo) / args.ckpt_dir / f"{args.name}_init{s}" / "best.pt"
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        assert ck["desc_dim"] == desc_dim, (
            f"desc_dim mismatch: checkpoint {ck['desc_dim']} vs attached {desc_dim} -- the descriptor "
            f"cache does not match the one the model was trained with."
        )
        model, backbone = build_model_from_checkpoint(ck)
        model = model.to(device).eval()
        per_seed[str(s)] = eval_accuracy(model, test_loader, device, backbone)
        print(f"[seed {s}] done", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"seeds": SEEDS, "desc_dim": desc_dim, "per_seed": per_seed}, indent=2)
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
