"""IG completeness-gap vs path-steps sweep on the VALIDATION split.

Characterises IG's exactness/runtime tradeoff WITHOUT touching the test split, so the step count the
paper reports (256) is never tuned on test data. For the pooled IG arm actually plotted in the
interpretability figure -- ig_pool_zeros: the pooled_none head, baseline=zeros -- and for each init
seed, it sweeps --steps and records, over the first --limit validation molecules:

  * the per-task completeness gap  |sum_i a_i - (y_hat - y_hat(0))|  (src.attribution.completeness),
  * the IG attribution wall-time (ms/molecule, CUDA-synchronised).

The gap falls as O(1/steps) with no floor (it is pure path-discretisation error); the cost is
steps x #tasks backward passes, so it grows ~linearly in steps. Together those two columns are the
exactness-vs-timing tradeoff.

Inference only; one GPU. Writes runs/ig_steps_sweep_val.json and REFUSES to overwrite it, so no
existing cache is touched.

    python scripts/score/ig_steps_sweep.py                  # steps 16..512, 300 val molecules, 3 seeds
"""

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.figures.build_test_split_jsons import SEEDS, model_ckpt_paths
from scripts.train.train_molledger import TASKS, subset_tasks
from src.attribution import attribute, completeness
from src.data.multitask import load_multitask_splits
from src.models.gnn import build_model_from_checkpoint


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512],
        help="Path-integral step counts to sweep.",
    )
    ap.add_argument("--limit", type=int, default=300, help="Validation molecules to attribute.")
    ap.add_argument(
        "--split",
        default="valid",
        choices=["valid", "train", "test"],
        help="Default 'valid' -- the whole point is to not touch 'test'.",
    )
    ap.add_argument(
        "--baseline",
        default="zeros",
        choices=["zeros", "mean"],
        help="Matches the plotted arm (ig_pool_zeros -> zeros).",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Scaffold-split seed (fixed at 42, must match training).",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/ig_steps_sweep_val.json")
    args = ap.parse_args()

    out = Path(args.repo) / args.out if not Path(args.out).is_absolute() else Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing {out}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # The pooled IG arm uses no 3D/descriptors, so one split load (no conformers) serves every seed.
    splits = load_multitask_splits(seed=args.split_seed, conformer_cache_path=None)
    splits = {s: subset_tasks(dl) for s, dl in splits.items()}
    data = splits[args.split][: args.limit]
    n_atoms = sum(d.num_nodes for d in data)
    all_idx = list(range(len(TASKS)))
    print(
        f"{args.split} subset: {len(data)} molecules, {n_atoms:,} atoms | "
        f"steps={args.steps} | baseline={args.baseline} | seeds={args.seeds}",
        flush=True,
    )

    result = {
        "arm": "ig_pool_zeros",
        "split": args.split,
        "split_seed": args.split_seed,
        "limit": args.limit,
        "n_molecules": len(data),
        "n_atoms": int(n_atoms),
        "baseline": args.baseline,
        "steps": args.steps,
        "seeds": args.seeds,
        "task_order": [t.name for t in TASKS],
        "per_seed": {},
    }

    for seed in args.seeds:
        path = model_ckpt_paths(args.repo, seed, 0.1, 0.1)["pooled_none"]  # anchor args unused here
        if not Path(path).exists():
            raise SystemExit(f"missing pooled_none checkpoint for seed {seed}: {path}")
        ck = torch.load(path, map_location=device, weights_only=False)
        model, backbone = build_model_from_checkpoint(ck)
        model = model.to(device).eval()
        print(f"\n[seed {seed}] {path}", flush=True)

        seed_out = {}
        for steps in args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            attrs, preds, aux = attribute(
                model,
                data,
                "integrated_gradients",
                backbone,
                device=device,
                task_idx=all_idx,
                steps=steps,
                baseline=args.baseline,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            base = aux.get("pred_base")
            per_task = {
                TASKS[k].name: completeness(attrs, preds, base, task_idx=[k]) for k in all_idx
            }
            mean_gap = sum(per_task.values()) / len(per_task)
            ms_per_mol = dt / len(data) * 1000.0
            seed_out[str(steps)] = {
                "gap_per_task": per_task,
                "gap_mean": mean_gap,
                "wall_s": dt,
                "ms_per_mol": ms_per_mol,
            }
            print(
                f"  steps {steps:4d}: mean|gap|={mean_gap:.4g}   {ms_per_mol:7.2f} ms/mol",
                flush=True,
            )
        result["per_seed"][str(seed)] = seed_out

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
