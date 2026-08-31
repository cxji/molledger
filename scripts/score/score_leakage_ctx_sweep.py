"""Score fragment-pair LEAKAGE for the MolLedger global-context WIDTH ladder
(ctx=0 sum-mean base, then gctx2/4/8/16, all uniform constant-lambda anchor), best-lambda per seed.
Mirrors build_val_global_context_ablation.NEW_ARMS but for leakage. Dumps per-pair jsonl into
runs/leakage_ctx_sweep/ ; scores are exact-additive so leakage = |D_core|/(|D_core|+|D_sub|).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
OUT = REPO / "runs/leakage_ctx_sweep"
OUT.mkdir(exist_ok=True)
SEEDS = (19, 209, 31)
TASKS = [
    "logd",
    "kinetic_solubility",
    "aqueous_solubility",
    "clint_mouse_liver",
    "clint_human_liver",
    "caco2_efflux_ratio",
    "caco2_papp_ab",
    "ppb_mouse_plasma",
    "ppb_mouse_brain",
    "ppb_mouse_muscle",
    "half_life",
]

# arm -> checkpoint dir template with {suffix}{seed}; suffix in {_anchor0.1-shape-rule,_anchor0.3-shape-rule}
ARMS = {
    # ctx=0 "No context": the sum-mean head used by the val figure's "No context" bar.
    "ctx0_summean": "checkpoints_anchor_sweep_summean/ablation_gin_additive_none{suffix}_init{seed}",
    "gctx2": "checkpoints_global_context_constlam/ablation_gin_additive_none{suffix}_gctx2_init{seed}",
    "gctx4": "checkpoints_global_context_constlam/ablation_gin_additive_none{suffix}_gctx4_init{seed}",
    "gctx8": "checkpoints_global_context_constlam/ablation_gin_additive_none{suffix}_gctx8_init{seed}",
    "gctx16": "checkpoints_global_context_constlam/ablation_gin_additive_none{suffix}_gctx16_init{seed}",
}


def best_lambda(tmpl, seed):
    best, bv = None, None
    for lam in (0.1, 0.3):
        suf = f"_anchor{lam}-shape-rule"
        ck = REPO / (tmpl.format(suffix=suf, seed=seed) + "/best.pt")
        if not ck.exists():
            continue
        m = torch.load(ck, map_location="cpu", weights_only=False).get("best_metric")
        if m is not None and (bv is None or m < bv):
            bv, best = float(m), lam
    return best


for arm, tmpl in ARMS.items():
    for seed in SEEDS:
        lam = best_lambda(tmpl, seed)
        if lam is None:
            print(f"SKIP {arm} init{seed}: no checkpoint", flush=True)
            continue
        ck = REPO / (tmpl.format(suffix=f"_anchor{lam}-shape-rule", seed=seed) + "/best.pt")
        dump = OUT / f"pairsd_{arm}_init{seed}.jsonl"
        if dump.exists():
            print(f"HAVE {dump.name}", flush=True)
            continue
        print(f"=== {arm} init{seed} best-lam={lam} ===", flush=True)
        subprocess.run(
            [
                PY,
                "scripts/score/matched_pair_attribution.py",
                "--pairs",
                "data/raw/multitask/fragment_pairs.pt",
                "--checkpoint",
                str(ck),
                "--method",
                "additive",
                "--tasks",
                *TASKS,
                "--classes",
                "class_d",
                "--split_filter",
                "testany",
                "--out",
                str(OUT / f"lkd_{arm}_init{seed}.json"),
                "--dump_pairs",
                str(dump),
            ],
            cwd=str(REPO),
            check=True,
        )
print("DONE")
