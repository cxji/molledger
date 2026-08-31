"""
Precompute and cache the WISP element-substitution mutant graphs for every molecule the held-out
interpretability grid scores. Model-independent, so computed once and reused across seeds by
matched_pair_attribution --method wisp --wisp_cache <path>.

The cache is a dict {canonical_smiles: packed}, `packed` = src.wisp_mutants.build_mutants output.
Molecule set = the union, over all tasks and every matched-pair collection (graph-identical and
fragment-swap), of both members of every held-out (testany) pair, deduplicated by canonical SMILES.

Usage:
    python scripts/score/precompute_wisp_mutants.py --out runs/ig_grid/wisp_mutants.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.score.matched_pair_attribution as M
from src.data.multitask import build_registry, featurize_registry
from src.wisp_mutants import build_mutants

DEFAULT_TASKS = [
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/ig_grid/wisp_mutants.pt")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--matched_pairs", default="data/raw/multitask/matched_pairs.pt")
    p.add_argument("--fragment_pairs", default="data/raw/multitask/fragment_pairs.pt")
    args = p.parse_args()

    t_setup = time.perf_counter()
    registry = build_registry()
    featurized = featurize_registry(registry)
    mp = torch.load(args.matched_pairs, map_location="cpu", weights_only=False)
    fp = torch.load(args.fragment_pairs, map_location="cpu", weights_only=False)
    print(f"  setup (registry + featurize + pairs): {time.perf_counter() - t_setup:.1f}s")

    # Held-out molecule set = union over tasks x classes of both members of every testany pair.
    # Graph-identical pairs match the pooled arms; fragment-swap pairs have no such filter.
    slices = [("class_a", mp, True), ("class_b", mp, True), ("class_d", fp, False)]
    keys = set()
    for task in args.tasks:
        for cls, art, gi in slices:
            for it in M.collect(art, cls, registry, task, "model", "testany", gi):
                if it["a"] in featurized and it["b"] in featurized:
                    keys.add(it["a"])
                    keys.add(it["b"])
    smiles = sorted({registry[q].smiles for q in keys})
    print(f"  held-out molecules to precompute: {len(keys)} keys -> {len(smiles)} unique SMILES")

    t_pre = time.perf_counter()
    cache, n_mut, n_fail = {}, 0, 0
    for j, smi in enumerate(smiles):
        packed = build_mutants(smi)
        if packed is None:
            n_fail += 1
            continue
        cache[smi] = packed
        n_mut += int(packed["mut_atom"].shape[0])
        if (j + 1) % 2000 == 0:
            dt = time.perf_counter() - t_pre
            print(
                f"    {j + 1}/{len(smiles)}  ({dt:.0f}s, {1000 * dt / (j + 1):.1f} ms/mol, "
                f"{n_mut} mutants so far)"
            )
    precompute_seconds = time.perf_counter() - t_pre

    torch.save(cache, args.out)
    import os

    size_gb = os.path.getsize(args.out) / 1e9
    print(
        f"\n  cached {len(cache)} molecules ({n_mut} mutants total, {n_fail} SMILES failed) "
        f"-> {args.out}  [{size_gb:.2f} GB]"
    )
    ms_per_mol = 1000 * precompute_seconds / max(len(smiles), 1)
    print(f"  PRECOMPUTE_SECONDS {precompute_seconds:.1f}   ({ms_per_mol:.1f} ms/molecule)")
    # Timing sidecar: raw per-molecule build cost (total precompute seconds / unique molecules).
    # src/metrics.attr_timing adds it to WISP's forward-scoring time.
    import json as _json

    timing_path = os.path.splitext(args.out)[0] + "_timing.json"
    with open(timing_path, "w") as fh:
        _json.dump(
            {
                "precompute_seconds": precompute_seconds,
                "n_unique_molecules": len(smiles),
                "ms_per_molecule": ms_per_mol,
            },
            fh,
            indent=1,
        )
    print(f"  wrote precompute timing sidecar -> {timing_path}")


if __name__ == "__main__":
    main()
