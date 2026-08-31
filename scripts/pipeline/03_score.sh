#!/usr/bin/env bash
# Stage 3 -- score the trained checkpoints into per-pair / per-molecule attribution dumps.
#
# scripts/score/matched_pair_attribution.py scores both members of every mined pair and decomposes the
# predicted delta into per-atom / fragment contributions. It is run once per arm x seed, on the
# graph-identical pairs (--graph_identical_only) and on the fragment-swap pairs (--split_filter testany).
# The --method selects the attribution: additive (exact per-atom scores), ig_fixed, grad_cam, lime, wisp,
# attention.
#
# Dump filenames follow build_matched_pair_jsons.py's accessors: pairs_{name}.jsonl / pairs_{name}_d.jsonl
# for the named-arm accessors, and pairs_init{seed}_{label}.jsonl / pairsd_init{seed}_{label}.jsonl for
# the exact-additive gctx8/summean grid (_exact_files).
#
# Arguments users typically set:
#   --checkpoint CKPT --method {additive,ig_fixed,grad_cam,lime,wisp,attention}
#   --tasks ... --classes class_a class_b [--graph_identical_only]   (graph-identical set)
#   --pairs FILE --classes class_d --split_filter testany           (fragment-swap set)
#   --ig_steps N --lime_samples N
#   --out FILE.json --dump_pairs FILE.jsonl
set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS=(19 209 31)
TASKS=(logd kinetic_solubility aqueous_solubility clint_mouse_liver clint_human_liver
       caco2_efflux_ratio caco2_papp_ab ppb_mouse_plasma ppb_mouse_brain ppb_mouse_muscle half_life)
FRAG=data/raw/multitask/fragment_pairs.pt

# score both pair sets for one checkpoint + method, named pairs_{name}[.jsonl|_d.jsonl]
score () {  # $1=checkpoint  $2=out_dir  $3=name  extra method args...
  local ck="$1" dir="$2" name="$3"; shift 3
  python scripts/score/matched_pair_attribution.py --checkpoint "$ck" "$@" \
    --tasks "${TASKS[@]}" --classes class_a class_b --graph_identical_only \
    --out "${dir}/lk_${name}.json" --dump_pairs "${dir}/pairs_${name}.jsonl"
  python scripts/score/matched_pair_attribution.py --checkpoint "$ck" "$@" --pairs "$FRAG" \
    --classes class_d --split_filter testany --tasks "${TASKS[@]}" \
    --out "${dir}/lkd_${name}.json" --dump_pairs "${dir}/pairs_${name}_d.jsonl"
}

# exact-additive grid: pairs_init{seed}_{label}.jsonl / pairsd_init{seed}_{label}.jsonl
score_exact () {  # $1=checkpoint  $2=out_dir  $3=seed  $4=label
  local ck="$1" dir="$2" seed="$3" label="$4"
  python scripts/score/matched_pair_attribution.py --checkpoint "$ck" --method additive \
    --tasks "${TASKS[@]}" --classes class_a class_b --graph_identical_only \
    --out "${dir}/lk_init${seed}_${label}.json" --dump_pairs "${dir}/pairs_init${seed}_${label}.jsonl"
  python scripts/score/matched_pair_attribution.py --checkpoint "$ck" --method additive --pairs "$FRAG" \
    --classes class_d --split_filter testany --tasks "${TASKS[@]}" \
    --out "${dir}/lkd_init${seed}_${label}.json" --dump_pairs "${dir}/pairsd_init${seed}_${label}.jsonl"
}

# WISP element-substitution mutants are model-INDEPENDENT, so precompute them ONCE (over the held-out
# molecule set) and reuse the cache across every seed's wisp arm below.
mkdir -p runs/ig_grid
python scripts/score/precompute_wisp_mutants.py --out runs/ig_grid/wisp_mutants.pt

mkdir -p runs/exact_grid_gctx8_constlam runs/exact_grid_summean runs/ig_grid runs/ig_grid_zeros \
         runs/gnan_grid runs/ligandformer_grid

for S in "${SEEDS[@]}"; do
  pooled="checkpoints_pooled_seeds/ablation_gin_pooled_none_init${S}/best.pt"
  gnan="checkpoints_gnan_seeds/ablation_gin_gnan_none_init${S}/best.pt"
  lf="checkpoints_ligandformer_seeds/ablation_gin_ligandformer_none_init${S}/best.pt"

  # MolLedger exact per-atom scores (leakage / exactness gap / timing), gctx8 at all three lambdas.
  score_exact "checkpoints_global_context_none/ablation_gin_additive_none_gctx8_init${S}/best.pt" \
              runs/exact_grid_gctx8_constlam "$S" none
  score_exact "checkpoints_global_context_constlam/ablation_gin_additive_none_anchor0.1-shape-rule_gctx8_init${S}/best.pt" \
              runs/exact_grid_gctx8_constlam "$S" anchor0.1
  score_exact "checkpoints_global_context_constlam/ablation_gin_additive_none_anchor0.3-shape-rule_gctx8_init${S}/best.pt" \
              runs/exact_grid_gctx8_constlam "$S" anchor0.3
  # MolLedger (no context): sum-mean head, fixed anchor 0.3 (best by val MAE across seeds).
  score_exact "checkpoints_anchor_sweep_summean/ablation_gin_additive_none_anchor0.3-shape-rule_init${S}/best.pt" \
              runs/exact_grid_summean "$S" anchor0.3

  # Post-hoc pooled-head baselines: off-manifold zero-baseline integrated gradients, Grad-CAM, LIME.
  score "$pooled" runs/ig_grid_zeros "ig_pooled_init${S}"          --method ig_fixed --ig_steps 128
  score "$pooled" runs/ig_grid       "gradcam_signed_pooled_init${S}" --method grad_cam
  score "$pooled" runs/ig_grid       "lime_pooled_init${S}"        --method lime --lime_samples 300
  # WISP element-substitution occlusion (forward-only at scoring time; reuses the precomputed mutants).
  score "$pooled" runs/ig_grid       "wisp_pooled_init${S}"        --method wisp --wisp_cache runs/ig_grid/wisp_mutants.pt

  # Native per-atom read-outs of the interpretability baselines.
  score "$gnan" runs/gnan_grid "gnan_init${S}"        --method additive
  score "$lf"   runs/ligandformer_grid "ligandformer_init${S}" --method attention
done

# GNAN, anchored (best of lambda {0.1,0.3} by val, matching build_matched_pair_jsons' selection).
for S in "${SEEDS[@]}"; do
  best_lam=$(python - "$S" <<'PY'
import sys
import torch
seed = sys.argv[1]
best_lam, best_val = None, None
for lam in ("0.1", "0.3"):
    ck = f"checkpoints_gnan_seeds/ablation_gin_gnan_none_anchor{lam}-shape-rule_init{seed}/best.pt"
    try:
        m = torch.load(ck, map_location="cpu", weights_only=False).get("best_metric")
    except FileNotFoundError:
        continue
    if m is not None and (best_val is None or float(m) < best_val):
        best_val, best_lam = float(m), lam
print(best_lam or "0.1")
PY
  )
  gnan_best="checkpoints_gnan_seeds/ablation_gin_gnan_none_anchor${best_lam}-shape-rule_init${S}/best.pt"
  score "$gnan_best" runs/gnan_grid "gnan_best_init${S}" --method additive
done

# Per-atom interpretation examples for the pair figures (select the pairs, then score them).
python scripts/score/select_pair_examples.py
python scripts/score/score_pair_examples.py --seed 19 \
  --pairs_json runs/pair_examples.json --out runs/pair_examples_attr.json

# IG steps vs wall-time trade-off (validation split) -> runs/ig_steps_sweep_val.json.
python scripts/score/ig_steps_sweep.py --steps 16 32 64 128 256 512 \
  --limit 300 --split valid --baseline zeros --out runs/ig_steps_sweep_val.json

# Class_d leakage vs global-context width (for the val ablation figure) -> runs/leakage_ctx_sweep/.
python scripts/score/score_leakage_ctx_sweep.py

# Predicted-delta baselines (GBT + pooled+descriptors) and pooled+descriptor accuracy.
for S in "${SEEDS[@]}"; do
  python scripts/score/eval_pair_delta_baselines.py --arm gbt        --seed "$S" --device cpu
  python scripts/score/eval_pair_delta_baselines.py --arm pooled_desc --seed "$S" --device cuda
done
python scripts/score/eval_pooled_desc_accuracy.py
