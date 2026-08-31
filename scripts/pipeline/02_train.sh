#!/usr/bin/env bash
# Stage 2 -- train MolLedger and all baselines
#
# All arms are trained by scripts/train/train_molledger.py (GINEConv 2D backbone) except the GBT
# descriptor baseline (scripts/train/descriptor_baseline.py). Each is trained at three init seeds {19,209,31}
# on the fixed scaffold split (--seed 42) so every metric can be reported as mean +/- sd.
#
# Arguments users typically set:
#   --head {additive,pooled,gnan,ligandformer}   which model family
#   --additive_readout {sum,summean}             additive score head (summean adds the 1/N mean branch)
#   --additive_global_context --additive_context_dim D   global-context (MolLedger headline, D=8)
#   --anchor_rule --lambda_anchor L                      Crippen/TPSA shape anchor at strength L
#   --descriptors {none,2d3d}                            RDKit descriptor injection (needs --conformer_cache)
#   --epochs --batch_size --hidden_dim --num_layers --dropout --lr --init_seed --seed --checkpoint_dir
set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS=(19 209 31)
COMMON=(--backbone gin --descriptors none --epochs 300 --seed 42
        --hidden_dim 128 --num_layers 4 --batch_size 64 --dropout 0.0)

for S in "${SEEDS[@]}"; do
  # --- MolLedger: global-context additive head, D=8 (the headline model) ---
  #   unanchored control (lambda 0) -> ours_none / additive_none
  python scripts/train/train_molledger.py "${COMMON[@]}" --head additive \
    --additive_global_context --additive_context_dim 8 --init_seed "$S" --lambda_anchor 0 \
    --checkpoint_dir checkpoints_global_context_none
  #   anchored (best of lambda {0.1,0.3} by val, selected downstream) -> ours_best / additive_best
  for LAM in 0.1 0.3; do
    python scripts/train/train_molledger.py "${COMMON[@]}" --head additive \
      --additive_global_context --additive_context_dim 8 --anchor_rule \
      --lambda_anchor "$LAM" --init_seed "$S" --checkpoint_dir checkpoints_global_context_constlam
  done

  # --- Global-context width sweep (ctx in {2,4,16}), anchored only -> the context-size ablation figure ---
  for CTX in 2 4 16; do
    for LAM in 0.1 0.3; do
      python scripts/train/train_molledger.py "${COMMON[@]}" --head additive \
        --additive_global_context --additive_context_dim "$CTX" --anchor_rule \
        --lambda_anchor "$LAM" --init_seed "$S" --checkpoint_dir checkpoints_global_context_constlam
    done
  done

  # --- MolLedger (no context): sum-mean head, no global-context vector -> summean_none / summean_best ---
  for LAM in 0 0.1 0.3; do
    python scripts/train/train_molledger.py "${COMMON[@]}" --head additive --additive_readout summean \
      --anchor_rule --lambda_anchor "$LAM" --init_seed "$S" \
      --checkpoint_dir checkpoints_anchor_sweep_summean
  done

  # --- Plain additive-sum head, no context (anchor-strength faithfulness ladder) -> sum_none/weak/strong ---
  for LAM in 0 0.1 0.3; do
    python scripts/train/train_molledger.py "${COMMON[@]}" --head additive \
      --anchor_rule --lambda_anchor "$LAM" --init_seed "$S" \
      --checkpoint_dir checkpoints_anchor_sweep
  done

  # --- Pooled GNN (accuracy ceiling / IG-Grad-CAM-LIME attribution backbone) -> pooled_none ---
  python scripts/train/train_molledger.py "${COMMON[@]}" --head pooled --init_seed "$S" \
    --checkpoint_dir checkpoints_pooled_seeds

  # --- Pooled + descriptors (2D+3D RDKit descriptors injected at the pooled head) -> pooled_desc ---
  python scripts/train/train_molledger.py --backbone gin --head pooled \
    --descriptors 2d3d --conformer_cache data/raw/multitask/conformers.pt \
    --epochs 300 --seed 42 --hidden_dim 128 --num_layers 4 --batch_size 64 --dropout 0.0 \
    --init_seed "$S" --checkpoint_dir checkpoints_pooled_desc_seeds

  # --- GNAN (additive-over-features baseline), unanchored -> gnan ---
  python scripts/train/train_molledger.py --backbone gin --head gnan --descriptors none \
    --lambda_anchor 0 \
    --epochs 300 --seed 42 --batch_size 64 --lr 1e-3 --init_seed "$S" \
    --checkpoint_dir checkpoints_gnan_seeds

  # --- GNAN, anchored (best of lambda {0.1,0.3} by val, selected downstream) -> gnan_best ---
  for LAM in 0.1 0.3; do
    python scripts/train/train_molledger.py --backbone gin --head gnan --descriptors none \
      --anchor_rule \
      --lambda_anchor "$LAM" --epochs 300 --seed 42 --batch_size 64 --lr 1e-3 --init_seed "$S" \
      --checkpoint_dir checkpoints_gnan_seeds
  done

  # --- LigandFormer (attention readout baseline) -> ligandformer ---
  python scripts/train/train_molledger.py --backbone gin --head ligandformer --descriptors none \
    --lambda_anchor 0 \
    --epochs 300 --seed 42 --batch_size 64 --lr 1e-3 --init_seed "$S" \
    --checkpoint_dir checkpoints_ligandformer_seeds

  # --- GBT descriptor baseline (227 RDKit descriptors, gradient-boosted trees) -> gbt_227 ---
  python scripts/train/descriptor_baseline.py --seed 42 --init_seed "$S" \
    --out "results_descriptor_seeds/results_descriptor_baseline_init${S}.json"
done
