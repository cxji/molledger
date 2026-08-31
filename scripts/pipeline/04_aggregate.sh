#!/usr/bin/env bash
# Stage 4 -- aggregate the score dumps into the JSON tables the figures read.
#
# These read the stage-3 dumps (and re-score accuracy/faithfulness from the checkpoints) and write the
# runs/*.json that src/metrics.py and the figure scripts consume. All are seed-aware (mean +/- sd
# over {19,209,31}).
#
# Arguments users typically set:
#   build_test_split_jsons.py  --ig_steps N --lime_samples M --json_out runs/test_split_tables.json
#   build_matched_pair_jsons.py --json_out runs/matched_pair_tables.json
set -euo pipefail
cd "$(dirname "$0")/../.."

# Test-split accuracy + faithfulness + single-molecule exactness -> runs/test_split_tables.json
# (also writes runs/test_sample_counts.json).
python scripts/figures/build_test_split_jsons.py --ig_steps 256 --lime_samples 300 \
  --json_out runs/test_split_tables.json

# Matched-pair localization (leakage) / predicted-delta accuracy / completeness gap. The pooled IG arm
# is the off-manifold zero-baseline IG (runs/ig_grid_zeros); Grad-CAM / LIME / WISP come from runs/ig_grid.
python scripts/figures/build_matched_pair_jsons.py --json_out runs/matched_pair_tables.json

# Validation-split accuracy for the two non-additive reference bars in the val ablation figures.
python scripts/figures/build_val_accuracy_jsons.py
