#!/usr/bin/env bash
# Stage 5 -- render the 11 paper figures into plots/ (created if absent).
#
# plot_results.py and plot_pair_interpretations.py read the stage-4 JSON tables. The two val_* scripts
# score their own arms from the checkpoints and then plot; pass --plot_only to re-plot from their
# cached runs/val_*.json without re-scoring (fast, no GPU).
set -euo pipefail
cd "$(dirname "$0")/../.."

# perf_mae_spearman(+appendix), interp_heldout(+appendix), attr_timing_ms_per_molecule, ig_steps_tradeoff
python scripts/figures/plot_results.py

# pair_interp_main, pair_interp_appendix_permeability_solubility, pair_interp_plasma_brain_aqueous
python scripts/figures/plot_pair_interpretations.py

# val_anchor_faithfulness  (anchor strength: faithfulness vs accuracy on the val split)
python scripts/figures/build_val_anchor_faithfulness.py

# val_additive_ablation_appendix  (global-context width sweep on the val split)
python scripts/figures/build_val_global_context_ablation.py
