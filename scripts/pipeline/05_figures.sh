#!/usr/bin/env bash
# Stage 5 -- render the paper figures into plots/ (created if absent) and the LaTeX tables into runs/.
#
# plot_results.py and plot_pair_interpretations.py read the stage-4 JSON tables. plot_noncircular_
# concordance.py and the two val_* scripts score their own arms from the checkpoints and then plot;
# pass --plot_only to re-plot from their cached runs/*.json without re-scoring (fast, no GPU). The
# build_*_latex_tables / build_anchor_relevance_table scripts read the stage-4 JSON tables and write
# runs/*.tex.
set -euo pipefail
cd "$(dirname "$0")/../.."

# perf_mae_spearman(+appendix), interp_heldout(+appendix), attr_timing_ms_per_molecule, ig_steps_tradeoff
python scripts/figures/plot_results.py

# pair_interp_main, pair_interp_appendix_permeability_solubility, pair_interp_plasma_brain_aqueous
python scripts/figures/plot_pair_interpretations.py

# noncircular_concordance  (per-atom attribution vs independent physical descriptors)
python scripts/figures/plot_noncircular_concordance.py

# val_anchor_faithfulness  (anchor strength: faithfulness vs accuracy on the val split)
python scripts/figures/build_val_anchor_faithfulness.py

# val_additive_ablation_appendix  (global-context width sweep on the val split)
python scripts/figures/build_val_global_context_ablation.py

# LaTeX tables -> runs/perf_tables.tex, runs/interp_tables.tex, runs/anchor_relevance_table.tex
python scripts/figures/build_perf_latex_tables.py
python scripts/figures/build_interp_latex_tables.py
python scripts/figures/build_anchor_relevance_table.py
