#!/usr/bin/env bash
# Stage 1 -- data preparation.
#
# Downloads the raw ADME datasets, builds the shared 3D conformer cache, and mines the matched
# molecular pairs used for the localization / per-atom interpretation figures. 
# Outputs land under data/raw/.
#
set -euo pipefail
cd "$(dirname "$0")/../.."

# 1. Raw datasets (ExpansionRx, mollipo, ESOL, TDC) -> data/raw/
python scripts/data/download.py --all

# 2. 3D conformer cache (RDKit ETKDG) for descriptors
python scripts/data/build_multitask_conformer_cache.py --out data/raw/multitask/conformers.pt

# 3. Matched pairs. matched_pairs.pt = single-atom / H-analog pairs 
# fragment_pairs.pt = larger fragment swaps. Both are read by scripts/score/matched_pair_attribution.py in stage 3.
python scripts/data/mine_matched_pairs.py  --out data/raw/multitask/matched_pairs.pt
python scripts/data/mine_fragment_pairs.py --out data/raw/multitask/fragment_pairs.pt
