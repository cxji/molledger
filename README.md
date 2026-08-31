# MolLedger: An Additive Graph Neural Network with Chemically Grounded ADME Attributions

MolLedger is a graph neural network that provides chemically-grounded attributions for predicting absorption, distribution, metabolism, and excretion (ADME). Optimizing ADME is an important part of small molecule drug discovery. Many machine learning models have been built to predict ADME properties to facilitate this optimization process, but explaining model predictions is challenging. MolLedger solves this problem with two novel components: 1) An additive head that produces per-atom scores given the output from message passing and a global context vector. 2) An auxiliary loss that anchors per-atom scores to physical properties. MolLedger's additive framework obtains exact interpretability at no cost to performance because of the global context vector. Furthermore, MolLedger produces more faithful attributions than any other interpretability method because of the auxiliary loss. Our case studies comparing interpretations from multiple methods on matched pairs reveal that MolLedger is much better at producing sensible explanations for predicted property changes. This repository contains the code to run MolLedger and reproduce the experiments in the paper.

## Setup

```bash
conda env create -f environment.yml
conda activate molledger
```

## Training models

To train MolLedger or any of the variants in the paper, run `python scripts/train/train_molledger.py` with the following arguments

| Arm (figure label) | How it is trained |
|---|---|
| **MolLedger** | `--head additive --additive_global_context --additive_context_dim 8 --anchor_rule --lambda_anchor {0.1, 0.3}` |
| Unanchored MolLedger | `--head additive --additive_global_context --additive_context_dim 8` |
| MolLedger (no context) | `--head additive --additive_readout summean --anchor_rule` |
| Pooled GNN | `--head pooled` |
| Pooled + descriptors | `--head pooled --descriptors 2d3d` |
| Anchored GNAN | `--head gnan --anchor_rule --lambda_anchor {0.1, 0.3}` |
| Unanchored GNAN | `--head gnan` | 
| LigandFormer | `--head ligandformer` |

To train a gradient-boosted tree with descriptors, run `python scripts/train/descriptor_baseline.py`.

## Pipeline

To reproduce the full pipeline in the paper, run the 5 stages below

```bash
bash scripts/pipeline/01_data.sh        # download datasets, build conformer cache, mine matched pairs
bash scripts/pipeline/02_train.sh       # train every figure arm x 3 seeds
bash scripts/pipeline/03_score.sh       # attribute checkpoints -> per-pair / per-molecule score dumps
bash scripts/pipeline/04_aggregate.sh   # aggregate dumps -> runs/*.json tables
bash scripts/pipeline/05_figures.sh     # render the 11 figures into plots/
```

What each stage produces:

- **01_data** → `data/raw/` datasets, `data/raw/multitask/conformers.pt`, and the mined
  `matched_pairs.pt` / `fragment_pairs.pt`.
- **02_train** → `checkpoints_*/` (one dir per arm) and `results_descriptor_seeds/` (GBT).
- **03_score** → `runs/exact_grid_*`, `runs/ig_grid*`, `runs/gnan_grid`, `runs/ligandformer_grid`,
  `runs/pair_examples_attr.json`, `runs/ig_steps_sweep_val.json`, `runs/leakage_ctx_sweep/`,
  `runs/pairdelta_grid`, `runs/pooled_desc_accuracy.json`.
- **04_aggregate** → `runs/test_split_tables.json`, `runs/matched_pair_tables{,_igzeros}.json`,
  `runs/val_accuracy_appendix.json`.
- **05_figures** → the 11 PDFs in `plots/`.

## Repository layout

```
src/                model + data code
  data/             dataset build, scaffold split, conformers, descriptors, matched-pair maps, anchors
  models/           gnn.py (AdditiveGNN / Pooled / GNAN), ligandformer.py
  attribution.py    per-atom attribution (additive, IG, Grad-CAM, LIME, attention) + faithfulness
  losses.py         masked multitask loss + the Crippen/TPSA anchor terms
  metrics.py        loaders/accessors over runs/*.json
scripts/            stage entrypoints, grouped by pipeline stage
  pipeline/         the five stage runners above
  data/             download, conformer cache, matched-pair mining
  train/            train_molledger.py, descriptor_baseline.py
  score/            attribution scoring + per-pair/per-molecule dumps + the IG steps sweep
  figures/          plots + tables
  tests/            pytest suite
```

The figures in the paper are produced by the following scripts in stage 5:
| Figures | Script |
|---|---|
| Figures 1 and 8: Performance metrics | `scripts/figures/plot_results.py` |
| Figures 2 and 9: Interpretability metrics | `scripts/figures/plot_results.py` |
| Figures 3, 10, and 11: Case studies | `scripts/figures/plot_pair_interpretations.py` |
| Figure 4: Ablation of global context vector | `scripts/figures/build_val_global_context_ablation.py` |
| Figure 5: Anchor strength and head design | `scripts/figures/build_val_anchor_faithfulness.py` |
| Figure 6: IG exactness vs time trade-off | `scripts/figures/plot_results.py` |
| Figure 7: Runtime of interpretability methods | `scripts/figures/plot_results.py` |
