"""
Unified multi-task molecule registry across ogbg-mollipo, ExpansionRx, TDC, and ESOL. Builds one
InChIKey-keyed registry with a [K]-shaped (NaN-masked) label vector per molecule, one global scaffold
split, and featurizes everything via ogb.utils.mol.smiles2graph.
"""

import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd
import torch
from ogb.utils.mol import smiles2graph
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

from src.data.conformers import build_conformer_cache_keyed
from src.data.dataset import _crippen_contribs, _tpsa_contribs, load_mollipo
from src.data.transforms import forward_transform, inverse_transform

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# 1. Property registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    name: str  # e.g. "logd", "kinetic_solubility"
    sources: list  # e.g. ["mollipo", "expansionrx"]
    anchor: str  # "crippen" | "tpsa" | "none"
    label_kind: str  # "continuous" | "binary"
    label_level: str = "graph"  # "graph" (sum-of-atoms matches a scalar batch.y label) |
    # "atom" (per-atom labels, no graph-level sum term). For an "atom" task, `anchor`
    # names the per-atom target tensor ("crippen"/"tpsa") and is that channel's target.
    label_transform: str = "none"  # "none" | "log" | "log1p" -- native->model-space transform,
    # applied in featurize_registry, inverted for reporting. See src/data/transforms.py.
    anchor_sign: int = +1  # sign multiplying `anchor` in the anchor term. Only ANCHOR_RULE sets -1.


# label_transform (last field): see src/data/transforms.py. logd & aqueous_solubility(ESOL) are
# already log; hia is binary; caco2_papp_ab is log10(cm/s) from both sources (see
# _papp_micro_to_log10_cm_s) -> "none". ppb x3 use LOQ/2-substituted values (see
# _substitute_censored_zeros) -> log. clint x2 still contain exact zeros -> log1p. half_life,
# caco2_efflux_ratio, kinetic_solubility are strictly positive -> log.
#
# NAMING WARNING: the three ppb_* tasks are named for the ASSAY (MPPB/MBPB/MGMB = mouse
# plasma/brain/gastrocnemius-muscle protein binding) but the values look like fraction UNBOUND
# (fu), not percent bound -- unconfirmed (no source data dictionary in data/raw). Any
# interpretation of these three tasks is currently INVERTED, and the Crippen anchor points the
# wrong way on them.
TASK_REGISTRY = [
    TaskSpec("logd", ["mollipo", "expansionrx"], "crippen", "continuous", label_transform="none"),
    TaskSpec("kinetic_solubility", ["expansionrx"], "tpsa", "continuous", label_transform="log"),
    TaskSpec("aqueous_solubility", ["esol"], "tpsa", "continuous", label_transform="none"),
    TaskSpec("clint_mouse_liver", ["expansionrx"], "none", "continuous", label_transform="log1p"),
    TaskSpec("clint_human_liver", ["expansionrx"], "none", "continuous", label_transform="log1p"),
    TaskSpec("caco2_efflux_ratio", ["expansionrx"], "crippen", "continuous", label_transform="log"),
    TaskSpec(
        "caco2_papp_ab", ["expansionrx", "tdc"], "crippen", "continuous", label_transform="none"
    ),  # log10(cm/s) from both sources; see _papp_micro_to_log10_cm_s
    TaskSpec("ppb_mouse_plasma", ["expansionrx"], "crippen", "continuous", label_transform="log"),
    TaskSpec("ppb_mouse_brain", ["expansionrx"], "crippen", "continuous", label_transform="log"),
    TaskSpec("ppb_mouse_muscle", ["expansionrx"], "crippen", "continuous", label_transform="log"),
    TaskSpec("hia", ["tdc"], "tpsa", "binary", label_transform="none"),
    TaskSpec("half_life", ["tdc"], "none", "continuous", label_transform="log"),
]
TASK_NAMES = [t.name for t in TASK_REGISTRY]
TASK_INDEX = {name: i for i, name in enumerate(TASK_NAMES)}
NUM_TASKS = len(TASK_REGISTRY)

# Canonical graph-task index space shared by the trainer, scorers, and aggregation code: every
# non-binary task (HIA excluded). Different index space than TASK_REGISTRY -- HIA sits at index
# 10, not the end, so it's not a prefix slice. GRAPH_TASK_COLS records which TASK_REGISTRY column
# each GRAPH_TASK_SPECS entry came from (used to slice `y`).
GRAPH_TASK_SPECS = [t for t in TASK_REGISTRY if t.label_kind != "binary"]
GRAPH_TASK_COLS = [i for i, t in enumerate(TASK_REGISTRY) if t.label_kind != "binary"]


# ---------------------------------------------------------------------------
# 1b. ANCHOR_RULE -- the corrected anchor assignment, frozen by hand
# ---------------------------------------------------------------------------
#
# task -> (anchor, sign). Applied opt-in via apply_anchor_rule(); TASK_REGISTRY is left untouched.
# Frozen table: train-split Spearman rho plus literature overrides for unstable measurements.
# `hia` is absent (binary label); apply_anchor_rule raises on it.
ANCHOR_RULE = {
    "logd": ("crippen", +1),  # near-definitional: logD is logP adjusted for ionization
    "kinetic_solubility": (
        "crippen",
        -1,
    ),  # override: GSE carries logP at -1 (same axis as aqueous_sol);
    # rho -0.193, below the deadband but sign-correct
    "aqueous_solubility": ("crippen", -1),  # GSE carries logP with coefficient -1; rho -0.845
    "clint_mouse_liver": ("crippen", +1),  # override: lipophilicity drives CYP clearance
    "clint_human_liver": ("crippen", +1),  # override: same assay/anchor as mouse
    "caco2_efflux_ratio": ("tpsa", +1),  # P-gp recognition; loosest mechanism of the anchored set
    "caco2_papp_ab": ("tpsa", -1),  # Veber: polarity impedes passive permeability
    "ppb_mouse_plasma": ("crippen", -1),  # values are fraction unbound, not percent bound
    "ppb_mouse_brain": ("crippen", -1),  # (see NAMING WARNING above TASK_REGISTRY)
    "ppb_mouse_muscle": ("crippen", -1),
    "half_life": ("none", 0),  # override: t_half = 0.693*Vd/CL, logP lifts both
}


def apply_anchor_rule(specs: list, rule: dict | None = None) -> list:
    """Return copies of `specs` with anchor + anchor_sign taken from ANCHOR_RULE. Raises on any spec
    the rule does not cover. Callers pass the spec list they train on (GRAPH_TASK_SPECS), never
    TASK_REGISTRY."""
    rule = ANCHOR_RULE if rule is None else rule
    missing = [s.name for s in specs if s.name not in rule]
    if missing:
        raise KeyError(
            f"ANCHOR_RULE does not cover {missing}. Add an explicit entry -- do not fall back to "
            f"the hand anchor."
        )
    out = []
    for spec in specs:
        anchor, sign = rule[spec.name]
        out.append(replace(spec, anchor=anchor, anchor_sign=sign))
    return out


# ---------------------------------------------------------------------------
# 2. Per-source raw loaders -- each returns list[(smiles, {task_name: value})]
# ---------------------------------------------------------------------------


def _load_mollipo_raw(root: str = "data/raw/mollipo"):
    splits = load_mollipo(root)
    out = []
    for split in ("train", "valid", "test"):
        for data in splits[split]:
            out.append((None, {"logd": data.y.item()}, data))
    # mollipo Data objects already carry x/edge_index/edge_attr from OGB's featurizer -- reuse
    # directly rather than re-featurizing via smiles2graph. Still need SMILES for InChIKey.
    root_path = Path(root)
    smiles_path = root_path / "ogbg_mollipo" / "mapping" / "mol.csv.gz"
    smiles_list = pd.read_csv(smiles_path)["smiles"].tolist()
    # load_mollipo doesn't expose global index -> re-derive split_idx directly.
    from ogb.graphproppred import PygGraphPropPredDataset

    split_idx = PygGraphPropPredDataset(name="ogbg-mollipo", root=root).get_idx_split()

    records = []
    for split in ("train", "valid", "test"):
        for data, gidx in zip(splits[split], split_idx[split].tolist()):
            records.append((smiles_list[gidx], {"logd": data.y.item()}, data))
    return records


EXPANSIONRX_COLUMN_MAP = {
    "logd": "LogD",
    "kinetic_solubility": "KSOL",
    "clint_human_liver": "HLM CLint",
    "clint_mouse_liver": "MLM CLint",
    "caco2_papp_ab": "Caco-2 Permeability Papp A>B",
    "caco2_efflux_ratio": "Caco-2 Permeability Efflux",
    "ppb_mouse_plasma": "MPPB",
    "ppb_mouse_brain": "MBPB",
    "ppb_mouse_muscle": "MGMB",
}
EXPANSIONRX_SMILES_COL = "SMILES"

# Per-source unit harmonization: every source enters caco2_papp_ab in ONE convention.
#
#     expansionrx  Papp in 1e-6 cm/s, raw     n=3773   min 0.00   median  6.14   max 51.41
#     tdc          log10(Papp in cm/s)        n= 910   min -7.76  median -5.13   max -3.51
#
# Convert ExpansionRx into TDC's log10(cm/s) convention: log10(6.14e-6) = -5.21 matches the TDC
# median of -5.13.
_LOG10_OF_MICRO = math.log10(1e-6)  # -6.0


def _papp_micro_to_log10_cm_s(value: float) -> float | None:
    """Papp in 1e-6 cm/s -> log10(Papp in cm/s). None for non-positive values (Papp == 0, i.e.
    below the assay's limit of quantification -- a censored reading, not zero permeability)."""
    if value <= 0:
        return None
    return math.log10(value) + _LOG10_OF_MICRO


# Tasks whose exact zeros are LEFT-CENSORED readings (below the assay's limit of quantification),
# not measurements of zero -- the smallest non-zero value per column is a round reporting floor.
# Substituted with LOQ/2 (per-task observed floor), then log-transformed.
#
# Not applied to clint_mouse_liver / clint_human_liver (same pattern, left on log1p).
_CENSORED_ZERO_TASKS = ("ppb_mouse_plasma", "ppb_mouse_brain", "ppb_mouse_muscle")


def _substitute_censored_zeros(records, task_names=_CENSORED_ZERO_TASKS):
    """Replace exact zeros with LOQ/2, LOQ estimated per task as the smallest positive value
    observed in that column. Mutates and returns `records` (list of (smiles, labels, extra))."""
    for task in task_names:
        positives = [r[1][task] for r in records if r[1].get(task) is not None and r[1][task] > 0]
        if not positives:
            continue
        floor = min(positives) / 2.0
        n = 0
        for r in records:
            if r[1].get(task) is not None and r[1][task] <= 0:
                r[1][task] = floor
                n += 1
        if n:
            print(f"  {task}: {n} censored zeros -> LOQ/2 = {floor:.5g} (LOQ {min(positives):.5g})")
    return records


def _load_expansionrx_raw(root: str = "data/raw/expansionrx"):
    root_path = Path(root)
    csvs = sorted(root_path.glob("expansion_data_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No expansion_data_*.csv files found under {root_path}. "
            f"Run `python scripts/data/download.py --dataset expansionrx` first."
        )
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)

    if EXPANSIONRX_SMILES_COL not in df.columns:
        raise ValueError(
            f"Expected SMILES column '{EXPANSIONRX_SMILES_COL}' not found in ExpansionRx "
            f"data. Actual columns: {list(df.columns)}"
        )
    missing = [col for col in EXPANSIONRX_COLUMN_MAP.values() if col not in df.columns]
    if missing:
        raise ValueError(
            f"Expected ExpansionRx columns {missing} not found. "
            f"Actual columns: {list(df.columns)}. Update EXPANSIONRX_COLUMN_MAP in "
            f"src/data/multitask.py if the schema changed."
        )

    out = []
    for _, row in df.iterrows():
        smiles = row[EXPANSIONRX_SMILES_COL]
        labels = {}
        for task_name, col in EXPANSIONRX_COLUMN_MAP.items():
            val = row[col]
            if pd.notna(val):
                val = float(val)
                if task_name == "caco2_papp_ab":
                    val = _papp_micro_to_log10_cm_s(val)  # -> TDC's convention; None if <= 0
                    if val is None:
                        continue
                labels[task_name] = val
        out.append((smiles, labels, None))
    return _substitute_censored_zeros(out)


_TDC_DATASET_TASK_MAP = {
    "Caco2_Wang": "caco2_papp_ab",
    "HIA_Hou": "hia",
    "Half_Life_Obach": "half_life",
}


def _load_tdc_raw(root: str = "data/raw/tdc"):
    root_path = Path(root)
    out = []
    for tdc_name, task_name in _TDC_DATASET_TASK_MAP.items():
        for part in ("train", "valid", "test"):
            path = root_path / f"{tdc_name}_{part}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Expected TDC file {path} not found. "
                    f"Run `python scripts/data/download.py --dataset tdc` first."
                )
            df = pd.read_csv(path)
            missing = [c for c in ("Drug_ID", "Drug", "Y") if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Expected TDC columns {missing} not found in {path}. "
                    f"Actual columns: {list(df.columns)}"
                )
            for _, row in df.iterrows():
                out.append((row["Drug"], {task_name: float(row["Y"])}, None))
    return out


_ESOL_SMILES_COL = "smiles"
_ESOL_LABEL_COL = "measured log solubility in mols per litre"


def _load_esol_raw(root: str = "data/raw/esol"):
    path = Path(root) / "delaney-processed.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected ESOL file {path} not found. "
            f"Run `python scripts/data/download.py --dataset esol` first."
        )
    df = pd.read_csv(path)
    missing = [c for c in (_ESOL_SMILES_COL, _ESOL_LABEL_COL) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected ESOL columns {missing} not found. Actual columns: {list(df.columns)}"
        )
    out = []
    for _, row in df.iterrows():
        out.append(
            (row[_ESOL_SMILES_COL], {"aqueous_solubility": float(row[_ESOL_LABEL_COL])}, None)
        )
    return out


# ---------------------------------------------------------------------------
# 3. Merge into one registry, keyed by InChIKey
# ---------------------------------------------------------------------------


@dataclass
class MoleculeRecord:
    smiles: str
    inchikey: str
    labels: dict = field(default_factory=dict)
    label_source: dict = field(
        default_factory=dict
    )  # task_name -> source name, for collision logging
    mollipo_data: object = None  # pre-featurized Data from load_mollipo, if sourced from mollipo


_parse_fail_count = 0
_tautomer_collision_count = 0
_label_collision_count = 0


def build_registry(
    mollipo_root="data/raw/mollipo",
    expansionrx_root="data/raw/expansionrx",
    tdc_root="data/raw/tdc",
    esol_root="data/raw/esol",
):
    """Merge all four sources into one dict[inchikey, MoleculeRecord]."""
    global _parse_fail_count, _tautomer_collision_count, _label_collision_count
    _parse_fail_count = 0
    _tautomer_collision_count = 0
    _label_collision_count = 0

    sources = {
        "mollipo": _load_mollipo_raw(mollipo_root),
        "expansionrx": _load_expansionrx_raw(expansionrx_root),
        "tdc": _load_tdc_raw(tdc_root),
        "esol": _load_esol_raw(esol_root),
    }

    registry = {}
    for source_name, records in sources.items():
        for smiles, labels, mollipo_data in records:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                _parse_fail_count += 1
                continue
            canon_smiles = Chem.MolToSmiles(mol)
            inchikey = Chem.MolToInchiKey(mol)

            if inchikey not in registry:
                registry[inchikey] = MoleculeRecord(
                    smiles=canon_smiles,
                    inchikey=inchikey,
                    mollipo_data=mollipo_data if source_name == "mollipo" else None,
                )
            else:
                record = registry[inchikey]
                if record.smiles != canon_smiles:
                    _tautomer_collision_count += 1
                if source_name == "mollipo" and record.mollipo_data is None:
                    record.mollipo_data = mollipo_data

            record = registry[inchikey]
            for task_name, value in labels.items():
                if task_name in record.labels:
                    _label_collision_count += 1
                    continue  # first-source-wins; assay conditions may differ across sources
                record.labels[task_name] = value
                record.label_source[task_name] = source_name

    print(
        f"  Registry: {len(registry)} unique molecules "
        f"({_parse_fail_count} SMILES failed to parse, "
        f"{_tautomer_collision_count} InChIKey collisions with differing canonical SMILES, "
        f"{_label_collision_count} cross-source label collisions resolved first-source-wins)"
    )

    for task in TASK_REGISTRY:
        n = sum(1 for r in registry.values() if task.name in r.labels)
        print(f"    {task.name}: {n} labeled molecules")

    return registry


# ---------------------------------------------------------------------------
# 4. Global scaffold split (deterministic, no deepchem)
# ---------------------------------------------------------------------------


def compute_global_scaffold_split(registry, frac_train=0.8, frac_valid=0.1, seed=42):
    scaffolds = {}
    for inchikey, record in registry.items():
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(record.smiles, includeChirality=False)
            if not scaffold:
                scaffold = record.smiles
        except Exception:
            scaffold = record.smiles
        scaffolds.setdefault(scaffold, []).append(inchikey)

    scaffold_keys = list(scaffolds.keys())
    random.Random(seed).shuffle(scaffold_keys)
    scaffold_keys.sort(key=lambda k: len(scaffolds[k]), reverse=True)

    n_total = len(registry)
    n_train_target = int(frac_train * n_total)
    n_valid_target = int(frac_valid * n_total)

    train, valid, test = [], [], []
    for key in scaffold_keys:
        group = scaffolds[key]
        if len(train) < n_train_target:
            train.extend(group)
        elif len(valid) < n_valid_target:
            valid.extend(group)
        else:
            test.extend(group)

    print(
        f"  Scaffold split: {len(scaffold_keys)} scaffold groups -> "
        f"train={len(train)} valid={len(valid)} test={len(test)}"
    )

    return {"train": train, "valid": valid, "test": test}


# ---------------------------------------------------------------------------
# 5. Featurization
# ---------------------------------------------------------------------------

_featurize_fail_count = 0


def featurize_registry(registry, conformer_cache_path: str | None = None):
    """
    conformer_cache_path: optional path to a conformer cache built by
    build_multitask_conformer_cache (InChIKey-keyed {"coords","dropped",...} schema). When
    given, joins a `pos` [N,3] field onto each Data object, dropping molecules without a
    usable conformer. Defaults to None (2D-only, no `pos` field attached).
    """
    global _featurize_fail_count
    _featurize_fail_count = 0
    out = {}

    conformer_coords, conformer_dropped = {}, {}
    if conformer_cache_path is not None:
        cache = torch.load(conformer_cache_path)
        conformer_coords, conformer_dropped = cache["coords"], cache["dropped"]
    n_no_conformer = 0

    for inchikey, record in registry.items():
        # [1, K] not [K]: PyG's default batching concatenates flat [K] tensors along dim 0,
        # producing [B*K] instead of [B, K]. An explicit leading batch dim of 1 stacks correctly.
        y = torch.tensor(
            [record.labels.get(t.name, float("nan")) for t in TASK_REGISTRY], dtype=torch.float
        )
        # Per-task native->model-space transform, applied once here (NaN passes through
        # unchanged). Reporting inverts it back.
        for k, t in enumerate(TASK_REGISTRY):
            if t.label_transform != "none":
                y[k] = forward_transform(t.label_transform, y[k])
        y = y.unsqueeze(0)

        # Featurize EVERY source (mollipo included) from the canonical SMILES via smiles2graph, so
        # x shares one atom order with crippen/tpsa and the cached conformer `pos`.
        #
        # BUGFIX: mollipo previously reused its OGB-pre-featurized graph (ORIGINAL-SMILES atom
        # order), which differs from canonical order for ~73% of mollipo molecules -- silently
        # misattaching crippen/tpsa and 3D coordinates to the wrong atoms. smiles2graph(original)
        # reproduces PygGraphPropPredDataset's stored features exactly, so this only reorders
        # mollipo's features into the canonical order; it does not change feature values.
        try:
            graph = smiles2graph(record.smiles)
        except Exception:
            _featurize_fail_count += 1
            continue
        x = torch.from_numpy(graph["node_feat"]).long()
        edge_index = torch.from_numpy(graph["edge_index"]).long()
        edge_attr = torch.from_numpy(graph["edge_feat"]).long()
        num_nodes = graph["num_nodes"]

        crippen = _crippen_contribs(record.smiles, num_nodes)
        if crippen is None:
            crippen = torch.zeros(num_nodes)
        tpsa = _tpsa_contribs(record.smiles, num_nodes)
        if tpsa is None:
            tpsa = torch.zeros(num_nodes)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            crippen=crippen,
            tpsa=tpsa,
        )

        if conformer_cache_path is not None:
            if inchikey in conformer_dropped or inchikey not in conformer_coords:
                n_no_conformer += 1
                continue
            pos = conformer_coords[inchikey]
            if pos.shape[0] != num_nodes:
                n_no_conformer += 1
                continue
            data.pos = pos

        out[inchikey] = data

    if _featurize_fail_count:
        print(f"  Featurization: {_featurize_fail_count} molecules dropped (smiles2graph failure)")
    if conformer_cache_path is not None and n_no_conformer:
        print(f"  Featurization: {n_no_conformer} molecules dropped (no usable conformer)")

    return out


# ---------------------------------------------------------------------------
# 5b. Per-task / per-anchor scale statistics (TRAIN split only)
# ---------------------------------------------------------------------------


def compute_label_scales(
    train_data: list,
    min_scale: float = 1e-3,
    num_tasks: int | None = None,
    specs: list | None = None,
) -> torch.Tensor:
    """
    Per-task label std from the TRAIN split only. Used to normalize masked_multitask_loss's
    L_prop terms (dividing squared error by std^2 puts every task in z-score units, so a
    task on a bigger raw scale doesn't dominate the summed loss). Clamped to min_scale.

    num_tasks: defaults to NUM_TASKS; pass a smaller count when `train_data`'s y is already
    sliced to a task subset (e.g. HIA excluded).

    specs: per-column TaskSpec list. If given, each column is inverse-transformed to NATIVE
    units before taking its std (for reporting/selection). Omit for transformed-space scales
    (the default, used to normalize the training loss).
    """
    num_tasks = NUM_TASKS if num_tasks is None else num_tasks
    y_all = torch.cat([d.y for d in train_data], dim=0)  # [n_train, K]
    scales = torch.ones(num_tasks)
    for k in range(num_tasks):
        col = y_all[:, k]
        col = col[~torch.isnan(col)]
        if specs is not None and specs[k].label_transform != "none":
            col = inverse_transform(specs[k].label_transform, col)
        if col.numel() > 1:
            scales[k] = col.std().clamp(min=min_scale)
    return scales


def compute_anchor_scales(train_data: list, min_scale: float = 1e-3) -> dict:
    """
    Per-atom std of Crippen and TPSA contributions (very different scales: Crippen ~-0.5 to
    0.5, TPSA 0 to ~20+), computed from the TRAIN split's atoms only.
    """
    crippen_all = torch.cat([d.crippen for d in train_data])
    tpsa_all = torch.cat([d.tpsa for d in train_data])
    return {
        "crippen": crippen_all.std().clamp(min=min_scale).item(),
        "tpsa": tpsa_all.std().clamp(min=min_scale).item(),
    }


# ---------------------------------------------------------------------------
# 6. Top-level entrypoint
# ---------------------------------------------------------------------------


def load_multitask_splits(
    mollipo_root="data/raw/mollipo",
    expansionrx_root="data/raw/expansionrx",
    tdc_root="data/raw/tdc",
    esol_root="data/raw/esol",
    seed=42,
    conformer_cache_path: str | None = None,
    return_keys: bool = False,
):
    """
    conformer_cache_path: optional -- see featurize_registry's docstring. None (default) is
    2D-only; pass a path (from build_multitask_conformer_cache) to join 3D conformers.

    return_keys: when True, also return {split: [inchikey, ...]}, aligned position-for-position
    with `out`'s Data lists -- for mapping each Data back to its matched-pair partners.
    """
    registry = build_registry(mollipo_root, expansionrx_root, tdc_root, esol_root)
    split_keys = compute_global_scaffold_split(registry, seed=seed)
    featurized = featurize_registry(registry, conformer_cache_path=conformer_cache_path)

    out, keys = {}, {}
    for split in ("train", "valid", "test"):
        total = len(split_keys[split])
        kept = [k for k in split_keys[split] if k in featurized]
        keys[split] = kept
        out[split] = [featurized[k] for k in kept]
        if conformer_cache_path is not None:
            print(
                f"  [multitask 3D] {split}: kept {len(out[split])}/{total} "
                f"(dropped {total - len(out[split])} without usable conformers/featurization)"
            )
        else:
            print(f"  [multitask] {split}: {len(out[split])} molecules")
    return (out, keys) if return_keys else out


def build_multitask_conformer_cache(
    out_path: str = "data/raw/multitask/conformers.pt",
    mollipo_root="data/raw/mollipo",
    expansionrx_root="data/raw/expansionrx",
    tdc_root="data/raw/tdc",
    esol_root="data/raw/esol",
    max_heavy_atoms: int = 50,
    seed: int = 42,
    limit: int | None = None,
):
    """Builds the merged registry and generates one conformer per unique InChIKey via
    build_conformer_cache_keyed. Run once, offline, before any 3D run.

    limit: optional cap on molecules processed, for smoke-testing before the full
    ~14,500-molecule run."""
    registry = build_registry(mollipo_root, expansionrx_root, tdc_root, esol_root)
    items = [(inchikey, record.smiles) for inchikey, record in registry.items()]
    if limit is not None:
        items = items[:limit]
    build_conformer_cache_keyed(items, out_path, max_heavy_atoms=max_heavy_atoms, seed=seed)
