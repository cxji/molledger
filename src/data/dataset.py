import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from pathlib import Path

import pandas as pd
from ogb.graphproppred import PygGraphPropPredDataset
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors
from torch_geometric.data import Data

# torch >= 2.6 defaults torch.load(weights_only=True), which rejects the
# PyG objects pickled inside OGB's preprocessed dataset cache. OGB calls
# torch.load without weights_only=False, so allowlist those classes here.
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except Exception:
    pass

_crippen_error_shown = False


def _crippen_contribs(smiles: str, expected_n_atoms: int) -> torch.Tensor | None:
    """Per-atom logP contributions from Crippen fragment model. Returns None on failure."""
    global _crippen_error_shown
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # Try rdMolDescriptors first (works across RDKit versions)
        try:
            contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        except AttributeError:
            contribs = Crippen._CalcCrippenContribs(mol)

        t = torch.tensor([c[0] for c in contribs], dtype=torch.float)
        if t.shape[0] != expected_n_atoms:
            if not _crippen_error_shown:
                print(
                    f"  [Crippen] atom count mismatch: RDKit={t.shape[0]}, OGB={expected_n_atoms} for SMILES: {smiles[:60]}"
                )
                _crippen_error_shown = True
            return None
        return t
    except Exception as e:
        if not _crippen_error_shown:
            print(f"  [Crippen] error: {e} for SMILES: {smiles[:60]}")
            _crippen_error_shown = True
        return None


_tpsa_error_shown = False


def _tpsa_contribs(smiles: str, expected_n_atoms: int) -> torch.Tensor | None:
    """
    Per-atom TPSA (topological polar surface area) contributions. Complements Crippen for
    multi-task anchoring: Crippen anchors lipophilicity-driven properties, TPSA anchors
    polarity-driven ones (solubility, HIA). Returns None on failure.
    """
    global _tpsa_error_shown
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        contribs = rdMolDescriptors._CalcTPSAContribs(mol)
        t = torch.tensor(list(contribs), dtype=torch.float)
        if t.shape[0] != expected_n_atoms:
            if not _tpsa_error_shown:
                print(
                    f"  [TPSA] atom count mismatch: RDKit={t.shape[0]}, expected={expected_n_atoms} for SMILES: {smiles[:60]}"
                )
                _tpsa_error_shown = True
            return None
        return t
    except Exception as e:
        if not _tpsa_error_shown:
            print(f"  [TPSA] error: {e} for SMILES: {smiles[:60]}")
            _tpsa_error_shown = True
        return None


def load_mollipo(root: str):
    """
    Load ogbg-mollipo with per-atom Crippen logP contributions attached.

    Returns dict with keys 'train', 'valid', 'test', each a list of PyG Data objects.
    Each Data has:
        x           [N, 9]   OGB atom features (integer-encoded)
        edge_index  [2, E]
        edge_attr   [E, 3]   OGB bond features (integer-encoded)
        y           [1]      molecular logD
        crippen     [N]      per-atom Crippen logP contributions (0 if unavailable)
    """
    root = Path(root)
    dataset = PygGraphPropPredDataset(name="ogbg-mollipo", root=str(root))
    split_idx = dataset.get_idx_split()

    smiles_path = root / "ogbg_mollipo" / "mapping" / "mol.csv.gz"
    smiles_list = pd.read_csv(smiles_path)["smiles"].tolist()

    data_list = []
    skipped = 0
    for i, pyg_data in enumerate(dataset):
        smiles = smiles_list[i]
        contribs = _crippen_contribs(smiles, pyg_data.x.shape[0])
        if contribs is None:
            contribs = torch.zeros(pyg_data.x.shape[0])
            skipped += 1

        data = Data(
            x=pyg_data.x,
            edge_index=pyg_data.edge_index,
            edge_attr=pyg_data.edge_attr,
            y=pyg_data.y.squeeze(-1).float(),  # [1]
            crippen=contribs,
        )
        data_list.append(data)

    if skipped:
        print(f"  Crippen fallback to zeros for {skipped}/{len(data_list)} molecules.")

    return {
        split: [data_list[i] for i in split_idx[split].tolist()]
        for split in ("train", "valid", "test")
    }
