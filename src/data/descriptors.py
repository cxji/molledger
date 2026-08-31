"""
Whole-molecule descriptor features (RDKit 2D + 3D), shared by the GBT baseline and the models.
The 2D block is the ~208 RDKit descriptors; the 3D block adds shape/contact features.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Descriptors3D
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

_BOND_TYPES = {
    0: Chem.BondType.SINGLE,
    1: Chem.BondType.DOUBLE,
    2: Chem.BondType.TRIPLE,
    3: Chem.BondType.AROMATIC,
}


def mol_from_data(data, with_conformer: bool = False):
    """Rebuild an RDKit Mol from a PyG `Data` produced by `featurize_registry`.

    The Data objects carry OGB feature vectors, not SMILES, so chemical perception (substructure
    matching, 3D descriptors) has to reconstruct the molecule. Atom features used: x[:, 0] atomic
    number index (z - 1), x[:, 3] formal charge index (charge + 5), x[:, 4] attached-H count,
    x[:, 7] aromaticity; bonds from edge_attr[:, 0].
    """
    rw = Chem.RWMol()
    for k in range(data.num_nodes):
        atom = Chem.Atom(int(data.x[k, 0]) + 1)
        atom.SetFormalCharge(int(data.x[k, 3]) - 5)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(int(data.x[k, 4]))
        atom.SetIsAromatic(bool(data.x[k, 7]))
        rw.AddAtom(atom)

    seen = set()
    for e in range(data.edge_index.size(1)):
        i, j = int(data.edge_index[0, e]), int(data.edge_index[1, e])
        key = (min(i, j), max(i, j))
        if key in seen:
            continue  # edge_index stores both directions
        seen.add(key)
        bond_type = _BOND_TYPES.get(int(data.edge_attr[e, 0]), Chem.BondType.SINGLE)
        rw.AddBond(i, j, bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            rw.GetBondBetweenAtoms(i, j).SetIsAromatic(True)

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )

    if with_conformer and getattr(data, "pos", None) is not None:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for k in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(k, Point3D(*[float(v) for v in data.pos[k]]))
        mol.AddConformer(conf, assignId=True)
    return mol


# Ipc/AvgIpc SEGFAULT (not raise -- so this cannot be handled with try/except) on the 1-heavy-atom
# molecules in this registry: they derive information content from the characteristic polynomial of
# the adjacency matrix, which is degenerate for a graph with no bonds. They are routinely dropped
# from QSAR descriptor sets anyway, since Ipc also overflows to ~1e30 on larger molecules.
EXCLUDE_2D = {"Ipc", "AvgIpc"}
DESC_2D = [(name, fn) for name, fn in Descriptors._descList if name not in EXCLUDE_2D]
FEATURE_NAMES_2D = [name for name, _ in DESC_2D]

SHAPE_3D_NAMES = [
    "Asphericity",
    "Eccentricity",
    "InertialShapeFactor",
    "NPR1",
    "NPR2",
    "PBF",
    "PMI1",
    "PMI2",
    "PMI3",
    "RadiusOfGyration",
    "SpherocityIndex",
]
CONTACT_NAMES = [
    "extent",
    "extent_per_atom",
    "rgyr_per_cbrt_atoms",
    "n_contacts",
    "contacts_per_atom",
    "mean_contact_len",
    "n_polar_contacts",
    "n_donor_acceptor_contacts",
]
FEATURE_NAMES_3D = SHAPE_3D_NAMES + CONTACT_NAMES
FEATURE_NAMES = FEATURE_NAMES_2D + FEATURE_NAMES_3D
DIM_2D, DIM_3D = len(FEATURE_NAMES_2D), len(FEATURE_NAMES_3D)
DIM_ALL = DIM_2D + DIM_3D


# Contact-graph atom typing: heavy atoms that can carry an H-bond (N O F P S Cl Br I). OGB stores
# the atomic number as an index into list(range(1, 119)), so z = x[:, 0] + 1; x[:, 4] is attached-H count.
_POLAR_Z = (7, 8, 9, 15, 16, 17, 35, 53)
_DONOR_Z = (7, 8, 16)
_ACCEPTOR_Z = (7, 8, 9, 16, 17, 35, 53)


def _atom_flags(data):
    """(polar, donor, acceptor) boolean masks per atom."""
    z = data.x[:, 0] + 1
    num_h = data.x[:, 4]
    polar = torch.isin(z, torch.tensor(_POLAR_Z, device=z.device))
    donor = torch.isin(z, torch.tensor(_DONOR_Z, device=z.device)) & (num_h > 0)
    acceptor = torch.isin(z, torch.tensor(_ACCEPTOR_Z, device=z.device))
    return polar, donor, acceptor


def _same_fragment(edge_index, n_nodes, device):
    """Boolean [n, n]: are i and j in the same covalently bonded fragment? (excludes salts/co-crystals
    from the contact search, since ETKDG can embed disconnected fragments on top of each other)."""
    row, col = edge_index
    reach = torch.eye(n_nodes, dtype=torch.bool, device=device)
    reach[row, col] = True
    for _ in range(int(n_nodes).bit_length()):  # squaring -> transitive closure
        nxt = reach | (reach.float() @ reach.float() > 0)
        if torch.equal(nxt, reach):
            break
        reach = nxt
    return reach


def _within_n_bonds(edge_index, n_nodes, n_bonds, device):
    """Boolean [n, n]: is j reachable from i in <= n_bonds bonds? (identity included)."""
    row, col = edge_index
    reach = torch.eye(n_nodes, dtype=torch.bool, device=device)
    reach[row, col] = True
    step = reach.clone()
    for _ in range(max(n_bonds - 1, 0)):
        reach = reach | (reach.float() @ step.float() > 0)
    return reach


def _contact_edges(data, topo_min=4, r_polar=4.6, r_nonpolar=4.0, r_min=2.4):
    """Through-space atom-pair contacts for the 3D descriptor summaries: pairs >= topo_min bonds
    apart in the covalent graph (so bonded/ring neighbors, whose distance bond geometry already
    implies, are excluded), within a polar/nonpolar distance cutoff (r_polar covers a one-water
    H-bond bridge, r_nonpolar a van-der-Waals contact), and no closer than r_min (drops
    sub-van-der-Waals clashes). Returns (dist, is_polar_pair, is_donor_acceptor) per contact edge
    (both directions included)."""
    pos = data.pos
    n = pos.size(0)
    d = torch.cdist(pos, pos)
    polar, donor, acceptor = _atom_flags(data)
    polar_pair = polar.unsqueeze(1) & polar.unsqueeze(0)
    donor_acceptor = (donor.unsqueeze(1) & acceptor.unsqueeze(0)) | (
        acceptor.unsqueeze(1) & donor.unsqueeze(0)
    )
    hbond_like = polar_pair | donor_acceptor

    too_close_in_graph = _within_n_bonds(data.edge_index, n, topo_min - 1, pos.device)
    cutoff = torch.where(hbond_like, torch.full_like(d, r_polar), torch.full_like(d, r_nonpolar))
    same_frag = _same_fragment(data.edge_index, n, pos.device)
    keep = (d <= cutoff) & (d >= r_min) & ~too_close_in_graph & same_frag
    si, sj = keep.nonzero(as_tuple=True)
    return d[si, sj], polar_pair[si, sj].float(), donor_acceptor[si, sj].float()


def descriptors_2d(mol):
    out = []
    for _, fn in DESC_2D:
        try:
            v = fn(mol)
            out.append(float(v) if v is not None and math.isfinite(float(v)) else np.nan)
        except Exception:
            out.append(np.nan)
    return out


def descriptors_3d(data, mol):
    """RDKit 3D shape descriptors + scalar summaries of the chemically filtered contact graph."""
    n = data.num_nodes
    # Shape descriptors need a non-degenerate inertia tensor; with 1-2 heavy atoms (methane,
    # ethane and the other ESOL solvents) they are meaningless at best.
    degenerate = n < 3
    out = []
    for name in SHAPE_3D_NAMES:
        if degenerate:
            out.append(np.nan)
            continue
        try:
            v = getattr(Descriptors3D, name)(mol)
            out.append(float(v) if v is not None and math.isfinite(float(v)) else np.nan)
        except Exception:
            out.append(np.nan)

    d = torch.cdist(data.pos, data.pos)
    extent = float(d.max())
    try:
        rgyr = float(Descriptors3D.RadiusOfGyration(mol)) if not degenerate else np.nan
    except Exception:
        rgyr = np.nan

    dist, polar_pair, donor_acceptor = _contact_edges(data)
    n_contacts = dist.size(0) // 2  # both directions stored
    mean_len = float(dist.mean()) if n_contacts else np.nan
    n_polar = float(polar_pair.sum()) / 2 if n_contacts else 0.0
    n_da = float(donor_acceptor.sum()) / 2 if n_contacts else 0.0

    out += [
        extent,
        extent / n,
        rgyr / (n ** (1 / 3)) if rgyr == rgyr else np.nan,
        float(n_contacts),
        n_contacts / n,
        mean_len,
        n_polar,
        n_da,
    ]
    return out


def descriptor_matrix(data_list, log_every: int = 4000):
    """[len(data_list), DIM_ALL] float64 array, NaNs preserved (callers impute)."""
    rows = []
    for i, data in enumerate(data_list):
        mol = mol_from_data(data, with_conformer=True)
        rows.append(descriptors_2d(mol) + descriptors_3d(data, mol))
        if log_every and (i + 1) % log_every == 0:
            print(f"    descriptors: {i + 1}/{len(data_list)}")
    return np.asarray(rows, dtype=np.float64)


def attach_descriptors(splits, cache_path: str | None = None):
    """Attach a standardized 2D+3D descriptor vector to every Data as `.desc` [1, D].

    Standardization and NaN imputation use TRAIN-split statistics only (median impute, then
    z-score); constant columns are dropped so the network never sees a zero-variance input.
    Returns the surviving feature dimension."""
    if cache_path and Path(cache_path).exists():
        blob = torch.load(cache_path, weights_only=False)
        raw = blob["raw"]
        if set(raw) != set(splits) or any(raw[s].shape[0] != len(splits[s]) for s in splits):
            raise ValueError(f"{cache_path} does not match these splits -- delete it and re-run.")
        print(f"  Descriptors: loaded from {cache_path}")
    else:
        raw = {}
        for split, data_list in splits.items():
            print(f"  Descriptors: computing for {split} ({len(data_list)} molecules)...")
            raw[split] = descriptor_matrix(data_list)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"raw": raw, "names": FEATURE_NAMES}, cache_path)
            print(f"  Descriptors: cached to {cache_path}")

    train = raw["train"]
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(train), train, median)
    mean, std = filled.mean(axis=0), filled.std(axis=0)
    live = std > 1e-8  # drop constant columns
    mean, std, median = mean[live], std[live], median[live]

    for split, data_list in splits.items():
        block = raw[split][:, live]
        block = np.where(np.isfinite(block), block, median)
        block = (block - mean) / std
        block = np.clip(block, -10.0, 10.0)  # bound test-split outliers
        for data, row in zip(data_list, block):
            data.desc = torch.tensor(row, dtype=torch.float32).unsqueeze(0)

    dim = int(live.sum())
    print(f"  Descriptors: {dim} features attached ({int((~live).sum())} constant columns dropped)")
    return dim
