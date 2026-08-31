"""
WISP element-substitution mutant builder (Janssen et al., Digital Discovery 2026, eqn 8).

Each heavy atom is mutated, one element at a time, to every element in `WISP_ELEMENTS` except its own.
Each mutant is RDKit-sanitised (invalid valences dropped) and featurised directly from the mol via the
same ogb feature functions the training data used (bit-identical to `ogb.utils.mol.smiles2graph`).

Output is a single packed batch of all of a molecule's mutants, scored in one forward pass. Arrays use
int16 to keep the on-disk cache small. Model-independent, so precomputed once and reused across seeds
(scripts/score/precompute_wisp_mutants.py).
"""

from __future__ import annotations

import numpy as np
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# WISP's substitution alphabet: H, B, C, N, O, F, Si, P, S, Cl, Br, I (the 12 organic elements).
WISP_ELEMENTS = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)


def _featurize_mol(mol):
    """(x, edge_index, edge_attr) for a sanitised mol, matching ogb.smiles2graph exactly but without
    the SMILES round-trip. edge_index is [2, 2E] (both directions), as smiles2graph emits."""
    x = np.array([atom_to_feature_vector(a) for a in mol.GetAtoms()], dtype=np.int16)
    ei, ef = [], []
    for b in mol.GetBonds():
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_to_feature_vector(b)
        ei.append((s, e))
        ei.append((e, s))
        ef.append(bf)
        ef.append(bf)
    if ei:
        edge_index = np.array(ei, dtype=np.int64).T
        edge_attr = np.array(ef, dtype=np.int16)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 3), dtype=np.int16)
    return x, edge_index, edge_attr


def build_mutants(smiles):
    """
    All valid single-atom element-substitution mutants of `smiles`, packed into one batch.

    Returns a dict (or None if the SMILES will not parse):
        n_atoms     int   -- heavy-atom count of the ORIGINAL molecule (== number of attributable atoms)
        mut_atom    [M]   -- source atom index (0..n_atoms-1) each of the M valid mutants came from
        node_batch  [sumN]-- mutant id (0..M-1) for each packed node
        x           [sumN, 9]  int16  -- packed atom features of all mutants
        edge_index  [2, sumE]   int64 -- packed, node ids already offset into the packed batch
        edge_attr   [sumE, 3]   int16
    A molecule with no valid mutants (rare) returns M == 0 arrays with n_atoms set, so callers still
    know how many atoms to emit (as zeros).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    N = mol.GetNumAtoms()
    xs, eis, efs, nbs, mut_atom = [], [], [], [], []
    off, mid = 0, 0
    for i in range(N):
        zi = mol.GetAtomWithIdx(i).GetAtomicNum()
        for z in WISP_ELEMENTS:
            if z == zi:
                continue
            rw = Chem.RWMol(mol)
            a = rw.GetAtomWithIdx(i)
            a.SetAtomicNum(z)
            a.SetFormalCharge(0)
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(False)
            try:
                m2 = rw.GetMol()
                Chem.SanitizeMol(m2)  # WISP's validity check + feature re-perception
                x, ei, ef = _featurize_mol(m2)
            except Exception:
                continue
            n = x.shape[0]
            if n < 1:
                continue
            xs.append(x)
            efs.append(ef)
            eis.append(ei + off)  # offset node ids into the packed batch
            nbs.append(np.full(n, mid, dtype=np.int64))
            mut_atom.append(i)
            off += n
            mid += 1

    if mid == 0:
        return {
            "n_atoms": N,
            "mut_atom": np.zeros(0, np.int64),
            "node_batch": np.zeros(0, np.int64),
            "x": np.zeros((0, 9), np.int16),
            "edge_index": np.zeros((2, 0), np.int64),
            "edge_attr": np.zeros((0, 3), np.int16),
        }
    return {
        "n_atoms": N,
        "mut_atom": np.array(mut_atom, dtype=np.int64),
        "node_batch": np.concatenate(nbs, 0),
        "x": np.concatenate(xs, 0),
        "edge_index": np.concatenate(eis, 1),
        "edge_attr": np.concatenate(efs, 0),
    }
