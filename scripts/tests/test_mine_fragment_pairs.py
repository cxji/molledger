"""Class-D fragment pairing logic (scripts/data/mine_fragment_pairs.py).

Guards the invariant that pairs are grouped by the `*`-pinned canonical core SMILES, which pins the
linker atom without reconstructing a bare core (the reconstruction path raised on cores that do not
kekulize after atom removal -- imides, lactams, tautomeric azoles).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from scripts.data.mine_fragment_pairs import mine, single_cuts  # noqa: E402

# A pyrazolo-triazine-dione pair differing by one substituent at one imide N. The bare core does not
# kekulize once the substituent is removed; the deleted assert used to reject this valid pair.
MOL_A = "C#CC(C)n1c(=O)c2c(-c3cncn3C)n(Cc3ccnc4ccc(Cl)cc34)nc2n(CC2CC2)c1=O"
MOL_B = "Cn1cncc1-c1c2c(=O)n(CC3CC3)c(=O)n(CC3CC3)c2nn1Cc1ccnc2ccc(Cl)cc12"


def _registry(**smiles):
    return {k: SimpleNamespace(smiles=v, labels={}) for k, v in smiles.items()}


def test_mine_pairs_valid_nonkekulizable_core():
    pairs, _ = mine(_registry(a=MOL_A, b=MOL_B), 1, 12)
    assert list(pairs) == [("a", "b")]
    rec = pairs[("a", "b")]
    assert {rec["from"], rec["to"]} == {"*C(C)C#C", "*CC1CC1"}
    assert rec["attach_a"] == 4 and rec["attach_b"] == 10


def test_single_cuts_pins_linker_atom():
    # mol_b's two imide N each bear cyclopropylmethyl but are not symmetry-equivalent: distinct cores.
    mol = Chem.MolFromSmiles(MOL_B)
    cores = {core for core, r, _, _ in single_cuts(mol, 1, 12) if r == "*CC1CC1"}
    assert len(cores) == 2


def test_symmetric_sites_share_core_string():
    # p-xylene's two methyls are symmetry-equivalent: one shared core (benign symmetry tie).
    mol = Chem.MolFromSmiles("Cc1ccc(C)cc1")
    cores = {core for core, r, _, _ in single_cuts(mol, 1, 12) if r == "*C"}
    assert cores == {"*c1ccc(C)cc1"}


def test_mine_rejects_different_linker_core():
    # Same substituent (methyl) on different cores must not pair: grouping keys differ.
    pairs, _ = mine(_registry(f="Cc1ccc(F)cc1", ef="CCc1ccc(F)cc1", tol="Cc1ccccc1"), 1, 12)
    assert list(pairs) == [("ef", "f")]  # 4-F ethyl/methyl share a core; toluene does not
