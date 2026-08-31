"""RDKit conformer generation + on-disk caching."""

from collections.abc import Iterable
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def generate_conformer(mol, seed: int = 42):
    """Embed + MMFF-optimize a conformer, return heavy-atom coords [N, 3] or None on failure."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        return None
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    return conf.GetPositions()


def build_conformer_cache_keyed(
    items: Iterable[tuple],
    out_path: str,
    max_heavy_atoms: int = 50,
    seed: int = 42,
    log_every: int = 500,
):
    """
    Keyed by an arbitrary string key (e.g. InChIKey) so it can cover the merged multi-source
    registry (mollipo + ExpansionRx + TDC + ESOL), rather than only one source. Downstream code
    (src.data.multitask.featurize_registry) looks up by key, dropping any that are missing or
    have a mismatched atom count.

    `items` is a list, not a lazy generator, in practice (multitask.py's registry is fully
    built before this is called) -- but accepting Iterable keeps the signature honest about
    what's actually required (single pass).
    """
    items = list(items)
    coords = {}
    dropped = {}
    for i, (key, smiles) in enumerate(items):
        if log_every and i > 0 and i % log_every == 0:
            print(
                f"  ...{i}/{len(items)} molecules processed "
                f"({len(coords)} kept, {len(dropped)} dropped so far)"
            )

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            dropped[key] = "invalid_smiles"
            continue
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy > max_heavy_atoms:
            dropped[key] = "too_large"
            continue

        xyz = generate_conformer(mol, seed=seed)
        if xyz is None:
            dropped[key] = "embed_failed"
            continue
        if xyz.shape[0] != n_heavy:
            dropped[key] = "atom_count_mismatch"
            continue

        coords[key] = torch.from_numpy(xyz).float()

    reasons = {}
    for reason in dropped.values():
        reasons[reason] = reasons.get(reason, 0) + 1

    print(f"Conformer cache: kept {len(coords)}/{len(items)}")
    for reason, count in sorted(reasons.items()):
        print(f"  dropped ({reason}): {count}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "coords": coords,
            "dropped": dropped,
            "n_total": len(items),
            "seed": seed,
            "max_heavy_atoms": max_heavy_atoms,
        },
        out_path,
    )
    print(f"Saved to {out_path}")
