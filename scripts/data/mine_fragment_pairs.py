"""
Mine MULTI-ATOM fragment matched pairs (Class D) -- CF3, OCH3, cyclopropyl, sulfonamide, ... --
the R-group swaps that mine_matched_pairs.py (single-atom Class A/B) leaves out.

Single-cut Hussain-Rea fragmentation, using rdMMPA's own cut definition:

    [#6+0;!$(*=,#[!#6])]!@!=!#[*]

i.e. an acyclic, non-ring, single, non-conjugated bond from an uncharged carbon that is not
double/triple bonded to a heteroatom. The fragment decomposition needs ATOM INDICES to partition
each molecule into (core, substituent), so the bond is cut with `FragmentOnBonds` and the pieces
read back with `GetMolFrags(..., fragsMolAtomMapping=...)`, which preserves the original indices.

Two molecules sharing a core with different R-groups form a pair; the larger piece is the core, the
smaller is the substituent whose atom indices `matched_pair_decomposition(site_a=[...])` consumes.
Class D pairs differ in atom count and connectivity, so there is no atom bijection: only the
additive fragment decomposition and fixed-baseline IG (which carries the `R` term) apply.

Usage:
    python scripts/data/mine_fragment_pairs.py --out data/raw/multitask/fragment_pairs.pt
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.multitask import TASK_NAMES, build_registry, compute_global_scaffold_split

RDLogger.DisableLog("rdApp.*")

# rdMMPA's default single-cut pattern (see rdMMPA.FragmentMol's signature).
MMPA_CUT_SMARTS = "[#6+0;!$(*=,#[!#6])]!@!=!#[*]"


def _canonical_fragment(piece):
    """Canonical SMILES with attachment-point isotopes cleared, so a fragment is identified by its
    chemistry rather than by which bond index happened to produce it."""
    piece = Chem.RWMol(piece)
    for atom in piece.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(piece)


def single_cuts(mol, min_frag_atoms=1, max_frag_atoms=12):
    """Yield (core_smiles, r_smiles, r_atom_indices, attachment_atom) for every valid single cut.

    The SMALLER piece is taken as the substituent. `max_frag_atoms` keeps the substituent an
    R-group rather than half the molecule -- without it the "core" of a symmetric molecule is
    ambiguous and the pairs stop being matched pairs in any useful sense.
    """
    patt = Chem.MolFromSmarts(MMPA_CUT_SMARTS)
    n_atoms = mol.GetNumAtoms()
    seen = set()
    for a1, a2 in mol.GetSubstructMatches(patt):
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is None or bond.GetIdx() in seen:
            continue
        seen.add(bond.GetIdx())
        try:
            frag_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True)
            mapping = []
            pieces = Chem.GetMolFrags(
                frag_mol, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=mapping
            )
        except Exception:
            continue
        if len(pieces) != 2:
            continue

        # Original indices survive as the first n_atoms entries; dummies are appended after.
        idx_sets = [[i for i in m if i < n_atoms] for m in mapping]
        order = (0, 1) if len(idx_sets[0]) >= len(idx_sets[1]) else (1, 0)
        ci, ri = order
        r_idx = idx_sets[ri]
        if not (min_frag_atoms <= len(r_idx) <= max_frag_atoms):
            continue
        if len(idx_sets[ci]) < len(r_idx):
            continue

        # FragmentOnBonds labels each dummy with the BOND index, so the same core canonicalizes to
        # `*C(F)(F)...`, `[2*]C(F)(F)...`, `[3*]C(F)(F)...` depending on which bond was cut --
        # identical chemistry, three different strings, and the context index would fragment and
        # lose almost every pair. Strip the isotope so the attachment point is just `*`.
        core_smi = _canonical_fragment(pieces[ci])
        r_smi = _canonical_fragment(pieces[ri])
        attach = a1 if a1 in idx_sets[ci] else a2
        yield core_smi, r_smi, sorted(r_idx), attach


def mine(registry, min_frag_atoms, max_frag_atoms):
    mols = {}
    contexts = defaultdict(list)
    for ik, rec in registry.items():
        mol = Chem.MolFromSmiles(rec.smiles)
        if mol is None:
            continue
        mols[ik] = mol
        for core, r_smi, r_idx, attach in single_cuts(mol, min_frag_atoms, max_frag_atoms):
            contexts[core].append((ik, r_smi, r_idx, attach))

    # Per (molecule, core), keep only the largest core seen (smallest substituent).
    candidates = defaultdict(dict)  # core_smi -> {ik: (r_smi, r_idx, attach)}
    for core, members in contexts.items():
        best = candidates[core]
        for ik, r_smi, r_idx, attach in members:
            prev = best.get(ik)
            if prev is None or len(r_idx) < len(prev[1]):
                best[ik] = (r_smi, r_idx, attach)

    # Resolve each (i, j) pair to its largest shared core across every core group they co-occur in.
    best_pair = {}  # (i, j) -> (sub_size, core_smi, ri, ia, attach_a, rj, ja, attach_b)
    transforms = Counter()
    for core, by_ik in candidates.items():
        if len(by_ik) < 2:
            continue
        for (i, (ri, ia, attach_a)), (j, (rj, ja, attach_b)) in itertools.combinations(
            sorted(by_ik.items()), 2
        ):
            if ri == rj:
                continue
            sub_size = len(ia) + len(
                ja
            )  # want the SMALLEST substituent total = largest shared core
            cand = (sub_size, core, ri, ia, attach_a, rj, ja, attach_b)
            prev = best_pair.get((i, j))
            if prev is None or cand[0] < prev[0]:
                best_pair[(i, j)] = cand

    # Attachment correspondence holds by construction: i and j are grouped under one `core` string,
    # the canonical SMILES of the core piece with `*` pinned at the attachment atom.
    pairs = {}
    for (i, j), (_, core, ri, ia, attach_a, rj, ja, attach_b) in best_pair.items():
        pairs[(i, j)] = {
            "from": ri,
            "to": rj,
            "site_atoms_a": ia,
            "site_atoms_b": ja,
            "attach_a": attach_a,
            "attach_b": attach_b,
            "n_atoms_a": len(ia),
            "n_atoms_b": len(ja),
        }
        transforms[(ri, rj)] += 1
    return pairs, transforms


def report(pairs, registry, where, name):
    loc = Counter()
    for i, j in pairs:
        wi, wj = where.get(i), where.get(j)
        loc[wi if wi == wj else "cross"] += 1
    same_n = sum(1 for v in pairs.values() if v["n_atoms_a"] == v["n_atoms_b"])
    print(f"\n=== {name}: {len(pairs)} unique pairs ===")
    print("  split placement: " + "  ".join(f"{k}={v}" for k, v in loc.most_common()))
    print(
        f"  substituent size: same on both sides in {same_n} pairs "
        f"({100 * same_n / max(len(pairs), 1):.0f}%); N-CHANGING in {len(pairs) - same_n}"
    )
    print(f"  {'assay':<22} {'pairs':>6} {'cross':>6}")
    for t in TASK_NAMES:
        n = n_cross = 0
        for i, j in pairs:
            if t in registry[i].labels and t in registry[j].labels:
                n += 1
                n_cross += where.get(i) != where.get(j)
        if n:
            print(f"  {t:<22} {n:>6} {n_cross:>6}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/multitask/fragment_pairs.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_frag_atoms", type=int, default=1)
    p.add_argument("--max_frag_atoms", type=int, default=12)
    p.add_argument("--top_transforms", type=int, default=25)
    args = p.parse_args()

    registry = build_registry()
    split = compute_global_scaffold_split(registry, seed=args.seed)
    where = {k: name for name, keys in split.items() for k in keys}

    pairs, transforms = mine(registry, args.min_frag_atoms, args.max_frag_atoms)

    print(f"\nTop {args.top_transforms} fragment transforms:")
    for (a, b), n in transforms.most_common(args.top_transforms):
        print(f"   {a:>18} -> {b:<18} {n}")

    report(pairs, registry, where, "CLASS D (multi-atom fragment swap)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "class_d": pairs,
            "smiles": {k: r.smiles for k, r in registry.items()},
            "split": where,
            "seed": args.seed,
            "transforms": dict(transforms),
            "min_frag_atoms": args.min_frag_atoms,
            "max_frag_atoms": args.max_frag_atoms,
        },
        out,
    )
    print(f"\nWrote {len(pairs)} Class-D pairs to {out}")


if __name__ == "__main__":
    main()
