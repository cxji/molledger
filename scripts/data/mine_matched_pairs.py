"""
Mine matched molecular pairs (MMPs) for counterfactual-baseline attribution. A matched pair
replaces the IG baseline with a REAL analog, so the decomposed quantity becomes the matched-pair
delta a chemist reasons about: sum_i a_i = y_hat(F analog) - y_hat(Cl analog).

Two classes are mined, and only the first supports the exact machinery:

  CLASS A -- one TERMINAL heavy atom swapped for another at the same position, same bond order
             (F->Cl, CH3->OH, ...). Heavy-atom count and connectivity are identical and only one
             row of `x` differs, so the atom bijection is exact. edge_attr is not always identical
             (an attachment bond's `is_conjugated` can flip); each pair carries a `graph_identical`
             flag to filter on before the paired-baseline IG route, which overrides node embeddings.

  CLASS B -- a terminal heavy atom deleted (H <-> substituent). Heavy-atom count differs by 1, so
             there is no bijection; mined and counted for reference, usable only via fragment-level
             attribution, not the counterfactual IG path.

Pass --conformer_rmsd to report cross-pair core RMSD (conformers are embedded independently, so a
pair's two members are unrelated poses -- keep 3D-consuming paths off matched-pair attribution).

Usage:
    python scripts/data/mine_matched_pairs.py --out data/raw/multitask/matched_pairs.pt
    python scripts/data/mine_matched_pairs.py --out /tmp/pairs.pt --conformer_rmsd
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from ogb.utils.mol import smiles2graph
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.multitask import (
    TASK_NAMES,
    TASK_REGISTRY,
    build_registry,
    compute_global_scaffold_split,
)

RDLogger.DisableLog("rdApp.*")

BOND_SYM = {Chem.BondType.SINGLE: "-", Chem.BondType.DOUBLE: "=", Chem.BondType.TRIPLE: "#"}


def substituent_label(atom, bond):
    """'-F', '-CH3', '=O' -- bond order included, so single- and double-bonded terminals never
    group together (they are different contexts and a different edit)."""
    nh = atom.GetTotalNumHs()
    s = atom.GetSymbol() + (f"H{nh}" if nh > 1 else "H" if nh == 1 else "")
    q = atom.GetFormalCharge()
    if q:
        s += ("+" * q) if q > 0 else ("-" * -q)
    return BOND_SYM.get(bond.GetBondType(), "?") + s


def _canonical_order(mol):
    """Atom indices in the order MolToSmiles emitted them.

    `_smilesAtomOutputOrder` is set as a side effect of MolToSmiles, so it must be read from the
    same object immediately after. It is a COMPUTED private property -- `GetPropsAsDict` hides it
    unless both includePrivate and includeComputed are set -- and it comes back as an RDKit vector
    rather than a list.
    """
    smi = Chem.MolToSmiles(mol)
    props = mol.GetPropsAsDict(includePrivate=True, includeComputed=True)
    order = props.get("_smilesAtomOutputOrder")
    if order is None:
        raise RuntimeError("MolToSmiles did not set _smilesAtomOutputOrder")
    if isinstance(order, str):
        order = [int(v) for v in order.strip("[]").rstrip(",").split(",") if v.strip()]
    return smi, [int(v) for v in order]


def build_contexts(registry):
    """molecule -> every (context SMILES, swapped atom, canonical order) it can produce."""
    mols, contexts, deletions = {}, defaultdict(list), defaultdict(list)
    for ik, rec in registry.items():
        mol = Chem.MolFromSmiles(rec.smiles)
        if mol is None:
            continue
        mols[ik] = mol
        for atom in mol.GetAtoms():
            if atom.GetDegree() != 1 or atom.GetAtomicNum() == 1:
                continue
            idx, bond = atom.GetIdx(), atom.GetBonds()[0]
            label = substituent_label(atom, bond)
            # The linker atom: the swapped/deleted atom's one core-side neighbour.
            linker = bond.GetOtherAtomIdx(idx)

            # Class A context: the shared skeleton, varying atom -> dummy. The dummy keeps the
            # bond (and so the neighbour's implicit-H count) intact, which is what makes the two
            # members of a pair differ in exactly one row of `x`.
            rw = Chem.RWMol(mol)
            a = rw.GetAtomWithIdx(idx)
            a.SetAtomicNum(0)
            a.SetFormalCharge(0)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
            try:
                ctx, order = _canonical_order(rw)
            except Exception:
                continue
            contexts[ctx].append((ik, label, idx, order, linker))

            # Class B context: delete the atom outright -> the H analog, keyed by its InChIKey.
            if bond.GetBondType() == Chem.BondType.SINGLE and mol.GetNumAtoms() > 1:
                rw2 = Chem.RWMol(mol)
                rw2.RemoveAtom(idx)
                try:
                    m2 = rw2.GetMol()
                    Chem.SanitizeMol(m2)
                    deletions[Chem.MolToInchiKey(m2)].append((ik, label, idx, linker))
                except Exception:
                    pass
    return mols, contexts, deletions


def _graph_identical(smiles_a, smiles_b, amap):
    """Do the two members have byte-identical bond sets once A is relabelled into B's indexing?

    Class A guarantees same N, same connectivity and same bond ORDER (the context SMILES keys on
    bond order, so single- and double-bonded terminals never group). It does NOT guarantee equal
    `edge_attr`: ~14% of Class-A pairs differ in the OGB `is_conjugated` bond feature, because
    swapping CH3->OH or CH3->NH2 onto an aromatic ring changes conjugation of the attachment bond.
    Bond type and stereo never differ.

    This matters for exactly one analysis route. Running B's node embeddings through
    `_EmbeddingOverride` on A's graph is only *molecule B* when the graphs agree in edge_attr too;
    otherwise it is a chimera carrying A's bond features, and paired-baseline IG then decomposes
    `y_hat(A) - y_hat(chimera)` rather than the true pair delta. The atom bijection and the
    fragment decomposition read no edge features and are unaffected either way.
    """
    ga, gb = smiles2graph(smiles_a), smiles2graph(smiles_b)
    m = np.asarray(amap)
    ea = {
        tuple(sorted((int(m[u]), int(m[v])))) + tuple(int(t) for t in f)
        for (u, v), f in zip(ga["edge_index"].T, ga["edge_feat"])
    }
    eb = {
        tuple(sorted((int(u), int(v)))) + tuple(int(t) for t in f)
        for (u, v), f in zip(gb["edge_index"].T, gb["edge_feat"])
    }
    return ea == eb


def pair_class_a(contexts, registry):
    """Unique Class-A pairs with an exact atom bijection.

    `order_a` and `order_b` are the two molecules' atom orders in the SAME canonical context
    string, so position k in that string is `order_a[k]` in A and `order_b[k]` in B. Composing
    them gives amap[i_in_A] = i_in_B for every atom, including the swapped one.

    Each pair carries `graph_identical` (see _graph_identical): True means the two graphs agree in
    edge_attr as well as connectivity, which is the precondition for paired-baseline IG. The pairs
    are NOT split into separate classes -- the chemistry is the same and only that one route cares
    -- so filter on the flag at analysis time.
    """
    pairs, transforms = {}, Counter()
    for members in contexts.values():
        by_ik = {}
        for ik, label, idx, order, linker in members:
            by_ik.setdefault(ik, (label, idx, order, linker))
        if len(by_ik) < 2:
            continue
        for (i, (li, xi, oi, ki)), (j, (lj, xj, oj, kj)) in itertools.combinations(
            sorted(by_ik.items()), 2
        ):
            if li == lj or (i, j) in pairs:
                continue
            if len(oi) != len(oj):
                continue
            amap = [0] * len(oi)
            for k in range(len(oi)):
                amap[oi[k]] = oj[k]
            # The bijection must carry the swapped atom to the swapped atom; anything else means
            # the two contexts canonicalized to the same string by coincidence.
            if amap[xi] != xj:
                continue
            try:
                identical = _graph_identical(registry[i].smiles, registry[j].smiles, amap)
            except Exception:
                identical = False
            pairs[(i, j)] = {
                "from": li,
                "to": lj,
                "site_a": xi,
                "site_b": xj,
                "atom_map": amap,
                "linker_a": ki,
                "linker_b": kj,
                "graph_identical": identical,
            }
            transforms[(li, lj)] += 1
    return pairs, transforms


def pair_class_b(deletions, registry):
    """H <-> substituent pairs.

    No atom bijection exists (N differs by 1), but the fragment-level decomposition does not need
    one -- summing scores over a set is order-independent. All it needs is `site`, the index of the
    substituent atom in the SUBSTITUTED member, which is recorded here along with which member that
    is (`heavy`); the other member is all core.
    """
    pairs = {}
    for ik_parent, kids in deletions.items():
        if ik_parent not in registry:
            continue
        for ik_child, label, idx, linker in kids:
            if ik_child == ik_parent:
                continue
            key = (min(ik_child, ik_parent), max(ik_child, ik_parent))
            pairs.setdefault(
                key, {"from": "-H", "to": label, "heavy": ik_child, "site": idx, "linker": linker}
            )
    return pairs


def _fwd(v, kind):
    if kind == "log":
        return float(np.log(max(v, 1e-9)))
    if kind == "log1p":
        return float(np.log1p(max(v, 0.0)))
    return float(v)


def report(pairs, registry, where, name):
    """Assay coverage, split placement, and delta size relative to each task's spread.

    Deltas are reported in MODEL space (the label_transform applied), because that is where the
    model's error lives and where `sd` is the scale the delta has to beat to be worth scoring.
    """
    tr = {t.name: t.label_transform for t in TASK_REGISTRY}
    loc = Counter()
    for i, j in pairs:
        wi, wj = where.get(i), where.get(j)
        loc[wi if wi == wj else "cross"] += 1

    print(f"\n=== {name}: {len(pairs)} unique pairs ===")
    print("  split placement: " + "  ".join(f"{k}={v}" for k, v in loc.most_common()))
    print(
        f"  {'assay':<22} {'pairs':>6} {'cross':>6} {'test':>5} | "
        f"{'med|dy|':>8} {'sd':>7} {'ratio':>6} {'>0.5sd':>7}"
    )
    for t in TASK_NAMES:
        kind, d, n_cross, n_test = tr[t], [], 0, 0
        for i, j in pairs:
            vi = registry[i].labels.get(t)
            vj = registry[j].labels.get(t)
            if vi is None or vj is None:
                continue
            d.append(abs(_fwd(float(vi), kind) - _fwd(float(vj), kind)))
            wi, wj = where.get(i), where.get(j)
            n_cross += wi != wj
            n_test += wi == wj == "test"
        if not d:
            continue
        allv = [_fwd(float(r.labels[t]), kind) for r in registry.values() if t in r.labels]
        sd, med = float(np.std(allv)), float(np.median(d))
        print(
            f"  {t:<22} {len(d):>6} {n_cross:>6} {n_test:>5} | {med:>8.3f} {sd:>7.3f} "
            f"{med / sd:>6.2f} {100 * np.mean(np.array(d) > 0.5 * sd):>6.0f}%"
        )


def conformer_rmsd(pairs, registry, cache_path, limit=400):
    """Core RMSD between the two independently-embedded cached conformers of a pair."""
    from rdkit.Chem import rdFMCS, rdMolAlign

    coords = torch.load(cache_path, map_location="cpu", weights_only=False)["coords"]

    def with_conf(ik):
        m = Chem.MolFromSmiles(registry[ik].smiles)
        if m is None or ik not in coords:
            return None
        xyz = coords[ik].numpy().astype(float)
        if xyz.shape[0] != m.GetNumAtoms():
            return None
        conf = Chem.Conformer(m.GetNumAtoms())
        for i in range(m.GetNumAtoms()):
            conf.SetAtomPosition(i, xyz[i].tolist())
        m.RemoveAllConformers()
        m.AddConformer(conf, assignId=True)
        return m

    rms, core_sizes, skipped = [], [], 0
    for i, j in list(pairs)[:limit]:
        a, b = with_conf(i), with_conf(j)
        if a is None or b is None:
            skipped += 1
            continue
        try:
            res = rdFMCS.FindMCS(
                [a, b], timeout=5, ringMatchesRingOnly=True, completeRingsOnly=True
            )
            if res.canceled or res.numAtoms < 4:
                skipped += 1
                continue
            patt = Chem.MolFromSmarts(res.smartsString)
            ma, mb = a.GetSubstructMatch(patt), b.GetSubstructMatch(patt)
            if not ma or len(ma) != len(mb):
                skipped += 1
                continue
            rms.append(rdMolAlign.AlignMol(b, a, atomMap=list(zip(mb, ma))))
            core_sizes.append(len(ma))
        except Exception:
            skipped += 1
    r = np.array(rms)
    print(f"\n=== cached-conformer core RMSD, {len(r)} pairs (skipped {skipped}) ===")
    if r.size == 0:
        print("  no pairs measurable -- nothing to report")
        return r
    print(f"  median core: {np.median(core_sizes):.0f} atoms")
    print(
        "  RMSD (A): " + "  ".join(f"p{p}={np.percentile(r, p):.2f}" for p in (10, 25, 50, 75, 90))
    )
    print(
        f"  >0.5A {100 * np.mean(r > 0.5):.0f}%   >1A {100 * np.mean(r > 1.0):.0f}%   "
        f">2A {100 * np.mean(r > 2.0):.0f}%"
    )
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/multitask/matched_pairs.pt")
    p.add_argument("--seed", type=int, default=42, help="scaffold-split seed; must match training.")
    p.add_argument(
        "--conformer_rmsd",
        action="store_true",
        help="Measure how far apart a pair's cached conformers are (slow-ish).",
    )
    p.add_argument("--conformer_cache", default="data/raw/multitask/conformers.pt")
    p.add_argument("--top_transforms", type=int, default=25)
    args = p.parse_args()

    registry = build_registry()
    split = compute_global_scaffold_split(registry, seed=args.seed)
    where = {k: name for name, keys in split.items() for k in keys}

    mols, contexts, deletions = build_contexts(registry)
    print(f"\nParsed {len(mols)} molecules into {len(contexts)} Class-A contexts.")

    pairs_a, transforms = pair_class_a(contexts, registry)
    pairs_b = pair_class_b(deletions, registry)

    print(f"\nTop {args.top_transforms} Class-A transforms:")
    for (a, b), n in transforms.most_common(args.top_transforms):
        print(f"   {a:>6} -> {b:<6} {n}")

    n_ident = sum(1 for v in pairs_a.values() if v["graph_identical"])
    print(
        f"\nClass-A graph identity: {n_ident}/{len(pairs_a)} pairs have identical edge_attr "
        f"({100 * n_ident / max(len(pairs_a), 1):.0f}%); the remainder differ in `is_conjugated` "
        f"and are invalid for paired-baseline IG only."
    )

    report(pairs_a, registry, where, "CLASS A (same-N terminal swap)")
    report(pairs_b, registry, where, "CLASS B (H <-> substituent)")

    if args.conformer_rmsd:
        conformer_rmsd(pairs_a, registry, args.conformer_cache)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "class_a": pairs_a,
            "class_b": {k: v for k, v in pairs_b.items()},
            "smiles": {ik: r.smiles for ik, r in registry.items()},
            "split": where,
            "seed": args.seed,
            "transforms": dict(transforms),
        },
        out,
    )
    print(f"\nWrote {len(pairs_a)} Class-A + {len(pairs_b)} Class-B pairs to {out}")


if __name__ == "__main__":
    main()
