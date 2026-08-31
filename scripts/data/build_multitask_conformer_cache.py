"""
Build a 3D conformer cache for the unified multi-task registry (mollipo + ExpansionRx +
TDC + ESOL, ~14,500 molecules), keyed by InChIKey. Required before loading 3D data.

Usage:
    python scripts/data/build_multitask_conformer_cache.py
    python scripts/data/build_multitask_conformer_cache.py --out data/raw/multitask/conformers.pt
"""

import argparse

from src.data.multitask import build_multitask_conformer_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/multitask/conformers.pt")
    parser.add_argument("--mollipo_root", default="data/raw/mollipo")
    parser.add_argument("--expansionrx_root", default="data/raw/expansionrx")
    parser.add_argument("--tdc_root", default="data/raw/tdc")
    parser.add_argument("--esol_root", default="data/raw/esol")
    parser.add_argument("--max_heavy_atoms", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap molecule count for smoke-testing before the full run.",
    )
    args = parser.parse_args()

    build_multitask_conformer_cache(
        out_path=args.out,
        mollipo_root=args.mollipo_root,
        expansionrx_root=args.expansionrx_root,
        tdc_root=args.tdc_root,
        esol_root=args.esol_root,
        max_heavy_atoms=args.max_heavy_atoms,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
