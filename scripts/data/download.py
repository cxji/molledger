"""
Download all datasets for the MolLedger ADME regression pipeline.

Usage:
    python scripts/data/download.py --all
    python scripts/data/download.py --dataset mollipo
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).parent / "raw"

EXPANSIONRX_REPO_ID = "openadmet/openadmet-expansionrx-challenge-data"


def download_mollipo():
    """ogbg-mollipo: logD, ~4200 molecules, scaffold split."""
    print("Downloading ogbg-mollipo...")
    from ogb.graphproppred import GraphPropPredDataset

    GraphPropPredDataset(name="ogbg-mollipo", root=str(ROOT / "mollipo"))
    print("  ogbg-mollipo done.")


def download_expansionrx():
    """ExpansionRx: 9 ADME properties, 5326 train / 2280 test (confirmed).

    Uses huggingface_hub.hf_hub_download directly rather than datasets.load_dataset --
    the installed `datasets` version (2.2.1) can't resolve this repo's file layout
    (glob-pattern incompatibility with newer fsspec), but the repo is just plain CSVs,
    so fetching them directly sidesteps the dataset-builder machinery entirely.
    """
    print("Downloading ExpansionRx...")
    from huggingface_hub import hf_hub_download

    out_dir = ROOT / "expansionrx"
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["expansion_data_train.csv", "expansion_data_test.csv"]:
        path = hf_hub_download(EXPANSIONRX_REPO_ID, filename, repo_type="dataset")
        out_path = out_dir / filename
        out_path.write_bytes(Path(path).read_bytes())
        print(f"  Saved {out_path}")
    print("  ExpansionRx done.")


def download_tdc():
    """TDC datasets: Caco-2, HIA, half-life."""
    print("Downloading TDC datasets...")
    from tdc.single_pred import ADME

    out_dir = ROOT / "tdc"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ["Caco2_Wang", "HIA_Hou", "Half_Life_Obach"]:
        print(f"  Downloading {name}...")
        data = ADME(name=name, path=str(out_dir))
        split = data.get_split(method="scaffold")
        for part in ["train", "valid", "test"]:
            split[part].to_csv(out_dir / f"{name}_{part}.csv", index=False)
    print("  TDC done.")


def download_esol():
    """ESOL: aqueous solubility, ~1128 molecules."""
    print("Downloading ESOL...")
    import urllib.request

    out_dir = ROOT / "esol"
    out_dir.mkdir(exist_ok=True)
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    out_path = out_dir / "delaney-processed.csv"
    urllib.request.urlretrieve(url, out_path)
    print(f"  Saved ESOL to {out_path}")


DATASETS = {
    "mollipo": download_mollipo,
    "expansionrx": download_expansionrx,
    "tdc": download_tdc,
    "esol": download_esol,
}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Download all datasets")
    group.add_argument("--dataset", choices=list(DATASETS), help="Download a single dataset")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)

    fns = list(DATASETS.values()) if args.all else [DATASETS[args.dataset]]
    for fn in fns:
        fn()

    print("\nAll requested downloads complete.")


if __name__ == "__main__":
    main()
