"""
Descriptor + gradient-boosted-tree baseline testing whether 3D geometry carries signal for these
endpoints, independently of any GNN. Same molecules, same scaffold split, same model class; the
two arms differ only in whether the feature vector includes 3D descriptors.

Feature blocks (identical rows in both arms):
  2D   -- all ~210 RDKit 2D descriptors.
  +3D  -- RDKit's 3D descriptor set (Asphericity, Eccentricity, InertialShapeFactor, NPR1/2,
          PBF, PMI1-3, RadiusOfGyration, SpherocityIndex) plus molecular extent, size-normalised
          compactness, and counts/lengths of the through-space contacts from src/data/geom_edges.py
          (total, polar-polar, donor-acceptor).

Per task, rows with a non-NaN label; hyperparameters chosen on the scaffold VALID split; test MAE
reported in model space and native units, with a paired bootstrap CI on the 2D -> 2D+3D difference.
Runs on CPU in a few minutes.

    python scripts/train/descriptor_baseline.py
    python scripts/train/descriptor_baseline.py --conformer_cache data/raw/multitask/conformers.pt
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from scripts.train.train_molledger import NTASK, TASKS, subset_tasks
from src.data.descriptors import DIM_2D, FEATURE_NAMES, descriptor_matrix
from src.data.multitask import compute_label_scales, load_multitask_splits
from src.data.transforms import inverse_transform


def build_features(splits, cache_path=None):
    """Raw (unstandardized) descriptor blocks per split, sharing ONE cache file and one
    implementation with the trainers' `attach_descriptors` -- same `{"raw": {split: [n, DIM_ALL]}}`
    schema, so whichever runs first pays the ~2 min and the other reads it.

    Trees need no standardization and handle NaN natively, so this returns the raw matrix split at
    DIM_2D; `attach_descriptors` applies the train-only impute/z-score that the networks need."""
    raw = None
    if cache_path and Path(cache_path).exists():
        blob = torch.load(cache_path, weights_only=False)
        raw = blob["raw"]
        if set(raw) != set(splits) or any(raw[s].shape[0] != len(splits[s]) for s in splits):
            raise ValueError(f"{cache_path} does not match these splits -- delete it and re-run.")
        print(f"Loaded cached descriptors from {cache_path}")

    if raw is None:
        raw = {}
        for split, data_list in splits.items():
            t0 = time.time()
            raw[split] = descriptor_matrix(data_list)
            print(
                f"  {split}: {raw[split].shape[0]} molecules, "
                f"{raw[split].shape[1]} features ({time.time() - t0:.0f}s)"
            )
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"raw": raw, "names": FEATURE_NAMES}, cache_path)
            print(f"Cached descriptors to {cache_path}")

    X2d = {s: raw[s][:, :DIM_2D] for s in splits}
    X3d = {s: raw[s][:, DIM_2D:] for s in splits}
    Y = {s: np.asarray([d.y.view(-1).numpy() for d in splits[s]], dtype=np.float64) for s in splits}
    return X2d, X3d, Y


GRID = [
    {"learning_rate": lr, "max_iter": it, "max_leaf_nodes": leaves}
    for lr in (0.05, 0.1)
    for it in (200, 500)
    for leaves in (15, 31)
]


def _corr(a, b, kind):
    """Pearson/Spearman with a NaN guard for constant vectors (scipy warns and returns nan)."""
    from scipy.stats import pearsonr, spearmanr

    if a.std() < 1e-12 or b.std() < 1e-12 or len(a) < 3:
        return float("nan")
    r = (pearsonr if kind == "pearson" else spearmanr)(a, b)[0]
    return float(r)


def fit_select(Xtr, ytr, Xva, yva, seed):
    """Grid-search on the scaffold validation split (not a random internal split, so the selection
    respects the same scaffold discipline as GNN checkpoint selection)."""
    best, best_mae, best_cfg = None, math.inf, None
    for cfg in GRID:
        model = HistGradientBoostingRegressor(
            loss="absolute_error", random_state=seed, early_stopping=False, **cfg
        )
        model.fit(Xtr, ytr)
        mae = float(np.abs(model.predict(Xva) - yva).mean())
        if mae < best_mae:
            best, best_mae, best_cfg = model, mae, cfg
    return best, best_mae, best_cfg


def paired_bootstrap(err_a, err_b, n_boot, seed):
    """95% CI on mean(err_a) - mean(err_b), resampling test molecules (paired)."""
    rng = np.random.default_rng(seed)
    n = len(err_a)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = err_a[idx].mean() - err_b[idx].mean()
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conformer_cache", default="data/raw/multitask/conformers.pt")
    p.add_argument(
        "--feature_cache",
        default="data/raw/multitask/descriptor_cache.pt",
        help="Shared with the trainers' --descriptor_cache; same schema, one file.",
    )
    p.add_argument("--out", default="results_descriptor_baseline.json")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Scaffold-split seed. Keep fixed (42) so every seed replicate scores on "
        "the same held-out molecules as the GNN checkpoints.",
    )
    p.add_argument(
        "--init_seed",
        type=int,
        default=None,
        help="Estimator random_state (histogram subsampling), analogous to the GNN "
        "--init_seed. Defaults to --seed. Vary this across {19,209,31} for a "
        "seed spread that holds the split fixed.",
    )
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()

    est_seed = args.init_seed if args.init_seed is not None else args.seed
    splits = load_multitask_splits(seed=args.seed, conformer_cache_path=args.conformer_cache)
    splits = {s: subset_tasks(dl) for s, dl in splits.items()}
    label_scales = compute_label_scales(splits["train"], num_tasks=NTASK).numpy()

    X2d, X3d, Y = build_features(splits, args.feature_cache)
    XB = {s: np.hstack([X2d[s], X3d[s]]) for s in splits}

    results, agg = {}, {"2d": [], "3d": []}
    print(
        f"\n{'task':22s} {'n_train':>8s} | {'2D test':>9s} {'+3D test':>9s} {'delta':>8s} "
        f"{'95% CI of delta':>20s}"
    )
    print("-" * 84)
    for k, spec in enumerate(TASKS):
        m = {s: ~np.isnan(Y[s][:, k]) for s in splits}
        ytr, yva, yte = (Y[s][m[s], k] for s in ("train", "valid", "test"))
        if len(yte) < 10 or len(ytr) < 50:
            print(f"{spec.name:22s} skipped (too few labels)")
            continue

        row = {"n_train": int(len(ytr)), "n_test": int(len(yte))}
        # HistGradientBoosting is deterministic at this data size (histogram binning subsamples only
        # above ~200k rows; our tasks are ~9-16k, so random_state is a no-op and every init_seed
        # gives an identical fit). To get a genuine seed spread -- the descriptor-model analog of the
        # GNN's init-seed variation -- resample the training rows with replacement per seed (a
        # bootstrap / bagging draw). init_seed=None keeps the full training set (the deterministic
        # single-run baseline); the same draw is shared by the 2d and 3d arms so their comparison
        # stays paired within a seed.
        boot = (
            np.random.default_rng(est_seed + k).integers(0, len(ytr), len(ytr))
            if args.init_seed is not None
            else None
        )
        errs = {}
        for arm, X in (("2d", X2d), ("3d", XB)):
            Xtr, ytr_a = X["train"][m["train"]], ytr
            if boot is not None:
                Xtr, ytr_a = Xtr[boot], ytr_a[boot]
            model, va_mae, cfg = fit_select(Xtr, ytr_a, X["valid"][m["valid"]], yva, est_seed)
            pred = model.predict(X["test"][m["test"]])
            errs[arm] = np.abs(pred - yte)
            # native-unit metrics, matching build_test_split_jsons._metrics so the GBT row is
            # directly comparable to the GNN rows (MAE/RMSE native; Pearson/Spearman/R^2 scale-free).
            pnat = inverse_transform(spec.label_transform, torch.tensor(pred)).numpy()
            ynat = inverse_transform(spec.label_transform, torch.tensor(yte)).numpy()
            enat = pnat - ynat
            sst = float(((ynat - ynat.mean()) ** 2).sum())
            row[arm] = {
                "val_mae": va_mae,
                "test_mae_model": float(errs[arm].mean()),
                "test_mae_native": float(np.abs(enat).mean()),
                "test_rmse_native": float(np.sqrt((enat**2).mean())),
                "test_pearson": _corr(pnat, ynat, "pearson"),
                "test_spearman": _corr(pnat, ynat, "spearman"),
                "test_r2": float(1.0 - (enat**2).sum() / sst) if sst > 0 else float("nan"),
                "config": cfg,
            }
            agg[arm].append(float(errs[arm].mean()) / float(label_scales[k]))

        delta = row["3d"]["test_mae_model"] - row["2d"]["test_mae_model"]
        lo, hi = paired_bootstrap(errs["3d"], errs["2d"], args.n_boot, args.seed)
        row["delta_model_mae"] = delta
        row["delta_ci95"] = [lo, hi]
        row["significant"] = bool(hi < 0 or lo > 0)
        results[spec.name] = row
        flag = "  *" if row["significant"] else ""
        print(
            f"{spec.name:22s} {len(ytr):8d} | {row['2d']['test_mae_model']:9.4f} "
            f"{row['3d']['test_mae_model']:9.4f} {delta:+8.4f} "
            f"[{lo:+.4f}, {hi:+.4f}]{flag}"
        )

    print("-" * 84)
    a2, a3 = float(np.mean(agg["2d"])), float(np.mean(agg["3d"]))
    print(f"{'mean-normalized':22s} {'':8s} | {a2:9.4f} {a3:9.4f} {a3 - a2:+8.4f}")
    print("\n(* = paired bootstrap CI excludes 0. Negative delta favours +3D.)")

    Path(args.out).write_text(
        json.dumps(
            {
                "per_task": results,
                "mean_normalized": {"2d": a2, "3d": a3},
                "seed": args.seed,
                "init_seed": est_seed,
                "conformer_cache": args.conformer_cache,
            },
            indent=2,
        )
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
