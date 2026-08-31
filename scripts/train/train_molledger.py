"""
Trains MolLedger: one GINEConv (2D) backbone with a choice of readout head --
additive sum-of-atom-scores (MolLedger), non-additive pooled MLP, GNAN, or LigandFormer --
over the SAME global scaffold split, so every arm's numbers are directly comparable.

Reading the ablation (all trained property-only by default, --lambda_anchor 0):
    pooled  vs additive  (same backbone)  -> cost of the additive-decomposition constraint

Usage:
    python scripts/train/train_molledger.py --head pooled --epochs 100
    python scripts/train/train_molledger.py --head additive --additive_global_context \\
        --additive_context_dim 8 --anchor_rule --lambda_anchor 0.3
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.descriptors import attach_descriptors
from src.data.multitask import (
    GRAPH_TASK_COLS,
    GRAPH_TASK_SPECS,
    apply_anchor_rule,
    compute_anchor_scales,
    compute_label_scales,
    load_multitask_splits,
)
from src.data.transforms import inverse_transform
from src.logger import MetricLogger
from src.losses import masked_multitask_loss
from src.models.gnn import AdditiveGNN, GNANModel, PooledGNN
from src.models.ligandformer import LigandFormerGNN

# The 11 non-binary tasks, HIA dropped (binary), so every arm is scored on the same task set.
# TASK_COLS maps each of these 11 tasks back to its column in the full 12-task registry `y`.
TASKS = GRAPH_TASK_SPECS
NTASK = len(TASKS)
TASK_COLS = GRAPH_TASK_COLS


def subset_tasks(data_list):
    """Slice each Data's y down from the full 12-task registry to the 11 non-binary tasks
    (HIA dropped), keeping x/edges/crippen/tpsa and pos when present. New Data objects so
    the shared split lists (reused across runs) aren't mutated."""
    out = []
    for d in data_list:
        kw = dict(
            x=d.x,
            edge_index=d.edge_index,
            edge_attr=d.edge_attr,
            y=d.y[:, TASK_COLS],
            crippen=d.crippen,
            tpsa=d.tpsa,
        )
        if getattr(d, "pos", None) is not None:
            kw["pos"] = d.pos
        out.append(Data(**kw))
    return out


def build_model(args, edge_in_dim, desc_dim=0):
    if args.head == "gnan":
        # GNAN baseline: additive-over-features GAM. Exact-sum like AdditiveGNN, so it returns
        # (pred, scores) and rides the additive attribution path. No message passing / edge
        # features / descriptors -- structure enters only via the per-hop decay.
        return GNANModel(node_in_dim=9, edge_in_dim=edge_in_dim, num_tasks=NTASK, desc_dim=desc_dim)
    if args.head == "pooled":
        return PooledGNN(
            node_in_dim=9,
            edge_in_dim=edge_in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_tasks=NTASK,
            dropout=args.dropout,
            desc_dim=desc_dim,
        )
    if args.head == "ligandformer":
        # LigandFormer (HAG-Net self-attention) reproduction: its own architecture, not a head on the
        # GIN trunk. No edge features / descriptors, so edge_in_dim/desc_dim are accepted-but-ignored.
        # Pooled-style readout (returns (pred, None)).
        return LigandFormerGNN(
            node_in_dim=9, edge_in_dim=edge_in_dim, num_tasks=NTASK, dropout=args.dropout
        )
    return AdditiveGNN(
        node_in_dim=9,
        edge_in_dim=edge_in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_tasks=NTASK,
        dropout=args.dropout,
        desc_dim=desc_dim,
        readout=args.additive_readout,
        global_context=args.additive_global_context,
        context_dim=args.additive_context_dim,
    )


def _model_forward(model, batch):
    """Raw model call. Returns whatever the model returns -- a (pred, scores) pair (scores is None
    for the pooled/ligandformer heads that have no per-atom scores)."""
    desc = getattr(batch, "desc", None)
    return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, desc)


def forward_model(model, batch, backbone="gin"):
    """(pred, scores) for every caller (train, eval, external scoring). `backbone` is accepted for
    the checkpoint's recorded-backbone contract (always "gin")."""
    return _model_forward(model, batch)


def evaluate(model, loader, device, backbone):
    """Per-task MAE over the 11 (continuous) tasks, native units + model/log space, over non-NaN
    entries only (all 11 are continuous, so there is no binary/accuracy branch here)."""
    model.eval()
    preds_by_task = [[] for _ in range(NTASK)]
    labels_by_task = [[] for _ in range(NTASK)]
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _ = forward_model(model, batch, backbone)
            for k in range(NTASK):
                mask = ~torch.isnan(batch.y[:, k])
                if mask.any():
                    preds_by_task[k].append(pred[mask, k].cpu())
                    labels_by_task[k].append(batch.y[mask, k].cpu())

    metrics, metrics_model = {}, {}
    for k, spec in enumerate(TASKS):
        if not preds_by_task[k]:
            metrics[spec.name] = metrics_model[spec.name] = float("nan")
            continue
        p = torch.cat(preds_by_task[k])
        y = torch.cat(labels_by_task[k])
        metrics_model[spec.name] = (p - y).abs().mean().item()
        pn = inverse_transform(spec.label_transform, p)
        yn = inverse_transform(spec.label_transform, y)
        metrics[spec.name] = (pn - yn).abs().mean().item()
    return metrics, metrics_model


def mean_normalized_continuous_metric(metrics, label_scales):
    vals = [
        v / label_scales[k].item()
        for k, (spec, v) in enumerate(zip(TASKS, metrics.values()))
        if not math.isnan(v)
    ]
    return sum(vals) / len(vals) if vals else math.inf


def train(args):
    if args.additive_global_context:
        if args.head != "additive":
            raise SystemExit("--additive_global_context requires --head additive.")
        if args.additive_context_dim <= 0:
            raise SystemExit("--additive_global_context requires --additive_context_dim > 0.")
        if args.additive_readout == "summean":
            raise SystemExit(
                "--additive_global_context wires into the plain additive readout only "
                "(no --additive_readout summean)."
            )
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device} | backbone={args.backbone} head={args.head}")

    # Swap in the corrected anchor assignment.
    if args.anchor_rule:
        global TASKS
        TASKS = apply_anchor_rule(GRAPH_TASK_SPECS)
        # anchor_sign is meaningless on an unanchored task, so normalize it away before diffing
        ident = lambda s: (s.anchor, s.anchor_sign if s.anchor != "none" else 0)
        changed = [
            (o.name, o.anchor, n.anchor, n.anchor_sign)
            for o, n in zip(GRAPH_TASK_SPECS, TASKS)
            if ident(o) != ident(n)
        ]
        print(f"ANCHOR_RULE applied: {len(changed)}/{len(TASKS)} tasks changed anchor")
        for name, old, new, sign in changed:
            print(f"    {name:<22} {old:<8} -> {new}{'+' if sign > 0 else '-' if sign < 0 else ''}")
        if not args.lambda_anchor:
            # Not an error -- the control arm of the sweep is exactly this -- but worth saying,
            # since it is also what a forgotten --lambda_anchor looks like.
            print("    (lambda_anchor=0, so the anchor terms are inert and this changes nothing)")

    # 3D descriptors read data.pos, so a run still needs the conformer cache when --descriptors 2d3d
    # is requested (the molecule set then matches the 3D splits, not the 2D ones).
    needs_3d = args.descriptors == "2d3d"
    conf = args.conformer_cache if needs_3d else None
    print(f"Loading unified multi-task registry ({'3D' if needs_3d else '2D'})...")
    splits = load_multitask_splits(seed=args.seed, conformer_cache_path=conf)
    splits = {s: subset_tasks(dl) for s, dl in splits.items()}  # drop HIA -> 11 tasks
    edge_in_dim = 3  # OGB bond features (bond type, stereo, conjugation)

    desc_dim = 0
    if args.descriptors != "none":
        desc_dim = attach_descriptors(splits, cache_path=args.descriptor_cache)

    train_loader = DataLoader(splits["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(splits["valid"], batch_size=args.batch_size)
    test_loader = DataLoader(splits["test"], batch_size=args.batch_size)

    label_scales = compute_label_scales(splits["train"], num_tasks=NTASK).to(device)
    anchor_scales = compute_anchor_scales(splits["train"])

    # Two INDEPENDENT seeds, deliberately. `--seed` fixes only the scaffold split (it is passed to
    # load_multitask_splits above and nowhere else); `--init_seed` fixes everything about the
    # training run -- weight init and DataLoader shuffle order.
    random.seed(args.init_seed)
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    torch.cuda.manual_seed_all(args.init_seed)

    model = build_model(args, edge_in_dim, desc_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    # Anything that changes what the run IS belongs in the directory name
    suffix = "" if args.descriptors == "none" else f"_desc{args.descriptors}"
    if args.lambda_anchor:
        suffix += f"_anchor{args.lambda_anchor:g}-shape"
        # Only marked when the anchor is actually on: at lambda_anchor=0 the rule is inert, so a
        # control arm keeps its plain name
        if args.anchor_rule:
            suffix += "-rule"
    # global-context head: context_dim changes what the run IS (and is a swept var), so it must be in
    # the path
    if args.additive_global_context:
        suffix += f"_gctx{args.additive_context_dim}"
    if args.init_seed:
        suffix += f"_init{args.init_seed}"
    checkpoint_dir = (
        Path(args.checkpoint_dir) / f"ablation_{args.backbone}_{args.head}_none{suffix}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricLogger(f"{checkpoint_dir}/metrics.json")
    ckpt_path = checkpoint_dir / "best.pt"
    ckpt_last_path = checkpoint_dir / "last.pt"

    best_metric, best_epoch, start_epoch = math.inf, 0, 1
    resume_path = ckpt_last_path if ckpt_last_path.exists() else ckpt_path
    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        if "optimizer" not in ckpt or "scheduler" not in ckpt:
            raise ValueError(
                f"{resume_path} predates optimizer/scheduler state -- start fresh "
                f"rather than restarting the cosine LR schedule mid-run."
            )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_metric = ckpt["best_metric"]
        best_epoch = ckpt.get("best_epoch", 0)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']} (best {best_metric:.4f} @ {best_epoch})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred, scores = forward_model(model, batch)
            # scores is None for pooled/ligandformer heads -> masked_multitask_loss skips the anchor.
            loss, log = masked_multitask_loss(
                pred,
                scores,
                batch.y,
                batch.crippen,
                batch.tpsa,
                TASKS,
                lambda_prop=args.lambda_prop,
                lambda_anchor=args.lambda_anchor,
                label_scales=label_scales,
                batch_index=batch.batch,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics, val_metrics_model = evaluate(model, val_loader, device, args.backbone)
            norm_agg = mean_normalized_continuous_metric(val_metrics_model, label_scales.cpu())
            print(
                f"Epoch {epoch:03d} | loss {total_loss / len(train_loader):.4f} | "
                f"mean normalized val MAE {norm_agg:.4f}"
            )
            for spec in TASKS:
                print(f"    {spec.name}: {val_metrics[spec.name]:.4f}")
            logger.log(
                {
                    "epoch": epoch,
                    "val_mean_normalized_mae": norm_agg,
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    **log,
                }
            )

            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "label_scales": label_scales.cpu(),
                "anchor_scales": anchor_scales,
                "backbone": args.backbone,
                "head": args.head,
                "additive_readout": args.additive_readout,  # sum|summean (intensive mean branch)
                "additive_global_context": args.additive_global_context,  # mean-field global-context head
                "additive_context_dim": args.additive_context_dim,  # context g bottleneck width
                "task_names": [t.name for t in TASKS],
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "descriptors": args.descriptors,
                "desc_dim": desc_dim,
                # GNAN forward-time config not recoverable from weights (normalize_rho has no params);
                # rho_hidden/layers/per_task ARE inferable from shapes at reload.
                "gnan_normalize_rho": getattr(model, "normalize_rho", None),
                "lambda_anchor": args.lambda_anchor,
                "seed": args.seed,
                "init_seed": args.init_seed,
                # The anchors actually trained against, not just the flag: scoring scripts and
                # anyone reading a checkpoint months from now should not have to infer them from
                # the directory name or trust that ANCHOR_RULE never changed.
                "anchor_rule": args.anchor_rule,
                "task_anchors": {t.name: [t.anchor, t.anchor_sign] for t in TASKS},
            }
            torch.save(state, ckpt_last_path)
            if norm_agg < best_metric:
                best_metric, best_epoch = norm_agg, epoch
                state["best_metric"], state["best_epoch"] = best_metric, best_epoch
                torch.save(state, ckpt_path)

    print(f"\nBest mean normalized val MAE: {best_metric:.4f} at epoch {best_epoch}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test_metrics, _ = evaluate(model, test_loader, device, args.backbone)
    print("\nTest metrics:")
    for spec in TASKS:
        print(f"  {spec.name}: {test_metrics[spec.name]:.4f}")
    logger.set_summary("best_val_mean_normalized_mae", best_metric)
    logger.set_summary("best_epoch", best_epoch)
    for spec in TASKS:
        logger.set_summary(f"test_{spec.name}", test_metrics[spec.name])


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backbone",
        choices=["gin"],
        default="gin",
        help="GINEConv 2D backbone (the only backbone).",
    )
    p.add_argument("--head", choices=["additive", "pooled", "gnan", "ligandformer"], required=True)
    p.add_argument(
        "--additive_global_context",
        action="store_true",
        help="--head additive only: global-context additive head. s_i = MLP([h_i;g]), "
        "y_hat_k = sum_i s_i, where g is a per-molecule context vector broadcast to every atom. "
        "Requires --additive_readout sum (default).",
    )
    p.add_argument(
        "--additive_context_dim",
        type=int,
        default=0,
        help="--additive_global_context context width (g in R^dim). Required (>0) when "
        "--additive_global_context is set.",
    )
    p.add_argument(
        "--additive_readout",
        choices=["sum", "summean"],
        default="sum",
        help="--head additive only: 'sum' (default) is y_hat_k = sum_i s_i. 'summean' adds a "
        "parallel per-molecule-mean branch, s_i = head_sum(h_i) + head_mean(h_i)/N.",
    )
    p.add_argument(
        "--conformer_cache",
        default="data/raw/multitask/conformers.pt",
        help="3D conformer cache (required for --descriptors 2d3d).",
    )
    p.add_argument(
        "--descriptors",
        choices=["none", "2d3d"],
        default="none",
        help="Inject RDKit whole-molecule descriptors (--head pooled only): the 208 2D "
        "descriptors plus 19 shape/contact 3D features, concatenated to the pooled graph vector.",
    )
    p.add_argument("--descriptor_cache", default="data/raw/multitask/descriptor_cache.pt")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_prop", type=float, default=1.0)
    p.add_argument(
        "--lambda_anchor",
        type=float,
        default=0.0,
        help="Weight for the Crippen/TPSA anchor term. Centres each side per molecule and "
        "normalizes by its std, supervising the per-molecule atom ordering. Only affects "
        "tasks whose TaskSpec.anchor is not 'none'; see src/losses.py.",
    )
    p.add_argument(
        "--anchor_rule",
        action="store_true",
        help="Train against src.data.multitask.ANCHOR_RULE instead of the hand-written "
        "TASK_REGISTRY anchors. Off by default; the registry itself is never mutated.",
    )
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Scaffold-SPLIT seed. Fixes the train/val/test partition.",
    )
    p.add_argument(
        "--init_seed",
        type=int,
        default=0,
        help="Seeds weight init and DataLoader shuffle order, independently of --seed. "
        "Appended to the checkpoint directory when non-zero.",
    )
    p.add_argument("--resume", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
