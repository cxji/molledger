import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


def _graph_mean(v: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    """Per-molecule mean of a per-atom vector, broadcast back to atoms."""
    return scatter(v, batch_index, dim=0, reduce="mean")[batch_index]


def _graph_rms(v: torch.Tensor, batch_index: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-molecule RMS of an already-centred per-atom vector, broadcast back to atoms.
    Clamps the mean-of-squares at eps**2 before the sqrt."""
    meansq = scatter(v**2, batch_index, dim=0, reduce="mean")[batch_index]
    return meansq.clamp(min=eps**2).sqrt()


def masked_multitask_loss(
    pred: torch.Tensor,
    scores: torch.Tensor | None,
    y: torch.Tensor,
    crippen: torch.Tensor,
    tpsa: torch.Tensor,
    task_specs: list,
    lambda_prop: float = 1.0,
    lambda_anchor: float = 0.1,
    label_scales: torch.Tensor | None = None,
    batch_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Masked multi-task loss over K ADME properties.

    Args:
        pred          [B, K] molecular predictions
        scores        [N, K] per-atom scores, or None for a non-additive model (e.g. PooledGNN)
                      with no per-atom scores -- the anchor term is then skipped and only the
                      masked property term contributes
        y             [B, K] molecular labels, NaN where a task is unobserved for that molecule
        crippen       [N] per-atom Crippen contributions (lipophilicity anchor)
        tpsa          [N] per-atom TPSA contributions (polarity anchor)
        task_specs    list of src.data.multitask.TaskSpec, length K
        lambda_prop   weight for the property prediction term
        lambda_anchor weight for the Crippen/TPSA anchoring term
        label_scales  [K] per-task label std (train-split only, from
                      src.data.multitask.compute_label_scales). Continuous tasks' L_prop is
                      divided by scale^2 so tasks of different label magnitude (e.g. CLint
                      ~1000s vs. logD ~1) contribute comparably to the summed loss. None = no
                      correction (all scales 1.0).
        batch_index   [N] graph assignment; required (centring/scaling is per molecule)

    Anchor term: supervises (s_i - mean_j s_j) against (anchor_i - mean_j anchor_j), each side
    divided by its per-molecule std.

    Returns:
        total loss, dict of per-task loss components for logging
    """
    anchor_src = scores
    if anchor_src is not None and batch_index is None:
        raise ValueError(
            "the anchor term needs batch_index (centring is per molecule, not per batch)"
        )
    if label_scales is None:
        label_scales = torch.ones(len(task_specs), device=pred.device)

    total = torch.zeros((), device=pred.device)
    metrics = {}

    for k, spec in enumerate(task_specs):
        col_y = y[:, k]
        mask = ~torch.isnan(col_y)
        if mask.sum() == 0:
            # No labeled examples for task k in this batch -- omit the term entirely.
            metrics[f"loss/{spec.name}/prop"] = float("nan")
            continue

        if spec.label_kind == "binary":
            # Already scale-free (logits/probabilities), no label_scales correction needed.
            L_prop_k = F.binary_cross_entropy_with_logits(pred[mask, k], col_y[mask])
        else:
            L_prop_k = F.mse_loss(pred[mask, k], col_y[mask]) / (label_scales[k] ** 2)
        metrics[f"loss/{spec.name}/prop"] = L_prop_k.item()
        total = total + lambda_prop * L_prop_k

        if anchor_src is not None and spec.anchor != "none":
            anchor_tensor = crippen if spec.anchor == "crippen" else tpsa
            # anchor_sign is the DIRECTION of the claim, applied before any centring or scaling.
            s_k, a_k = anchor_src[:, k], spec.anchor_sign * anchor_tensor

            # Centre both sides per molecule, then divide by their per-molecule std
            s_k = s_k - _graph_mean(s_k, batch_index)
            a_k = a_k - _graph_mean(a_k, batch_index)
            s_k = s_k / _graph_rms(s_k, batch_index)
            a_k = a_k / _graph_rms(a_k, batch_index)

            L_anchor_k = F.mse_loss(s_k, a_k)
            metrics[f"loss/{spec.name}/anchor_{spec.anchor}"] = L_anchor_k.item()
            total = total + lambda_anchor * L_anchor_k

    metrics["loss/total"] = total.item()
    return total, metrics
