"""
Per-atom attribution harness for the deterministic GNN regressors.

Every method here maps (model, molecules) -> a list of [N_atoms_i, K] attribution matrices, one
per molecule, so they are directly comparable on one axis. The primary methods:

    additive              scores emitted directly by an additive head. Free, exact-sum.
    integrated_gradients  path integral of dy/dh_i from a baseline embedding. Completeness axiom.
    wisp                  WISP's atom attributor (Janssen et al., Digital Discovery 2026): occlusion
                          by ELEMENT SUBSTITUTION -- mutate atom i to each organic element, re-featurise,
                          average the prediction drop (their eqn 8). Needs batch.smiles; off-manifold,
                          NOT exact-sum. Lets us run WISP through this same harness (leakage + gap).

The gradient methods attribute with respect to the post-embedding node representation, not the raw
atom features.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

__all__ = [
    "attribute",
    "ATTRIBUTION_METHODS",
    "completeness",
    "faithfulness",
    "matched_pair_decomposition",
    "score_molecules",
]


# --------------------------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------------------------


def _forward(model, batch, backbone):
    """Uniform call across the two backbone signatures (mirrors train_molledger)."""
    desc = getattr(batch, "desc", None)
    if backbone == "gin":
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, desc)
    return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pos, desc)


class _EmbeddingOverride:
    """
    Forward hook on `model.node_emb` with two modes:
      capture  -- record node_emb's output, change nothing (`self.value` is None).
      override -- return `self.value` in place of node_emb's output.
    """

    def __init__(self, model):
        self.model = model
        self.value = None
        self.captured = None
        self._handle = None

    def _hook(self, module, inputs, output):
        self.captured = output
        return self.value if self.value is not None else output

    def __enter__(self):
        self._handle = self.model.node_emb.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        return False


def _split_by_graph(t, batch_vec, num_graphs):
    """[N, K] + batch assignment -> list of per-molecule [N_i, K]."""
    return [t[batch_vec == g] for g in range(num_graphs)]


# --------------------------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------------------------


def _additive(model, batch, backbone, task_idx=None, **kw):
    pred, scores = _forward(model, batch, backbone)
    if scores is None:
        raise ValueError(
            "method='additive' requires an additive head; this model returned "
            "scores=None. Use integrated_gradients or wisp for a pooled head."
        )
    return scores.detach(), pred.detach(), {}


def _path_gradients(model, batch, backbone, baseline_h, steps, task_idx=None):
    """
    Core of IG: gradient of the prediction w.r.t. the node representation, evaluated at `steps`
    points equally spaced from `baseline_h` to h (midpoint rule, alpha = (m + 0.5)/steps), averaged
    and scaled by (h - baseline_h). Returns [N, K], columns outside `task_idx` left NaN.
    """
    ks = list(range(model.num_tasks)) if task_idx is None else list(task_idx)
    with _EmbeddingOverride(model) as ov:
        with torch.no_grad():
            _forward(model, batch, backbone)
        h_real = ov.captured.detach()

        delta = h_real - baseline_h
        total = torch.zeros(h_real.size(0), h_real.size(1), len(ks), device=h_real.device)

        for m in range(steps):
            alpha = (m + 0.5) / steps
            h_a = (baseline_h + alpha * delta).detach().requires_grad_(True)
            ov.value = h_a
            pred, _ = _forward(model, batch, backbone)
            for j, k in enumerate(ks):
                (g,) = torch.autograd.grad(pred[:, k].sum(), h_a, retain_graph=(j + 1 < len(ks)))
                total[:, :, j] += g.detach()
            ov.value = None

    partial = (delta.unsqueeze(-1) * (total / steps)).sum(dim=1)  # [N, len(ks)]
    out = torch.full((h_real.size(0), model.num_tasks), float("nan"), device=partial.device)
    out[:, ks] = partial
    return out


def _baseline_pred(model, batch, backbone, base_h):
    """Prediction when every atom is replaced by the baseline representation."""
    with _EmbeddingOverride(model) as ov:
        ov.value = base_h
        with torch.no_grad():
            pred_base, _ = _forward(model, batch, backbone)
    return pred_base.detach()


def _integrated_gradients(model, batch, backbone, steps=256, baseline="mean", task_idx=None, **kw):
    """
    IG with baseline "mean" (mean atom embedding over the batch), "zeros", or a supplied tensor.
    aux carries `pred_base` = y_hat(baseline); IG's completeness target is y_hat - pred_base.
    """
    with _EmbeddingOverride(model) as ov:
        with torch.no_grad():
            _forward(model, batch, backbone)
        h_real = ov.captured.detach()

    if baseline == "mean":
        base = h_real.mean(dim=0, keepdim=True).expand_as(h_real)
    elif baseline == "zeros":
        base = torch.zeros_like(h_real)
    else:
        base = baseline.to(h_real).expand_as(h_real)

    attrs = _path_gradients(model, batch, backbone, base, steps, task_idx)
    with torch.no_grad():
        pred, _ = _forward(model, batch, backbone)
    return attrs, pred.detach(), {"pred_base": _baseline_pred(model, batch, backbone, base)}


#: Max mutants forwarded at once (splits only pathologically large molecules; normal ones never reach it).
_WISP_MUT_CHUNK = 4096


def _wisp_occlusion(model, batch, backbone, task_idx=None, wisp_mutants=None, **kw):
    """
    WISP atom attributor (Janssen et al., Digital Discovery 2026, eqn 8). Occlusion by element
    substitution: each heavy atom i is mutated, one element at a time, to every element in WISP's
    12-element alphabet except its own; each mutant is RDKit-sanitised and featurised, and the
    attribution is the mean prediction drop over the G valid mutants:

        attr_i = (1/G) * sum_h ( y_hat(original) - y_hat(mutant_{i,h}) )

    Mutant graphs come from `src.wisp_mutants.build_mutants`; pass a precomputed {smiles: packed} dict
    as `wisp_mutants`, or they are built on the fly. All of a molecule's mutants go through the model
    in one forward (chunked only past `_WISP_MUT_CHUNK`). Requires `batch.smiles`. NOT exact-sum.
    """
    from src.wisp_mutants import build_mutants

    assert batch.num_graphs == 1, (
        "_wisp_occlusion expects one molecule per call (attribute() enforces it)"
    )
    smi = getattr(batch, "smiles", None)
    if isinstance(smi, (list, tuple)):
        smi = smi[0] if smi else None
    if smi is None:
        raise ValueError(
            "method='wisp' needs batch.smiles set to the canonical SMILES each Data was "
            "featurised from (attach record.smiles before calling attribute())."
        )
    if backbone != "gin":
        raise ValueError(
            "method='wisp' supports the gin backbone only (mutants carry no 3D coords)."
        )

    device = batch.x.device
    N = batch.num_nodes
    with torch.no_grad():
        pred, _ = _forward(model, batch, backbone)  # [1, K] = y_hat(original)
    y_org = pred[0]
    attrs = torch.zeros(N, model.num_tasks, device=device)

    packed = (wisp_mutants or {}).get(smi)
    if packed is None:
        packed = build_mutants(smi)
    if packed is None or int(packed["n_atoms"]) != N or packed["mut_atom"].shape[0] == 0:
        # SMILES/graph atom counts disagree, or no valid mutant -> emit zeros rather than crash a sweep.
        return attrs, pred.detach(), {}

    x = torch.as_tensor(packed["x"]).long().to(device)
    edge_index = torch.as_tensor(packed["edge_index"]).long().to(device)
    edge_attr = torch.as_tensor(packed["edge_attr"]).long().to(device)
    node_batch = torch.as_tensor(packed["node_batch"]).long().to(device)
    mut_atom = torch.as_tensor(packed["mut_atom"]).long().to(device)
    M = int(mut_atom.numel())

    # Score every mutant. One forward normally (M <= _WISP_MUT_CHUNK); split only enormous molecules.
    # edge_index nodes are packed-global and node_batch is contiguous per mutant, so a mutant slice
    # [m0, m1) is re-based by subtracting its first node index (node_first[m0]) and mutant id m0.
    y_mut = torch.empty(M, model.num_tasks, device=device)
    if M <= _WISP_MUT_CHUNK:
        with torch.no_grad():
            y_mut, _ = model(x, edge_index, edge_attr, node_batch, None)
    else:
        node_first = torch.searchsorted(node_batch, torch.arange(M, device=device))
        edge_mut = node_batch[edge_index[0]]
        for m0 in range(0, M, _WISP_MUT_CHUNK):
            m1 = min(m0 + _WISP_MUT_CHUNK, M)
            nmask = (node_batch >= m0) & (node_batch < m1)
            emask = (edge_mut >= m0) & (edge_mut < m1)
            base = int(node_first[m0])
            with torch.no_grad():
                yk, _ = model(
                    x[nmask],
                    edge_index[:, emask] - base,
                    edge_attr[emask],
                    node_batch[nmask] - m0,
                    None,
                )
            y_mut[m0:m1] = yk

    drop = y_org.unsqueeze(0) - y_mut  # [M, K], WISP eqn-8 summands
    sums = torch.zeros(N, model.num_tasks, device=device).index_add_(0, mut_atom, drop)
    counts = torch.zeros(N, device=device).index_add_(
        0, mut_atom, torch.ones(M, device=device)
    )  # G per atom
    nz = counts > 0
    attrs[nz] = sums[nz] / counts[nz].unsqueeze(1)
    return attrs, pred.detach(), {}


def _grad_cam(model, batch, backbone, task_idx=None, **kw):
    """
    Signed node-level Grad-CAM (Pope et al. 2019) on the last message-passing layer.

    For task k:  A = last-conv node activations [N, H];
                 alpha_{g,c} = mean_{i in graph g} d(sum_i y_k)/d A_{i,c}   (per-graph GAP of grads);
                 score_i = sum_c alpha_{g(i),c} * A_{i,c}.

    Columns outside task_idx are left NaN. Returns the zero-activation prediction y_hat(A=0) as
    `pred_base`. NOT exact-sum.
    """
    ks = list(range(model.num_tasks)) if task_idx is None else list(task_idx)
    cap = {}
    handle = model.convs[-1].register_forward_hook(lambda m, i, o: cap.__setitem__("A", o))
    try:
        with torch.enable_grad():
            pred, _ = _forward(model, batch, backbone)
            A = cap["A"]  # [N, H]
            out = torch.full((A.size(0), model.num_tasks), float("nan"), device=A.device)
            for j, k in enumerate(ks):
                (g,) = torch.autograd.grad(pred[:, k].sum(), A, retain_graph=(j + 1 < len(ks)))
                alpha = scatter(g, batch.batch, dim=0, reduce="mean")  # [B, H] per-graph GAP
                cam = (alpha[batch.batch] * A).sum(dim=1)  # [N]
                out[:, k] = cam.detach()
    finally:
        handle.remove()

    # y_hat when the hooked layer emits 0 for every atom, returned as the completeness baseline.
    zh = model.convs[-1].register_forward_hook(lambda m, i, o: torch.zeros_like(o))
    try:
        with torch.no_grad():
            pred_base, _ = _forward(model, batch, backbone)  # [B, K]
    finally:
        zh.remove()
    return out, pred.detach(), {"pred_base": pred_base.detach()}


def _lime(
    model,
    batch,
    backbone,
    task_idx=None,
    num_samples=500,
    kernel_width=0.25,
    lime_alpha=1e-2,
    lime_seed=0,
    **kw,
):
    """
    LIME (Ribeiro et al. 2016) with atoms as the interpretable units.

    Draws `num_samples` binary masks z in {0,1}^N (each atom kept with prob 0.5; sample 0 is the full
    molecule), zeroes the absent atoms' input features, queries the model, weights each sample by
    pi = exp( -(removed/N)^2 / kernel_width^2 ), and fits a ridge surrogate y ~ w . z + b per task by
    weighted least squares. The per-atom coefficients w are the attribution; the intercept b is
    returned as `pred_base`. NOT exact-sum. Run at batch_size 1.
    """
    assert batch.num_graphs == 1, "_lime expects one molecule per call (attribute() enforces this)"
    ks = list(range(model.num_tasks)) if task_idx is None else list(task_idx)
    device = batch.x.device
    N = batch.num_nodes
    x0, ei, ea = batch.x, batch.edge_index, batch.edge_attr
    desc0, pos0 = getattr(batch, "desc", None), getattr(batch, "pos", None)

    gen = torch.Generator().manual_seed(int(lime_seed) + N)  # draw varies per molecule
    z = (torch.rand(num_samples, N, generator=gen) > 0.5).float()  # cpu [S, N]
    z[0] = 1.0  # always include full molecule
    zdev = z.to(device)

    x_big = (x0.float().unsqueeze(0) * zdev.unsqueeze(-1)).reshape(num_samples * N, x0.size(1))
    ei_big = torch.cat([ei + s * N for s in range(num_samples)], dim=1)
    ea_big = ea.repeat(num_samples, 1) if ea is not None else None
    batch_big = torch.arange(num_samples, device=device).repeat_interleave(N)
    desc_big = desc0.repeat(num_samples, 1) if desc0 is not None else None
    pos_big = pos0.repeat(num_samples, 1) if pos0 is not None else None

    with torch.no_grad():
        if backbone == "gin":
            preds, _ = model(x_big, ei_big, ea_big, batch_big, desc_big)
        else:
            preds, _ = model(x_big, ei_big, ea_big, batch_big, pos_big, desc_big)
    preds = preds.detach().cpu().numpy()  # [S, K]

    removed = (N - z.sum(1)).numpy()
    pi = np.exp(-((removed / max(N, 1)) ** 2) / (kernel_width**2))  # [S]
    Z = np.concatenate([z.numpy(), np.ones((num_samples, 1))], axis=1)  # [S, N+1] (intercept)
    G = Z.T @ (pi[:, None] * Z) + lime_alpha * np.eye(N + 1)
    G[-1, -1] -= lime_alpha  # leave the intercept unpenalized
    out = torch.full((N, model.num_tasks), float("nan"))
    intercept = torch.full(
        (1, model.num_tasks), float("nan")
    )  # [1, K] graph-level, like IG's pred_base
    for k in ks:
        coef = np.linalg.solve(G, Z.T @ (pi * preds[:, k]))
        out[:, k] = torch.from_numpy(coef[:N]).float()
        intercept[0, k] = float(coef[N])  # the surrogate's non-atom term b
    return (
        out.to(device),
        torch.from_numpy(preds[0:1]).float().to(device),
        {"pred_base": intercept.to(device)},
    )


def _attention(model, batch, backbone, task_idx=None, **kw):
    """
    LigandFormer per-atom attention attribution: attention received (a_i = mean over query rows of
    each block's [N,N] map, averaged over blocks; model.attention_received). Task-independent, so the
    same per-atom vector fills every requested task column. Magnitude-only (>= 0, sums to 1),
    unsigned, NOT exact-sum.
    """
    if not hasattr(model, "attention_received"):
        raise TypeError(
            "method 'attention' requires a LigandFormer model exposing attention_received()"
        )
    with torch.no_grad():
        pred, _ = _forward(model, batch, backbone)
        a = model.attention_received(
            batch.x, batch.edge_index, batch.batch
        )  # [N], task-independent
    ks = list(range(model.num_tasks)) if task_idx is None else list(task_idx)
    out = torch.full((a.size(0), model.num_tasks), float("nan"), device=a.device)
    for k in ks:
        out[:, k] = a
    return out, pred, {}


ATTRIBUTION_METHODS = {
    "additive": _additive,
    "integrated_gradients": _integrated_gradients,
    "wisp": _wisp_occlusion,
    "grad_cam": _grad_cam,
    "lime": _lime,
    "attention": _attention,
}


def attribute(
    model, data_list, method, backbone="gin", batch_size=32, device="cuda", task_idx=None, **kw
):
    """
    Returns (attrs, preds, aux):
        attrs  list of [N_i, K] per molecule; columns outside `task_idx` are NaN for the
               gradient methods, which only compute what was asked for
        preds  [n_molecules, K]
        aux    dict. "pred_base" [n_molecules, K] for IG -- its completeness target is
               y_hat - pred_base, not y_hat -- and for grad_cam/lime, whose pred_base is the non-atom
               residual (the zero-activation baseline / surrogate intercept), same role.

    task_idx: restrict the gradient methods to these task columns.
    """
    if method not in ATTRIBUTION_METHODS:
        raise KeyError(f"unknown method {method!r}; have {sorted(ATTRIBUTION_METHODS)}")
    fn = ATTRIBUTION_METHODS[method]
    model.eval()

    # lime fits a per-molecule surrogate, and wisp enumerates per-molecule element-substitution
    # mutants, so both need one molecule per batch; the rest can batch freely.
    loader = DataLoader(
        data_list, batch_size=1 if method in ("lime", "wisp") else batch_size, shuffle=False
    )
    all_attrs, all_preds, aux_chunks = [], [], {}
    for batch in loader:
        batch = batch.to(device)
        attrs, pred, aux = fn(model, batch, backbone, task_idx=task_idx, **kw)
        all_attrs.extend(_split_by_graph(attrs.cpu(), batch.batch.cpu(), batch.num_graphs))
        all_preds.append(pred.cpu())
        for key, val in (aux or {}).items():
            val = val.cpu()
            # graph-level ([B, K]) vs atom-level ([N, K]) aux; a lone graph is graph-level
            # (single-atom molecule makes num_graphs == num_nodes == 1)
            is_graph = val.size(0) == batch.num_graphs and (
                batch.num_graphs == 1 or val.size(0) != batch.num_nodes
            )
            aux_chunks.setdefault(key, []).append(
                val if is_graph else _split_by_graph(val, batch.batch.cpu(), batch.num_graphs)
            )

    out_aux = {}
    for key, chunks in aux_chunks.items():
        out_aux[key] = (
            torch.cat(chunks, dim=0)
            if isinstance(chunks[0], torch.Tensor)
            else [m for c in chunks for m in c]
        )
    return all_attrs, torch.cat(all_preds, dim=0), out_aux


# --------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------


def _pearson(a, b, eps=1e-9):
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=eps)
    return (a @ b / denom).item()


def faithfulness(attrs, data_list, task_idx, anchor_attr, min_atoms=3, abs_ref=False):
    """
    Per-molecule Pearson of the attribution against a per-atom chemical reference (`crippen` or
    `tpsa`), averaged over molecules. Returns (mean Pearson, n_usable). Molecules with fewer than
    `min_atoms` atoms or a constant anchor/column are skipped.

    `abs_ref` correlates against |reference| instead of the signed reference (for the unsigned
    attention attribution).
    """
    pear = []
    for a, d in zip(attrs, data_list):
        ref = getattr(d, anchor_attr)
        ref = ref.squeeze(-1) if ref.dim() > 1 else ref
        if abs_ref:
            ref = ref.abs()
        if a.size(0) < min_atoms or ref.std() < 1e-8:
            continue
        col = a[:, task_idx]
        if col.std() < 1e-8:
            continue
        pear.append(_pearson(col, ref))
    if not pear:
        return float("nan"), 0
    return (sum(pear) / len(pear), len(pear))


def completeness(attrs, preds, baseline_preds=None, task_idx=None):
    """
    Gap between the summed attribution and what it should reconstruct: the prediction, or
    prediction-minus-baseline when `baseline_preds` is given. Returns the mean absolute gap over
    molecules and the selected tasks.
    """
    target = preds if baseline_preds is None else preds - baseline_preds
    summed = torch.stack([a.sum(dim=0) for a in attrs], dim=0)
    if task_idx is not None:
        ks = list(task_idx)
        summed, target = summed[:, ks], target[:, ks]
    gap = (summed - target).abs()
    return gap.mean().item()


# --------------------------------------------------------------------------------------------
# matched-pair decomposition
# --------------------------------------------------------------------------------------------


def score_molecules(model, data_list, backbone="gin", batch_size=64, device="cuda"):
    """Per-atom scores for a list of molecules -> list of [N_i, K]. Additive heads only."""
    model = model.to(device).eval()
    out = []
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, scores = _forward(model, batch, backbone)
            if scores is None:
                raise ValueError(
                    "score_molecules requires an additive head (scores=None here). "
                    "The fragment decomposition is exact only for y_hat = sum_i s_i."
                )
            out.extend(
                s.cpu() for s in _split_by_graph(scores.detach(), batch.batch, batch.num_graphs)
            )
    return out


def _union_site(site, linker):
    """Merge `linker` into `site` (None | int | iterable of ints) without mutating either input."""
    if linker is None:
        return site
    if site is None:
        return [linker]
    if isinstance(site, (int, np.integer)):
        return [int(site), linker] if int(site) != linker else [int(site)]
    out = list(site)
    if linker not in out:
        out.append(linker)
    return out


def matched_pair_decomposition(
    scores_a, scores_b, site_a=None, site_b=None, atom_map=None, linker_a=None, linker_b=None
):
    """
    Exact fragment-level split of a matched-pair prediction delta, for an additive head (y_hat =
    sum_i s_i):

        dy      = (sum_i s^A_i) - (sum_i s^B_i)  =  d_sub + d_core
        d_sub   = s^A[site_a] - s^B[site_b]
        d_core  = (sum_i s^A_i - s^A[site_a]) - (sum_i s^B_i - s^B[site_b])
        leakage = |d_core| / (|d_core| + |d_sub|)

    `site` is an atom index, an iterable of indices (multi-atom substituent), or None (fragment
    absent, e.g. the H side of an H<->X pair). Only a partition into (core, substituent) is needed,
    so this covers N-changing transforms with no atom correspondence.

    atom_map (only when an atom bijection exists): amap[i_in_A] = i_in_B; per-atom core deltas
    s^A_i - s^B_amap[i] are returned as `per_atom`. None otherwise.

    linker_a / linker_b: the atom bonded to the substituent across the cut, folded into d_sub.
    Does not mutate site_a/site_b.

    Returns a dict of [K] tensors (plus `per_atom` [N_A, K] when atom_map is given).
    """
    sum_a, sum_b = scores_a.sum(0), scores_b.sum(0)
    zero = torch.zeros_like(sum_a)

    def _frag(scores, site):
        """Sum of scores over the substituent. `site` is None (absent) | int (one atom) | iterable
        of ints (multiple atoms)."""
        if site is None:
            return zero
        if isinstance(site, (int, np.integer)):
            return scores[site]
        idx = torch.as_tensor(list(site), dtype=torch.long)
        return scores[idx].sum(0) if idx.numel() else zero

    site_a_sub = _union_site(site_a, linker_a)
    site_b_sub = _union_site(site_b, linker_b)
    s_sub_a, s_sub_b = _frag(scores_a, site_a_sub), _frag(scores_b, site_b_sub)

    d_sub = s_sub_a - s_sub_b
    d_core = (sum_a - s_sub_a) - (sum_b - s_sub_b)
    dy = sum_a - sum_b

    out = {
        "dy": dy,
        "d_core": d_core,
        "d_sub": d_sub,
        "leakage": d_core.abs() / (d_core.abs() + d_sub.abs()).clamp_min(1e-12),
        "y_a": sum_a,
        "y_b": sum_b,
    }
    if atom_map is not None:
        idx = torch.as_tensor(atom_map, dtype=torch.long)
        out["per_atom"] = scores_a - scores_b[idx]
    return out




def graph_predictions(model, data_list, backbone="gin", batch_size=64, device="cuda"):
    """Plain `ŷ` per molecule -- works for pooled heads, which have no per-atom scores."""
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for batch in DataLoader(data_list, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            pred, _ = _forward(model, batch, backbone)
            out.append(pred.detach().cpu())
    return torch.cat(out, dim=0)
