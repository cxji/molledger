#!/usr/bin/env python
"""Verify GNANModel against a LITERAL port of the reference repo's GNAN.forward
(github.com/mayabechlerspeicher/Graph-Neural-Additive-Networks---GNAN, GNAN.py + pre_process_datasets.py).

The reference forward, per query node v (their code):
    fx[:,k]        = fs[k](x[:,k].view(-1,1))          # per-feature shape function
    f_sums         = fx.sum(dim=1)                      # g(x_u) = Σ_k f_k(x_u[k])
    rho_dist[v]    = rho(node_distances[v].view(-1,1))  # node_distances = 1/(hops+1)  (preprocessing)
    if normalize_rho: rho_dist[v] /= normalization_matrix[v]   # = #nodes at that shortest-path dist
    pred_for_node[v] = Σ_u rho_dist[v,u] * f_sums[u]
Graph prediction = Σ_v pred_for_node[v]; per-atom contribution = Σ_v rho_dist[v,u] * f_sums[u].

We run the reference with the SAME learnable modules (model.shape_fns, model.rho) as GNANModel, so any
mismatch is in the aggregation/normalisation, which is the only thing GNANModel reorganises (sparse
distance shells instead of the per-node Python loop). Tests normalize_rho in {True, False}.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ogb.utils.features import get_atom_feature_dims  # noqa: E402

from src.models.gnn import GNANModel  # noqa: E402


def reference_scores(model, x, edge_index, n):
    """Literal port of GNAN.forward, node-summed to a graph prediction; returns (per_atom_c, pred)."""
    T = model.num_tasks
    g = sum(emb(x.long()[:, k]) for k, emb in enumerate(model.shape_fns))  # [N, T]

    adj = np.zeros((n, n))
    r, c = edge_index.numpy()
    adj[r, c] = 1.0
    adj[c, r] = 1.0
    hops = shortest_path(adj, method="D", unweighted=True)  # [N, N], inf if apart
    node_distances = 1.0 / (hops + 1.0)  # preprocessing transform
    node_distances[np.isinf(hops)] = 0.0
    normalization = np.zeros((n, n))  # #nodes at that distance
    for v in range(n):
        vals, counts = np.unique(hops[v], return_counts=True)
        cmap = dict(zip(vals.tolist(), counts.tolist()))
        for u in range(n):
            normalization[v, u] = cmap[hops[v, u]]

    nd = torch.tensor(node_distances, dtype=torch.float32)
    rho_dist = model.rho(nd.view(-1, 1)).reshape(n, n, -1)  # [N, N, rho_out]
    if not model.rho_per_task:
        rho_dist = rho_dist.expand(n, n, T)
    if model.normalize_rho:
        rho_dist = rho_dist / torch.tensor(normalization, dtype=torch.float32).view(n, n, 1)

    w = rho_dist.sum(dim=0)  # Σ_v  -> [N, T]
    c = w * g
    return c, c.sum(0)


def random_connected_graph(n, extra_edges, seed):
    g = torch.Generator().manual_seed(seed)
    # spanning path guarantees connectivity, then a few random chords
    a = torch.arange(n - 1)
    b = torch.arange(1, n)
    for _ in range(extra_edges):
        i, j = torch.randint(0, n, (2,), generator=g)
        if i != j:
            a = torch.cat([a, i.view(1)])
            b = torch.cat([b, j.view(1)])
    ei = torch.stack([torch.cat([a, b]), torch.cat([b, a])])
    dims = get_atom_feature_dims()[:9]
    x = torch.stack([torch.randint(0, d, (n,), generator=g) for d in dims], dim=1)
    return x, ei


def main():
    torch.manual_seed(0)
    ok = True
    for normalize in (True, False):
        for per_task in (True, False):
            for n, extra, seed in [(6, 0, 1), (9, 3, 2), (13, 5, 3), (20, 8, 4)]:
                x, ei = random_connected_graph(n, extra, seed)
                m = GNANModel(
                    node_in_dim=9,
                    num_tasks=4,
                    rho_hidden=16,
                    rho_layers=2,
                    normalize_rho=normalize,
                    rho_per_task=per_task,
                ).eval()
                batch = torch.zeros(n, dtype=torch.long)
                with torch.no_grad():
                    pred, scores = m(x, ei, None, batch)
                    c_ref, pred_ref = reference_scores(m, x, ei, n)
                ds = (scores - c_ref).abs().max().item()
                dp = (pred[0] - pred_ref).abs().max().item()
                exact = (pred[0] - scores.sum(0)).abs().max().item()
                tag = f"norm={normalize} per_task={per_task} n={n}"
                if max(ds, dp) > 1e-5:
                    ok = False
                    print(f"  MISMATCH {tag}: max|score-ref|={ds:.2e} max|pred-ref|={dp:.2e}")
                else:
                    print(
                        f"  ok {tag}: score/pred match ref (<= {max(ds, dp):.1e}); exact-sum {exact:.1e}"
                    )
    print("ALL MATCH" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
