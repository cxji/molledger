import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.utils.features import get_atom_feature_dims
from torch_geometric.nn import GINEConv
from torch_geometric.utils import scatter


def _trunk_dims_from_state_dict(sd):
    """Shared GINEConv-trunk shape inference for AdditiveGNN/PooledGNN checkpoints: node/edge/hidden
    dims and layer count from the trunk weights, desc_dim from the optional descriptor encoder."""
    n_layers = max(int(k.split(".")[1]) for k in sd if k.startswith("convs.")) + 1
    return dict(
        node_in_dim=sd["node_emb.weight"].shape[1],
        edge_in_dim=sd["edge_emb.weight"].shape[1],
        hidden_dim=sd["node_emb.weight"].shape[0],
        num_layers=n_layers,
        desc_dim=sd["desc_encoder.0.weight"].shape[1] if "desc_encoder.0.weight" in sd else 0,
    )


class AdditiveGNN(nn.Module):
    """
    GINEConv message-passing network with a per-atom score head and a global context vector.

    Molecular prediction for task k: y_hat_k = sum_i s_{i,k},
    """

    def __init__(
        self,
        node_in_dim: int = 9,
        edge_in_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_tasks: int = 1,
        dropout: float = 0.0,
        desc_dim: int = 0,  # accepted for signature parity; unused (additive head has no descriptor path)
        readout: str = "sum",
        global_context: bool = False,
        context_dim: int = 0,
    ):
        super().__init__()
        assert readout in ("sum", "summean"), readout
        self.num_tasks = num_tasks
        self.desc_dim = desc_dim
        self.readout = readout

        self.node_emb = nn.Linear(node_in_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_in_dim, hidden_dim)

        self.convs = nn.ModuleList(
            [
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.BatchNorm1d(hidden_dim * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                    ),
                    edge_dim=hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # Per-atom score head: y_hat_k = sum_i (W h_i)_k.
        head_in = hidden_dim

        def _make_head():
            return nn.Linear(head_in, num_tasks)

        # Global-context head: context_mlp reads the pooled [sum;mean] trunk vector, emits a
        # per-molecule context g broadcast to every atom:
        #     g = context_mlp([sum_i h_i ; mean_i h_i])
        #     s_i = score_head([h_i ; g]),   y_hat_k = sum_i s_{i,k}
        # g is a per-molecule constant, so sum_i s_i = pred still holds exactly. Plain additive
        # readout only (asserted: no summean).
        self.global_context = global_context
        self.context_dim = context_dim
        if global_context:
            assert context_dim > 0, "global_context requires context_dim > 0"
            assert readout == "sum", (
                "global_context wires into the plain additive readout only (no summean)"
            )
            self.context_mlp = nn.Sequential(
                nn.Linear(2 * head_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, context_dim),
            )
            self.score_head = nn.Sequential(
                nn.Linear(head_in + context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_tasks),
            )
        else:
            self.context_mlp = None
            self.score_head = _make_head()  # extensive (sum) branch
        # Intensive (mean) branch: with readout="summean",
        #     s_{i,k} = head_sum(h_i)_k + head_mean(h_i)_k / N   (N = #atoms)
        # 1/N is a per-molecule constant, so y_hat_k = sum_i s_{i,k} still holds exactly.
        self.mean_head = _make_head() if readout == "summean" else None

    def forward(self, x, edge_index, edge_attr, batch, desc=None):
        """
        Args:
            x           [N, node_in_dim]  atom features
            edge_index  [2, E]
            edge_attr   [E, edge_in_dim]  bond features
            batch       [N]               batch assignment
            desc        unused (accepted for call-signature parity with PooledGNN/GNANModel)

        Returns:
            pred    [B, num_tasks]   molecular predictions (sum of atom scores)
            scores  [N, num_tasks]   per-atom scores
        """
        feat = self._trunk(x, edge_index, edge_attr, batch, desc)
        scores = self._unary_scores(feat, batch)  # [N, num_tasks]  per-atom
        pred = scatter(scores, batch, dim=0, reduce="sum")  # [B, num_tasks]
        return pred, scores

    def _unary_scores(self, feat, batch):
        """Per-atom score s_i = head_sum(feat_i) [+ head_mean(feat_i)/N_i under readout=summean].
        N_i = #atoms in the molecule; sum_i s_i = pred holds exactly. With global_context, the score
        head instead reads [feat_i ; g], g = context_mlp([sum feat ; mean feat]) (see __init__)."""
        if self.context_mlp is not None:
            g_sum = scatter(feat, batch, dim=0, reduce="sum")  # [B, head_in]
            g_mean = scatter(feat, batch, dim=0, reduce="mean")  # [B, head_in]
            g = self.context_mlp(torch.cat([g_sum, g_mean], dim=-1))  # [B, context_dim]
            return self.score_head(torch.cat([feat, g[batch]], dim=-1))  # [N, num_tasks]
        s = self.score_head(feat)
        if self.mean_head is not None:
            ones = torch.ones(feat.size(0), 1, device=feat.device, dtype=feat.dtype)
            n_per_graph = scatter(ones, batch, dim=0, reduce="sum")  # [B, 1]
            inv_n = (1.0 / n_per_graph.clamp_min(1.0))[batch]  # [N, 1]
            s = s + self.mean_head(feat) * inv_n
        return s

    def _trunk(self, x, edge_index, edge_attr, batch, desc=None):
        """Shared message-passing trunk. Returns the final per-atom embedding [N, hidden] the score
        head reads. `desc` accepted for call-signature parity; unused."""
        x = self.node_emb(x.float())
        edge_attr = self.edge_emb(edge_attr.float())

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return x

    @classmethod
    def from_checkpoint(cls, ck):
        """Reconstruct from a training checkpoint's config + weight shapes. global_context/readout
        are read from checkpoint metadata, falling back to weight presence (context_mlp./mean_head.)
        if the flag does not exist."""
        sd = ck["model"] if "model" in ck else ck
        global_context = ck.get(
            "additive_global_context", any(k.startswith("context_mlp.") for k in sd)
        )
        context_dim = ck.get("additive_context_dim", 0) or (
            sd["context_mlp.3.weight"].shape[0] if "context_mlp.3.weight" in sd else 0
        )
        readout = ck.get("additive_readout") or (
            "summean" if any(k.startswith("mean_head.") for k in sd) else "sum"
        )
        num_tasks = (
            sd["score_head.3.weight"].shape[0]
            if "score_head.3.weight" in sd
            else sd["score_head.weight"].shape[0]
        )
        model = cls(
            num_tasks=num_tasks,
            readout=readout,
            global_context=global_context,
            context_dim=context_dim,
            **_trunk_dims_from_state_dict(sd),
        )
        model.load_state_dict(sd)
        return model


class PooledGNN(nn.Module):
    """
    Non-additive counterpart to AdditiveGNN: same GINEConv backbone, graph-level pooled MLP head
    instead of the per-atom-score-sum readout.

        AdditiveGNN:  y_hat_k = sum_i s_{i,k}                     (linear in independent atom scores)
        PooledGNN:    y_hat   = MLP( [mean_i h_i ; sum_i h_i] )   (nonlinear graph-level head)

    No per-atom scores, so no Crippen/TPSA anchor term.

    forward returns (pred, None): drop-in for the (pred, scores) convention the training loop and
    masked_multitask_loss use, which skips the anchor term when scores is None.
    """

    def __init__(
        self,
        node_in_dim: int = 9,
        edge_in_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_tasks: int = 1,
        dropout: float = 0.0,
        desc_dim: int = 0,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.desc_dim = desc_dim

        self.node_emb = nn.Linear(node_in_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_in_dim, hidden_dim)
        # Descriptors, if present, are concatenated to the pooled graph vector (see
        # src/data/descriptors.py) -- not available to AdditiveGNN (would break y_hat_k = sum_i s_{i,k}).
        self.desc_encoder = (
            nn.Sequential(
                nn.Linear(desc_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if desc_dim
            else None
        )

        self.convs = nn.ModuleList(
            [
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.BatchNorm1d(hidden_dim * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                    ),
                    edge_dim=hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # Graph head reads [mean_i h_i ; sum_i h_i] (sum = N*mean, but the head weights them
        # independently: (W_m + N*W_s)*mean); + hidden dim when descriptors are present.
        head_in = hidden_dim * 2 + (hidden_dim if self.desc_encoder is not None else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, x, edge_index, edge_attr, batch, desc=None):
        """Returns (pred [B, num_tasks], None). No per-atom scores by construction."""
        x = self.node_emb(x.float())
        edge_attr = self.edge_emb(edge_attr.float())

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        def _pool(h):
            return torch.cat(
                [scatter(h, batch, dim=0, reduce="mean"), scatter(h, batch, dim=0, reduce="sum")],
                dim=-1,
            )

        g = _pool(x)
        if self.desc_encoder is not None:
            # One encoded descriptor vector per GRAPH -- no broadcast, no re-pooling.
            g = torch.cat([g, self.desc_encoder(desc)], dim=-1)  # [B, 3*hidden]
        pred = self.head(g)  # [B, num_tasks]
        return pred, None

    @classmethod
    def from_checkpoint(cls, ck):
        """Reconstruct from a training checkpoint's config + weight shapes."""
        sd = ck["model"] if "model" in ck else ck
        num_tasks = sd["head.3.weight"].shape[0]
        model = cls(num_tasks=num_tasks, **_trunk_dims_from_state_dict(sd))
        head_in = sd["head.0.weight"].shape[1]
        if model.head[0].in_features != head_in:
            raise ValueError(
                f"reconstructed pooled head expects {model.head[0].in_features} "
                f"inputs but the checkpoint has {head_in}"
            )
        model.load_state_dict(sd)
        return model


class GNANModel(nn.Module):
    """
    Graph Neural Additive Network (Bechler-Speicher et al., NeurIPS 2024, arXiv:2406.01317),
    reimplemented on our 9-dim OGB atom features and 11-task registry. Matches the reference repo's
    GNAN class (github.com/mayabechlerspeicher/Graph-Neural-Additive-Networks---GNAN, GNAN.py) in
    its default configuration (normalize_rho=True); verified against a literal port in
    scripts/tests/test_gnan_reference.py.

    A GAM over node features (no feature interactions) with graph structure entering only through a
    learned distance decay rho. Reference per-node prediction:

        pred_v = sum_u  rho(dist(v,u)) / c_v(dist(v,u))  *  g(x_u)     [g(x_u) = sum_k f_k(x_u[k])]

    dist = shortest-path hop count; with normalize_rho, c_v(h) = #nodes at distance h from v. Graph
    prediction sums over query nodes v. Regrouping onto the feature-bearing atom u gives per-atom
    terms:

        pred_t = sum_u w_{u,t} * g_{u,t},   w_{u,t} = sum_v rho_t(dist(v,u)) / c_v(dist(v,u))

    c_{u,t} = w_{u,t}*g_{u,t} is an exact per-atom contribution, so forward returns (pred, c) -- the
    same (pred, scores) convention as AdditiveGNN, so score_molecules, --method additive,
    matched_pair_decomposition, and the anchor faithfulness all apply unchanged.

    Adaptations from the reference:
      * Feature functions are per-category Embeddings, not MLPs on a scalar (all 9 OGB atom features
        are categorical integer codes: atomic num, chirality, degree, formal charge, num-H, radical
        electrons, hybridization, is-aromatic, is-in-ring; cardinalities from get_atom_feature_dims).
      * rho is per-task; rho_per_task=False recovers the shared-rho reference default.
      * No message passing, no edge features, no descriptors. edge_attr/desc accepted for call-
        signature parity and ignored.
    """

    def __init__(
        self,
        node_in_dim: int = 9,
        edge_in_dim: int = 3,  # accepted for signature parity; unused (GNAN has no edge features)
        num_tasks: int = 1,
        rho_hidden: int = 32,
        rho_layers: int = 2,
        normalize_rho: bool = True,
        rho_per_task: bool = True,
        hidden_dim: int = 128,  # accepted for signature parity; unused
        desc_dim: int = 0,  # accepted for signature parity; unused
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.normalize_rho = normalize_rho
        self.rho_per_task = rho_per_task
        self.desc_dim = desc_dim

        # Per-feature shape functions f_{k,t}: g_{u,t} = sum_k f_k(x_u[k]) (see class docstring).
        # Small non-zero init: pred is bilinear in (rho, g); zero g would zero rho's gradient too.
        dims = get_atom_feature_dims()[:node_in_dim]
        self.shape_fns = nn.ModuleList([nn.Embedding(c, num_tasks) for c in dims])
        for emb in self.shape_fns:
            nn.init.normal_(emb.weight, std=0.1)

        # rho: MLP on the distance transform 1/(hop+1) (hop 0 -> 1.0, farther -> smaller). Evaluated
        # on the integer hops present per batch -> [H+1, T] table; no fixed hop cap.
        rho_out = num_tasks if rho_per_task else 1
        layers, prev = [], 1
        for _ in range(max(rho_layers - 1, 0)):
            layers += [nn.Linear(prev, rho_hidden), nn.ReLU()]
            prev = rho_hidden
        layers += [nn.Linear(prev, rho_out)]
        self.rho = nn.Sequential(*layers)

    @torch.no_grad()
    def _distance_shells(self, edge_index, num_nodes, device):
        """Sparse shortest-path shells. Returns (shells, hist):
          shells[h] : sparse bool [N, N], shells[h][v,u] = 1 iff hops(v,u) == h  (h = 0..H)
          hist      : dense [N, H+1], hist[v, h] = #nodes at distance h from v = rowsum(shells[h])
        Boolean reachability expansion on the batched block-diagonal adjacency (stays sparse, no
        dense [N, N]). Expands to the full diameter (H = last non-empty shell)."""
        n = num_nodes
        r, c = edge_index
        self_i = torch.arange(n, device=device)
        ri = torch.cat([r, c, self_i])
        ci = torch.cat([c, r, self_i])  # symmetric + self loops
        A = torch.sparse_coo_tensor(
            torch.stack([ri, ci]), torch.ones(ri.numel(), device=device), (n, n)
        ).coalesce()

        def _bool(sp):
            return torch.sparse_coo_tensor(
                sp.indices(), (sp.values() > 0).float(), sp.shape
            ).coalesce()

        eye = torch.sparse_coo_tensor(
            torch.stack([self_i, self_i]), torch.ones(n, device=device), (n, n)
        ).coalesce()
        shells = [eye]  # shell 0 = identity (self, hop 0)
        hist = [torch.ones(n, device=device)]
        cum = eye
        h = 0
        while True:
            reached = _bool(torch.sparse.mm(A, cum))  # reached in <= h+1 hops
            layer = _bool((reached - cum).coalesce())  # exactly h+1 (reached includes cum)
            cnt = torch.sparse.sum(layer, dim=1).to_dense()
            if float(cnt.sum()) == 0.0:
                break  # no farther nodes -> full diameter
            shells.append(layer)
            hist.append(cnt)
            cum = reached
            h += 1
        return shells, torch.stack(hist, dim=1)  # [N, H+1]

    def forward(self, x, edge_index, edge_attr, batch, desc=None):
        """Returns (pred [B, num_tasks], scores [N, num_tasks]) with pred = sum_i scores_i exactly."""
        device = x.device
        g = sum(
            emb(x.long()[:, k]) for k, emb in enumerate(self.shape_fns)
        )  # [N, T]  sum_k f_k(x[k])
        shells, hist = self._distance_shells(edge_index, x.size(0), device)  # H+1 shells, [N,H+1]
        H = len(shells) - 1

        # rho table over integer hops present: rho_table[h] = rho(1/(h+1)).  [H+1, T] (or [H+1,1]).
        hops = torch.arange(H + 1, device=device, dtype=torch.float32)
        rho_table = self.rho((1.0 / (hops + 1.0)).view(-1, 1))  # [H+1, rho_out]
        if not self.rho_per_task:
            rho_table = rho_table.expand(-1, self.num_tasks)

        # w_u = sum_h rho[h] * sum_v shells[h][v,u] * n_v(h),  n_v(h) = 1/hist_v[h] (normalize) or 1.
        # shells symmetric, so sum_v shells[h][v,u]*n_v = (shells[h] @ n_h)_u; exact node-collapse of
        # GNAN's per-query-node sum, so pred = sum_u w_u*g_u holds exactly.
        w = torch.zeros(x.size(0), self.num_tasks, device=device)
        for h in range(H + 1):
            if self.normalize_rho:
                inv = torch.where(
                    hist[:, h] > 0, 1.0 / hist[:, h].clamp_min(1.0), torch.zeros_like(hist[:, h])
                )
                col = torch.sparse.mm(shells[h], inv.view(-1, 1)).squeeze(1)  # [N]
            else:
                col = hist[:, h]  # sum_v shells[h][v,u] = hist_u[h]
            w = w + rho_table[h].unsqueeze(0) * col.unsqueeze(1)  # [N, T]

        scores = w * g  # [N, T] exact per-atom
        pred = scatter(scores, batch, dim=0, reduce="sum")  # [B, T]
        return pred, scores

    @classmethod
    def from_checkpoint(cls, ck):
        """Reconstruct from a training checkpoint's config + weight shapes. normalize_rho is
        forward-time config with no weights, so it comes from checkpoint metadata."""
        sd = ck["model"] if "model" in ck else ck
        model = cls(
            **gnan_kwargs_from_state_dict(sd), normalize_rho=ck.get("gnan_normalize_rho", True)
        )
        model.load_state_dict(sd)
        return model


def gnan_kwargs_from_state_dict(sd):
    """Infer the GNANModel weight-shape constructor kwargs (node_in_dim, num_tasks, rho_hidden,
    rho_layers, rho_per_task) from a saved state_dict."""
    node_in_dim = sum(1 for k in sd if k.startswith("shape_fns.") and k.endswith(".weight"))
    num_tasks = sd["shape_fns.0.weight"].shape[1]  # Embedding weight [C_0, T]
    rho_lin = sorted(
        int(k.split(".")[1]) for k in sd if k.startswith("rho.") and k.endswith(".weight")
    )
    rho_out = sd[f"rho.{rho_lin[-1]}.weight"].shape[0]
    rho_hidden = sd[f"rho.{rho_lin[0]}.weight"].shape[0] if len(rho_lin) > 1 else 32
    return dict(
        node_in_dim=node_in_dim,
        num_tasks=num_tasks,
        rho_hidden=rho_hidden,
        rho_layers=len(rho_lin),
        rho_per_task=(rho_out == num_tasks),
    )


def build_model_from_checkpoint(ck):
    """Reconstruct whichever architecture (AdditiveGNN / PooledGNN / GNANModel / LigandFormerGNN) a
    checkpoint was trained with, dispatching on its state-dict key prefixes. Returns (model,
    backbone); backbone is "gin" for every architecture here (kept for call-signature parity with
    the training loop's forward_model(model, batch, backbone))."""
    from src.models.ligandformer import LigandFormerGNN

    sd = ck["model"] if "model" in ck else ck
    if any(k.startswith("shape_fns.") for k in sd):
        return GNANModel.from_checkpoint(ck), "gin"
    if any(k.startswith("spatial_atts.") for k in sd):
        return LigandFormerGNN.from_checkpoint(ck), "gin"
    if "score_head.weight" in sd or "score_head.0.weight" in sd:
        return AdditiveGNN.from_checkpoint(ck), "gin"
    return PooledGNN.from_checkpoint(ck), "gin"
