"""LigandFormer (HAG-Net self-attention) baseline, faithfully reproduced from
github.com/GHDDI-AILab/LigandFormer (model.py) for the interpretability comparison.

Architecture is a near-verbatim copy of their `LigandFormer` / `SpatialAttentionConv` /
`SpatialAttentionBlockGIN` / `HAGConv` (module + attribute names preserved so a reference test can
transfer weights). Three deliberate adaptations:

  1. `HAGConv.aggregate` uses `torch_geometric.utils.scatter` (max+sum) instead of the upstream
     `torch_scatter.scatter` / `segment_csr` (torch_scatter is not installed here); numerically
     identical on standard `edge_index` inputs.
  2. Regression: `output_dim = num_tasks` (11), no sigmoid; the property MSE lives in the loss.
  3. Per-atom attribution = attention RECEIVED (mean over query rows of each block's [N,N] map,
     averaged over blocks) -- NOT their `atom_attention_scores` (`attention.mean(dim=1)` over a
     row-stochastic matrix, which is EXACTLY 1/N per atom, i.e. uniform/degenerate; verified). This
     is the standard, non-degenerate attention-importance. See `attention_received()`.

The readout is theirs: pyramid MEAN-pool (concat of every block's mean-pooled hidden) into a
nonlinear MLP head, so this is a POOLED-style model -- `forward` returns (pred, None) and there are
no exact per-atom contributions (attention weights are not prediction contributions: no
completeness). No edge features (message = neighbour features), no descriptors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import GELU, BatchNorm1d, LeakyReLU, Linear, ModuleList, Sequential
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import scatter


class SpatialAttentionConv(nn.Module):
    """Self-attention over atoms within each molecular graph (verbatim from upstream)."""

    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.attention_dim = 128

        self.query = Sequential(
            nn.GroupNorm(input_dim, input_dim, affine=True),
            nn.Conv1d(input_dim, self.attention_dim, kernel_size=1),
            LeakyReLU(0.1, inplace=True),
        )
        self.key = Sequential(
            nn.GroupNorm(input_dim, input_dim, affine=True),
            nn.Conv1d(input_dim, self.attention_dim, kernel_size=1),
            LeakyReLU(0.1, inplace=True),
        )
        self.val = Sequential(
            nn.GroupNorm(input_dim, input_dim, affine=True),
            nn.Conv1d(input_dim, output_dim, kernel_size=1),
            LeakyReLU(0.1, inplace=True),
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=num_heads,
            kdim=self.attention_dim,
            vdim=output_dim,
            batch_first=True,
        )
        self.output_proj = Linear(self.attention_dim, output_dim)

    def forward(
        self, x: Tensor, batch_index: Tensor, return_attention: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        query_v = self.query(x).squeeze(0).T
        key_v = self.key(x).squeeze(0).T
        val_v = self.val(x).squeeze(0).T

        graph_num = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        graph_counts = torch.bincount(batch_index, minlength=graph_num).tolist()

        reweighted, attention_blocks, start = [], [], 0
        for graph_count in graph_counts:
            end = start + graph_count
            graph_output, graph_attn = self.multihead_attn(
                query_v[start:end].unsqueeze(0),
                key_v[start:end].unsqueeze(0),
                val_v[start:end].unsqueeze(0),
                need_weights=return_attention,
                average_attn_weights=False,
            )
            reweighted.append(self.output_proj(graph_output.squeeze(0)))
            if return_attention:
                attention_blocks.append(
                    graph_attn.mean(dim=1).squeeze(0)
                )  # mean over heads -> [n,n]
            start = end

        reweighted_v = torch.cat(reweighted, dim=0)
        if not return_attention:
            return reweighted_v, None
        n = reweighted_v.shape[0]
        attention = reweighted_v.new_zeros((n, n))  # block-diagonal [total_N, total_N]
        start = 0
        for block in attention_blocks:
            end = start + block.shape[0]
            attention[start:end, start:end] = block
            start = end
        return reweighted_v, attention


class SpatialAttentionBlockGIN(nn.Module):
    """Bottleneck projection + atom self-attention for one block (verbatim)."""

    def __init__(self, input_dim: int, output_dim: int, viz_att: bool = False, num_heads: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.viz_att = viz_att
        self.bottle_neck = Sequential(
            nn.GroupNorm(input_dim, input_dim, affine=True),
            nn.Conv1d(input_dim, output_dim, kernel_size=1),
            LeakyReLU(0.1, inplace=True),
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=output_dim)
        self.leak_relu = LeakyReLU(0.1, inplace=True)
        self.spatial_attention = SpatialAttentionConv(output_dim, output_dim, num_heads=num_heads)

    def forward(self, x: Tensor, batch_index: Tensor, return_attention: bool = False):
        x = x.T.unsqueeze(0)
        x = self.bottle_neck(x)
        reweighted_x, attention = self.spatial_attention(
            x, batch_index, return_attention=return_attention or self.viz_att
        )
        x = reweighted_x.T.unsqueeze(0)
        x = self.leak_relu(x).squeeze(0).T
        x = self.layer_norm(x)
        if return_attention or self.viz_att:
            return x, attention
        return x


class HAGConv(MessagePassing):
    """GIN-like conv with max+sum aggregation (upstream; aggregate ported to tg scatter)."""

    def __init__(
        self,
        nn_module,
        aggregation_methods=None,
        multiple_aggregation_merge_method: str = "sum",
        node_feature_update_method: str = "cat",
    ):
        super().__init__()
        self.nn = nn_module
        self.aggregation_methods = aggregation_methods or ["max", "sum"]
        self.multiple_aggregation_merge_method = multiple_aggregation_merge_method
        self.node_feature_update_method = node_feature_update_method
        self.reset_parameters()

    def reset_parameters(self) -> None:
        reset(self.nn)

    def forward(self, x, edge_index, size=None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, size=size)
        x_r = x[1]
        if self.node_feature_update_method == "cat":
            out = torch.cat([x_r, out], dim=-1)
        else:
            raise ValueError(f"Unsupported update method: {self.node_feature_update_method}")
        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor, ptr=None, dim_size=None) -> Tensor:
        # tg scatter for each method, then sum-merge (upstream uses torch_scatter/segment_csr; a plain
        # index scatter is numerically identical on standard edge_index and needs no torch_scatter).
        outs = [
            scatter(inputs, index, dim=0, dim_size=dim_size, reduce=m)
            for m in self.aggregation_methods
        ]
        if self.multiple_aggregation_merge_method != "sum":
            raise ValueError(f"Unsupported merge: {self.multiple_aggregation_merge_method}")
        merged = outs[0]
        for o in outs[1:]:
            merged = merged + o
        return merged


class LigandFormerGNN(nn.Module):
    """LigandFormer graph model adapted to our multitask regression pipeline.

    forward(x, edge_index, edge_attr, batch, desc=None) -> (pred [B, num_tasks], None).
    `edge_attr` and `desc` are accepted for gin call-signature parity and ignored (LigandFormer uses
    neither). `attention_received(...)` returns the per-atom attention attribution.
    """

    def __init__(
        self,
        node_in_dim: int = 9,
        edge_in_dim: int = 3,
        num_tasks: int = 1,
        block_num: int = 3,
        embedding_dim: int = 75,
        conv_hidden_dim: int = 256,
        classifier_hidden_dim: int = 256,
        readout_methods: str = "mean",
        pyramid_feature: bool = True,
        att_num_heads: int = 1,
        dropout: float = 0.1,
        hidden_dim: int = 128,
        desc_dim: int = 0,
        **_: object,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.node_feature_dim = node_in_dim
        self.embedding_dim = embedding_dim
        self.block_num = block_num
        self.readout_methods = readout_methods
        self.pyramid_feature = pyramid_feature
        self.dropout = dropout
        self.desc_dim = desc_dim
        conv_input_dim = embedding_dim * 2  # node_feature_update_method="cat"

        self.node_embedding = Sequential(
            Linear(node_in_dim, embedding_dim),
            LeakyReLU(0.1, inplace=True),
            Linear(embedding_dim, embedding_dim),
        )
        self.conv_blocks = ModuleList()
        self.spatial_atts = ModuleList()
        for i in range(block_num):
            self.conv_blocks.append(
                HAGConv(
                    Sequential(
                        Linear(conv_input_dim, conv_hidden_dim),
                        LeakyReLU(),
                        BatchNorm1d(conv_hidden_dim, momentum=0.01),
                        Linear(conv_hidden_dim, embedding_dim),
                        LeakyReLU(),
                        BatchNorm1d(embedding_dim, momentum=0.01),
                    )
                )
            )
            self.spatial_atts.append(
                SpatialAttentionBlockGIN(
                    input_dim=(i + 2) * embedding_dim,
                    output_dim=embedding_dim,
                    num_heads=att_num_heads,
                )
            )

        classifier_input_dim = (1 + block_num) * embedding_dim if pyramid_feature else embedding_dim
        self.dense_0 = Sequential(
            Linear(classifier_input_dim, classifier_hidden_dim),
            BatchNorm1d(classifier_hidden_dim, momentum=0.01),
            GELU(),
        )
        self.dense_1 = Sequential(
            Linear(classifier_hidden_dim, classifier_hidden_dim),
            BatchNorm1d(classifier_hidden_dim, momentum=0.01),
            GELU(),
        )
        self.dense_2 = Linear(classifier_hidden_dim, num_tasks)

    @classmethod
    def from_checkpoint(cls, ck):
        """Reconstruct from a training checkpoint's weight shapes (node_in_dim/num_tasks are the
        only load-bearing dims; the rest are architecture defaults)."""
        sd = ck["model"] if "model" in ck else ck
        model = cls(
            node_in_dim=sd["node_embedding.0.weight"].shape[1],
            num_tasks=sd["dense_2.weight"].shape[0],
        )
        model.load_state_dict(sd)
        return model

    def _blocks(self, x, edge_index, batch, return_attention=False):
        """Run node embedding + the block_num conv/attention blocks. Returns (hiddens, attn_maps)."""
        x = self.node_embedding(x.float())
        block_input, block_fusion, hiddens, attn_maps = x, x, [x], []
        for i in range(self.block_num):
            x = self.conv_blocks[i](x=block_input, edge_index=edge_index)
            block_input = x
            block_fusion = torch.cat((block_fusion, x), dim=1)
            if return_attention:
                reweighted, attn = self.spatial_atts[i](block_fusion, batch, return_attention=True)
                attn_maps.append(attn)
            else:
                reweighted = self.spatial_atts[i](block_fusion, batch)
            x = block_input + reweighted
            hiddens.append(x)
        return hiddens, attn_maps

    def forward(self, x, edge_index, edge_attr=None, batch=None, desc=None):
        hiddens, _ = self._blocks(x, edge_index, batch, return_attention=False)
        if self.pyramid_feature:
            pooled = []
            for h in hiddens:
                if self.training and self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=True)
                pooled.append(scatter(h, batch, dim=0, reduce=self.readout_methods))
            graph_feature = torch.cat(pooled, dim=1)
        else:
            h = hiddens[-1]
            if self.training and self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=True)
            graph_feature = scatter(h, batch, dim=0, reduce=self.readout_methods)

        y = self.dense_0(graph_feature)
        if self.training and self.dropout > 0:
            y = F.dropout(y, p=self.dropout * 2, training=True)
        y = self.dense_1(y)
        if self.training and self.dropout > 0:
            y = F.dropout(y, p=self.dropout * 2, training=True)
        pred = self.dense_2(y)
        return pred, None  # pooled-style: no exact per-atom scores

    @torch.no_grad()
    def attention_received(self, x, edge_index, batch):
        """Per-atom attention attribution [N]: attention RECEIVED = mean over query rows of each
        block's [N,N] map, averaged over blocks. Non-degenerate (unlike upstream's row-mean = 1/N).
        Block maps are block-diagonal, so off-molecule rows are 0 and the within-molecule mean is
        recovered by dividing the column sum by the molecule size."""
        self.eval()
        _, attn_maps = self._blocks(x, edge_index, batch, return_attention=True)
        size = scatter(torch.ones(x.size(0), device=x.device), batch, dim=0, reduce="sum")[batch]
        per_block = [
            attn.sum(dim=0) / size for attn in attn_maps
        ]  # column sum / mol size = mean recv
        return torch.stack(per_block, dim=0).mean(dim=0)  # average over blocks -> [N]
