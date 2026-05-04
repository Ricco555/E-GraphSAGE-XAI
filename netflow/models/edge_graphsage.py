"""EdgeGraphSAGE: multi-layer GraphSAGE encoder with a per-edge MLP classifier.

Nodes carry no structural features; a learned constant embedding is used
instead (one Embedding(1, hidden) shared across all nodes). Edge features
(numeric + OHE concatenated) are injected at the edge-head stage, not during
graph convolution.

Forward contract:
    logits = model(blocks, x_nodes, pair_graph, e_feat)
    - blocks:      list of DGL message-flow graphs from NeighborSampler
    - x_nodes:     (N_src, d_node) float32 tensor, or None for learned embedding
    - pair_graph:  DGL graph whose edges are the current mini-batch
    - e_feat:      (B, d_edge) float32 edge features aligned to pair_graph.edges()
"""
from __future__ import annotations

import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn

from models.edge_head import EdgeHead


class EdgeGraphSAGE(nn.Module):
    def __init__(
        self,
        in_node: int = 0,
        hidden: int = 128,
        num_layers: int = 2,
        aggregator: str = "mean",
        edge_in: int = 0,
        edge_mlp_hidden: int = 128,
        num_classes: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_dummy_nodes = (in_node == 0)

        self.sage = nn.ModuleList()
        self.norms = nn.ModuleList()
        for li in range(num_layers):
            in_dim = (in_node if in_node > 0 else hidden) if li == 0 else hidden
            self.sage.append(dglnn.SAGEConv(in_dim, hidden, aggregator_type=aggregator))
            self.norms.append(nn.BatchNorm1d(hidden))

        if self.use_dummy_nodes:
            self.node_embed = nn.Embedding(1, hidden)

        self.head = EdgeHead(hidden * 2 + edge_in, edge_mlp_hidden, num_classes, dropout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_node_features(self, blocks: list, x_nodes) -> torch.Tensor:
        if x_nodes is not None:
            return x_nodes
        return self.node_embed.weight[0].expand(
            blocks[0].num_src_nodes(), self.node_embed.embedding_dim
        )

    def _run_sage(self, blocks: list, h: torch.Tensor) -> torch.Tensor:
        for conv, bn, block in zip(self.sage, self.norms, blocks):
            h_dst = h[: block.num_dst_nodes()]
            h = conv(block, (h, h_dst))
            h = bn(h)
            h = torch.relu(h)
        return h

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        blocks: list,
        x_nodes,
        pair_graph: dgl.DGLGraph,
        e_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = self._init_node_features(blocks, x_nodes)
        h = self._run_sage(blocks, h)
        u, v = pair_graph.edges(form="uv")
        return self.head(torch.cat([h[u], h[v], e_feat], dim=1))

    def encode(
        self,
        blocks: list,
        x_nodes,
        return_src: bool = False,
    ):
        """Return per-node embeddings for the destination nodes of the final block.

        If return_src=True, returns (h_src_init, h_dst) where h_src_init are the
        initial node features for blocks[0].src nodes — used by structural XAI.
        """
        h = self._init_node_features(blocks, x_nodes)
        h_src_init = h.clone() if return_src else None

        last_h_dst = None
        for conv, bn, block in zip(self.sage, self.norms, blocks):
            last_h_dst = h[: block.num_dst_nodes()]
            h = conv(block, (h, last_h_dst))
            h = bn(h)
            h = torch.relu(h)

        if return_src:
            return h_src_init, last_h_dst
        return last_h_dst

    def predict_from_embeddings(
        self,
        h_dst: torch.Tensor,
        pair_graph: dgl.DGLGraph,
        e_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits from pre-computed endpoint embeddings (used by XAI)."""
        with pair_graph.local_scope():
            src, dst = pair_graph.edges(form="uv")
            return self.head(torch.cat([h_dst[src], h_dst[dst], e_feat], dim=1))
