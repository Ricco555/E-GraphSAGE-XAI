"""GATEdgeClassifier: multi-layer GAT encoder with a per-edge MLP classifier.

New baseline alongside GCN and GraphSAGE. Uses multi-head attention
(GATConv) with head-concatenation at each intermediate layer and head-
averaging at the final layer so the output dimension stays `hidden`
regardless of num_heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn

from models.edge_head import EdgeHead


class GATEdgeClassifier(nn.Module):
    def __init__(
        self,
        in_node: int = 0,
        hidden: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        edge_in: int = 0,
        edge_mlp_hidden: int = 128,
        num_classes: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert hidden % num_heads == 0, "hidden must be divisible by num_heads"
        head_dim = hidden // num_heads

        self.use_dummy_nodes = (in_node == 0)
        if self.use_dummy_nodes:
            self.node_embed = nn.Embedding(1, hidden)

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for li in range(num_layers):
            in_dim = (in_node if in_node > 0 else hidden) if li == 0 else hidden
            # All layers except the last concatenate heads; last layer averages.
            is_last = (li == num_layers - 1)
            self.gat_layers.append(
                dglnn.GATConv(
                    in_dim, head_dim, num_heads=num_heads,
                    feat_drop=dropout, attn_drop=dropout,
                    residual=(li > 0), activation=None,
                    allow_zero_in_degree=True,
                )
            )
            # After concat: num_heads * head_dim = hidden. After average: hidden.
            self.norms.append(nn.BatchNorm1d(hidden))

        self.head = EdgeHead(hidden * 2 + edge_in, edge_mlp_hidden, num_classes, dropout)
        self.dropout = nn.Dropout(dropout)

    def _init_node_features(self, blocks: list, x_nodes) -> torch.Tensor:
        if x_nodes is not None:
            return x_nodes
        return self.node_embed.weight[0].expand(
            blocks[0].num_src_nodes(), self.node_embed.embedding_dim
        )

    def encode(self, blocks: list, x_nodes) -> torch.Tensor:
        h = self._init_node_features(blocks, x_nodes)
        for conv, bn, block in zip(self.gat_layers, self.norms, blocks):
            h_dst = h[: block.num_dst_nodes()]
            out = conv(block, (h, h_dst))  # (N_dst, num_heads, head_dim)
            h   = out.flatten(1)           # (N_dst, hidden) — concat heads every layer
            h   = bn(h)
            h   = torch.relu(h)
            h   = self.dropout(h)
        return h

    def forward(
        self,
        blocks: list,
        x_nodes,
        pair_graph: dgl.DGLGraph,
        e_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode(blocks, x_nodes)
        u, v = pair_graph.edges(form="uv")
        return self.head(torch.cat([h[u], h[v], e_feat], dim=1))
