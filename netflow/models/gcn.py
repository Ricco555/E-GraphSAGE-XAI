"""GCNEdgeClassifier: 2-layer GraphConv encoder with a per-edge MLP classifier.

Extracted from netflow/notebooks/legacy/04_baseline-xg-gcn.ipynb (cell 11).
Fixes: removed erroneous @torch.no_grad() on encode() which blocked gradient
flow through the GCN layers during training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

from models.edge_head import EdgeHead


class GCNEdgeClassifier(nn.Module):
    def __init__(
        self,
        node_in: int = 0,
        edge_in: int = 0,
        hidden: int = 128,
        num_classes: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.node_in = node_in
        # Project node features (or constant 1) into hidden space.
        self.feat_proj = nn.Linear(node_in if node_in > 0 else 1, hidden)

        self.gcn1 = dglnn.GraphConv(
            hidden, hidden, norm="both", weight=True, bias=True,
            allow_zero_in_degree=True, activation=F.relu,
        )
        self.gcn2 = dglnn.GraphConv(
            hidden, hidden, norm="both", weight=True, bias=True,
            allow_zero_in_degree=True, activation=None,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = EdgeHead(2 * hidden + edge_in, hidden, num_classes, dropout)

    def encode(self, blocks: list, x_nodes) -> torch.Tensor:
        if self.node_in > 0:
            h = self.feat_proj(x_nodes)
        else:
            h = self.feat_proj(
                torch.ones((blocks[0].num_src_nodes(), 1), device=blocks[0].device)
            )
        h = self.gcn1(blocks[0], h)
        h = self.dropout(h)
        h = self.gcn2(blocks[1], h)
        return h

    def forward(
        self,
        blocks: list,
        x_nodes,
        pair_graph: dgl.DGLGraph,
        e_feat: torch.Tensor,
    ) -> torch.Tensor:
        h_dst = self.encode(blocks, x_nodes)
        src, dst = pair_graph.edges()
        return self.head(torch.cat([h_dst[src], h_dst[dst], e_feat], dim=1))
