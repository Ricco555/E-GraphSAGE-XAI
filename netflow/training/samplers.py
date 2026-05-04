"""Sampling pipeline builders for E-GraphSAGE training.

Provides the DGL NeighborSampler-based edge loader that reads from the
OnDiskDataset layout produced by scripts/prepare_data.py.

The loader API deliberately matches train_edgecls_dbg.py so that a
seed=42 run produces the same mini-batch sequence for validation.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import dgl
from dgl.dataloading import NeighborSampler, as_edge_prediction_sampler
from dgl.dataloading import DataLoader as DGLDataLoader


def build_dgl_graph(root: str, n_nodes: int) -> dgl.DGLGraph:
    """Construct a DGL graph from the OnDiskDataset edge list.

    The resulting graph has no node/edge data attached — callers should
    add `g.edata["y"]` before passing to a DataLoader.
    """
    edges = np.load(os.path.join(root, "graph", "edges.npy"))  # (2, E)
    return dgl.graph(
        (edges[0].astype(np.int64), edges[1].astype(np.int64)),
        num_nodes=int(n_nodes),
    )


def make_edge_loader(
    g: dgl.DGLGraph,
    eids: torch.Tensor,
    *,
    fanouts: tuple[int, ...],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 42,
) -> DGLDataLoader:
    """Return a DGL DataLoader that iterates over `eids` with neighbor sampling.

    Yields (input_nodes, pair_graph, blocks) — identical shape to the legacy
    train_edgecls_dbg.py loader, so existing run_epoch logic is unchanged.
    """
    sampler = as_edge_prediction_sampler(NeighborSampler(list(fanouts)))
    gen = torch.Generator().manual_seed(seed)
    return DGLDataLoader(
        g, eids, sampler,
        batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers, use_ddp=False, generator=gen,
    )
