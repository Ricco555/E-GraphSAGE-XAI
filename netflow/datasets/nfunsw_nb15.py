"""
NF-UNSW-NB15 GraphBolt dataset wiring.

Provides:
- CSRFeature: a `dgl.graphbolt.Feature` subclass that holds a `scipy.sparse.csr_matrix`
  on the host and densifies a slice per `read()` call. This keeps the wide one-hot
  categorical block (~580 dims, mostly zero) off the GPU until a batch needs it.
- write_ondisk_layout(...): writes the GraphBolt OnDiskDataset metadata.yaml and
  numpy artifacts. The CSR is written separately as `features/x_cat.npz` and is
  attached at load time (it is not declared in metadata.yaml because GraphBolt's
  on-disk feature schema does not natively support sparse formats).
- load_dataset(root): loads the OnDiskDataset and attaches the CSRFeature.

Layout under <root>:
  metadata.yaml              # GraphBolt OnDiskDataset descriptor
  graph/edges.npy            # int64, shape (2, E) — src/dst node ids
  features/x_num.npy         # float32 memmap, (E, d_num)
  features/x_cat.npz         # scipy.sparse.csr_matrix, (E, d_cat) — attached at load
  features/y.npy             # int64, (E,)
  features/ts.npy            # float32, (E,)
  tasks/train_eids.npy       # int64, LOCAL edge ids
  tasks/val_eids.npy
  tasks/test_eids.npy
  store_meta.json            # n_nodes, n_edges, d_num, d_cat, num_classes

NOTE: GraphBolt's API for custom features and attachment evolves between minor
versions; this module is written against the DGL 2.x stable surface. If your
local `dgl.graphbolt` differs, adjust `load_dataset()` accordingly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import yaml
from scipy import sparse


# -----------------------------------------------------------------------------
# CSR-backed Feature
# -----------------------------------------------------------------------------

def _gb():
    """Lazy import so this module is importable without GraphBolt during scaffolding."""
    import dgl.graphbolt as gb  # noqa: F401
    return gb


class CSRFeature:
    """A GraphBolt-compatible Feature backed by `scipy.sparse.csr_matrix`.

    Inherits the `gb.Feature` ABC at instantiation time so this module can be
    imported in environments where GraphBolt is not built. `read()` slices the
    CSR by row ids and returns a dense float32 tensor — this is the sole point
    where the one-hot block becomes dense.
    """

    def __init__(self, csr: sparse.csr_matrix):
        if not sparse.isspmatrix_csr(csr):
            csr = sparse.csr_matrix(csr)
        if csr.dtype != np.float32:
            csr = csr.astype(np.float32, copy=False)
        self._csr = csr
        self._n_rows, self._n_cols = csr.shape

    def read(self, ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if ids is None:
            return torch.from_numpy(self._csr.toarray())
        if isinstance(ids, torch.Tensor):
            ids_np = ids.detach().cpu().numpy()
        else:
            ids_np = np.asarray(ids)
        rows = self._csr[ids_np].toarray()
        return torch.from_numpy(rows)

    def size(self) -> torch.Size:
        return torch.Size([self._n_cols])

    def count(self) -> int:
        return self._n_rows

    def metadata(self) -> dict:
        return {"backend": "csr", "n_cols": self._n_cols, "nnz": int(self._csr.nnz)}


def make_csr_feature(csr: sparse.csr_matrix):
    """Build a `gb.Feature` wrapping a CSR. Subclasses `gb.Feature` lazily so this
    module is importable without DGL/GraphBolt on the path."""
    gb = _gb()

    class _CSRFeatureGB(CSRFeature, gb.Feature):
        def __init__(self, csr_):
            CSRFeature.__init__(self, csr_)
            gb.Feature.__init__(self)

    return _CSRFeatureGB(csr)


# -----------------------------------------------------------------------------
# On-disk layout writer
# -----------------------------------------------------------------------------

@dataclass
class StoreMeta:
    n_nodes: int
    n_edges: int
    d_num: int
    d_cat: int
    num_classes: int


def write_ondisk_layout(
    root: str,
    *,
    src_ids: np.ndarray,
    dst_ids: np.ndarray,
    n_nodes: int,
    x_num: np.ndarray,         # (E, d_num) float32, train+val+test concatenated in graph order
    x_cat: sparse.csr_matrix,  # (E, d_cat) float32
    y: np.ndarray,             # (E,) int64
    ts: np.ndarray,            # (E,) float32
    train_eids: np.ndarray,    # int64 LOCAL edge ids in this graph
    val_eids: np.ndarray,
    test_eids: np.ndarray,
    num_classes: int,
    dataset_name: str = "nfunsw_nb15",
) -> StoreMeta:
    """Write the GraphBolt OnDiskDataset layout described at the top of this file.

    Edge ordering: `src_ids[i]`, `dst_ids[i]`, `x_num[i]`, `x_cat[i]`, `y[i]`,
    `ts[i]` must all describe the same edge i. `train/val/test_eids` index into
    this same edge order (LOCAL ids).
    """
    e = src_ids.shape[0]
    assert dst_ids.shape == (e,)
    assert x_num.shape[0] == e
    assert x_cat.shape[0] == e
    assert y.shape == (e,)
    assert ts.shape == (e,)
    assert x_num.dtype == np.float32
    assert y.dtype == np.int64
    assert ts.dtype == np.float32

    os.makedirs(os.path.join(root, "graph"), exist_ok=True)
    os.makedirs(os.path.join(root, "features"), exist_ok=True)
    os.makedirs(os.path.join(root, "tasks"), exist_ok=True)

    edges = np.stack([src_ids.astype(np.int64), dst_ids.astype(np.int64)], axis=0)
    np.save(os.path.join(root, "graph", "edges.npy"), edges)

    np.save(os.path.join(root, "features", "x_num.npy"), x_num)
    sparse.save_npz(os.path.join(root, "features", "x_cat.npz"),
                    x_cat.astype(np.float32, copy=False))
    np.save(os.path.join(root, "features", "y.npy"), y)
    np.save(os.path.join(root, "features", "ts.npy"), ts)

    np.save(os.path.join(root, "tasks", "train_eids.npy"), train_eids.astype(np.int64))
    np.save(os.path.join(root, "tasks", "val_eids.npy"),   val_eids.astype(np.int64))
    np.save(os.path.join(root, "tasks", "test_eids.npy"),  test_eids.astype(np.int64))

    metadata = {
        "dataset_name": dataset_name,
        "graph": {
            "nodes": [{"num": int(n_nodes)}],
            "edges": [{"format": "numpy", "path": "graph/edges.npy"}],
        },
        "feature_data": [
            {"domain": "edge", "type": None, "name": "x_num",
             "format": "numpy", "in_memory": False, "path": "features/x_num.npy"},
            {"domain": "edge", "type": None, "name": "y",
             "format": "numpy", "in_memory": True, "path": "features/y.npy"},
            {"domain": "edge", "type": None, "name": "ts",
             "format": "numpy", "in_memory": True, "path": "features/ts.npy"},
            # x_cat is attached at load time (CSRFeature) — not declared here.
        ],
        "tasks": [
            {
                "name": "edge_classification",
                "num_classes": int(num_classes),
                "train_set":      [{"data": [{"name": "seeds", "format": "numpy", "path": "tasks/train_eids.npy"}]}],
                "validation_set": [{"data": [{"name": "seeds", "format": "numpy", "path": "tasks/val_eids.npy"}]}],
                "test_set":       [{"data": [{"name": "seeds", "format": "numpy", "path": "tasks/test_eids.npy"}]}],
            }
        ],
    }
    with open(os.path.join(root, "metadata.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    meta = StoreMeta(
        n_nodes=int(n_nodes),
        n_edges=int(e),
        d_num=int(x_num.shape[1]),
        d_cat=int(x_cat.shape[1]),
        num_classes=int(num_classes),
    )
    with open(os.path.join(root, "store_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta.__dict__, f, indent=2)
    return meta


# -----------------------------------------------------------------------------
# Loader (attaches CSRFeature post-load)
# -----------------------------------------------------------------------------

def load_dataset(root: str):
    """Load the GraphBolt OnDiskDataset and attach the CSR-backed `x_cat` feature.

    Returns the loaded `gb.OnDiskDataset` instance. Access edge features via
    `ds.feature.read("edge", None, "x_num" | "x_cat" | "y" | "ts")`.
    """
    gb = _gb()
    ds = gb.OnDiskDataset(root).load()

    csr_path = os.path.join(root, "features", "x_cat.npz")
    if os.path.exists(csr_path):
        csr = sparse.load_npz(csr_path)
        feat = make_csr_feature(csr)
        # Attach under the standard ("edge", None, "x_cat") key.
        ds.feature[("edge", None, "x_cat")] = feat
    return ds


def load_store_meta(root: str) -> StoreMeta:
    with open(os.path.join(root, "store_meta.json"), "r", encoding="utf-8") as f:
        d = json.load(f)
    return StoreMeta(**d)
