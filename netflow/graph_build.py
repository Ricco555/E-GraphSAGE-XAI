# graph_build.py
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import dgl
import torch


def _ensure_string(s: pd.Series) -> pd.Series:
    return s.astype("string")  # keeps NA as <NA>


def _make_ip_ids(
    s_src: pd.Series,
    s_dst: pd.Series,
    ip2id: Dict[str, int] | None = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Map IPv4 strings to contiguous node IDs.
    - Reuses an existing map if provided (so val/test reuse train’s node IDs when possible).
    - Adds unseen IPs with new IDs.
    """
    if ip2id is None:
        ip2id = {}

    src_ids = np.empty(len(s_src), dtype=np.int64)
    dst_ids = np.empty(len(s_dst), dtype=np.int64)

    for i, (a, b) in enumerate(zip(s_src.array, s_dst.array)):
        # Defensive: treat NA/empty as string "NA" (should be rare after cleaning)
        a = str(a) if a is not pd.NA else "NA"
        b = str(b) if b is not pd.NA else "NA"

        if a in ip2id:
            src_ids[i] = ip2id[a]
        else:
            nid = len(ip2id)
            ip2id[a] = nid
            src_ids[i] = nid

        if b in ip2id:
            dst_ids[i] = ip2id[b]
        else:
            nid = len(ip2id)
            ip2id[b] = nid
            dst_ids[i] = nid

    return src_ids, dst_ids, ip2id


def build_light_graph_for_split(
    df_split: pd.DataFrame,
    split_dir: str,
    *,
    src_col: str = "IPV4_SRC_ADDR",
    dst_col: str = "IPV4_DST_ADDR",
    start_ms_col: str = "FLOW_START_MILLISECONDS",
    label_col: str = "y",                 # int64 labels already mapped
    ip2id: Dict[str, int] | None = None,  # pass train's map to val/test
    device: str = "cpu",
    save_path: str | None = None          # e.g., "graphs/train.bin"
) -> Tuple[dgl.DGLGraph, Dict[str, int]]:
    """
    Build a lightweight directed multigraph:
      - nodes: unique IPs (string → int ID via ip2id)
      - edges: flows in df_split order (we do NOT add reverse duplicates)
      - g.edata:
          * dgl.EID : GLOBAL edge IDs (edge_indices.npy) for this split
          * 't'     : start time in seconds (float32)
          * 'y'     : class label per edge (int64)
    Does NOT store heavy edge features; those come from the feature store on demand.

    Returns (graph, ip2id).
    """
    # ----- load alignment arrays from feature_store -----
    edge_indices = np.load(os.path.join(split_dir, "edge_indices.npy"))        # (n_edges,)
    y = np.load(os.path.join(split_dir, "y.npy")).astype(np.int64, copy=False) # (n_edges,)
    ts_path = os.path.join(split_dir, "ts.npy")
    if os.path.exists(ts_path):
        t_sec = np.load(ts_path).astype(np.float32, copy=False)                # (n_edges,)
    else:
        # derive if needed; but normally saved by feature_store
        t_sec = (df_split[start_ms_col].to_numpy("int64", copy=False) / 1000.0).astype(np.float32)

    assert len(df_split) == edge_indices.shape[0] == y.shape[0] == t_sec.shape[0], \
        "Split dataframe and feature_store arrays must match in length."

    # ----- map IPs to node IDs (reuse/extend ip2id) -----
    s_src = _ensure_string(df_split[src_col])
    s_dst = _ensure_string(df_split[dst_col])
    src_ids, dst_ids, ip2id = _make_ip_ids(s_src, s_dst, ip2id)

    num_nodes = len(ip2id)
    num_edges = src_ids.shape[0]

    # ----- build dgl graph (directed, allow multi-edges) -----
    g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int64, device="cpu")
    # Edge order in g matches df_split row order by construction.

    # ----- minimal edata for training -----
    g.edata[dgl.EID] = torch.from_numpy(edge_indices)          # global EIDs, int64
    g.edata["y"]     = torch.from_numpy(y)                     # labels,    int64
    g.edata["t"]     = torch.from_numpy(t_sec)                 # time (s),  float32

    # (Optional) you can add degree-based node features later; for now keep it light.
    # Move only when needed; sampling works on CPU just fine.
    if device != "cpu":
        g = g.to(device)

    # ----- persist graph (optional) -----
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dgl.save_graphs(save_path, [g])
        # also persist the ip2id mapping for reuse
        # (numpy save of strings+ids is simple and portable)
        names = np.fromiter(ip2id.keys(), dtype=object)
        ids   = np.fromiter(ip2id.values(), dtype=np.int64)
        np.savez_compressed(save_path + ".ip2id.npz", names=names, ids=ids)

    return g, ip2id


def load_ip2id(path_npz: str) -> Dict[str, int]:
    z = np.load(path_npz, allow_pickle=True)
    names = z["names"]
    ids = z["ids"]
    return {str(k): int(v) for k, v in zip(names.tolist(), ids.tolist())}
