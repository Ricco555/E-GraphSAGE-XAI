# debug_align.py
from __future__ import annotations
import os, numpy as np, dgl

def check_graph_vs_store(g: dgl.DGLGraph, split_dir: str, *, verbose=True) -> np.ndarray:
    """
    Ensure every GLOBAL EID used by graph 'g' exists in feature_store split_dir.
    Returns: store_ids (np.ndarray of GLOBAL EIDs in this split).
    """
    g_eids = g.edata[dgl.EID].cpu().numpy().astype(np.int64)

    idx_path = os.path.join(split_dir, "edge_indices.npy")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Missing split file: {idx_path}")
    store_ids = np.load(idx_path).astype(np.int64)

    missing_in_store = np.setdiff1d(g_eids, store_ids, assume_unique=False)
    extra_in_store   = np.setdiff1d(store_ids, g_eids, assume_unique=False)

    if verbose:
        print(f"[align] Graph edges: {g_eids.size} | Store rows: {store_ids.size}")
        print(f"[align] Missing in store: {missing_in_store.size} | Extra in store: {extra_in_store.size}")
        if missing_in_store.size:
            print("        First missing:", missing_in_store[:10].tolist())
        if extra_in_store.size:
            print("        First extra  :", extra_in_store[:10].tolist())

    if missing_in_store.size:
        raise RuntimeError(
            f"[align] {missing_in_store.size} GLOBAL EIDs used by graph are not present in store '{split_dir}'. "
            f"Rebuild graph and feature_store from the SAME cleaned DF and SAME split indices."
        )
    return store_ids