# feature_store.py
from __future__ import annotations

import os, torch, dgl
from typing import Tuple, Optional, Dict

import numpy as np
from scipy import sparse

from feature_numeric import fit_numeric_transform, transform_numeric
from categorical_encoding import fit_categorical_transform, transform_categorical


# ---------- low-level I/O ----------

def _save_memmap_float32(path: str, X: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert X.dtype == np.float32, "numeric memmap must be float32"
    m = np.memmap(path, dtype="float32", mode="w+", shape=X.shape)
    m[:] = X
    m.flush()


def _load_memmap_float32(path: str, shape: Tuple[int, int]) -> np.memmap:
    return np.memmap(path, dtype="float32", mode="r", shape=shape)


def _save_csr(path: str, X_csr: sparse.csr_matrix) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sparse.save_npz(path, X_csr.astype(np.float32, copy=False))


def _load_csr(path: str) -> sparse.csr_matrix:
    return sparse.load_npz(path)

def _map_global_to_local(g: dgl.DGLGraph, split_dir: str, require_full_cover: bool = True) -> torch.Tensor:
    """
    Map the split's GLOBAL edge IDs (feature_store/<split>/edge_indices.npy) to this graph's
    LOCAL edge IDs (0..E-1). Returns a torch.int64 tensor ordered exactly like the store file.

    If require_full_cover=True, raise if any store EID is not present in the graph.
    """
    # 1) GLOBAL EIDs from the feature store (this split)
    store_global = np.load(os.path.join(split_dir, "edge_indices.npy")).astype(np.int64)

    # 2) GLOBAL EIDs carried by the graph (one per LOCAL edge id)
    g_global = g.edata[dgl.EID].cpu().numpy().astype(np.int64)  # shape (E_local,)

    # 3) Build a fast lookup: sort graph-global once, then search positions of store ids
    order = np.argsort(g_global, kind="mergesort")
    g_sorted = g_global[order]
    pos = np.searchsorted(g_sorted, store_global)  # candidate positions in sorted array
    ok = (pos < g_sorted.size) & (g_sorted[pos] == store_global)

    if require_full_cover and not np.all(ok):
        missing = store_global[~ok]
        raise RuntimeError(
            f"[align] {missing.size} store EIDs not present in graph. "
            f"First few: {missing[:10].tolist()}"
        )

    # keep only the ones that matched (normally all of them)
    pos = pos[ok]
    local = order[pos]  # map sorted-pos --> LOCAL edge ids (0..E-1)
    return torch.from_numpy(local.astype(np.int64))


# Call this once per split after you save edge_indices.npy
def build_id_map(split_dir: str) -> None:
    """
    Persist a sorted index for global EIDs in this split so we can map any batch of
    global EIDs -> local row indices exactly and quickly.
    Saves: split_dir/eid_map.npz  with arrays: 'global_sorted', 'rows_sorted'
    """
    edge_indices = np.load(os.path.join(split_dir, "edge_indices.npy"))  # shape (N,)
    rows = np.arange(edge_indices.shape[0], dtype=np.int64)
    order = np.argsort(edge_indices, kind="mergesort")
    global_sorted = edge_indices[order]
    rows_sorted = rows[order]
    np.savez_compressed(os.path.join(split_dir, "eid_map.npz"),
                        global_sorted=global_sorted, rows_sorted=rows_sorted)

def _load_id_map(split_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(os.path.join(split_dir, "eid_map.npz"))
    global_sorted, rows_sorted = z["global_sorted"], z["rows_sorted"]
    # small consistency check: global_sorted must be sorted(edge_indices)
    eids = np.load(os.path.join(split_dir, "edge_indices.npy"))
    if global_sorted.shape != eids.shape or not np.array_equal(global_sorted, np.sort(eids, kind="mergesort")):
        raise RuntimeError(f"[feature_store] eid_map.npz is stale or mismatched in {split_dir}.")
    return global_sorted, rows_sorted

def map_eids_to_rows(eids_global: np.ndarray, split_dir: str) -> np.ndarray:
    """
    Map a vector of GLOBAL EIDs (from pair_graph.edata[dgl.EID]) to LOCAL row indices
    in this split's feature arrays. Returns an array of length B (batch), one row per input EID.
    Raises a clear error if any EID is missing (misalignment).
    """
    global_sorted, rows_sorted = _load_id_map(split_dir)
    eids_global = np.asarray(eids_global, dtype=np.int64, order="C")
    idx = np.searchsorted(global_sorted, eids_global)
    # Guard: idx in-bounds and matches value
    in_bounds = (idx >= 0) & (idx < global_sorted.shape[0])
    ok = in_bounds & (global_sorted[idx] == eids_global)
    if not np.all(ok):
        missing = eids_global[~ok]
        raise RuntimeError(
            f"[feature_store] Missing {missing.size} EIDs in split={split_dir}. "
            f"First few: {missing[:10].tolist()}. "
            "This indicates a split/graph/feature alignment issue. "
            "Rebuild the graph and feature_store consistently."
        )
    return rows_sorted[idx]

# ---------- public build API ----------

def build_feature_store(
    df_train,
    df_val,
    df_test,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    # numeric-categoricals are EXCLUDED from numeric transform and INCLUDED in categorical OHE
    numeric_categoricals: list,
    categorical_cols: list,               # typically includes numeric_categoricals (+ any other true categoricals)
    out_dir: str = "feature_store",
    numeric_artifacts_dir: str = "artifacts/numeric",
    categorical_artifacts_dir: str = "artifacts/categorical",
    use_port_buckets: bool = True,
    rare_min_freq: Optional[int] = 50,
    rare_top_k: Optional[int] = None,
    save_timestamps: bool = True,
    start_ms_col: str = "FLOW_START_MILLISECONDS",
    dur_ms_col: str = "FLOW_DURATION_MILLISECONDS",
) -> Dict[str, tuple]:
    """
    Build per-split feature stores and return a dict with shapes.
    Files written under out_dir/{train|val|test}/:
      - numeric.dat           (float32 memmap [n, d_num])
      - categorical.npz       (CSR float32   [n, d_cat])
      - edge_indices.npy      (int64         [n])
      - y.npy                 (int64         [n])
      - ts.npy                (float32       [n])  optional; start time in seconds

    Returns: dict split_name -> (n_rows, d_num, d_cat)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------- NUMERIC ----------
    Xnum_train, _ = fit_numeric_transform(
        df_train,
        exclude_numeric_categoricals=numeric_categoricals,
        scaler_type="standard",
        apply_corr_prune=True,
        corr_threshold=0.995,
        artifacts_dir=numeric_artifacts_dir,
    )
    Xnum_val,  _ = transform_numeric(df_val,  artifacts_dir=numeric_artifacts_dir)
    Xnum_test, _ = transform_numeric(df_test, artifacts_dir=numeric_artifacts_dir)

    # ---------- CATEGORICAL ----------
    Xcat_train, _ = fit_categorical_transform(
        df_train,
        cat_cols=categorical_cols,
        use_port_buckets=use_port_buckets,
        min_freq=rare_min_freq,
        top_k=rare_top_k,
        artifacts_dir=categorical_artifacts_dir,
    )
    Xcat_val = transform_categorical(df_val,  artifacts_dir=categorical_artifacts_dir)
    Xcat_test= transform_categorical(df_test, artifacts_dir=categorical_artifacts_dir)

    # ---------- write per split ----------
    def _save_split(name, Xnum, Xcat, idx, y_split, df_split):
        split_dir = os.path.join(out_dir, name)
        os.makedirs(split_dir, exist_ok=True)

        # numeric
        _save_memmap_float32(os.path.join(split_dir, "numeric.dat"),
                             Xnum.astype(np.float32, copy=False))

        # categorical
        _save_csr(os.path.join(split_dir, "categorical.npz"), Xcat)

        # labels
        np.save(os.path.join(split_dir, "y.npy"), y_split.astype(np.int64, copy=False))

        # timestamps (seconds) for optional temporal analysis
        if save_timestamps:
            if start_ms_col in df_split.columns:
                ts = (df_split[start_ms_col].to_numpy("int64", copy=False) / 1000.0).astype("float32")
            else:
                # fallback: derive from duration if needed
                if dur_ms_col in df_split.columns:
                    ts = (df_split[dur_ms_col].to_numpy("int64", copy=False) / 1000.0).astype("float32")
                else:
                    ts = np.zeros(df_split.shape[0], dtype=np.float32)
            np.save(os.path.join(split_dir, "ts.npy"), ts)

        # global edge ids (row positions from original df)
        np.save(os.path.join(split_dir, "edge_indices.npy"), idx.astype(np.int64, copy=False))

        return (Xnum.shape[0], Xnum.shape[1], Xcat.shape[1])

    shapes = {}
    shapes["train"] = _save_split("train", Xnum_train, Xcat_train, train_idx, y_train, df_train)
    shapes["val"]   = _save_split("val",   Xnum_val,   Xcat_val,   val_idx,   y_val,   df_val)
    shapes["test"]  = _save_split("test",  Xnum_test,  Xcat_test,  test_idx,  y_test,  df_test)

    print(f"[feature_store] train: n={shapes['train'][0]} d_num={shapes['train'][1]} d_cat={shapes['train'][2]}")
    print(f"[feature_store] val  : n={shapes['val'][0]}   d_num={shapes['val'][1]}   d_cat={shapes['val'][2]}")
    print(f"[feature_store] test : n={shapes['test'][0]}  d_num={shapes['test'][1]}  d_cat={shapes['test'][2]}")
    
    # build global->local eid maps for fast batch lookup
    for s in ("train", "val", "test"):
        build_id_map(os.path.join(out_dir, s))

    return shapes


# ---------- batch fetch API for training ----------

def _load_numeric_memmap(split_dir: str) -> np.memmap:
    """Load numeric memmap using shape from file size."""
    path = os.path.join(split_dir, "numeric.dat")
    # Infer shape: we persist shape by also having y.npy (n,) to get n_rows
    y = np.load(os.path.join(split_dir, "y.npy"))
    n = int(y.shape[0])
    # cross-check with split EIDs length
    eids = np.load(os.path.join(split_dir, "edge_indices.npy"))
    if eids.shape[0] != n:
        raise RuntimeError(f"[feature_store] n_rows mismatch in {split_dir}: y={n} vs edge_indices={eids.shape[0]}")
    # we also need d_num; infer via file size
    num_bytes = os.path.getsize(path)
    d = num_bytes // (4 * n)  # float32 bytes=4
    return _load_memmap_float32(path, (n, d))


def _load_cat_csr(split_dir: str) -> sparse.csr_matrix:
    return _load_csr(os.path.join(split_dir, "categorical.npz"))

def fetch_edge_features(eids_global: np.ndarray, split_dir: str) -> np.ndarray:
    """
    Strictly fetch features for a batch of GLOBAL EIDs, in the same order, 1 row per EID.
    Requires that build_id_map(split_dir) was called during feature store creation.
    """
    local_rows = map_eids_to_rows(eids_global, split_dir)  # length B

    Xnum = _load_numeric_memmap(split_dir)  # (N, d_num)
    x_num = Xnum[local_rows]               # (B, d_num)

    Xcat = _load_cat_csr(split_dir)         # (N, d_cat) CSR (may be 0 cols)
    if Xcat.shape[1] > 0:
        x_cat = Xcat[local_rows].toarray().astype(np.float32, copy=False)  # (B, d_cat)
    else:
        x_cat = np.zeros((local_rows.shape[0], 0), dtype=np.float32)

    x = np.hstack([x_num, x_cat]).astype(np.float32, copy=False)
    assert x.shape[0] == eids_global.shape[0], "Feature batch must match EID batch size"
    return x

""" How to use: 
from feature_store import build_feature_store

# We already have:
# df_train, df_val, df_test       (chronological splits)
# y_train, y_val, y_test          (int labels via label_mapping)
# train_idx, val_idx, test_idx    (global edge indices from persist_splits)

numeric_cats = [
    "PROTOCOL","L7_PROTO","ICMP_TYPE","ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE","DNS_QUERY_ID","FTP_COMMAND_RET_CODE",
    "L4_SRC_PORT","L4_DST_PORT",  # if you use bucket/one-hot ports - otherwise exclude
]

cat_cols = numeric_cats  # plus any extra true categoricals if you have them

shapes = build_feature_store(
    df_train, df_val, df_test,
    y_train, y_val, y_test,
    train_idx, val_idx, test_idx,
    numeric_categoricals=numeric_cats,
    categorical_cols=cat_cols,
    out_dir="feature_store",
    numeric_artifacts_dir="artifacts/numeric",
    categorical_artifacts_dir="artifacts/categorical",
    use_port_buckets=False,
    rare_min_freq=50,
    rare_top_k=None,
    save_timestamps=True,
)
print(shapes)

"""