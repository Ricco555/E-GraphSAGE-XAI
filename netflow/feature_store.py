# feature_store.py
from __future__ import annotations

import os
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


def _save_id_fastmap(out_path: str, ids: np.ndarray) -> None:
    """
    Given global edge IDs for this split (row order), save arrays so we can map
    global eids -> local row indices quickly:
       - ids_sorted: sorted copy of ids
       - order: array s.t. order[rank(global_id)] = local_row
    """
    ids = ids.astype(np.int64, copy=False)
    order = np.empty_like(ids)
    # argsort over ids gives ranks; place local row positions at those ranks
    ranks = np.argsort(ids, kind="mergesort")
    order[ranks] = np.arange(ids.shape[0], dtype=np.int64)
    ids_sorted = np.sort(ids, kind="mergesort")
    np.savez_compressed(out_path, ids_sorted=ids_sorted, order=order)


def _map_global_to_local(eids_global: np.ndarray, ids_sorted: np.ndarray, order: np.ndarray) -> np.ndarray:
    """
    Map global edge IDs to local row indices.
    Returns -1 for any ID not found.
    """
    pos = np.searchsorted(ids_sorted, eids_global)
    in_range = (pos < ids_sorted.shape[0]) & (ids_sorted[pos] == eids_global)
    local = np.full(eids_global.shape[0], -1, dtype=np.int64)
    local[in_range] = order[pos[in_range]]
    return local


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
      - id_fastmap.npz        (ids_sorted, order)
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
        _save_id_fastmap(os.path.join(split_dir, "id_fastmap.npz"), idx.astype(np.int64, copy=False))

        return (Xnum.shape[0], Xnum.shape[1], Xcat.shape[1])

    shapes = {}
    shapes["train"] = _save_split("train", Xnum_train, Xcat_train, train_idx, y_train, df_train)
    shapes["val"]   = _save_split("val",   Xnum_val,   Xcat_val,   val_idx,   y_val,   df_val)
    shapes["test"]  = _save_split("test",  Xnum_test,  Xcat_test,  test_idx,  y_test,  df_test)

    print(f"[feature_store] train: n={shapes['train'][0]} d_num={shapes['train'][1]} d_cat={shapes['train'][2]}")
    print(f"[feature_store] val  : n={shapes['val'][0]}   d_num={shapes['val'][1]}   d_cat={shapes['val'][2]}")
    print(f"[feature_store] test : n={shapes['test'][0]}  d_num={shapes['test'][1]}  d_cat={shapes['test'][2]}")

    return shapes


# ---------- batch fetch API for training ----------

def _load_numeric_memmap(split_dir: str) -> np.memmap:
    """Load numeric memmap using shape from file size."""
    path = os.path.join(split_dir, "numeric.dat")
    # Infer shape: we persist shape by also having y.npy (n,) to get n_rows
    y = np.load(os.path.join(split_dir, "y.npy"))
    n = int(y.shape[0])
    # we also need d_num; infer via file size
    num_bytes = os.path.getsize(path)
    d = num_bytes // (4 * n)  # float32 bytes=4
    return _load_memmap_float32(path, (n, d))


def _load_cat_csr(split_dir: str) -> sparse.csr_matrix:
    return _load_csr(os.path.join(split_dir, "categorical.npz"))


def _load_fastmap(split_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(os.path.join(split_dir, "id_fastmap.npz"))
    return z["ids_sorted"], z["order"]


def fetch_edge_features(eids_global: np.ndarray, split_dir: str) -> np.ndarray:
    """
    Given GLOBAL edge IDs for this split, return a dense float32 (B, d_edge)
    by concatenating numeric memmap rows with categorical CSR rows (toarray()).
    Any eids not present in this split are ignored (dropped).
    """
    eids_global = np.asarray(eids_global, dtype=np.int64)
    ids_sorted, order = _load_fastmap(split_dir)
    local = _map_global_to_local(eids_global, ids_sorted, order)

    keep = (local >= 0)
    if not keep.any():
        return np.zeros((0, 0), dtype=np.float32)

    local_rows = local[keep]

    Xnum = _load_numeric_memmap(split_dir)
    x_num = Xnum[local_rows]  # (B, d_num)

    Xcat = _load_cat_csr(split_dir)      # (N, d_cat)
    if Xcat.shape[1] > 0:
        x_cat = Xcat[local_rows].toarray().astype(np.float32, copy=False)
    else:
        x_cat = np.zeros((local_rows.shape[0], 0), dtype=np.float32)

    return np.hstack([x_num, x_cat]).astype(np.float32, copy=False)

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