"""End-to-end data preparation: raw NetFlow CSV -> GraphBolt OnDiskDataset.

Runs:
  1) Clean (drop NA IDs, ±inf -> NA, fillna(0))
  2) Build IP:port endpoint identifiers
  3) Chronological 60/30/10 split
  4) Fit string -> int label map on TRAIN
  5) Fit numeric pipeline on TRAIN; transform val/test
  6) Fit categorical OHE on TRAIN; transform val/test
  7) Concatenate splits in (train, val, test) order to define a single graph
  8) Build node id mapping (IP:port -> contiguous int) from the concatenation
  9) Write GraphBolt OnDiskDataset under <out.root> with CSR-backed x_cat

Usage:
  python scripts/prepare_data.py --config configs/data.yaml
  python scripts/prepare_data.py --config configs/data.yaml --frac 0.1   # subsample for tuning
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

# Allow running from netflow/ directly: `python scripts/prepare_data.py ...`
HERE = Path(__file__).resolve().parent
NETFLOW_ROOT = HERE.parent
if str(NETFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(NETFLOW_ROOT))

from datasets.data_cleaning import clean_nfunsw_nb15
from datasets.chronological_split import (
    make_chronological_split_indices,
    save_split_indices,
)
from datasets.label_mapping import (
    fit_label_map,
    transform_labels,
    save_label_map,
)
from datasets.feature_numeric import fit_numeric_transform, transform_numeric
from datasets.categorical_encoding import (
    fit_categorical_transform,
    transform_categorical,
)
from datasets.nfunsw_nb15 import write_ondisk_layout
from utils.corr_viz import plot_numeric_corr_heatmap


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, type=str,
                    help="Path to YAML config (see configs/data.yaml).")
    ap.add_argument("--frac", default=None, type=float,
                    help="If set, stratified-sample this fraction of TRAIN (and VAL) "
                         "edges before building the dataset. For tuning sweeps.")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--skip_corr_plot", action="store_true",
                    help="Skip the (slow) Spearman correlation heatmap.")
    return ap.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_endpoint_ids(df: pd.DataFrame, ep_cfg: dict) -> pd.DataFrame:
    """Replace src/dst IP columns with `IP:port` strings, drop the port columns."""
    src_ip = ep_cfg["src_ip_col"]
    dst_ip = ep_cfg["dst_ip_col"]
    src_port = ep_cfg["src_port_col"]
    dst_port = ep_cfg["dst_port_col"]
    df = df.copy()
    df[src_ip] = df[src_ip].astype(str) + ":" + df[src_port].astype(str)
    df[dst_ip] = df[dst_ip].astype(str) + ":" + df[dst_port].astype(str)
    df = df.drop(columns=[src_port, dst_port])
    return df


def stratified_subsample(idx: np.ndarray, y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
    pos = np.arange(len(idx))
    keep_pos, _ = next(sss.split(pos, y))
    keep_pos = np.sort(keep_pos)
    return keep_pos


def factorize_endpoints(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    src_col: str,
    dst_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build a single shared endpoint -> int id space across all three splits.

    Order matters: train endpoints get the lowest ids (so train ip2id matches
    legacy build_light_graph_for_split when val/test reuse train's map). All
    endpoints from val/test that didn't appear in train append to the tail.
    """
    cat = pd.concat([
        df_train[[src_col, dst_col]],
        df_val[[src_col, dst_col]],
        df_test[[src_col, dst_col]],
    ], axis=0, ignore_index=False)

    flat = pd.concat([cat[src_col], cat[dst_col]], axis=0, ignore_index=True).astype(str)
    codes, _ = pd.factorize(flat, sort=False)  # first-seen order — train comes first
    n_per = [len(df_train), len(df_val), len(df_test)]

    # Reshape codes back: (E_total * 2,) -> (E_total, 2). Order is: src col then dst col.
    n_total = sum(n_per)
    src_codes = codes[:n_total]
    dst_codes = codes[n_total:]

    s_tr = src_codes[: n_per[0]].astype(np.int64)
    s_va = src_codes[n_per[0]: n_per[0] + n_per[1]].astype(np.int64)
    s_te = src_codes[n_per[0] + n_per[1]:].astype(np.int64)
    d_tr = dst_codes[: n_per[0]].astype(np.int64)
    d_va = dst_codes[n_per[0]: n_per[0] + n_per[1]].astype(np.int64)
    d_te = dst_codes[n_per[0] + n_per[1]:].astype(np.int64)

    n_nodes = int(codes.max()) + 1
    return s_tr, d_tr, s_va, d_va, s_te, d_te, n_nodes


def main():
    args = parse_args()
    cfg = load_config(args.config)

    raw_csv = cfg["raw"]["csv_path"]
    label_col_raw = cfg["raw"]["label_col"]
    drop_cols = cfg["raw"].get("drop_cols", [])
    ep_cfg = cfg["endpoint"]
    split_cfg = cfg["split"]
    num_cfg = cfg["numeric"]
    cat_cfg = cfg["categorical"]
    out_cfg = cfg["out"]

    print(f"[prepare_data] config: {args.config}")
    print(f"[prepare_data] reading {raw_csv}")
    df = pd.read_csv(raw_csv)

    # 1) clean
    df = clean_nfunsw_nb15(df)

    # 2) endpoints
    df = build_endpoint_ids(df, ep_cfg)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.rename(columns={label_col_raw: "label"})

    # 3) chronological split
    train_idx, val_idx, test_idx, meta = make_chronological_split_indices(
        df,
        start_col=split_cfg["start_col"],
        end_col=split_cfg["end_col"],
        dur_col=split_cfg["duration_col"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
    )
    save_split_indices(out_cfg["split_indices_dir"], train_idx, val_idx, test_idx, meta)

    df_train = df.loc[train_idx]
    df_val = df.loc[val_idx]
    df_test = df.loc[test_idx]

    # Optional subsampling for hyperparameter sweeps.
    if args.frac is not None and args.frac < 1.0:
        # Apply stratified subsampling to TRAIN and VAL only (keep TEST full).
        from datasets.label_mapping import _clean_label_series
        y_tr_str = _clean_label_series(df_train["label"]).fillna("__NA__").to_numpy()
        y_va_str = _clean_label_series(df_val["label"]).fillna("__NA__").to_numpy()
        keep_tr = stratified_subsample(train_idx, y_tr_str, args.frac, args.seed)
        keep_va = stratified_subsample(val_idx, y_va_str, args.frac, args.seed + 1)
        df_train = df_train.iloc[keep_tr]
        df_val = df_val.iloc[keep_va]
        train_idx = train_idx[keep_tr]
        val_idx = val_idx[keep_va]
        print(f"[prepare_data] subsampled to frac={args.frac}: "
              f"train={len(df_train)} val={len(df_val)} test={len(df_test)}")

    # 4) labels
    label2id = fit_label_map(df_train["label"], order="alpha")
    save_label_map(out_cfg["label_map_path"], label2id)
    y_train = transform_labels(df_train["label"], label2id)
    y_val = transform_labels(df_val["label"], label2id)
    y_test = transform_labels(df_test["label"], label2id)
    assert (y_train >= 0).all(), "Train has unseen/missing labels."
    num_classes = len(label2id)
    print(f"[prepare_data] num_classes={num_classes}")

    # 5) optional pre-pruning correlation plot
    if not args.skip_corr_plot:
        plot_numeric_corr_heatmap(
            df_train,
            exclude_numeric_categoricals=num_cfg["excluded_categoricals"],
            out_dir=out_cfg["corr_artifacts_dir"],
            threshold=num_cfg["corr_threshold"],
            nonneg_frac=num_cfg["nonneg_frac"],
        )

    # 6) numeric
    Xnum_train, _ = fit_numeric_transform(
        df_train,
        exclude_numeric_categoricals=num_cfg["excluded_categoricals"],
        scaler_type=num_cfg["scaler"],
        apply_corr_prune=num_cfg["apply_corr_prune"],
        corr_threshold=num_cfg["corr_threshold"],
        artifacts_dir=out_cfg["numeric_artifacts_dir"],
    )
    Xnum_val, _ = transform_numeric(df_val, artifacts_dir=out_cfg["numeric_artifacts_dir"])
    Xnum_test, _ = transform_numeric(df_test, artifacts_dir=out_cfg["numeric_artifacts_dir"])
    print(f"[prepare_data] numeric: train={Xnum_train.shape} val={Xnum_val.shape} test={Xnum_test.shape}")

    # 7) categorical
    Xcat_train, _ = fit_categorical_transform(
        df_train,
        cat_cols=num_cfg["excluded_categoricals"],
        use_port_buckets=cat_cfg["use_port_buckets"],
        min_freq=cat_cfg["rare_min_freq"],
        top_k=cat_cfg["rare_top_k"],
        artifacts_dir=out_cfg["categorical_artifacts_dir"],
    )
    Xcat_val = transform_categorical(df_val, artifacts_dir=out_cfg["categorical_artifacts_dir"])
    Xcat_test = transform_categorical(df_test, artifacts_dir=out_cfg["categorical_artifacts_dir"])
    print(f"[prepare_data] categorical: train={Xcat_train.shape} val={Xcat_val.shape} test={Xcat_test.shape}")

    # 8) endpoint factorization shared across splits
    s_tr, d_tr, s_va, d_va, s_te, d_te, n_nodes = factorize_endpoints(
        df_train, df_val, df_test, ep_cfg["src_ip_col"], ep_cfg["dst_ip_col"],
    )
    print(f"[prepare_data] n_nodes={n_nodes}")

    # 9) concatenate splits in (train, val, test) order; this defines the graph edge order.
    src_all = np.concatenate([s_tr, s_va, s_te])
    dst_all = np.concatenate([d_tr, d_va, d_te])
    x_num_all = np.vstack([Xnum_train, Xnum_val, Xnum_test]).astype(np.float32, copy=False)
    x_cat_all = sparse.vstack([Xcat_train, Xcat_val, Xcat_test]).tocsr()
    y_all = np.concatenate([y_train, y_val, y_test]).astype(np.int64, copy=False)

    # Timestamps in seconds.
    start_ms = split_cfg["start_col"]
    ts_all = (
        pd.concat([df_train[start_ms], df_val[start_ms], df_test[start_ms]])
        .to_numpy("int64", copy=False) / 1000.0
    ).astype(np.float32, copy=False)

    # Local edge ids per split.
    n_tr, n_va, n_te = len(s_tr), len(s_va), len(s_te)
    train_eids = np.arange(0, n_tr, dtype=np.int64)
    val_eids   = np.arange(n_tr, n_tr + n_va, dtype=np.int64)
    test_eids  = np.arange(n_tr + n_va, n_tr + n_va + n_te, dtype=np.int64)

    # 10) write OnDiskDataset
    root = out_cfg["root"]
    os.makedirs(root, exist_ok=True)
    meta = write_ondisk_layout(
        root,
        src_ids=src_all, dst_ids=dst_all, n_nodes=n_nodes,
        x_num=x_num_all, x_cat=x_cat_all, y=y_all, ts=ts_all,
        train_eids=train_eids, val_eids=val_eids, test_eids=test_eids,
        num_classes=num_classes,
    )
    print(f"[prepare_data] wrote OnDiskDataset to {root}")
    print(f"[prepare_data] meta: {json.dumps(meta.__dict__, indent=2)}")


if __name__ == "__main__":
    main()
