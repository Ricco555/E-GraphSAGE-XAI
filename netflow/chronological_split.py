# persist_splits.py
# makes a chronological 60/30/10 split (boundary-safe for identical timestamps),
# saves index arrays (.npz) to be able to rebuild graphs from the exact same rows later,
# saves the split datasets (Parquet or CSV) so they can be loaded without re-splitting,
# writes a tiny meta.json with time thresholds and counts for auditability.
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChronoMeta:
    start_col: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    n_total: int
    n_train: int
    n_val: int
    n_test: int
    t_min: int
    t_train_max: int
    t_val_max: int
    t_max: int


def _resolve_start_ms(
    df: pd.DataFrame,
    start_col: str = "FLOW_START_MILLISECONDS",
    end_col: str = "FLOW_END_MILLISECONDS",
    dur_col: str = "FLOW_DURATION_MILLISECONDS",
) -> pd.Series:
    if start_col in df.columns:
        s = pd.to_numeric(df[start_col], errors="coerce")
        return s.astype("Int64")
    if end_col in df.columns and dur_col in df.columns:
        end = pd.to_numeric(df[end_col], errors="coerce")
        dur = pd.to_numeric(df[dur_col], errors="coerce")
        return (end - dur).astype("Int64")
    raise KeyError(f"Missing time columns: need '{start_col}' or ('{end_col}','{dur_col}')")


def _adjust_cut_at_equal_timestamps(t_sorted: np.ndarray, cut_idx: int) -> int:
    n = t_sorted.shape[0]
    if cut_idx <= 0 or cut_idx >= n:
        return cut_idx
    t_cut = t_sorted[cut_idx - 1]
    j = cut_idx
    while j < n and t_sorted[j] == t_cut:
        j += 1
    return j


def make_chronological_split_indices(
    df: pd.DataFrame,
    start_col: str = "FLOW_START_MILLISECONDS",
    end_col: str = "FLOW_END_MILLISECONDS",
    dur_col: str = "FLOW_DURATION_MILLISECONDS",
    train_ratio: float = 0.60,
    val_ratio: float = 0.30,
    test_ratio: float = 0.10,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ChronoMeta]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1."
    t_start = _resolve_start_ms(df, start_col, end_col, dur_col)

    if t_start.isna().any():
        keep = ~t_start.isna()
        dropped = int((~keep).sum())
        if verbose and dropped:
            print(f"[split] Dropping {dropped} rows with missing start time.")
        df = df.loc[keep].copy()
        t_start = t_start.loc[keep]

    order = np.argsort(t_start.to_numpy(dtype=np.int64), kind="mergesort")
    idx_sorted = df.index.to_numpy()[order]
    t_sorted = t_start.to_numpy(dtype=np.int64)[order]
    n = len(idx_sorted)
    if n == 0:
        raise ValueError("Empty dataframe after filtering for start time.")

    cut_train = int(round(n * train_ratio))
    cut_val = int(round(n * (train_ratio + val_ratio)))
    cut_train = max(1, min(cut_train, n - 2))
    cut_val = max(cut_train + 1, min(cut_val, n - 1))
    cut_train = _adjust_cut_at_equal_timestamps(t_sorted, cut_train)
    cut_val = _adjust_cut_at_equal_timestamps(t_sorted, cut_val)

    train_idx = idx_sorted[:cut_train].astype(np.int64, copy=False)
    val_idx = idx_sorted[cut_train:cut_val].astype(np.int64, copy=False)
    test_idx = idx_sorted[cut_val:].astype(np.int64, copy=False)

    meta = ChronoMeta(
        start_col=start_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_total=n,
        n_train=int(train_idx.size),
        n_val=int(val_idx.size),
        n_test=int(test_idx.size),
        t_min=int(t_sorted[0]),
        t_train_max=int(t_sorted[cut_train - 1]),
        t_val_max=int(t_sorted[cut_val - 1]),
        t_max=int(t_sorted[-1]),
    )

    if verbose:
        print("[split] Chronological 60/30/10 with boundary safety")
        print(f"        TRAIN: {meta.n_train:>8}  t∈[{meta.t_min}, {meta.t_train_max}]")
        print(f"        VAL:   {meta.n_val:>8}  t∈({meta.t_train_max}, {meta.t_val_max}]")
        print(f"        TEST:  {meta.n_test:>8}  t∈({meta.t_val_max}, {meta.t_max}]")

    return train_idx, val_idx, test_idx, meta


def save_split_indices(out_dir: str, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, meta: ChronoMeta) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "split_indices.npz"),
                        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)
    print(f"[split] Saved indices → {out_dir}/split_indices.npz and meta → {out_dir}/meta.json")


def load_split_indices(out_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    zp = np.load(os.path.join(out_dir, "split_indices.npz"))
    with open(os.path.join(out_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return zp["train_idx"], zp["val_idx"], zp["test_idx"], meta


def save_split_frames(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    out_dir: str,
    fmt: str = "parquet", # "parquet" or "csv", parquet for better performance
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df_train = df.loc[train_idx]
    df_val = df.loc[val_idx]
    df_test = df.loc[test_idx]

    if fmt == "parquet":
        df_train.to_parquet(os.path.join(out_dir, "train.parquet"), engine="pyarrow", index=False)
        df_val.to_parquet(os.path.join(out_dir, "val.parquet"), engine="pyarrow", index=False)
        df_test.to_parquet(os.path.join(out_dir, "test.parquet"), engine="pyarrow", index=False)
    elif fmt == "csv":
        df_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        df_val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
        df_test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    else:
        raise ValueError("fmt must be 'parquet' or 'csv'.")

    print(f"[split] Saved materialized splits to {out_dir} ({fmt}).")

""" 
#how to use in Jupyter notebook or script:
from persist_splits import (
    make_chronological_split_indices,
    save_split_indices, load_split_indices,
    save_split_frames,
)

# df_clean = ...  # output of your data_cleaning.clean_nfunsw_nb15(df_raw)

# 1) Create indices (once)
train_idx, val_idx, test_idx, meta = make_chronological_split_indices(
    df_clean,
    start_col="FLOW_START_MILLISECONDS",  # in ms
    end_col="FLOW_END_MILLISECONDS",
    dur_col="FLOW_DURATION_MILLISECONDS",
    train_ratio=0.60, val_ratio=0.30, test_ratio=0.10,
)

# 2) Persist indices for future runs
save_split_indices("artifacts/splits", train_idx, val_idx, test_idx, meta)

# (Optional) also dump the materialized split datasets now
save_split_frames(df_clean, train_idx, val_idx, test_idx, out_dir="artifacts/splits", fmt="parquet")

# 3) In any future run (graph build, feature store, training), load the SAME indices:
train_idx2, val_idx2, test_idx2, meta2 = load_split_indices("artifacts/splits")

# Use them to subset df_clean deterministically
df_train = df_clean.loc[train_idx2]
df_val   = df_clean.loc[val_idx2]
df_test  = df_clean.loc[test_idx2] """