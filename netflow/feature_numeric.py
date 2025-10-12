# feature_numeric.py
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler


@dataclass(frozen=True)
class NumericArtifacts:
    """Everything needed to reproduce numeric transforms exactly."""
    num_cols_original: List[str]       # numeric columns INPUT (before any drops)
    num_cols_final: List[str]          # numeric columns kept after var + corr pruning
    nonneg_mask: List[bool]            # len = len(num_cols_original); True => log1p applied
    scaler_type: str                   # 'standard' or 'robust'
    scaler_state_path: str             # path to joblib scaler (stored separately)
    drop_zero_var: bool
    corr_prune: bool
    corr_threshold: float
    # masks refer to ORIGINAL numeric order
    keep_mask_var: List[bool]          # after zero-variance
    keep_mask_corr: List[bool]         # after correlation (applied after var mask)


def _select_numeric_columns(
    df: pd.DataFrame,
    exclude: Iterable[str]
) -> List[str]:
    """Return numeric columns excluding any provided (numeric-coded categoricals)."""
    ex = set(exclude)
    num_cols = []
    for c in df.columns:
        if c in ex:
            continue
        if pd.api.types.is_numeric_dtype(df[c].dtype):
            num_cols.append(c)
    return num_cols


def _infer_nonneg_mask(
    df_num: pd.DataFrame,
    frac_threshold: float = 0.995
) -> np.ndarray:
    """
    Infer which numeric columns are non-negative. A column is treated as non-negative
    if at least `frac_threshold` of its values are >= 0. (Robust to rare negatives.)
    """
    arr = df_num.to_numpy(dtype="float64", copy=False)
    ge0 = (arr >= 0).mean(axis=0)  # fraction >= 0 per column
    return (ge0 >= frac_threshold)


def _apply_log1p_where_nonneg(arr: np.ndarray, nonneg_mask: np.ndarray) -> np.ndarray:
    out = arr.copy()
    if nonneg_mask.any():
        cols = np.where(nonneg_mask)[0]
        out[:, cols] = np.log1p(np.maximum(out[:, cols], 0.0))
    return out


def _fit_scaler(X: np.ndarray, scaler_type: str):
    if scaler_type == "robust":
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    else:
        scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X)
    return scaler


def _transform_with_scaler(X: np.ndarray, scaler) -> np.ndarray:
    return scaler.transform(X)


def _drop_zero_variance(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    var = X.var(axis=0)
    keep = (var > eps)
    return X[:, keep], keep


def _spearman_corr_prune(
    X: np.ndarray,
    thr: float = 0.995,
    max_cols_for_dense: int = 5000
) -> np.ndarray:
    """
    Conservative Spearman-based pruning.
    Returns boolean mask over columns (in the CURRENT matrix) indicating which to keep.
    """
    d = X.shape[1]
    if d <= 1:
        return np.ones(d, dtype=bool)

    # Spearman via ranks → Pearson on ranks
    # memory-aware: rank-transform column-wise
    X_rank = np.empty_like(X, dtype=np.float64)
    for j in range(d):
        col = X[:, j]
        # argsort twice to get ranks (average ranks not strictly needed for pruning)
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, X.shape[0] + 1, dtype=np.float64)
        X_rank[:, j] = ranks

    # Pearson corr on ranks
    df_rank = pd.DataFrame(X_rank)
    rho = df_rank.corr(method="pearson").abs()

    upper = rho.where(np.triu(np.ones(rho.shape), k=1).astype(bool))
    keep = np.ones(d, dtype=bool)

    # Greedy: if col j highly corr with first kept i, drop j
    # Deterministic order left-to-right
    for i in range(d):
        if not keep[i]:
            continue
        # columns >= i+1 that are highly correlated with i
        high = upper.iloc[i, (i + 1):] >= thr
        if high.any():
            drop_js = np.where(high.to_numpy())[0] + (i + 1)
            keep[drop_js] = False

    return keep


def fit_numeric_transform(
    df_train: pd.DataFrame,
    exclude_numeric_categoricals: Iterable[str],
    scaler_type: str = "standard",
    apply_corr_prune: bool = True,
    corr_threshold: float = 0.995,
    artifacts_dir: str = "artifacts/numeric"
) -> Tuple[np.ndarray, NumericArtifacts]:
    """
    Fit numeric pipeline on TRAIN ONLY:
      - select numeric columns (excluding numeric-coded categoricals)
      - infer non-negative columns → log1p on those
      - fit scaler (Standard/Robust)
      - drop zero-variance columns
      - optional Spearman correlation pruning
    Persist scaler + masks; return transformed TRAIN matrix (float32).
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1) select numeric-only columns (excluding numeric-coded categoricals)
    num_cols = _select_numeric_columns(df_train, exclude_numeric_categoricals)
    if not num_cols:
        raise ValueError("No numeric columns selected. Check exclusions.")
    X = df_train[num_cols].to_numpy(dtype=np.float64, copy=True)

    # 2) infer non-negative → log1p
    nonneg_mask = _infer_nonneg_mask(df_train[num_cols])
    X = _apply_log1p_where_nonneg(X, nonneg_mask)

    # 3) fit scaler on TRAIN
    scaler = _fit_scaler(X, scaler_type=scaler_type)
    X = _transform_with_scaler(X, scaler)

    # 4) zero-variance drop
    X, keep_var = _drop_zero_variance(X, eps=1e-12)

    # 5) correlation pruning (optional)
    if apply_corr_prune and X.shape[1] > 1:
        keep_corr = _spearman_corr_prune(X, thr=corr_threshold)
        X = X[:, keep_corr]
    else:
        keep_corr = np.ones(X.shape[1], dtype=bool)

    # 6) persist scaler separately + artifacts.json via joblib
    scaler_path = os.path.join(artifacts_dir, "numeric_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # compute final kept column names (apply masks in order)
    num_cols_arr = np.array(num_cols)
    num_cols_after_var = num_cols_arr[np.array(keep_var, dtype=bool)]
    num_cols_final = num_cols_after_var[np.array(keep_corr, dtype=bool)].tolist()

    arts = NumericArtifacts(
        num_cols_original=num_cols,
        num_cols_final=num_cols_final,
        nonneg_mask=nonneg_mask.astype(bool).tolist(),
        scaler_type=scaler_type,
        scaler_state_path=scaler_path,
        drop_zero_var=True,
        corr_prune=bool(apply_corr_prune),
        corr_threshold=float(corr_threshold),
        keep_mask_var=keep_var.astype(bool).tolist(),
        keep_mask_corr=keep_corr.astype(bool).tolist(),
    )
    joblib.dump(asdict(arts), os.path.join(artifacts_dir, "numeric_artifacts.joblib"))

    return X.astype(np.float32, copy=False), arts


def transform_numeric(
    df_split: pd.DataFrame,
    artifacts_dir: str = "artifacts/numeric"
) -> Tuple[np.ndarray, Dict]:
    """
    Apply TRAIN-fitted numeric pipeline to any split (VAL/TEST):
      - select original numeric columns, same order
      - log1p on columns flagged non-negative
      - apply stored scaler
      - apply stored masks (zero-variance + corr) in order
    Returns (X_float32, artifacts_dict_loaded).
    """
    arts_dict = joblib.load(os.path.join(artifacts_dir, "numeric_artifacts.joblib"))
    scaler = joblib.load(arts_dict["scaler_state_path"])

    num_cols_original = arts_dict["num_cols_original"]
    nonneg_mask = np.array(arts_dict["nonneg_mask"], dtype=bool)
    keep_var = np.array(arts_dict["keep_mask_var"], dtype=bool)
    keep_corr = np.array(arts_dict["keep_mask_corr"], dtype=bool)

    # build matrix in original numeric order
    X = df_split[num_cols_original].to_numpy(dtype=np.float64, copy=True)
    X = _apply_log1p_where_nonneg(X, nonneg_mask)
    X = _transform_with_scaler(X, scaler)
    # apply masks
    X = X[:, keep_var]
    X = X[:, keep_corr]
    return X.astype(np.float32, copy=False), arts_dict

""" 
How to use in your jupyter notebook or script:
from feature_numeric import fit_numeric_transform, transform_numeric

# 0) Columns to EXCLUDE here because they’re numeric-coded categoricals:
numeric_cats = [
    "PROTOCOL", "L7_PROTO",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE", "DNS_QUERY_ID",
    "FTP_COMMAND_RET_CODE",
    # any other *_TYPE / *_ID style columns you want one-hot later
]

# 1) TRAIN: fit numeric pipeline (log1p, scaler, zero-variance drop, optional Spearman prune)
Xnum_train, num_arts = fit_numeric_transform(
    df_train,
    exclude_numeric_categoricals=numeric_cats,
    scaler_type="standard",           # or "robust" if outliers are extreme
    apply_corr_prune=True,
    corr_threshold=0.995,
    artifacts_dir="artifacts/numeric"
)
print("[numeric] train shape:", Xnum_train.shape)

# 2) VAL/TEST: apply frozen transforms
Xnum_val, _  = transform_numeric(df_val,  artifacts_dir="artifacts/numeric")
Xnum_test, _ = transform_numeric(df_test, artifacts_dir="artifacts/numeric")
print("[numeric] val/test shapes:", Xnum_val.shape, Xnum_test.shape)

# Later you will handle categoricals (including numeric-coded ones) with OneHotEncoder,
# then horizontally stack: X_edge = np.hstack([Xnum, Xcat_dense_or_csr.toarray()]) """