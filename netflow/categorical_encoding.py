# feature_categorical.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

# ---------------------------
# Port bucketing (optional)
# ---------------------------

def bucket_ports(series: pd.Series, scheme: str = "iana") -> pd.Series:
    """
    Bucket L4 ports into coarse categories to avoid huge cardinality.
    scheme='iana':
        0        -> "port_0"
        1..1023  -> "well_known"
        1024..49151 -> "registered"
        49152..65535 -> "ephemeral"
        NA       -> NA
    """
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    out = out.mask(s.isna(), other=pd.NA)
    out = out.fillna(pd.NA)

    m = ~s.isna()
    if scheme == "iana":
        out.loc[m & (s == 0)] = "port_0"
        out.loc[m & (s >= 1) & (s <= 1023)] = "well_known"
        out.loc[m & (s >= 1024) & (s <= 49151)] = "registered"
        out.loc[m & (s >= 49152) & (s <= 65535)] = "ephemeral"
    else:
        raise ValueError("Unsupported scheme for bucket_ports.")
    return out.astype("string")


# ---------------------------
# Artifacts
# ---------------------------

@dataclass(frozen=True)
class CatArtifacts:
    cat_cols: List[str]                   # columns used as categorical inputs (after any bucketing)
    rare_token: str                       # token used for collapsed categories
    rare_applied: Dict[str, bool]         # per column, whether rare collapsing was used
    keep_sets: Dict[str, List[str]]       # per column, categories kept pre-encoder (strings)
    encoder_path: str                     # joblib path for OneHotEncoder
    feature_names: List[str]              # encoder.get_feature_names_out(cat_cols)
    output_dim: int                       # total OHE dimension


# ---------------------------
# Core utilities
# ---------------------------

def _as_string_series(s: pd.Series) -> pd.Series:
    """Normalize to pandas string dtype, preserve NAs, strip whitespace."""
    ss = s.astype("string")
    return ss.str.strip()


def _apply_rare_collapse(
    df_cat: pd.DataFrame,
    min_freq: Optional[int],
    top_k: Optional[int],
    rare_token: str = "__RARE__"
) -> Tuple[pd.DataFrame, Dict[str, bool], Dict[str, set]]:
    """
    Collapse infrequent categories to rare_token.
    - If top_k is set, keep only top_k most frequent categories; others -> rare_token.
    - Then apply min_freq threshold: any remaining category with count < min_freq -> rare_token.
    Returns:
      df_out, rare_applied (per col), keep_sets (per col)
    """
    rare_applied: Dict[str, bool] = {}
    keep_sets: Dict[str, set] = {}
    out = df_cat.copy()

    for c in df_cat.columns:
        vc = df_cat[c].value_counts(dropna=True)
        keep = set(vc.index.tolist())

        changed = False
        if top_k is not None and top_k > 0 and len(keep) > top_k:
            keep = set([k for k, _ in vc.sort_values(ascending=False).head(top_k).items()])
            changed = True

        if min_freq is not None and min_freq > 1:
            low = set(vc[vc < min_freq].index.tolist())
            if low:
                keep = keep - low
                changed = True

        keep_sets[c] = keep
        rare_applied[c] = changed

        if changed:
            out[c] = out[c].where(out[c].isin(keep) | out[c].isna(), other=rare_token)

    return out, rare_applied, keep_sets


# ---------------------------
# Public API
# ---------------------------

def fit_categorical_transform(
    df_train: pd.DataFrame,
    cat_cols: Iterable[str],
    *,
    use_port_buckets: bool = True,
    src_port_col: str = "L4_SRC_PORT",
    dst_port_col: str = "L4_DST_PORT",
    rare_token: str = "__RARE__",
    min_freq: Optional[int] = None,
    top_k: Optional[int] = None,
    artifacts_dir: str = "artifacts/categorical",
) -> Tuple[sparse.csr_matrix, CatArtifacts]:
    """
    Fit a categorical pipeline on TRAIN ONLY:
      - Optionally bucket ports to reduce cardinality.
      - Normalize to string dtype, strip whitespace.
      - Optionally collapse rare categories to `rare_token` (by `top_k` and/or `min_freq`).
      - Fit OneHotEncoder(sparse_output=True, handle_unknown='ignore', dtype=float32).
      - Persist encoder + metadata; return TRAIN CSR matrix.

    Returns (Xcat_train_csr, CatArtifacts)
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1) Prepare dataframe with chosen categorical columns (copy)
    df_cat = df_train[list(cat_cols)].copy()

    # 2) Optional port bucketing
    effective_cols: List[str] = list(cat_cols)
    if use_port_buckets:
        for col in (src_port_col, dst_port_col):
            if col in df_cat.columns:
                df_cat[col] = bucket_ports(df_cat[col], scheme="iana")

    # 3) Normalize to string dtype
    for c in df_cat.columns:
        df_cat[c] = _as_string_series(df_cat[c])

    # 4) Collapse rare categories (train only)
    df_collapsed, rare_applied, keep_sets = _apply_rare_collapse(
        df_cat,
        min_freq=min_freq,
        top_k=top_k,
        rare_token=rare_token
    )

    # Ensure rare_token exists in columns where we applied collapsing (so encoder learns a column for it)
    for c in df_collapsed.columns:
        if rare_applied.get(c, False):
            # If no row currently has rare_token (possible if top_k only, but some kept),
            # we force-inject one row via fillna trick so OHE learns the column; then revert.
            if not (df_collapsed[c] == rare_token).any():
                # Temporarily assign rare_token to one NA to register the column
                na_pos = df_collapsed[c].isna()
                if na_pos.any():
                    ix = na_pos.idxmax()
                    df_collapsed.loc[ix, c] = rare_token

    # 5) Fit OneHotEncoder
    ohe = OneHotEncoder(dtype=np.float32, sparse_output=True, handle_unknown="ignore")
    Xcat_train = ohe.fit_transform(df_collapsed)

    # 6) Persist artifacts
    enc_path = os.path.join(artifacts_dir, "onehot.joblib")
    joblib.dump(ohe, enc_path)

    art = CatArtifacts(
        cat_cols=df_collapsed.columns.tolist(),
        rare_token=rare_token,
        rare_applied={k: bool(v) for k, v in rare_applied.items()},
        keep_sets={k: sorted(list(v)) for k, v in keep_sets.items()},
        encoder_path=enc_path,
        feature_names=ohe.get_feature_names_out(df_collapsed.columns).tolist(),
        output_dim=int(Xcat_train.shape[1]),
    )
    with open(os.path.join(artifacts_dir, "categorical_artifacts.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(art), f, ensure_ascii=False, indent=2)

    return Xcat_train.tocsr(), art


def transform_categorical(
    df_split: pd.DataFrame,
    artifacts_dir: str = "artifacts/categorical",
) -> sparse.csr_matrix:
    """
    Apply TRAIN-fitted categorical pipeline to any split (VAL/TEST):
      - Rebuild the same categorical dataframe in the same column order.
      - Apply the same port bucketing choice (inferred from artifacts' cat_cols names).
      - Normalize to string dtype.
      - Map unseen categories to `rare_token` **if** that column used rare collapsing in TRAIN; else leave as-is.
      - OneHotEncoder.transform (sparse CSR, float32).
    """
    with open(os.path.join(artifacts_dir, "categorical_artifacts.json"), "r", encoding="utf-8") as f:
        arts = json.load(f)
    ohe: OneHotEncoder = joblib.load(arts["encoder_path"])

    cat_cols = arts["cat_cols"]
    rare_token = arts["rare_token"]
    rare_applied = arts["rare_applied"]
    keep_sets = {k: set(v) for k, v in arts["keep_sets"].items()}

    # Recreate categorical frame in the same order
    df_cat = df_split[cat_cols].copy()

    # If you renamed bucketed columns, they’re already in cat_cols; otherwise, just normalize
    for c in df_cat.columns:
        df_cat[c] = _as_string_series(df_cat[c])

    # Map categories not in keep_sets to rare_token, but only for columns that used rare collapsing
    for c in df_cat.columns:
        if rare_applied.get(c, False):
            ks = keep_sets.get(c, set())
            df_cat[c] = df_cat[c].where(df_cat[c].isna() | df_cat[c].isin(ks), other=rare_token)

    Xcat = ohe.transform(df_cat)
    return Xcat.tocsr()
