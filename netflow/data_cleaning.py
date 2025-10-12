# data_cleaning.py
# NF-UNSW-NB15 ingest & cleaning pipeline.
# Steps:
# 1) Drop rows missing required identifiers (IPs or ports).
# 2) Detect ±inf and existing missing values (per column), convert ±inf -> NA.
# 3) Fill NA to 0 for numeric throughput-like columns.
#
# Usage (in Jupyter notebook):
#   from data_cleaning import clean_nfunsw_nb15
#   df_clean = clean_nfunsw_nb15(df_raw)

from __future__ import annotations

import sys
from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

REQUIRED_ID_COLS = ("IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT")


def _ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _drop_missing_ids(df: pd.DataFrame,
                      ip_cols: Tuple[str, str] = ("IPV4_SRC_ADDR", "IPV4_DST_ADDR"),
                      port_cols: Tuple[str, str] = ("L4_SRC_PORT", "L4_DST_PORT")) -> pd.DataFrame:
    before = len(df)

    # Treat empty strings in IP columns as missing as well.
    miss_ip = pd.Series(False, index=df.index)
    for c in ip_cols:
        if c not in df.columns:
            raise KeyError(f"Expected IP column '{c}' not found.")
        col = df[c]
        if is_string_dtype(col.dtype):
            miss_ip = miss_ip | col.isna() | (col.str.len() == 0)
        else:
            miss_ip = miss_ip | col.isna()

    miss_port = pd.Series(False, index=df.index)
    for c in port_cols:
        if c not in df.columns:
            raise KeyError(f"Expected port column '{c}' not found.")
        # Ports can legitimately be 0 for some protocols; only drop true NA.
        miss_port = miss_port | df[c].isna()

    to_drop = miss_ip | miss_port
    dropped = int(to_drop.sum())
    df2 = df.loc[~to_drop].copy()

    print(f"[clean] Step 1/3: dropped rows missing IP/ports: {dropped} (from {before} -> {len(df2)})")
    return df2


def _mark_inf_as_na(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Convert ±inf in float columns to NA (pd.NA), conserving PyArrow dtypes where possible.
    Returns (df_out, columns_with_inf_or_na_found)
    """
    df2 = df.copy()
    cols_flagged = []

    for c in df2.columns:
        s = df2[c]

        # Only float columns can contain ±inf. Integer columns (including PyArrow ints) cannot.
        if is_float_dtype(s.dtype):
            # Use a float64 view to detect non-finite values (works for PyArrow floats too).
            arr = s.to_numpy(dtype="float64", copy=False)
            nonfinite_mask = ~np.isfinite(arr)  # True where inf, -inf, or NaN
            if nonfinite_mask.any():
                cols_flagged.append(c)
                # Replace ±inf with pd.NA; leave existing NaN as-is (still considered missing).
                inf_mask = np.isinf(arr)
                if inf_mask.any():
                    # Assign pd.NA; pandas with pyarrow floats supports NA scalars.
                    df2.loc[inf_mask, c] = pd.NA
        # For completeness, we don't expect string/object columns to have inf.
        # Integer columns won't have inf by construction.

    # Also report columns that already had NA (not just inf) to inform Step 3.
    for c in df2.columns:
        if df2[c].isna().any() and c not in cols_flagged:
            cols_flagged.append(c)

    if cols_flagged:
        cols_flagged_sorted = sorted(set(cols_flagged))
        print(f"[clean] Step 2/3: columns with ±inf and/or missing values detected:\n"
              f"        {cols_flagged_sorted}")
    else:
        print("[clean] Step 2/3: no ±inf or missing values detected in any column.")

    return df2, sorted(set(cols_flagged))


def _fill_numeric_na_with_zero(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Fill NA in numeric columns with 0. Leaves non-numeric columns unchanged.
    Works with PyArrow-backed numeric dtypes.
    """
    df2 = df.copy()
    num_cols = [c for c in df2.columns if is_numeric_dtype(df2[c].dtype)]
    if not num_cols:
        print("[clean] Step 3/3: no numeric columns found to fill.")
        return df2, 0

    # Count NA before filling (numeric columns only).
    na_before = sum(int(df2[c].isna().sum()) for c in num_cols)
    if na_before == 0:
        print("[clean] Step 3/3: no missing values in numeric columns; nothing to fill.")
        return df2, 0

    df2[num_cols] = df2[num_cols].fillna(0)
    print(f"[clean] Step 3/3: filled {na_before} missing numeric values with 0.")
    return df2, na_before


def clean_nfunsw_nb15(df: pd.DataFrame,
                      required_id_cols: Iterable[str] = REQUIRED_ID_COLS) -> pd.DataFrame:
    """
    Perform the three-step cleaning pipeline on an NF-UNSW-NB15 DataFrame:
      1) Drop rows missing IPs or ports (strict NA; IP empty-string is also dropped).
      2) Convert ±inf to NA; report columns containing ±inf/NA.
      3) Fill NA with 0 for numeric columns (throughput-like metrics).

    Returns a cleaned DataFrame, preserving PyArrow dtypes where possible.
    """
    _ensure_required_columns(df, required_id_cols)

    # Step 1
    df1 = _drop_missing_ids(df, ip_cols=(required_id_cols[0], required_id_cols[1]),
                               port_cols=(required_id_cols[2], required_id_cols[3]))

    # Step 2
    df2, _ = _mark_inf_as_na(df1)

    # Step 3
    df3, _ = _fill_numeric_na_with_zero(df2)

    print(f"[clean] Done. Final shape: {df3.shape}")
    return df3


# Optional: simple CLI for batch cleaning a CSV/Parquet file.
# Example:
#   python data_cleaning.py input.parquet output_clean.parquet
#   python data_cleaning.py input.csv output_clean.csv
def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path, engine="pyarrow")
    if path.lower().endswith(".csv"):
        # Low-memory False to keep dtypes stable with pyarrow backend (pandas>=2.0).
        return pd.read_csv(path, low_memory=False)
    raise ValueError("Unsupported file type. Use .parquet or .csv")


def _to_any(df: pd.DataFrame, path: str) -> None:
    if path.lower().endswith(".parquet"):
        df.to_parquet(path, engine="pyarrow", index=False)
    elif path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError("Unsupported file type. Use .parquet or .csv")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaning.py <input.(parquet|csv)> <output.(parquet|csv)>")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    df_in = _read_any(inp)
    df_out = clean_nfunsw_nb15(df_in)
    _to_any(df_out, outp)
    print(f"[clean] Saved: {outp}")
