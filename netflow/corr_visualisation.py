# numeric_corr_viz.py
from __future__ import annotations

import os
from typing import Iterable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _infer_nonneg_mask(df_num: pd.DataFrame, nonneg_frac: float = 0.995) -> np.ndarray:
    arr = df_num.to_numpy(dtype="float64", copy=False)
    ge0 = (arr >= 0).mean(axis=0)
    return (ge0 >= nonneg_frac)


def _log1p_where_nonneg(arr: np.ndarray, nonneg_mask: np.ndarray) -> np.ndarray:
    out = arr.copy()
    if nonneg_mask.any():
        cols = np.where(nonneg_mask)[0]
        out[:, cols] = np.log1p(np.maximum(out[:, cols], 0.0))
    return out


def _spearman_abs_corr(df_num_log: pd.DataFrame) -> pd.DataFrame:
    # Spearman via pandas (rank + Pearson internally)
    return df_num_log.corr(method="spearman").abs()


def _plot_heatmap(corr: pd.DataFrame, title: str, out_png: str) -> None:
    # Auto-size figure based on matrix size (kept within reasonable bounds)
    h = min(14, 0.18 * corr.shape[0] + 3)
    w = min(18, 0.18 * corr.shape[1] + 4)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(corr.values, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(corr.shape[1]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _top_pairs(corr_abs: pd.DataFrame, k: int = 50) -> List[Tuple[float, str, str]]:
    cols = corr_abs.columns.tolist()
    d = len(cols)
    upper = corr_abs.where(np.triu(np.ones_like(corr_abs, dtype=bool), k=1))
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            v = upper.iat[i, j]
            if pd.notna(v):
                pairs.append((float(v), cols[i], cols[j]))
    pairs.sort(reverse=True, key=lambda t: t[0])
    return pairs[:k]


def plot_numeric_corr_heatmap(
    df_train: pd.DataFrame,
    exclude_numeric_categoricals: Iterable[str],
    out_dir: str = "artifacts/corr",
    threshold: float = 0.995,
    max_features: int = 150,
    nonneg_frac: float = 0.995,
    topk_pairs: int = 100,
    filename_prefix: str = "spearman_corr",
) -> Dict:
    """
    Produce and save a Spearman |rho| heatmap for numeric features (pre-pruning).
    - Excludes numeric-coded categoricals.
    - Applies log1p to columns inferred as (mostly) nonnegative.
    - If too many features, plots subset that participate in pairs with |rho| >= threshold.
    Saves:
      - PNG heatmap under out_dir
      - CSV of top-k pairs (value, featA, featB) under out_dir

    Returns a dict with basic stats (paths, counts).
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) numeric-only selection (exclude numeric-coded categoricals)
    drop_cols = set(exclude_numeric_categoricals)
    df_num = df_train.select_dtypes(include="number").drop(columns=[c for c in drop_cols if c in df_train.columns],
                                                          errors="ignore")
    if df_num.shape[1] == 0:
        raise ValueError("No numeric features to visualize after excluding numeric-coded categoricals.")

    # 2) stabilization: log1p for (mostly) nonnegative columns
    arr = df_num.to_numpy(dtype="float64", copy=True)
    nonneg_mask = _infer_nonneg_mask(df_num, nonneg_frac=nonneg_frac)
    arr = _log1p_where_nonneg(arr, nonneg_mask)
    df_log = pd.DataFrame(arr, columns=df_num.columns, index=df_num.index)

    # 3) Spearman |rho|
    corr_abs = _spearman_abs_corr(df_log)

    # 4) Decide subset to plot
    cols = corr_abs.columns.tolist()
    d = len(cols)
    png_path = os.path.join(out_dir, f"{filename_prefix}_full.png")
    csv_path = os.path.join(out_dir, f"{filename_prefix}_top_pairs.csv")

    if d <= max_features:
        _plot_heatmap(corr_abs, title=f"Spearman |ρ| (numeric pre-pruning), features={d}", out_png=png_path)
        plotted = cols
    else:
        # pick only features that participate in high-corr pairs
        upper = corr_abs.where(np.triu(np.ones_like(corr_abs, dtype=bool), k=1))
        involved = set()
        for i, c in enumerate(cols):
            high = upper.iloc[i, (i + 1):] >= threshold
            if high.any():
                involved.add(c)
                for j, flag in enumerate(high.to_numpy(), start=i + 1):
                    if flag:
                        involved.add(cols[j])
        if not involved:
            # no pairs above threshold; plot first max_features columns for orientation
            subset = cols[:max_features]
            _plot_heatmap(corr_abs.loc[subset, subset],
                          title=f"Spearman |ρ| (subset; no pairs ≥ {threshold})",
                          out_png=png_path)
            plotted = subset
        else:
            subset = sorted(involved)
            _plot_heatmap(corr_abs.loc[subset, subset],
                          title=f"Spearman |ρ| (highly-correlated subset ≥ {threshold}); features={len(subset)}",
                          out_png=png_path)
            plotted = subset

    # 5) Save top-k pairs to CSV (full-space ranking)
    top_pairs = _top_pairs(corr_abs, k=topk_pairs)
    pd.DataFrame(top_pairs, columns=["abs_rho", "feat_a", "feat_b"]).to_csv(csv_path, index=False)

    return {
        "n_numeric": int(df_num.shape[1]),
        "n_plotted": int(len(plotted)),
        "threshold": float(threshold),
        "nonneg_frac": float(nonneg_frac),
        "png_path": png_path,
        "pairs_csv": csv_path,
    }
