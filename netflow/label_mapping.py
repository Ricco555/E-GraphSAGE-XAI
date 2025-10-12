# label_mapping.py
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def _clean_label_series(y: pd.Series) -> pd.Series:
    """Normalize label strings: strip whitespace; map empty strings to NA."""
    s = y.astype("string")  # preserves NA
    s = s.str.strip()
    s = s.mask(s == "", other=pd.NA)
    return s


def fit_label_map(
    y_train: pd.Series,
    order: str = "alpha",
    explicit_classes: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Build a deterministic string -> int mapping from TRAIN labels only.

    Parameters
    ----------
    y_train : pd.Series  (string-like)
    order   : {'alpha','freq','given'}
        alpha : alphabetical order (default, fully deterministic)
        freq  : descending frequency on train (ties broken alphabetically)
        given : use explicit_classes order (must pass explicit_classes)
    explicit_classes : list[str], required if order='given'

    Returns
    -------
    label2id : dict[str,int]
    """
    y = _clean_label_series(y_train).dropna()
    if order == "given":
        if not explicit_classes:
            raise ValueError("explicit_classes must be provided when order='given'.")
        classes = list(dict.fromkeys(explicit_classes))  # de-dup, keep order
    elif order == "freq":
        vc = y.value_counts(dropna=False)
        # sort by (-count, label) for stable order
        classes = sorted(vc.index.tolist(), key=lambda c: (-vc[c], c))
    else:
        # alpha
        classes = sorted(y.unique().tolist())

    label2id = {lbl: i for i, lbl in enumerate(classes)}
    return label2id


def transform_labels(y: pd.Series, label2id: Dict[str, int], unknown_policy: str = "error") -> np.ndarray:
    """
    Map labels to int IDs with a chosen policy for unseen labels at transform time.

    unknown_policy : {'error','ignore','map_to_last'}
        error      : raise if an unseen label is encountered.
        ignore     : leave unseen labels as -1 (you can filter later).
        map_to_last: map any unseen label to a special class ID = len(label2id).
                     (Useful only if you trained with that extra class.)
    """
    y_clean = _clean_label_series(y)
    ids = np.full(shape=len(y_clean), fill_value=-1, dtype=np.int64)

    for lbl, idx in label2id.items():
        mask = (y_clean == lbl)
        if mask.any():
            ids[mask.to_numpy()] = idx

    unseen_mask = (ids == -1) & y_clean.notna().to_numpy()
    if unseen_mask.any():
        unseen_vals = pd.unique(y_clean[unseen_mask])
        if unknown_policy == "error":
            raise KeyError(f"Unseen labels at transform time: {list(unseen_vals)[:10]} "
                           f"(total {len(unseen_vals)})")
        elif unknown_policy == "map_to_last":
            ids[unseen_mask] = len(label2id)
        elif unknown_policy == "ignore":
            # leave as -1
            pass
        else:
            raise ValueError(f"unknown_policy '{unknown_policy}' not recognized.")

    # Treat NA labels as -1 consistently
    ids[(y_clean.isna()).to_numpy()] = -1
    return ids


def inverse_transform(y_ids: np.ndarray, label2id: Dict[str, int]) -> List[Optional[str]]:
    """Map int IDs back to strings. -1 maps to None."""
    id2label = {v: k for k, v in label2id.items()}
    out: List[Optional[str]] = []
    for v in y_ids.tolist():
        out.append(id2label.get(int(v), None) if v != -1 else None)
    return out


def class_weights_from_train(y_train_ids: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """
    Balanced class weights suitable for torch.nn.CrossEntropyLoss(weight=...).
    Ignores -1 entries if present.
    """
    y = y_train_ids[y_train_ids >= 0]
    if num_classes is None:
        num_classes = int(y.max()) + 1 if y.size else 0
    if num_classes == 0:
        return np.array([], dtype=np.float32)

    classes = np.arange(num_classes, dtype=np.int64)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return w.astype(np.float32)


def save_label_map(path: str, label2id: Dict[str, int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)


def load_label_map(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

""" 
# How to use in Jupyter notebook / pipeline
from label_mapping import fit_label_map, transform_labels, save_label_map, load_label_map, class_weights_from_train

# 1) Fit mapping on TRAIN ONLY
label2id = fit_label_map(df_train["Attack"], order="alpha")  # or order="freq"
save_label_map("artifacts/label_map.json", label2id)

# 2) Apply mapping to all splits (consistent)
label2id = load_label_map("artifacts/label_map.json")
y_train = transform_labels(df_train["Attack"], label2id)  # np.int64
y_val   = transform_labels(df_val["Attack"], label2id)
y_test  = transform_labels(df_test["Attack"], label2id)

# (Optional) sanity check: ensure no -1 slipped in
assert (y_train >= 0).all(), "Train split has unseen/missing labels."

# 3) Class weights for loss
weights = class_weights_from_train(y_train, num_classes=len(label2id))
# In torch:
# criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device)) """