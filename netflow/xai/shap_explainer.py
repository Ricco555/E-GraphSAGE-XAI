"""SHAP-based feature explanations and structural neighbor-masking XAI.

Replaces the top-level xai.py.  All feature access goes through the
GraphBolt OnDiskDataset (no legacy feature_store/ directories).

Simplified return contracts vs the old xai.py:
  - compute_pair_embedding: direct call to model.encode, no try/except shape
    heuristics.  EdgeGraphSAGE.encode(return_src=True) always returns
    (h_src_init, h_dst) cleanly.
  - local_shap_for_edge: always returns shap_vals as np.ndarray (d,) — the
    caller no longer needs shape-dispatch logic.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import dgl
from scipy import sparse


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

def load_feature_names(artifacts_dir: str) -> tuple[list[str], int, int]:
    """Load edge feature column names from pipeline artifacts.

    Reads:
      <artifacts_dir>/numeric/numeric_artifacts.joblib  — numeric column list
      <artifacts_dir>/categorical/categorical_artifacts.json — OHE feature names

    Returns (feature_names, d_num, d_cat).
    Falls back to generic names if artifacts are missing or mismatched.
    """
    import joblib

    num_path = os.path.join(artifacts_dir, "numeric", "numeric_artifacts.joblib")
    cat_path = os.path.join(artifacts_dir, "categorical", "categorical_artifacts.json")

    # Numeric names.
    num_names: list[str] | None = None
    if os.path.exists(num_path):
        try:
            arts = joblib.load(num_path)
            for key in ("num_cols_final", "numeric_cols_final", "num_cols", "numeric_cols"):
                if isinstance(arts.get(key), (list, tuple)):
                    num_names = list(arts[key])
                    break
        except Exception:
            num_names = None

    # Categorical names.
    cat_names: list[str] | None = None
    if os.path.exists(cat_path):
        try:
            with open(cat_path, "r", encoding="utf-8") as f:
                cat_meta = json.load(f)
            fnh = cat_meta.get("feature_names")
            if isinstance(fnh, list):
                cat_names = list(fnh)
        except Exception:
            cat_names = None

    d_num = len(num_names) if num_names else 0
    d_cat = len(cat_names) if cat_names else 0

    if num_names is None:
        num_names = [f"num_{i}" for i in range(d_num)]
    if cat_names is None:
        cat_names = [f"cat_{i}" for i in range(d_cat)]

    return num_names + cat_names, d_num, d_cat


# ---------------------------------------------------------------------------
# Pair embedding (simplified)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_pair_embedding(
    model: nn.Module,
    blocks: list,
    x_nodes=None,
    return_src: bool = False,
):
    """Run model.encode and return node embeddings.

    With return_src=False  → returns h_dst (N_dst, hidden)
    With return_src=True   → returns (h_src_init, h_dst)

    EdgeGraphSAGE.encode always handles both forms — no defensive fallback needed.
    """
    return model.encode(blocks, x_nodes, return_src=return_src)


# ---------------------------------------------------------------------------
# Edge-head wrapper for KernelSHAP
# ---------------------------------------------------------------------------

def make_edge_head_fn(
    model: nn.Module,
    he_fixed: np.ndarray,
    device: str = "cpu",
    return_prob: bool = True,
    target_class: int | None = None,
):
    """Wrap the per-edge MLP as a SHAP-compatible numpy function.

    Freezes the pair embedding [h_src ‖ h_dst] and treats e_feat as the
    sole variable input.  Returns probabilities (or logits if return_prob=False).

    If target_class is None  → output shape (B, C).
    If target_class is set   → output shape (B,).
    """
    he_t = torch.from_numpy(he_fixed.astype(np.float32)).unsqueeze(0).to(device)

    def _f(X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        he  = he_t.expand(X_t.size(0), -1)
        with torch.no_grad():
            logits = model.head(torch.cat([he, X_t], dim=1))
            if return_prob:
                out = torch.softmax(logits, dim=1)
            else:
                out = logits
        out_np = out.cpu().numpy()
        return out_np if target_class is None else out_np[:, target_class]

    return _f


# ---------------------------------------------------------------------------
# Feature reading helper
# ---------------------------------------------------------------------------

def _read_edge_features(
    eids: np.ndarray,
    feature,
    x_cat_csr: sparse.csr_matrix,
) -> np.ndarray:
    """Return (N, d_num+d_cat) float32 array for the given LOCAL edge IDs."""
    ids_t = torch.from_numpy(eids.astype(np.int64))
    x_num = feature.read("edge", None, "x_num", ids=ids_t).numpy().astype(np.float32)
    x_cat = x_cat_csr[eids].toarray().astype(np.float32)
    return np.concatenate([x_num, x_cat], axis=1)


# ---------------------------------------------------------------------------
# Variable-level grouping for grouped KernelSHAP
# ---------------------------------------------------------------------------

def load_cat_variable_names(artifacts_dir: str) -> list[str]:
    """Return original categorical variable names (before OHE) from artifacts.

    Reads cat_cols from <artifacts_dir>/categorical/categorical_artifacts.json.
    Returns e.g. ['PROTOCOL', 'L7_PROTO', 'ICMP_TYPE', ...].
    """
    cat_path = os.path.join(artifacts_dir, "categorical", "categorical_artifacts.json")
    if not os.path.exists(cat_path):
        return []
    with open(cat_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return list(meta.get("cat_cols", []))


def build_variable_groups(
    feature_names: list[str],
    d_num: int,
    cat_cols: list[str],
) -> list[dict]:
    """Build a variable-level group structure for grouped KernelSHAP.

    Collapses the ~601-dim OHE feature space into one slot per original
    variable (38 numeric + len(cat_cols) categorical = ~45 total).

    OHE column names follow the convention ``{VARIABLE}_{VALUE}`` where
    VARIABLE is an entry in cat_cols.  Prefix matching is used so that
    multi-word variable names (e.g. ICMP_IPV4_TYPE) are handled correctly.

    Returns a list of dicts:
      {"name": str, "type": "numeric"|"categorical", "indices": [col_idx,...]}
    Indices are absolute positions in the full (d_num + d_cat) feature vector.
    """
    # Sort by length descending so longer prefixes match before shorter ones.
    sorted_vars = sorted(cat_cols, key=len, reverse=True)

    def _detect_var(ohe_name: str) -> str | None:
        for var in sorted_vars:
            if ohe_name.startswith(var + "_"):
                return var
        return None

    groups: list[dict] = []
    for i in range(d_num):
        groups.append({"name": feature_names[i], "type": "numeric", "indices": [i]})

    cat_order: dict[str, None] = {}
    cat_group2idx: dict[str, list[int]] = defaultdict(list)
    for j in range(d_num, len(feature_names)):
        name = feature_names[j]
        var  = _detect_var(name)
        if var is None:
            # Fallback: treat as its own group (should not happen with cat_cols).
            var = name
        if var not in cat_order:
            cat_order[var] = None
        cat_group2idx[var].append(j)

    for var in cat_order:
        groups.append({
            "name":    var,
            "type":    "categorical",
            "indices": sorted(cat_group2idx[var]),
        })
    return groups


def encode_to_variable_space(
    X_full: np.ndarray,
    variable_groups: list[dict],
) -> np.ndarray:
    """Encode full feature matrix (N, d_full) into variable space (N, n_vars).

    Numeric slots: copy value directly.
    Categorical slots: argmax of the OHE block → integer category index (float32).
    All-zero OHE rows (unknown / unseen category) get sentinel = len(group.indices)
    so that reconstruction leaves them as all-zero.
    """
    N = X_full.shape[0]
    X_var = np.zeros((N, len(variable_groups)), dtype=np.float32)
    for j, grp in enumerate(variable_groups):
        if grp["type"] == "numeric":
            X_var[:, j] = X_full[:, grp["indices"][0]]
        else:
            block     = X_full[:, grp["indices"]]
            X_var[:, j] = block.argmax(axis=1).astype(np.float32)
            zero_mask = block.sum(axis=1) == 0
            if zero_mask.any():
                X_var[zero_mask, j] = float(len(grp["indices"]))
    return X_var


def make_grouped_edge_head_fn(
    model: nn.Module,
    he_fixed: np.ndarray,
    variable_groups: list[dict],
    device: str = "cpu",
    return_prob: bool = True,
    target_class: int | None = None,
):
    """Edge-head wrapper that operates in variable space (~45 dims).

    Accepts X_var (N, n_vars) where categorical slots are integer category
    indices (float32).  Reconstructs the full OHE feature vector via vectorised
    numpy scatter before calling model.head.

    When KernelSHAP sets a categorical slot to its background value (= another
    row's category index), the entire OHE block is replaced atomically, so each
    categorical variable counts as one coalition member regardless of its
    one-hot cardinality.
    """
    d_full = sum(len(grp["indices"]) for grp in variable_groups)
    he_t   = torch.from_numpy(he_fixed.astype(np.float32)).unsqueeze(0).to(device)

    def _reconstruct(X_var: np.ndarray) -> np.ndarray:
        N2     = X_var.shape[0]
        X_out  = np.zeros((N2, d_full), dtype=np.float32)
        for j, grp in enumerate(variable_groups):
            if grp["type"] == "numeric":
                X_out[:, grp["indices"][0]] = X_var[:, j]
            else:
                idxs    = np.asarray(grp["indices"], dtype=np.int32)
                n_cats  = len(idxs)
                cat_int = np.round(X_var[:, j]).astype(np.int32)
                valid   = (cat_int >= 0) & (cat_int < n_cats)
                if valid.any():
                    rows = np.where(valid)[0]
                    X_out[rows, idxs[cat_int[valid]]] = 1.0
        return X_out

    def _f(X_var: np.ndarray) -> np.ndarray:
        X_full = _reconstruct(np.asarray(X_var, dtype=np.float32))
        X_t    = torch.from_numpy(X_full).to(device)
        he     = he_t.expand(X_t.size(0), -1)
        with torch.no_grad():
            logits = model.head(torch.cat([he, X_t], dim=1))
            out    = torch.softmax(logits, dim=1) if return_prob else logits
        out_np = out.cpu().numpy()
        return out_np if target_class is None else out_np[:, target_class]

    return _f


def local_shap_for_edge_grouped(
    model: nn.Module,
    g: dgl.DGLGraph,
    loader,
    eid_local: int,
    feature,
    x_cat_csr: sparse.csr_matrix,
    bg_eids: np.ndarray,
    variable_groups: list[dict],
    background_size: int = 100,
    target_class: int | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[np.ndarray, list[str], np.ndarray, int, float]:
    """KernelSHAP explanation at the VARIABLE level (~45 dims).

    Each categorical variable's OHE block is treated as a single coalition
    member, collapsing the ~601-dim problem to ~45 dims.  This makes
    KernelSHAP fast and stable and produces directly interpretable attributions
    (e.g. "L7_PROTO contributed X" rather than 33 individual one-hot values).

    Use build_variable_groups() to construct variable_groups.

    Returns
    -------
    shap_vals  : np.ndarray (n_vars,) — SHAP value per variable
    var_names  : list[str]            — variable name for each slot
    x_var      : np.ndarray (n_vars,) — query edge in variable space
    y_true     : int                  — true label
    base_value : float                — SHAP expected value
    """
    model.eval()
    he_fixed    = None
    x_edge_full = None
    y_true      = None

    for _input_nodes, pair_graph, blocks in loader:
        local_eids = pair_graph.edata[dgl.EID].cpu().numpy().astype(np.int64)
        if eid_local not in local_eids:
            continue

        blocks     = [b.to(device) for b in blocks]
        pair_graph = pair_graph.to(device)

        h_dst = compute_pair_embedding(model, blocks, x_nodes=None)
        src_pos, dst_pos = pair_graph.edges(form="uv")
        pos      = int(np.where(local_eids == eid_local)[0][0])
        he_fixed = torch.cat([h_dst[src_pos[pos]], h_dst[dst_pos[pos]]], dim=0).cpu().numpy()

        x_edge_full = _read_edge_features(local_eids[pos : pos + 1], feature, x_cat_csr)[0]
        y_true      = int(g.edata["y"][eid_local].item())
        break

    if he_fixed is None:
        raise RuntimeError(f"eid_local={eid_local} not found in any loader batch.")

    rng    = np.random.default_rng(seed)
    n_bg   = min(background_size, len(bg_eids))
    bg_idx = rng.choice(len(bg_eids), size=n_bg, replace=False)
    bg_X_var = encode_to_variable_space(
        _read_edge_features(bg_eids[bg_idx], feature, x_cat_csr),
        variable_groups,
    )
    x_var = encode_to_variable_space(x_edge_full[None, :], variable_groups)[0]

    f = make_grouped_edge_head_fn(
        model, he_fixed, variable_groups,
        device=device, return_prob=True, target_class=target_class,
    )
    explainer = shap.KernelExplainer(f, bg_X_var, seed=seed)
    sv_raw    = explainer.shap_values(x_var[None, :], nsamples="auto")

    if isinstance(sv_raw, list):
        if target_class is not None and target_class < len(sv_raw):
            sv = np.asarray(sv_raw[target_class]).reshape(-1)
        else:
            probs = f(x_var[None, :])
            pred  = int(np.argmax(probs[0])) if probs.ndim == 2 else 0
            sv    = np.asarray(sv_raw[pred]).reshape(-1)
        base_raw = explainer.expected_value
        base     = float(base_raw[pred] if isinstance(base_raw, (list, np.ndarray))
                         else base_raw)
    else:
        sv   = np.asarray(sv_raw).reshape(-1)
        base = float(np.asarray(explainer.expected_value).flat[0])

    var_names = [grp["name"] for grp in variable_groups]
    return sv, var_names, x_var, y_true, base


# ---------------------------------------------------------------------------
# Local SHAP for a single edge
# ---------------------------------------------------------------------------

def local_shap_for_edge(
    model: nn.Module,
    g: dgl.DGLGraph,
    loader,
    eid_local: int,
    feature,
    x_cat_csr: sparse.csr_matrix,
    bg_eids: np.ndarray,
    background_size: int = 100,
    target_class: int | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """KernelSHAP explanation for a single edge.

    Freezes [h_src ‖ h_dst] for `eid_local` and explains the edge MLP
    f(e_feat) → P(target_class) with KernelSHAP.

    Returns
    -------
    shap_vals : np.ndarray (d,)   — SHAP values for the chosen class
    x_edge    : np.ndarray (d,)   — edge feature vector
    y_true    : int               — true label
    base_value: float             — SHAP expected value
    """
    model.eval()

    he_fixed = None
    x_edge_row = None
    y_true = None

    for _input_nodes, pair_graph, blocks in loader:
        local_eids = pair_graph.edata[dgl.EID].cpu().numpy().astype(np.int64)
        if eid_local not in local_eids:
            continue

        blocks     = [b.to(device) for b in blocks]
        pair_graph = pair_graph.to(device)

        h_dst = compute_pair_embedding(model, blocks, x_nodes=None)
        src_pos, dst_pos = pair_graph.edges(form="uv")
        pos = int(np.where(local_eids == eid_local)[0][0])
        he_fixed = torch.cat([h_dst[src_pos[pos]], h_dst[dst_pos[pos]]], dim=0).cpu().numpy()

        x_edge_row = _read_edge_features(local_eids[pos : pos + 1], feature, x_cat_csr)[0]
        y_true     = int(g.edata["y"][eid_local].item())
        break

    if he_fixed is None:
        raise RuntimeError(f"eid_local={eid_local} not found in any loader batch.")

    # Background: sample from bg_eids.
    rng   = np.random.default_rng(seed)
    n_bg  = min(background_size, len(bg_eids))
    bg_idx = rng.choice(len(bg_eids), size=n_bg, replace=False)
    bg_X  = _read_edge_features(bg_eids[bg_idx], feature, x_cat_csr)

    # Explain.
    f = make_edge_head_fn(model, he_fixed, device=device,
                          return_prob=True, target_class=target_class)
    explainer = shap.KernelExplainer(f, bg_X, seed=seed)
    sv_raw = explainer.shap_values(x_edge_row[None, :], nsamples="auto")

    # Normalise to (d,) regardless of SHAP's output shape.
    if isinstance(sv_raw, list):
        # multi-output: pick target_class or the predicted class
        if target_class is not None and target_class < len(sv_raw):
            sv = np.asarray(sv_raw[target_class]).reshape(-1)
        else:
            probs = f(x_edge_row[None, :])
            pred  = int(np.argmax(probs[0])) if probs.ndim == 2 else 0
            sv = np.asarray(sv_raw[pred]).reshape(-1)
        base_raw = explainer.expected_value
        base = float(base_raw[pred] if isinstance(base_raw, (list, np.ndarray))
                     else base_raw)
    else:
        sv   = np.asarray(sv_raw).reshape(-1)
        base = float(np.asarray(explainer.expected_value).flat[0])

    return sv, x_edge_row, y_true, base


# ---------------------------------------------------------------------------
# Class-level SHAP aggregation
# ---------------------------------------------------------------------------

def sample_edges_by_class(
    g: dgl.DGLGraph,
    n_per_class: int = 200,
    seed: int = 0,
) -> dict[int, list[int]]:
    """Return {class_id: [local_eid, ...]} with at most n_per_class per class."""
    rng = np.random.default_rng(seed)
    y   = g.edata["y"].cpu().numpy()
    out: dict[int, list[int]] = {}
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]
        k   = min(n_per_class, idx.size)
        out[int(c)] = rng.choice(idx, size=k, replace=False).tolist()
    return out


def aggregate_shap_per_class(
    model: nn.Module,
    g: dgl.DGLGraph,
    loader,
    feature,
    x_cat_csr: sparse.csr_matrix,
    bg_eids: np.ndarray,
    n_per_class: int = 200,
    background_size: int = 100,
    device: str = "cpu",
    variable_groups: list[dict] | None = None,
) -> dict[int, np.ndarray]:
    """Mean |SHAP| per feature/variable per class.  Returns {class_id: array}.

    If variable_groups is provided (from build_variable_groups()), runs grouped
    KernelSHAP in the ~45-dim variable space so each categorical variable is
    one coalition member.  Output arrays are then (n_vars,).  Otherwise runs
    in the raw ~601-dim OHE space.
    """
    c2idx = sample_edges_by_class(g, n_per_class=n_per_class)
    result: dict[int, np.ndarray] = {}
    for c, eids in c2idx.items():
        rows = []
        for eid in eids:
            try:
                if variable_groups is not None:
                    sv, *_ = local_shap_for_edge_grouped(
                        model, g, loader, eid, feature, x_cat_csr,
                        bg_eids, variable_groups, background_size,
                        target_class=c, device=device,
                    )
                else:
                    sv, *_ = local_shap_for_edge(
                        model, g, loader, eid, feature, x_cat_csr,
                        bg_eids, background_size, target_class=c, device=device,
                    )
                rows.append(np.abs(sv))
            except Exception:
                continue
        if rows:
            result[c] = np.mean(np.vstack(rows), axis=0)
    return result


def aggregate_signed_shap_per_class(
    model: nn.Module,
    g: dgl.DGLGraph,
    loader,
    feature,
    x_cat_csr: sparse.csr_matrix,
    bg_eids: np.ndarray,
    n_per_class: int = 200,
    background_size: int = 100,
    device: str = "cpu",
    variable_groups: list[dict] | None = None,
) -> dict[int, np.ndarray]:
    """Mean (signed) SHAP per feature/variable per class.  Returns {class_id: array}.

    Pass variable_groups (from build_variable_groups()) to run in the ~45-dim
    variable space instead of the raw ~601-dim OHE space.
    """
    c2idx = sample_edges_by_class(g, n_per_class=n_per_class)
    result: dict[int, np.ndarray] = {}
    for c, eids in c2idx.items():
        rows = []
        for eid in eids:
            try:
                if variable_groups is not None:
                    sv, *_ = local_shap_for_edge_grouped(
                        model, g, loader, eid, feature, x_cat_csr,
                        bg_eids, variable_groups, background_size,
                        target_class=c, device=device,
                    )
                else:
                    sv, *_ = local_shap_for_edge(
                        model, g, loader, eid, feature, x_cat_csr,
                        bg_eids, background_size, target_class=c, device=device,
                    )
                rows.append(sv)
            except Exception:
                continue
        if rows:
            result[c] = np.mean(np.vstack(rows), axis=0)
    return result


def shap_matrix_for_class(
    model: nn.Module,
    g: dgl.DGLGraph,
    loader,
    feature,
    x_cat_csr: sparse.csr_matrix,
    bg_eids: np.ndarray,
    class_id: int,
    n_samples: int = 200,
    background_size: int = 100,
    feature_names: list[str] | None = None,
    device: str = "cpu",
    variable_groups: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect a SHAP matrix for all sampled edges of `class_id`.

    Returns (S, X, feature_names) where S and X are (n, d) float arrays.

    If variable_groups is provided, S and X are in the ~45-dim variable space
    and feature_names are the original variable names — directly usable in
    shap.summary_plot without any post-hoc aggregation.
    """
    rng   = np.random.default_rng(0)
    y     = g.edata["y"].cpu().numpy()
    idx_c = np.nonzero(y == class_id)[0]
    if idx_c.size == 0:
        raise ValueError(f"No edges with class={class_id}.")
    take = min(n_samples, idx_c.size)
    eids = rng.choice(idx_c, size=take, replace=False).astype(int)

    if variable_groups is not None:
        var_names = [grp["name"] for grp in variable_groups]
        feature_names = var_names
        d = len(variable_groups)

    S_rows, X_rows = [], []
    d_ref = None
    for eid in eids:
        try:
            if variable_groups is not None:
                sv, _, x_rep, *_ = local_shap_for_edge_grouped(
                    model, g, loader, int(eid), feature, x_cat_csr,
                    bg_eids, variable_groups, background_size,
                    target_class=class_id, device=device,
                )
            else:
                sv, x_rep, *_ = local_shap_for_edge(
                    model, g, loader, int(eid), feature, x_cat_csr,
                    bg_eids, background_size, target_class=class_id, device=device,
                )
            if d_ref is None:
                d_ref = sv.size
                if variable_groups is None:
                    if feature_names is None or len(feature_names) != d_ref:
                        feature_names = [f"f{i}" for i in range(d_ref)]
            if sv.size == d_ref:
                S_rows.append(sv)
                X_rows.append(x_rep)
        except Exception:
            continue

    if not S_rows:
        raise RuntimeError("No SHAP rows collected; increase n_samples.")
    return np.vstack(S_rows), np.vstack(X_rows), feature_names  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Structural XAI: neighbor masking
# ---------------------------------------------------------------------------

def get_neighbor_positions_from_batch(
    blocks: list,
    pair_graph: dgl.DGLGraph,
    local_eids: np.ndarray,
    eid_local: int,
) -> tuple[int, int, list[int], list[int]]:
    """Return (src_pos, dst_pos, neighbors_src_positions, neighbors_global_nids)."""
    src_t, dst_t = pair_graph.edges()
    pos     = int(np.where(np.asarray(local_eids, dtype=np.int64) == eid_local)[0][0])
    src_pos = int(src_t[pos].item())
    dst_pos = int(dst_t[pos].item())

    b0    = blocks[0]
    u, v  = b0.edges()
    u_np  = u.cpu().numpy().astype(np.int64)
    v_np  = v.cpu().numpy().astype(np.int64)
    src_global = b0.srcdata[dgl.NID].cpu().numpy()

    mask        = (v_np == dst_pos) | (v_np == src_pos)
    neigh_pos   = [int(x) for x in np.unique(u_np[mask])
                   if int(x) not in (src_pos, dst_pos)]
    neigh_gnids = [int(src_global[p]) for p in neigh_pos]
    return src_pos, dst_pos, neigh_pos, neigh_gnids


def neighbor_impact_approx(
    model: nn.Module,
    h_dst: torch.Tensor,
    src_pos: int,
    dst_pos: int,
    e_feat_row: torch.Tensor,
    neighbors_pos: list[int],
    device: str = "cpu",
    target_class: int | None = None,
    h_src: torch.Tensor | None = None,
) -> dict[int, float]:
    """Estimate each neighbor's structural contribution via zero-masking.

    Returns {neighbor_block_pos: delta_prob} where delta = base_prob - masked_prob.
    """
    model.eval()
    impacts: dict[int, float] = {}
    dst_len = int(h_dst.size(0))
    src_len = int(h_src.size(0)) if h_src is not None else 0

    with torch.no_grad():
        sv = h_dst[src_pos].unsqueeze(0).to(device)
        dv = h_dst[dst_pos].unsqueeze(0).to(device)
        ev = e_feat_row.unsqueeze(0).float().to(device)
        base_logits  = model.head(torch.cat([sv, dv, ev], dim=1))
        if target_class is None:
            target_class = int(base_logits.argmax(dim=1)[0].item())
        base_prob = torch.softmax(base_logits, dim=1)[0, target_class].item()

        for ni in neighbors_pos:
            if h_src is not None and 0 <= ni < src_len:
                h_s = h_src.clone().to(device); h_s[ni] = 0.0
                h_d = h_dst.to(device)
            elif 0 <= ni < dst_len:
                h_s = h_src.to(device) if h_src is not None else None
                h_d = h_dst.clone().to(device); h_d[ni] = 0.0
            else:
                continue
            logits = model.head(torch.cat([h_d[src_pos].unsqueeze(0),
                                            h_d[dst_pos].unsqueeze(0), ev], dim=1))
            impacts[ni] = base_prob - torch.softmax(logits, dim=1)[0, target_class].item()
    return impacts


def visualize_neighbor_impacts(
    model: nn.Module,
    loader,
    g: dgl.DGLGraph,
    eid_local: int,
    feature,
    x_cat_csr: sparse.csr_matrix,
    topk: int = 10,
    device: str = "cpu",
    target_class: int = 0,
):
    """Compute and plot top-k neighbor impacts for `eid_local`."""
    model.eval()
    torch.set_grad_enabled(False)

    for _in_nodes, pair_graph, blocks in loader:
        local_eids = pair_graph.edata[dgl.EID].cpu().numpy().astype(np.int64)
        if eid_local not in local_eids:
            continue

        pair_graph = pair_graph.to(device)
        blocks     = [b.to(device) for b in blocks]

        h_src, h_dst = compute_pair_embedding(model, blocks, x_nodes=None, return_src=True)
        src_pos, dst_pos, neigh_pos, neigh_gnids = get_neighbor_positions_from_batch(
            blocks, pair_graph, local_eids, eid_local,
        )
        pos = int(np.where(local_eids == eid_local)[0][0])
        e_feat_row = torch.from_numpy(
            _read_edge_features(local_eids[pos : pos + 1], feature, x_cat_csr)[0]
        ).to(device)

        impacts = neighbor_impact_approx(
            model, h_dst, src_pos, dst_pos, e_feat_row,
            neigh_pos, device=device, target_class=target_class, h_src=h_src,
        )
        if not impacts:
            raise RuntimeError("No neighbor impacts computed.")

        items  = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:topk]
        labels = [f"nid={neigh_gnids[neigh_pos.index(p)] if p in neigh_pos else p}"
                  for p, _ in items]
        vals   = [v for _, v in items]

        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(labels))))
        ax.barh(range(len(labels)), vals[::-1], color="C1")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1])
        ax.set_xlabel("Δ prob (base − masked)")
        ax.set_title(f"Top-{len(labels)} neighbor impacts — edge {eid_local} (class={target_class})")
        plt.tight_layout()
        plt.show()
        torch.set_grad_enabled(True)
        return impacts, labels, vals

    raise RuntimeError(f"eid_local={eid_local} not found in loader batches.")


# ---------------------------------------------------------------------------
# Feature grouping utilities
# ---------------------------------------------------------------------------

def build_group_index(
    feature_names: list[str],
    cat_delim: str = "=",
    numeric_as_own_group: bool = True,
    custom_map: dict | None = None,
    cat_vars: list[str] | None = None,
) -> tuple[dict[str, list[int]], list[str]]:
    """Map each feature to a group name.

    Two grouping modes:

    cat_vars (recommended): Provide the list of original categorical variable
      names (e.g. from load_cat_variable_names()).  OHE column names are matched
      by prefix ``{VAR}_``, so multi-word names like ICMP_IPV4_TYPE work
      correctly.

    cat_delim fallback: Group OHE columns by splitting on cat_delim (default
      "=").  Only works when the encoder uses "=" as separator.

    Returns (group2idx, col2group).
    """
    # Build a fast prefix matcher when cat_vars is provided.
    if cat_vars:
        sorted_vars = sorted(cat_vars, key=len, reverse=True)

        def _detect(name: str) -> str | None:
            for var in sorted_vars:
                if name.startswith(var + "_"):
                    return var
            return None
    else:
        _detect = None  # type: ignore[assignment]

    col2group: list[str] = []
    group2idx: dict[str, list[int]] = defaultdict(list)
    for j, name in enumerate(feature_names):
        if custom_map and name in custom_map:
            g = custom_map[name]
        elif _detect is not None:
            matched = _detect(name)
            if matched is not None:
                g = matched
                if custom_map and g in custom_map:
                    g = custom_map[g]
            else:
                g = name if numeric_as_own_group else "NUMERIC"
        elif cat_delim in name:
            g = name.split(cat_delim, 1)[0]
            if custom_map and g in custom_map:
                g = custom_map[g]
        else:
            g = name if numeric_as_own_group else "NUMERIC"
        col2group.append(g)
        group2idx[g].append(j)
    return dict(group2idx), col2group


def shap_to_group_values(
    shap_vec: np.ndarray,
    feature_names: list[str],
    mode: str = "signed_sum",
    **group_kwargs,
) -> tuple[list[str], np.ndarray, dict[str, list[int]]]:
    """Aggregate a SHAP vector over feature groups.

    mode: "signed_sum" | "signed_mean" | "abs_sum" | "abs_mean"
    Returns (groups, values, group2idx).
    """
    v = np.asarray(shap_vec, dtype=float).reshape(-1)
    assert v.size == len(feature_names)
    group2idx, _ = build_group_index(feature_names, **group_kwargs)
    groups, vals = [], []
    for g, idxs in group2idx.items():
        sub = v[idxs]
        agg = {"signed_sum":  sub.sum,
               "signed_mean": sub.mean,
               "abs_sum":     lambda: np.abs(sub).sum(),
               "abs_mean":    lambda: np.abs(sub).mean()}[mode]()
        groups.append(g); vals.append(agg)
    return groups, np.asarray(vals, dtype=float), group2idx


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

class CategoryColorMap:
    """Stable deterministic colour assignment for group labels across plots."""

    def __init__(self, palette=None):
        self.palette = list(palette or plt.get_cmap("tab20").colors)
        self._map: dict[str, object] = {}
        self._cursor = 0

    def color(self, group: str):
        if group not in self._map:
            self._map[group] = self.palette[self._cursor % len(self.palette)]
            self._cursor += 1
        return self._map[group]

    def colors(self, groups: list[str]) -> list:
        return [self.color(g) for g in groups]


def plot_topk_per_class(
    class2_vals: dict[int, np.ndarray],
    feature_names: list[str],
    id2label: dict[int, str],
    k: int = 10,
    save_dir: str = "artifacts/xai",
    title: str = "Top-k Feature Importances",
    signed: bool = False,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for c, vals_raw in class2_vals.items():
        vals_arr = np.asarray(vals_raw)
        if vals_arr.size == 0:
            continue
        top_idx = np.argsort(np.abs(vals_arr))[::-1][:k]
        names = [feature_names[i] for i in top_idx[::-1]]
        vals  = vals_arr[top_idx[::-1]]
        fig, ax = plt.subplots(figsize=(8, 5))
        if signed:
            colors = ["#1f77b4" if x > 0 else "red" for x in vals]
            ax.barh(names, vals, color=colors)
            ax.axvline(0, color="k", lw=1)
        else:
            ax.barh(names, np.abs(vals))
        label = id2label.get(c, str(c))
        ax.set_title(f"{title} — {label}")
        fig.tight_layout()
        safe = lambda s: "".join(ch if ch.isalnum() or ch in " _-" else "_" for ch in s).lower().replace(" ", "_")
        fig.savefig(os.path.join(save_dir, f"{safe(title)}_{safe(label)}.png"), dpi=150)
        plt.close(fig)


def plot_spider(
    class2_vals: dict[int, np.ndarray],
    feature_names: list[str],
    id2label: dict[int, str],
    k_union: int = 8,
    save_path: str = "artifacts/xai/spider.png",
) -> None:
    chosen = set()
    for imp in class2_vals.values():
        chosen.update(np.argsort(imp)[::-1][:k_union].tolist())
    chosen_list = sorted(chosen)
    labels_feat = [feature_names[i] for i in chosen_list]

    classes_order = sorted(class2_vals.keys())
    data = []
    for c in classes_order:
        imp = class2_vals[c][chosen_list]
        imp_norm = (imp - imp.min()) / (imp.max() - imp.min() + 1e-12)
        data.append(imp_norm)

    angles = np.linspace(0, 2 * np.pi, len(chosen_list), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.subplot(111, polar=True)
    for row, c in zip(data, classes_order):
        vals = row.tolist() + row[:1].tolist()
        ax.plot(angles, vals, lw=2, label=id2label.get(c, str(c)))
        ax.fill(angles, vals, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels_feat, fontsize=9)
    ax.set_title("Spider Chart — Normalized Feature Importance (mean |SHAP|)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_shap_bar(
    groups: list[str],
    values: np.ndarray,
    color_map: CategoryColorMap,
    top_k: int = 10,
    title: str = "",
    signed: bool = True,
    savepath: str | None = None,
) -> None:
    groups_a = np.asarray(groups)
    values_a = np.asarray(values, dtype=float)
    order = np.argsort(np.abs(values_a))[-top_k:][::-1]
    g_sel = groups_a[order]
    v_sel = (values_a if signed else np.abs(values_a))[order]
    cols  = color_map.colors(g_sel.tolist())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(list(g_sel[::-1]), list(v_sel[::-1]), color=cols[::-1])
    if signed:
        ax.axvline(0.0, color="k", lw=1)
    ax.set_title(title)
    fig.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def beeswarm_for_class(
    model: nn.Module,
    g: dgl.DGLGraph,
    loader,
    feature,
    x_cat_csr: sparse.csr_matrix,
    bg_eids: np.ndarray,
    class_id: int,
    n_samples: int = 200,
    background_size: int = 100,
    top_k: int = 20,
    feature_names: list[str] | None = None,
    device: str = "cpu",
    savepath: str | None = None,
    variable_groups: list[dict] | None = None,
) -> None:
    """SHAP beeswarm plot for a single class.

    Pass variable_groups (from build_variable_groups()) to plot in the ~45-dim
    variable space.  With only ~45 features, top_k filtering is less important
    but still honoured.
    """
    S, X, feat_names = shap_matrix_for_class(
        model, g, loader, feature, x_cat_csr, bg_eids,
        class_id, n_samples, background_size, feature_names, device,
        variable_groups=variable_groups,
    )
    mean_abs = np.abs(S).mean(axis=0)
    k        = min(top_k, S.shape[1])
    top_idx  = np.argsort(mean_abs)[::-1][:k]
    plt.figure(figsize=(8, 5))
    shap.summary_plot(S[:, top_idx], X[:, top_idx],
                      feature_names=[feat_names[i] for i in top_idx],
                      plot_type="dot", show=False)
    plt.title(f"SHAP beeswarm — class {class_id}", fontsize=12)
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
