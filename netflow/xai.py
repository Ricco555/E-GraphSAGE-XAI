import os, joblib, json, numpy as np, torch, shap, matplotlib.pyplot as plt
from typing import Tuple
import dgl
from traitlets import List
from feature_store import fetch_edge_features

def load_feature_names(feature_store_dir: str) -> Tuple[List[str], int, int]:
    """
    Build edge feature names (numeric + one-hot categorical) for a given split.
    Returns:
        feature_names: list[str] of length (d_num + d_cat)
        d_num:        int, numeric width in the split
        d_cat:        int, categorical (one-hot) width in the split

    This function is compatible with the artifacts written by:
      - feature_numeric.py  -> artifacts/numeric/numeric_artifacts.joblib (num_cols_final)  [numeric]  (float32)
      - categorical_encoding.py -> artifacts/categorical/categorical_artifacts.json (feature_names/encoder_path)  [CSR]
      - feature_store.py    -> {split}/numeric.dat, {split}/y.npy, {split}/categorical.npz (optional)
    """
    split_dir = os.path.abspath(feature_store_dir)
    fs_root = os.path.abspath(os.path.join(split_dir, os.pardir))  # e.g., .../feature_store

    # --- 1) Determine true shapes from the split (source of truth) ---
    # numeric: infer d_num from file size and #rows from y.npy (float32 -> 4 bytes)
    y_path = os.path.join(split_dir, "y.npy")
    num_path = os.path.join(split_dir, "numeric.dat")
    if not (os.path.exists(y_path) and os.path.exists(num_path)):
        raise FileNotFoundError(f"[feature_store] Missing {y_path} or {num_path}")

    n_rows = int(np.load(y_path).shape[0])
    num_bytes = os.path.getsize(num_path)
    if n_rows <= 0 or num_bytes % 4 != 0:
        raise RuntimeError(f"[feature_store] numeric.dat size or y.npy rows invalid in {split_dir}")
    d_num = int(num_bytes // (4 * n_rows))  # float32

    # categorical: read shape from CSR if present; else 0
    cat_path = os.path.join(split_dir, "categorical.npz")
    if os.path.exists(cat_path):
        import scipy.sparse as sp  # optional dep already used in your pipeline
        Xcat = sp.load_npz(cat_path)
        if Xcat.shape[0] != n_rows:
            raise RuntimeError(f"[feature_store] categorical rows={Xcat.shape[0]} != y rows={n_rows} in {split_dir}")
        d_cat = int(Xcat.shape[1])
    else:
        d_cat = 0

    # --- 2) Try to load metadata artifacts for real column names ---

    # Numeric artifacts (preferred): artifacts/numeric/numeric_artifacts.joblib
    num_names = None
    num_art_candidates = [
        os.path.join(fs_root, "artifacts", "numeric", "numeric_artifacts.joblib"),
        os.path.join("artifacts", "numeric", "numeric_artifacts.joblib"),  # relative fallback
    ]
    for p in num_art_candidates:
        if os.path.exists(p):
            try:
                arts = joblib.load(p)  # dict produced by feature_numeric.fit_numeric_transform
                # Prefer "num_cols_final" (after var/corr masks); fall back to "num_cols"
                for k in ("num_cols_final", "numeric_cols_final", "num_cols", "numeric_cols"):
                    if isinstance(arts.get(k), (list, tuple)):
                        if len(arts[k]) == d_num:
                            num_names = list(arts[k])
                        # length mismatch -> ignore and fallback
                        break
            except Exception:
                pass
            if num_names is not None:
                break

    # If not available or mismatched, fallback to generic
    if num_names is None:
        num_names = [f"num_{i}" for i in range(d_num)]

    # Categorical artifacts (preferred): artifacts/categorical/categorical_artifacts.json
    cat_names = []
    if d_cat > 0:
        cat_art_candidates = [
            os.path.join(fs_root, "artifacts", "categorical", "categorical_artifacts.json"),
            os.path.join("artifacts", "categorical", "categorical_artifacts.json"),  # relative fallback
        ]
        cat_meta = None
        for p in cat_art_candidates:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        cat_meta = json.load(f)
                    break
                except Exception:
                    cat_meta = None

        if cat_meta is not None:
            # (A) Use precomputed feature_names when they match d_cat
            fnh = cat_meta.get("feature_names", None)
            if isinstance(fnh, list) and len(fnh) == d_cat:
                cat_names = list(fnh)
            else:
                # (B) Rebuild names from saved encoder if present and compatible
                enc_path = cat_meta.get("encoder_path", "")
                if enc_path and not os.path.isabs(enc_path):
                    enc_path = os.path.join(fs_root, "artifacts", "categorical", enc_path)
                try:
                    enc = joblib.load(enc_path) if enc_path and os.path.exists(enc_path) else None
                except Exception:
                    enc = None
                if enc is not None and hasattr(enc, "categories_"):
                    # categories_ is list of arrays in the same order as cat_cols
                    cat_cols = cat_meta.get("cat_cols", [])
                    cat_names_tmp = []
                    try:
                        for col, cats in zip(cat_cols, enc.categories_):
                            for c in list(cats):
                                cat_names_tmp.append(f"{col}={c}")
                        if len(cat_names_tmp) == d_cat:
                            cat_names = cat_names_tmp
                    except Exception:
                        cat_names = []

        # Fallback: generic names if still empty or mismatched
        if not cat_names or len(cat_names) != d_cat:
            cat_names = [f"cat_{i}" for i in range(d_cat)]

    # --- 3) Finalize ---
    feature_names = num_names + cat_names
    # Guard against any drift: correct the length to exact (d_num + d_cat)
    total_dim = d_num + d_cat
    if len(feature_names) != total_dim:
        # Heuristic: trim or pad the tail with generic names to match the true width
        if len(feature_names) > total_dim:
            feature_names = feature_names[:total_dim]
        else:
            pad = [f"feat_{i}" for i in range(total_dim - len(feature_names))]
            feature_names = feature_names + pad

    return feature_names, d_num, d_cat

# Local SHAP for a single edge (feature-level)
# We freeze graph context (the pair embedding [h_src,h_dst]) 
# and explain only the effect of the edge feature vector on the logit/probability for a chosen class.
@torch.no_grad()
def compute_pair_embedding(model, blocks, x_nodes, return_src: bool = False):
    """
    Backwards-compatible wrapper around model.encode.
    - Default (return_src=False): returns whatever model.encode(...) returned (expected: h_dst tensor).
    - If return_src=True: attempts to return (h_src, h_dst).
        * If model.encode already returns a pair/tuple of two tensors, they are validated and ordered
          to match (num_src_nodes_in_block0, num_dst_nodes_in_last_block).
        * If model.encode returns a single tensor (legacy), a RuntimeError is raised with a short
          instruction to update model.encode to provide src embeddings too.

    This preserves the original behavior for callers that expect a single h_dst tensor.
    """
    # call model.encode and forward the return_src request when present
    try:
        res = model.encode(blocks, x_nodes, return_src=return_src)
    except TypeError:
        # model.encode doesn't accept return_src (legacy) — call without it
        res = model.encode(blocks, x_nodes)

    if not return_src:
        return res

    # want (h_src, h_dst)
    # If encode already returned a pair, try to interpret it
    if isinstance(res, (list, tuple)) and len(res) == 2:
        a, b = res[0], res[1]
        # determine expected sizes from blocks
        try:
            num_src = int(blocks[0].srcdata[dgl.NID].shape[0])
            num_dst = int(blocks[-1].dstdata[dgl.NID].shape[0])
        except Exception:
            # fallback if shapes can't be read — assume ordering (a -> h_src, b -> h_dst)
            return (a, b)
        # match by first-dimension
        if getattr(a, "size", None) and getattr(b, "size", None):
            a0 = int(a.size(0))
            b0 = int(b.size(0))
            if a0 == num_src and b0 == num_dst:
                return (a, b)
            if a0 == num_dst and b0 == num_src:
                return (b, a)
            # shapes don't match expected counts -> still return (a,b) but warn via RuntimeError for clarity
            raise RuntimeError(
                f"model.encode returned two tensors but their sizes don't match block src/dst counts "
                f"(got {a0},{b0} expected {num_src},{num_dst}). Inspect model.encode output."
            )

    # If encode returned a single tensor, treat as legacy h_dst and fail fast when src requested
    if not isinstance(res, (list, tuple)):
        raise RuntimeError(
            "compute_pair_embedding: model.encode currently returns a single tensor (h_dst). "
            "To enable structural XAI (neighbor masking) please update model.encode to return "
            "(h_src, h_dst) when called with return_src=True, or call compute_pair_embedding with "
            "return_src=False and provide h_src by other means."
        )

    # any other unexpected return type
    raise RuntimeError("compute_pair_embedding: unexpected return from model.encode")


def make_edge_head_fn(model, he_fixed: np.ndarray, device="cpu", return_prob=True, target_class=None):
    """
    Wrap the edge-head to accept X_edge (B, edge_in) and return:
      - if target_class is None:
          (B, C) probabilities (or logits if return_prob=False)
      - else:
          (B,) probabilities (or logits) for the target class
    NOTE: returns *NumPy* arrays, with no grad graph attached.
    """
    he_fixed_t = torch.from_numpy(he_fixed[None, :].astype(np.float32, copy=False)).to(device)
    he_fixed_t.requires_grad_(False)

    def f_edge(X: np.ndarray):
        # SHAP will pass a numpy array; ensure float32
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X).to(device)
        # Build fixed [h_src||h_dst] per row
        he = he_fixed_t.expand(X_t.size(0), -1)
        with torch.no_grad():
            logits = model.edge_mlp(torch.cat([he, X_t], dim=1))  # (B, C)
            if target_class is None:
                if return_prob:
                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                    return probs  # (B, C)
                return logits.detach().cpu().numpy()              # (B, C)
            # Single-class output
            if return_prob:
                probs = torch.softmax(logits, dim=1)[:, target_class].detach().cpu().numpy()
                return probs  # (B,)
            return logits[:, target_class].detach().cpu().numpy()  # (B,)
    return f_edge

def local_shap_for_edge(model, g, loader, edge_idx_local: int, split_dir: str,
                        background_size=100, target_class=None, device="cpu"):
    """
    Explain a single edge's prediction wrt edge features using KernelSHAP:
      1) Build blocks for that edge (sample a tiny batch that includes it).
      2) Compute pair embedding [h_src,h_dst] fixed for this edge.
      3) Build wrapper f_edge(X_edge) -> probability/score for target_class.
      4) Explain X_edge with background (from train or val split).

    Returns: shap_values (1, edge_in), feature_names list, base_value
    """
    model.eval()
    torch.set_grad_enabled(False)
    # 1) Grab one mini-batch that includes this edge
    # Simple way: iterate loader until this local eid appears
    he_fixed = None
    x_edge_row = None
    y_true = None
    for input_nodes, pair_graph, blocks in loader:
        local_eids = pair_graph.edata[dgl.EID].cpu().numpy().astype(np.int64)
        if edge_idx_local in local_eids:
            # 2) Build h_dst and recover src/dst row
            if blocks and (blocks[0].device.type != torch.device(device).type):
                blocks = [b.to(device) for b in blocks]
            pair_graph = pair_graph.to(device) if pair_graph.device.type != torch.device(device).type else pair_graph

            h_dst = compute_pair_embedding(model, blocks, x_nodes=None)   # if node_in=0
            src_pos, dst_pos = pair_graph.edges()
            # Position of our edge in this batch:
            pos = int(np.where(local_eids == edge_idx_local)[0][0])
            he = torch.cat([h_dst[src_pos][pos], h_dst[dst_pos][pos]], dim=0)  # (2*hidden,)
            he_fixed = he.detach().cpu().numpy()

            # 3) Edge features for this edge (GLOBAL EID)
            global_eids = g.edata[dgl.EID][torch.from_numpy(local_eids)].cpu().numpy().astype(np.int64)
            e_feat_np = fetch_edge_features(global_eids[pos:pos+1], split_dir).astype(np.float32, copy=False)
            x_edge_row = e_feat_np[0]
            # Label
            y_true = int(g.edata["y"][edge_idx_local].item())
            break

    if he_fixed is None:
        raise RuntimeError(f"Local eid {edge_idx_local} not found in loader batch.")

    # Background: sample background_size random edges from the same split
    store_eids = loader.store_eids if hasattr(loader, "store_eids") else \
                 np.load(os.path.join(split_dir, "edge_indices.npy")).astype(np.int64)
    bg_ids = store_eids[np.random.default_rng(0).choice(store_eids.size, size=min(background_size, store_eids.size), replace=False)]
    bg_X = fetch_edge_features(bg_ids, split_dir).astype(np.float32, copy=False)   # (B_bg, edge_in)

    # 4) Wrap the edge-head for SHAP
    f = make_edge_head_fn(model, he_fixed, device=device, return_prob=True, target_class=target_class)
    explainer = shap.KernelExplainer(f, bg_X)
    sv = explainer.shap_values(x_edge_row[None, :], nsamples='auto')  # returns array(s)

    # Helper to unify sv -> (1, d) array
    def _select_sv(sv, target_class, f, x_edge_row):
        # If wrapper returns single-output (target_class is not None),
        # KernelExplainer usually returns np.ndarray of shape (1, d)
        if isinstance(sv, np.ndarray):
            return sv  # (1, d)
        # Otherwise it's a list:
        if not isinstance(sv, list) or len(sv) == 0:
            raise RuntimeError("Unexpected SHAP return type.")
        if target_class is None:
            # choose predicted class
            probs = f(x_edge_row[None, :])  # (1, C)
            # probs may be (1, C) or (1,) if wrapper returns logits; enforce 2D
            if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[1] > 1:
                pred_c = int(np.argmax(probs[0]))
            else:
                # If not multi-output, there is only one class to explain
                pred_c = 0
            pred_c = min(pred_c, len(sv) - 1)
            return sv[pred_c]  # (1, d)
        # target_class is set: pick it if available, else fallback to the only element
        if target_class < len(sv):
            return sv[target_class]  # (1, d)
        else:
            # single-output list-of-one fallback
            return sv[0]

    # Pick the correct 2D array
    shap_vals = _select_sv(sv, target_class, f, x_edge_row)  # (1, d)
    base_value = explainer.expected_value
    # expected_value can be a scalar or list; normalize it to a scalar
    if isinstance(base_value, (list, tuple, np.ndarray)):
        # select consistent base for the chosen output
        if target_class is None:
            if isinstance(sv, list) and len(base_value) == len(sv):
                # predicted class base value
                probs = f(x_edge_row[None, :])
                if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[1] > 1:
                    pred_c = int(np.argmax(probs[0]))
                else:
                    pred_c = 0
                base_value = base_value[min(pred_c, len(base_value)-1)]
            else:
                base_value = base_value[0]
        else:
            idx = target_class if isinstance(sv, list) and target_class < len(base_value) else 0
            base_value = base_value[idx]

    return shap_vals, x_edge_row, y_true, base_value

# Global importance per class
# Aggregate absolute SHAP across many edges per class to get per-class top features.
def sample_edges_by_class(g, n_per_class=200, seed=0):
    """Return a dict: class_id -> list[local_edge_id]"""
    rng = np.random.default_rng(seed)
    y = g.edata["y"].cpu().numpy()
    out = {}
    for c in np.unique(y):  # iterating only existing classes is more efficient then iterate empty classes
        idx = np.nonzero(y == c)[0]
        k = min(n_per_class, idx.size)
        out[c] = rng.choice(idx, size=k, replace=False).tolist()
    return out

def aggregate_shap_per_class(model, g, loader, split_dir, feature_names, n_per_class=200, device="cpu"):
    """
    Compute mean |SHAP| per feature per class by sampling n_per_class edges per class.
    Returns: dict[class_id] -> (mean_abs_shap: np.ndarray edge_in,)
    """
    c2idx = sample_edges_by_class(g, n_per_class=n_per_class)
    class2_meanabs = {}
    for c, eids in c2idx.items():
        vals = []
        for le in eids:
            # local explanation for this edge on its predicted class or target c (pick one)
            shap_vals, x_edge, y_true, base_value = local_shap_for_edge(
                model, g, loader, le, split_dir, background_size=100, target_class=c, device=device
            )
            vals.append(np.abs(shap_vals[0]))
        if vals:
            class2_meanabs[c] = np.mean(np.vstack(vals), axis=0)
    return class2_meanabs

# Spider (radar) plot across classes
def plot_spider(class2_vals, feature_names, id2label, k_union=8, save_path="artifacts/xai/spider.png"):
    # 1) choose the union of top features
    chosen = set()
    for c, imp in class2_vals.items():
        top = np.argsort(imp)[::-1][:k_union]
        chosen.update(top.tolist())
    chosen = sorted(chosen)
    labels_feat = [feature_names[i] for i in chosen]

    # 2) normalize per class to [0,1]
    data = []
    classes_order = sorted(class2_vals.keys())
    for c in classes_order:
        imp = class2_vals[c][chosen]
        imp_norm = (imp - imp.min()) / (imp.max() - imp.min() + 1e-12)
        data.append(imp_norm)
    data = np.vstack(data)

    # 3) radar plot
    angles = np.linspace(0, 2*np.pi, len(chosen), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    for row, c in zip(data, classes_order):
        vals = row.tolist() + row[:1].tolist()
        ax.plot(angles, vals, linewidth=2, label=id2label[c])
        ax.fill(angles, vals, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels_feat, fontsize=9)
    ax.set_title("Spider Chart — Normalized Feature Importance (mean |SHAP|)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

# Feature impact per class (signed effects)
def aggregate_signed_shap_per_class(model, g, loader, split_dir, feature_names, n_per_class=200, device="cpu"):
    c2idx = sample_edges_by_class(g, n_per_class=n_per_class)
    class2_signed = {}
    for c, eids in c2idx.items():
        vals = []
        for le in eids:
            shap_vals, x_edge, y_true, base_value = local_shap_for_edge(
                model, g, loader, le, split_dir, background_size=100, target_class=c, device=device
            )
            vals.append(shap_vals[0])
        if vals:
            class2_signed[c] = np.mean(np.vstack(vals), axis=0)
    return class2_signed

# Plot top-k features per class
def plot_topk_per_class(class2_vals, feature_names, id2label, k=10, save_dir="artifacts/xai",
                        title="Top-k Feature Importances", signed=False):
    """
    Unified plotter for per-class top-k features.

    - class2_vals: dict[class_id] -> 1D array-like of values (mean|SHAP| or signed SHAP)
    - feature_names: list of feature names
    - id2label: mapping class_id -> human label
    - k: top-k features
    - title: chart title prefix (e.g. "Top-k Feature Importances" or "Top-k Feature Impacts")
    - signed: if True treat values as signed and color bars (green>0, red<=0) and draw zero line
    """
    os.makedirs(save_dir, exist_ok=True)
    for c, vals_raw in class2_vals.items():
        vals_arr = np.asarray(vals_raw)
        if vals_arr.size == 0:
            continue
        if signed:
            top_idx = np.argsort(np.abs(vals_arr))[::-1][:k]
            names = [feature_names[i] for i in top_idx][::-1]
            vals = vals_arr[top_idx][::-1]
            colors = ["green" if x > 0 else "red" for x in vals]
            plt.figure(figsize=(8, 5))
            plt.barh(names, vals, color=colors)
            plt.axvline(0, color="k", lw=1)
        else:
            top_idx = np.argsort(vals_arr)[::-1][:k]
            names = [feature_names[i] for i in top_idx][::-1]
            vals = vals_arr[top_idx][::-1]
            plt.figure(figsize=(8, 5))
            plt.barh(names, vals)

        label = id2label.get(c, str(c))
        plt.title(f"{title} — {label}")
        plt.tight_layout()
        # safe filename derived from title and label
        safe_title = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in title).strip().lower().replace(" ", "_")
        safe_label = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in str(label)).strip().lower().replace(" ", "_")
        fname = os.path.join(save_dir, f"{safe_title}_{safe_label}.png")
        plt.savefig(fname, dpi=150)
        plt.show()

# Structural XAI: neighbor masking
# estimates structural contribution by removing/masking neighbors and measuring the output delta.
# For a chosen edge (u,v), sample blocks as usual.
# Compute baseline logit/prob for class c.
# Identify 1-hop neighbors used in blocks[0] for u and v (via blocks[0].edges() and blocks[0].srcdata[dgl.NID]).
# For each neighbor n, rebuild a modified block where you remove edges from n→{u or v} (or zero out h_n in h_dst for a quick approximation), then recompute the logit delta for c.
# Rank neighbors by |delta|; group by attributes (e.g., protocol or subnet) for analyst-friendly views.
# “perturbation/ablation” estimate of structural contribution
def neighbor_impact_approx(model, h_dst, src_pos, dst_pos, e_feat_row, neighbors_pos, device="cpu", target_class=None, h_src=None):
    """
    h_dst: (num_dst, hidden) embeddings for dst nodes (as before)
    h_src: optional (num_src, hidden) embeddings for block src nodes (if available)
    neighbors_pos: list[int] positions in block.src (or block.dst) to "mask"
    If a neighbor index is valid in h_src it will be masked there; otherwise in h_dst.
    Returns: dict {neighbor_idx_in_block -> delta_prob[target_class]}
    """
    import torch
    impacts = {}
    model.eval()
    with torch.no_grad():
        # scalar indices
        s_idx = int(src_pos.item()) if isinstance(src_pos, torch.Tensor) else int(src_pos)
        d_idx = int(dst_pos.item()) if isinstance(dst_pos, torch.Tensor) else int(dst_pos)

        # source/dest vectors for the explained edge (use h_dst as before)
        src_vec = h_dst[s_idx].to(device).unsqueeze(0)   # (1, hidden)
        dst_vec = h_dst[d_idx].to(device).unsqueeze(0)   # (1, hidden)
        e_vec   = e_feat_row.to(device).unsqueeze(0).float()  # (1, feat_dim)

        inp = torch.cat([src_vec, dst_vec, e_vec], dim=1)
        base_logits = model.edge_mlp(inp)
        if target_class is None:
            target_class = int(torch.argmax(base_logits, dim=1)[0].item())
        base_prob = torch.softmax(base_logits, dim=1)[0, target_class].item()

        # lengths for safe indexing
        dst_len = int(h_dst.size(0))
        src_len = int(h_src.size(0)) if h_src is not None else 0

        for n in neighbors_pos:
            ni = int(n)
            # clone copies
            h_dst_mod = h_dst.clone().to(device)
            h_src_mod = h_src.clone().to(device) if h_src is not None else None

            # decide where to mask: prefer h_src if valid there (neighbors_pos are typically src positions)
            if h_src_mod is not None and 0 <= ni < src_len:
                h_src_mod[ni] = 0.0
            elif 0 <= ni < dst_len:
                h_dst_mod[ni] = 0.0
            else:
                # out-of-range: skip this neighbor
                continue

            # Build edge vectors: use h_dst_mod for endpoint vectors (keeps previous behaviour)
            src_vec_m = h_dst_mod[s_idx].unsqueeze(0)
            dst_vec_m = h_dst_mod[d_idx].unsqueeze(0)
            inp_m = torch.cat([src_vec_m, dst_vec_m, e_vec], dim=1)
            logits = model.edge_mlp(inp_m)
            prob = torch.softmax(logits, dim=1)[0, target_class].item()
            impacts[ni] = base_prob - prob
    return impacts

# Helper to get neighbor positions from a batch
def get_neighbor_positions_from_batch(blocks, pair_graph, local_eids, edge_idx_local):
    """
    Return tuple (src_pos, dst_pos, neighbors_pos, neighbors_global_nids)
    - src_pos, dst_pos: integer positions inside h_dst for the chosen edge
    - neighbors_pos: list of integer positions inside h_dst to perturb (src-side positions)
    - neighbors_global_nids: corresponding global node ids from blocks[0].srcdata[dgl.NID]
    Assumes blocks[0] is the sampled (bipartite) block used to compute h_dst.
    """
    # edges in the pair_graph batch (local positions inside pair_graph)
    src_pos_tensor, dst_pos_tensor = pair_graph.edges()
    local_eids_arr = np.asarray(local_eids, dtype=np.int64)
    pos = int(np.where(local_eids_arr == edge_idx_local)[0][0])
    src_pos = int(src_pos_tensor[pos].item())
    dst_pos = int(dst_pos_tensor[pos].item())

    # block-level source node global ids and edge lists (block 0)
    b0 = blocks[0]
    b0_u, b0_v = b0.edges()
    src_global = b0.srcdata[dgl.NID].cpu().numpy()  # array length = b0.num_src_nodes()
    # neighbors: any src node that connects to either dst_pos or src_pos (heuristic)
    # choose all src indices u where v == dst_pos OR v == src_pos
    u = b0_u.cpu().numpy().astype(np.int64)
    v = b0_v.cpu().numpy().astype(np.int64)
    # select positions in src node array (u are indices into src nodes)
    mask = (v == dst_pos) | (v == src_pos) | (u == dst_pos) | (u == src_pos)
    neigh_src_positions = np.unique(u[mask]).tolist()
    # remove the actual source and destination positions so we perturb "others"
    neigh_src_positions = [int(x) for x in neigh_src_positions if int(x) not in (src_pos, dst_pos)]
    neighbors_global_nids = [int(src_global[p]) for p in neigh_src_positions]
    return src_pos, dst_pos, neigh_src_positions, neighbors_global_nids

def visualize_neighbor_impacts(model, loader, g, edge_gid, split_dir, topk=10, device="cpu", target_class=0):
    import torch, numpy as np, matplotlib.pyplot as plt
    from feature_store import fetch_edge_features
    model.eval()
    torch.set_grad_enabled(False)

    for input_nodes, pair_graph, blocks in loader:
        local_eids = pair_graph.edata[dgl.EID].cpu().numpy().astype(np.int64)
        if int(edge_gid) not in local_eids:
            continue
        pair_graph = pair_graph.to(device)
        blocks = [b.to(device) for b in blocks]

        # request both src and dst embeddings (backwards compatible: compute_pair_embedding still returns h_dst when called elsewhere)
        try:
            h_src, h_dst = compute_pair_embedding(model, blocks, x_nodes=None, return_src=True)
        except ValueError:
            # if compute_pair_embedding cannot return src/dst, fall back to old behavior and raise clear error
            raise RuntimeError("compute_pair_embedding must be updated to support return_src=True (see xai.py instructions)")

        src_pos, dst_pos, neighbors_pos, neighbors_global_nids = get_neighbor_positions_from_batch(blocks, pair_graph, local_eids, int(edge_gid))

        # fetch edge features
        e_feat_np = fetch_edge_features(np.array([int(edge_gid)], dtype=np.int64), split_dir)
        e_feat = torch.from_numpy(e_feat_np[0]).to(device).float()

        # pass both embeddings to neighbor impact (now handles src/dst masking)
        impacts = neighbor_impact_approx(model, h_dst.to(device),
                                         torch.tensor(src_pos, dtype=torch.long, device=device),
                                         torch.tensor(dst_pos, dtype=torch.long, device=device),
                                         e_feat, neighbors_pos, device=device, target_class=target_class, h_src=h_src.to(device))

        if len(impacts) == 0:
            raise RuntimeError("no neighbor impacts computed (empty/invalid neighbors_pos)")

        items = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:topk]
        labels = []
        vals = []
        for pos_idx, delta in items:
            try:
                gid = neighbors_global_nids[neighbors_pos.index(pos_idx)]
            except Exception:
                try:
                    gid = int(blocks[0].srcdata[dgl.NID][pos_idx].item())
                except Exception:
                    gid = int(pos_idx)
            labels.append(f"nid={gid}")
            vals.append(delta)

        fig, ax = plt.subplots(figsize=(8, max(3, 0.35*len(labels))))
        y_pos = range(len(labels))
        ax.barh(y_pos, vals[::-1], color="C1")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels[::-1])
        ax.set_xlabel("delta prob (base - masked)")
        ax.set_title(f"Top-{len(labels)} neighbor impacts for edge {edge_gid} (class={target_class})")
        plt.tight_layout()
        plt.show()
        return impacts, labels, vals

    raise RuntimeError(f"global edge id {edge_gid} not found in loader batches")

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def build_group_index(feature_names, cat_delim="=", numeric_as_own_group=True, custom_map=None):
    """
    Build a mapping: group_name -> list of feature indices (cols).
    - Categorical one-hots are detected by 'cat_delim' (default '='), e.g. 'PROTOCOL=6' -> group 'PROTOCOL'.
    - Numeric features become their own groups (group == feature name), unless custom_map remaps them.
    - custom_map (dict[str->str]) lets you override grouping for any feature or category group.
    Returns:
      group2idx: dict[group_name] -> list[int]
      col2group: list[str] of length d (group name for each feature)
    """
    d = len(feature_names)
    col2group = [None]*d
    group2idx = defaultdict(list)

    for j, name in enumerate(feature_names):
        if custom_map and name in custom_map:
            g = custom_map[name]
        else:
            if cat_delim in name:                 # categorical one-hot
                g = name.split(cat_delim, 1)[0]   # prefix before '='
                if custom_map and g in custom_map:
                    g = custom_map[g]
            else:
                g = name if numeric_as_own_group else "NUMERIC"
        col2group[j] = g
        group2idx[g].append(j)

    return dict(group2idx), col2group


def shap_to_group_values(shap_vec, feature_names, mode="signed_sum", **group_kwargs):
    """
    Aggregate a SHAP vector over groups defined by build_group_index(...).
    Parameters
      shap_vec: np.ndarray shape (d,) or (1,d) (signed SHAP for a sample or mean over samples)
      feature_names: list[str] length d
      mode:
        - "signed_sum": sum of SHAP within group (keeps direction; good for signed mean SHAP)
        - "signed_mean": mean of SHAP within group
        - "abs_sum": sum of |SHAP| within group (good for feature ranking)
        - "abs_mean": mean of |SHAP| within group
      group_kwargs: passed to build_group_index(...)
    Returns
      groups: list[str] group names
      values: np.ndarray shape (G,) aggregated per group in the same order as 'groups'
      group2idx: dict[group -> list[int]] (for drill-down)
    """
    v = np.asarray(shap_vec, dtype=float).reshape(-1)  # (d,)
    assert v.size == len(feature_names), f"shap dim {v.size} != names {len(feature_names)}"

    group2idx, _ = build_group_index(feature_names, **group_kwargs)
    groups, vals = [], []
    for g, idxs in group2idx.items():
        sub = v[idxs]
        if mode == "signed_sum":
            agg = sub.sum()
        elif mode == "signed_mean":
            agg = sub.mean()
        elif mode == "abs_sum":
            agg = np.abs(sub).sum()
        elif mode == "abs_mean":
            agg = np.abs(sub).mean()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        groups.append(g); vals.append(agg)

    return groups, np.asarray(vals, dtype=float), group2idx

class CategoryColorMap:
    """
    Deterministic color assignment for category groups across multiple plots.
    Reuse the same instance in all your figures to keep colors stable.
    """
    def __init__(self, palette=None):
        # default palette: tab20
        if palette is None:
            # 20 distinct colors (repeat if you exceed 20 groups)
            palette = plt.get_cmap("tab20").colors
        self.palette = list(palette)
        self.map = {}     # group -> color (rgba)
        self._cursor = 0

    def color(self, group):
        if group not in self.map:
            self.map[group] = self.palette[self._cursor % len(self.palette)]
            self._cursor += 1
        return self.map[group]

    def colors(self, groups):
        return [self.color(g) for g in groups]

def plot_grouped_shap_bar(groups, values, color_map: CategoryColorMap,
                          top_k=10, title="", signed=True, savepath=None):
    """
    Plot aggregated SHAP by group.
    - If signed=True, bars can be positive or negative (directional effect).
    - If signed=False, plot absolute magnitudes only.
    """
    groups = np.asarray(groups)
    values = np.asarray(values, dtype=float)

    if signed:
        order = np.argsort(np.abs(values))[-top_k:][::-1]
    else:
        values = np.abs(values)
        order = np.argsort(values)[-top_k:][::-1]

    g_sel = groups[order]
    v_sel = values[order]
    cols  = color_map.colors(g_sel)

    plt.figure(figsize=(9, 5))
    plt.barh(list(g_sel[::-1]), list(v_sel[::-1]), color=cols[::-1])
    if signed:
        plt.axvline(0.0, color="k", lw=1)
    plt.title(title)
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
