import os, numpy as np, torch, shap, matplotlib.pyplot as plt
import dgl
from feature_store import fetch_edge_features

def load_feature_names(feature_store_dir: str):
    """
    Return: feature_names (list[str]), d_num (int), d_cat (int)
    Assumes artifacts saved encoder + numeric column names.
    """
    # adjust artifact path to where you saved them
    import joblib
    candidates = [
        os.path.join("artifacts", "scalers", "artifacts.pkl"),
        os.path.join(feature_store_dir, "artifacts.pkl"),
        os.path.join(feature_store_dir, "..", "artifacts.pkl"),
        os.path.join(feature_store_dir, "scalers", "artifacts.pkl"),
    ]
    arts_p = None
    for p in candidates:
        if os.path.exists(p):
            arts_p = p
            break
    if arts_p is None:
        raise FileNotFoundError(f"artifacts.pkl not found; tried the following paths: {candidates}")
    arts = joblib.load(arts_p)

    num_cols = arts.get("num_cols") or arts.get("numeric_cols")  # numeric column names
    d_num = len(num_cols)

    enc = arts.get("encoder")
    cat_cols = arts.get("cat_cols") or []
    cat_names = []
    if enc is not None and len(cat_cols) > 0:
        # OneHotEncoder.categories_ is a list of arrays in the same order as cat_cols
        idx = 0
        for col, cats in zip(cat_cols, enc.categories_):
            # each category produces a column
            for c in cats:
                cat_names.append(f"{col}={c}")
            idx += 1
    d_cat = len(cat_names)

    feature_names = list(num_cols) + cat_names
    return feature_names, d_num, d_cat

# Local SHAP for a single edge (feature-level)
# We freeze graph context (the pair embedding [h_src,h_dst]) 
# and explain only the effect of the edge feature vector on the logit/probability for a chosen class.
@torch.no_grad()
def compute_pair_embedding(model, blocks, x_nodes):
    """Return h_dst embeddings for the last block (D, hidden); uses your model.encode."""
    return model.encode(blocks, x_nodes)  # if node_in=0 your encode handles constant features

def make_edge_head_fn(model, he_fixed: np.ndarray, device="cpu", return_prob=True, target_class=None):
    """
    Wrap the edge-head to accept X_edge (B, edge_in) and output score/probability for target_class (B,).
    he_fixed: (2*hidden,) concatenated [h_src,h_dst] for the specific edge.
    """
    he_fixed_t = torch.from_numpy(he_fixed[None, :]).to(device)  # (1, 2*hidden)

    def f_edge(X: np.ndarray):
        X_t = torch.from_numpy(X.astype(np.float32, copy=False)).to(device)
        he = he_fixed_t.expand(X_t.size(0), -1)
        logits = model.edge_mlp(torch.cat([he, X_t], dim=1))  # (B, C)
        if target_class is None:
            # return max-logit or probabilities for SHAP multi-output
            if return_prob:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                return probs  # (B,C)
            return logits.detach().cpu().numpy()
        # single-class output
        if return_prob:
            probs = torch.softmax(logits, dim=1)[:, target_class]
            return probs.detach().cpu().numpy()
        return logits[:, target_class].detach().cpu().numpy()

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
            e_feat_np = fetch_edge_features(global_eids[pos:pos+1], split_dir)  # (1, edge_in)
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
    bg_X = fetch_edge_features(bg_ids, split_dir)  # (B_bg, edge_in)

    # 4) Wrap the edge-head for SHAP
    f = make_edge_head_fn(model, he_fixed, device=device, return_prob=True, target_class=target_class)
    explainer = shap.KernelExplainer(f, bg_X)
    sv = explainer.shap_values(x_edge_row[None, :], nsamples='auto')  # returns array(s)

    # If multi-output, pick target_class
    if isinstance(sv, list):
        if target_class is None:
            target_class = int(np.argmax(f(x_edge_row[None,:])[0]))
        shap_vals = sv[target_class]  # (1, edge_in)
        base_value = explainer.expected_value[target_class]
    else:
        shap_vals = sv  # (1, edge_in)
        base_value = explainer.expected_value

    return shap_vals, x_edge_row, y_true, base_value

# Global importance per class
# Aggregate absolute SHAP across many edges per class to get per-class top features.
def sample_edges_by_class(g, n_per_class=200, seed=0):
    """Return a dict: class_id -> list[local_edge_id]"""
    rng = np.random.default_rng(seed)
    y = g.edata["y"].cpu().numpy()
    out = {}
    for c in range(int(y.max())+1):
        idx = np.where(y == c)[0]
        if idx.size == 0: continue
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
            class2_meanabs[c] = np.mean(np.row_stack(vals), axis=0)
    return class2_meanabs

# Plot top-k features per class
def plot_topk_bar_per_class(class2_meanabs, feature_names, id2label, k=10, save_dir="artifacts/xai"):
    os.makedirs(save_dir, exist_ok=True)
    for c, imp in class2_meanabs.items():
        top_idx = np.argsort(imp)[::-1][:k]
        plt.figure(figsize=(8,5))
        plt.barh([feature_names[i] for i in top_idx][::-1], imp[top_idx][::-1])
        plt.title(f"Top-{k} Feature Importances (mean |SHAP|) — {id2label[c]}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"topk_shap_{id2label[c]}.png"), dpi=150)
        plt.show()

# Spider (radar) plot across classes
def plot_spider(class2_meanabs, feature_names, id2label, k_union=8, save_path="artifacts/xai/spider.png"):
    # 1) choose the union of top features
    chosen = set()
    for c, imp in class2_meanabs.items():
        top = np.argsort(imp)[::-1][:k_union]
        chosen.update(top.tolist())
    chosen = sorted(chosen)
    labels_feat = [feature_names[i] for i in chosen]

    # 2) normalize per class to [0,1]
    data = []
    classes_order = sorted(class2_meanabs.keys())
    for c in classes_order:
        imp = class2_meanabs[c][chosen]
        imp_norm = (imp - imp.min()) / (imp.max() - imp.min() + 1e-12)
        data.append(imp_norm)
    data = np.row_stack(data)

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
            class2_signed[c] = np.mean(np.row_stack(vals), axis=0)
    return class2_signed

def plot_signed_topk(class2_signed, feature_names, id2label, k=10, save_dir="artifacts/xai"):
    os.makedirs(save_dir, exist_ok=True)
    for c, v in class2_signed.items():
        # pick k with largest |effect|
        top_idx = np.argsort(np.abs(v))[::-1][:k]
        names = [feature_names[i] for i in top_idx][::-1]
        vals  = v[top_idx][::-1]
        colors = ["green" if x>0 else "red" for x in vals]
        plt.figure(figsize=(8,5))
        plt.barh(names, vals, color=colors)
        plt.axvline(0, color="k", lw=1)
        plt.title(f"Top-{k} Feature Impacts (signed SHAP) — {id2label[c]}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"signed_shap_{id2label[c]}.png"), dpi=150)
        plt.show()

# Structural XAI: neighbor masking
# estimates structural contribution by removing/masking neighbors and measuring the output delta.
# For a chosen edge (u,v), sample blocks as usual.
# Compute baseline logit/prob for class c.
# Identify 1-hop neighbors used in blocks[0] for u and v (via blocks[0].edges() and blocks[0].srcdata[dgl.NID]).
# For each neighbor n, rebuild a modified block where you remove edges from n→{u or v} (or zero out h_n in h_dst for a quick approximation), then recompute the logit delta for c.
# Rank neighbors by |delta|; group by attributes (e.g., protocol or subnet) for analyst-friendly views.
# “perturbation/ablation” estimate of structural contribution
def neighbor_impact_approx(model, h_dst, src_pos, dst_pos, e_feat_row, neighbors_pos, device="cpu", target_class=None):
    """
    h_dst: (D, hidden) embeddings for dst nodes in last block
    src_pos, dst_pos: tensors (B,), pick the one edge by index 'pos'
    neighbors_pos: list[int] positions in h_dst to "mask"
    Returns: dict {neighbor_idx_in_block -> delta_prob[target_class]}
    """
    impacts = {}
    base_logits = model.edge_mlp(torch.cat([torch.cat([h_dst[src_pos], h_dst[dst_pos]], dim=1),
                                            e_feat_row[None,:].to(device).expand(h_dst.size(0), -1)], dim=1))
    base_prob = torch.softmax(base_logits, dim=1)[0, target_class].item()

    for n in neighbors_pos:
        h_mod = h_dst.clone()
        h_mod[n] = 0.0
        logits = model.edge_mlp(torch.cat([torch.cat([h_mod[src_pos], h_mod[dst_pos]], dim=1),
                                           e_feat_row[None,:].to(device)], dim=1))
        prob   = torch.softmax(logits, dim=1)[0, target_class].item()
        impacts[int(n)] = base_prob - prob  # positive => neighbor increased prob for class
    return impacts
