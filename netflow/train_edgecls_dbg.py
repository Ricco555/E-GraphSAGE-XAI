# train_edgecls.py
from __future__ import annotations
import os, sys, argparse, logging
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl.dataloading import NeighborSampler, as_edge_prediction_sampler
from dgl.dataloading import DataLoader as DGLDataLoader  # avoid torch DataLoader clash

from feature_store import fetch_edge_features, _map_global_to_local
from debug_align import check_graph_vs_store

# Try to import your model; if unavailable, use a fallback.
try:
    from model import EdgeGraphSAGE  # must implement forward(blocks, x_nodes, pair_graph, e_feat)
except Exception:
    EdgeGraphSAGE = None


def _setup_logging(debug: bool):
    lvl = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout, force=True)
    os.environ["PYTHONUNBUFFERED"] = "1"

# ---------------- Fallback model (used only if you don't provide model.py) ----------------
class _FallbackEdgeGraphSAGE(nn.Module):
    def __init__(self, in_node=0, hidden=128, num_layers=2, aggregator='mean',
                 edge_in=0, edge_mlp_hidden=128, num_classes=8, dropout=0.3):
        super().__init__()
        import dgl.nn as dglnn
        self.sage = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_dummy_nodes = (in_node == 0)

        for li in range(num_layers):
            if li == 0:
                in_dim = in_node if in_node > 0 else hidden  # KEY: if no node feats, first layer takes `hidden`
            else:
                in_dim = hidden
            self.sage.append(dglnn.SAGEConv(in_dim, hidden, aggregator_type=aggregator))
            self.norms.append(nn.BatchNorm1d(hidden))

        if self.use_dummy_nodes:
            self.node_embed = nn.Embedding(1, hidden)  # learned constant node embedding

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + edge_in, edge_mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_mlp_hidden, num_classes),
        )

    def forward(self, blocks, x_nodes, pair_graph, e_feat):
        if x_nodes is None:
            h = self.node_embed.weight[0].expand(
                blocks[0].num_src_nodes(), self.node_embed.embedding_dim
            )
        else:
            h = x_nodes

        for conv, bn, block in zip(self.sage, self.norms, blocks):
            h_dst = h[:block.num_dst_nodes()]
            h = conv(block, (h, h_dst))
            h = bn(h)
            h = torch.relu(h)

        u, v = pair_graph.edges(form='uv')
        e_repr = torch.cat([h[u], h[v], e_feat], dim=1)
        return self.edge_mlp(e_repr)
    
    def encode(self, blocks: list[dgl.DGLBlock], x_nodes: torch.Tensor, return_src: bool = False) -> torch.Tensor:
        """
        Compute node embeddings for the last block's destination nodes by running the same
        per-block SAGE + norm + activation pipeline used in forward; returns the embeddings
        for the destination nodes of the final block.

        Backwards compatible:
        - Default (return_src=False): returns last_h_dst exactly as before.
        - If return_src=True: returns (h_src_init, last_h_dst) where h_src_init are the
          input node embeddings for blocks[0].src nodes (learned constant or x_nodes).
          This avoids breaking existing callers while enabling caller code that needs
          src-aligned embeddings (e.g. structural XAI helpers).
        """
        # Initialize node features (either provided or learned constant if no node feats)
        if x_nodes is None:
            # Expand learned constant node embedding to match the number of source nodes for the first block
            h = self.node_embed.weight[0].expand(
                blocks[0].num_src_nodes(), self.node_embed.embedding_dim
            )
        else:
            h = x_nodes

        # capture initial src-aligned embeddings if requested (keep a copy)
        h_src_init = h.clone() if return_src else None

        last_h_dst = None
        # Apply same conv -> norm -> relu sequence as in forward
        for conv, bn, block in zip(self.sage, self.norms, blocks):
            last_h_dst = h[:block.num_dst_nodes()]
            h = conv(block, (h, last_h_dst))
            h = bn(h)
            h = torch.relu(h)

        # return embeddings for destination nodes of the last block
        if return_src:
            return (h_src_init, last_h_dst)
        return last_h_dst

    def predict_from_embeddings(self, h_dst: torch.Tensor, pair_graph: dgl.DGLGraph, e_feat: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for edges in pair_graph given precomputed dst-endpoint embeddings `h_dst`.
        Assumes `pair_graph` node indexing aligns with rows of `h_dst` (i.e. edges refer to indices
        within the block that produced `h_dst`).
        """
        with pair_graph.local_scope():
            src, dst = pair_graph.edges(form='uv')
            # gather endpoint embeddings and concatenate with edge features
            he = torch.cat([h_dst[src], h_dst[dst]], dim=1)  # (B, 2*hidden)
            x = torch.cat([he, e_feat], dim=1)               # (B, 2*hidden + edge_in)
            return self.edge_mlp(x)

# ---------------- DataLoader (edge mode) ----------------
def make_edge_loader(g: dgl.DGLGraph, batch_size=2048, fanouts=(25, 15),
                     shuffle=True, num_workers=0, seed=42) -> callable:
    """
    Returns a function that builds a DGL DataLoader for given EIDs.
    We avoid passing device/pin_memory/persistent_workers to prevent version clashes.
    """
    base = NeighborSampler(list(fanouts))
    sampler = as_edge_prediction_sampler(base)  # yields (input_nodes, pair_graph, blocks)

    def _loader(eids: torch.Tensor) -> DGLDataLoader:
        gen = torch.Generator().manual_seed(seed)
        return DGLDataLoader(
            g, eids, sampler,
            batch_size=batch_size, shuffle=shuffle, drop_last=False,
            num_workers=num_workers, use_ddp=False, generator=gen
        )
    return _loader


def _print_block_shapes(blocks):
    for i, b in enumerate(blocks):
        logging.debug(f"  L{i}: src={b.num_src_nodes()} dst={b.num_dst_nodes()} E={b.num_edges()}")


# ---------------- One epoch (train or eval) ----------------
def run_epoch(model, loader: DGLDataLoader, g: dgl.DGLGraph, split_dir: str, device,
              optim: torch.optim.Optimizer | None, criterion: nn.Module | None, debug: bool = False):
    split_dir = getattr(loader, "split_dir", split_dir)  # prefer loader’s attached split_dir
    training = optim is not None
    model.train(training)
    total_loss, total_correct, total = 0.0, 0, 0

    for bidx, (input_nodes, pair_graph, blocks) in enumerate(loader):
        # Move to device explicitly
        if blocks and (blocks[0].device.type != torch.device(device).type):
            blocks = [b.to(device) for b in blocks]
        if pair_graph.device.type != torch.device(device).type:
            pair_graph = pair_graph.to(device)

        # Node features for this batch (or None if you don't have any)
        if "x" in g.ndata and g.ndata["x"].numel() > 0:
            x_nodes = g.ndata["x"][input_nodes].to(device)  # (src_nodes, d_node)
        else:
            x_nodes = None

        # 1) get LOCAL edge ids of the sampled edges (these are parent graph internal IDs)
        local_eids = pair_graph.edata[dgl.EID].to('cpu').numpy().astype(np.int64)

        # 2) map LOCAL edge ids -> GLOBAL flow ids via the split’s edge index array
        #    (edge_indices.npy is aligned with g’s local edge order)
        store_eids = np.load(os.path.join(split_dir, "edge_indices.npy")).astype(np.int64)
        global_eids = g.edata[dgl.EID][torch.from_numpy(local_eids)].cpu().numpy().astype(np.int64)
        # (optional debug—prove mapping == store mapping exactly)
        if debug:
            logging.info(f"[debug] local_eids[:8]={local_eids[:8].tolist()}")
            logging.info(f"[debug] mapped global_eids[:8]={global_eids[:8].tolist()}")
            logging.info(f"[debug] store_eids@local[:8]={store_eids[local_eids[:8]].tolist()}")
            ok = np.isin(global_eids, store_eids)
            logging.info(f"[debug] all mapped-in-store? {ok.all()} ({ok.sum()}/{ok.size})")
            # This must be True because `global_eids` are literally a subset of `store_eids`.
            assert ok.all(), "Mapping broken: mapped global ids must be in `store_eids`."
            store_eids = np.load(os.path.join(split_dir, "edge_indices.npy")).astype(np.int64)
            assert np.array_equal(global_eids, store_eids[local_eids]), "[debug] graph-based global_eids != store_eids[local_eids] (should never happen)"
        
        present = np.isin(global_eids, store_eids)
        if not present.all():
            bad = global_eids[~present][:10]
            raise RuntimeError(f"[batch-align] {bad.size} global ids not found in store: {bad.tolist()}")

        e_feat_np = fetch_edge_features(global_eids, split_dir)  # (B, d_edge)
        e_feat = torch.from_numpy(e_feat_np).to(device)

        # Labels aligned by LOCAL ids
        y = g.edata["y"][torch.from_numpy(local_eids)].to(device)

        if debug and bidx == 0:
            logging.debug(f"[DBG] batch={bidx} B={pair_graph.num_edges()} k={len(blocks)}")
            if x_nodes is None:
                logging.debug("       x_nodes=None (no node features)")
            else:
                logging.debug(f"       x_nodes={tuple(x_nodes.shape)} dtype={x_nodes.dtype}")
            _print_block_shapes(blocks)
            logging.debug(f"       e_feat={tuple(e_feat.shape)} dtype={e_feat.dtype}")
            logging.debug(f"       y={tuple(y.shape)} classes={torch.unique(y).tolist()}")

        if training:
            optim.zero_grad(set_to_none=True)

        try:
            logits = model(blocks, x_nodes, pair_graph, e_feat)  # (B, num_classes)
        except Exception:
            logging.error("Model forward failed. Shapes:")
            logging.error(f"  pair_graph.edges={pair_graph.num_edges()}")
            logging.error(f"  e_feat={tuple(e_feat.shape)}")
            if x_nodes is None:
                logging.error("  x_nodes=None")
            else:
                logging.error(f"  x_nodes={tuple(x_nodes.shape)}")
            _print_block_shapes(blocks)
            raise

        loss = (criterion(logits, y) if training else nn.functional.cross_entropy(logits, y))

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return (total_loss / max(total, 1), total_correct / max(total, 1))


# ---------------- Training runner ----------------
def run_training(args: argparse.Namespace | SimpleNamespace):
    _setup_logging(getattr(args, "debug", False))

    # Paths
    fs_train = os.path.join(args.feature_store, args.split_train)
    fs_val   = os.path.join(args.feature_store, args.split_val)

    # Load graphs (built previously and saved with dgl.save_graphs)
    g_train = dgl.load_graphs(os.path.join(args.graphs_dir, "train.bin"))[0][0]
    g_val   = dgl.load_graphs(os.path.join(args.graphs_dir, "val.bin"))[0][0]

    # If you drop invalid labels in VAL, filter *local* ids accordingly (example)
    if "y" in g_val.edata:
        keep_mask = (g_val.edata["y"] >= 0)              # per-local-edge boolean mask
    
    # 1) Preflight: verify graph ↔ store consistency; get the store’s GLOBAL EIDs
    store_train_ids = check_graph_vs_store(g_train, fs_train, verbose=True)
    store_val_ids   = check_graph_vs_store(g_val,   fs_val,   verbose=True)
    assert_graph_equals_store(g_train, fs_train, "train")
    assert_graph_equals_store(g_val,   fs_val,   "val")
    # Infer edge feature dim if needed
    if args.edge_in == 0:
        sample_eid = g_train.edata[dgl.EID][:1].cpu().numpy()
        e_feat_sample = fetch_edge_features(sample_eid, split_dir=fs_train)
        edge_in = int(e_feat_sample.shape[1])
    else:
        edge_in = int(args.edge_in)

    # --- Robust class mapping from TRAIN graph ---
    ytr_t = g_train.edata["y"].long().cpu()
    if ytr_t.numel() == 0:
        raise RuntimeError("[labels] g_train has zero edges/labels.")
    uniq_train = torch.unique(ytr_t)
    uniq_train = uniq_train[uniq_train >= 0]  # drop any negative labels if present
    if uniq_train.numel() < 2:
        raise RuntimeError(f"[labels] Need >=2 classes in TRAIN. Found: {uniq_train.tolist()}")

    # Build dense LUT old->new for train labels (contiguous 0..K-1)
    uniq_train_sorted = torch.sort(uniq_train).values
    remap = {int(o): int(i) for i, o in enumerate(uniq_train_sorted.tolist())}
    max_old = int(uniq_train_sorted.max().item())
    lut = torch.full((max_old + 1,), -1, dtype=torch.long)
    for o, n in remap.items():
        lut[o] = n

    # Remap TRAIN labels in-graph; assert no unknowns
    if ytr_t.max().item() > max_old:
        extend = ytr_t.max().item() - max_old
        lut = torch.cat([lut, torch.full((extend,), -1, dtype=torch.long)], dim=0)
    ytr_new = lut[ytr_t.clamp_min(0)]
    if (ytr_new < 0).any():
        bad = torch.unique(ytr_t[ytr_new < 0]).tolist()
        raise RuntimeError(f"[labels] TRAIN contains labels not in its own set: {bad}")
    g_train.edata["y"] = ytr_new

    # Remap VAL labels; unknown labels (not seen in train) → -1 and DROPPED
    yval_t = g_val.edata["y"].long().cpu()
    if yval_t.numel() == 0:
        raise RuntimeError("[labels] g_val has zero edges/labels.")
    if yval_t.max().item() > (lut.numel() - 1):
        extend = yval_t.max().item() - (lut.numel() - 1)
        lut = torch.cat([lut, torch.full((extend,), -1, dtype=torch.long)], dim=0)
    yval_new = lut[yval_t.clamp_min(0)]
    g_val.edata["y"] = yval_new

    num_classes = int(uniq_train.numel())

    device = torch.device(args.device)

    # Infer node feature dim
    if "x" in g_train.ndata and g_train.ndata["x"].numel() > 0:
        in_node = int(g_train.ndata["x"].shape[1])
    else:
        in_node = 0

    # Build model
    ModelClass = EdgeGraphSAGE if EdgeGraphSAGE is not None else _FallbackEdgeGraphSAGE
    model = ModelClass(
        in_node=in_node, hidden=args.hidden, num_layers=args.layers, aggregator=args.aggregator,
        edge_in=edge_in, edge_mlp_hidden=args.edge_mlp_hidden, num_classes=num_classes,
        dropout=args.dropout
    ).to(device)
    try:
        in0 = model.sage[0].in_feats if hasattr(model.sage[0], "in_feats") else None
        out0 = model.sage[0].out_feats if hasattr(model.sage[0], "out_feats") else None
        logging.info(f"[model] SAGE[0]: in={in0} out={out0} hidden={args.hidden} edge_in={edge_in} in_node={in_node}")
    except Exception:
        pass

    # Optim / loss (class weights from TRAIN after remap)
    binc = torch.bincount(g_train.edata["y"].cpu(), minlength=num_classes).float()
    weights = (binc.sum() / (binc + 1e-8)) / num_classes
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # DataLoaders (filter VAL to known labels)
    fanouts = tuple(int(x) for x in args.fanouts.split(","))
    train_loader_fn = make_edge_loader(g_train, batch_size=args.batch_size, fanouts=fanouts,
                                       shuffle=True, num_workers=args.num_workers, seed=args.seed)
    val_loader_fn   = make_edge_loader(g_val,   batch_size=args.batch_size, fanouts=fanouts,
                                       shuffle=False, num_workers=args.num_workers, seed=args.seed)

    # Map store GLOBAL → graph LOCAL ids
    eids_train_local = _map_global_to_local(g_train, fs_train)
    eids_val_local   = _map_global_to_local(g_val,   fs_val)
    train_loader = train_loader_fn(eids_train_local)
    val_loader   = val_loader_fn(eids_val_local)
    # attach split dirs to the loaders (so run_epoch can pick the correct store)
    train_loader.split_dir = fs_train
    val_loader.split_dir   = fs_val
    # Run the check right after creation:
    assert_loader_seed_alignment(g_train, eids_train_local, os.path.join(args.feature_store, "train"), name="train")
    assert_loader_seed_alignment(g_val,   eids_val_local,   os.path.join(args.feature_store, "val"),   name="val")

    # Optional first-batch inspection
    if getattr(args, "debug", False):
        logging.info("[debug] Inspecting first training batch shapes...")
        try:
            run_epoch(model, train_loader, g_train, fs_train, device, optim=None, criterion=None, debug=True)
        except Exception:
            logging.exception("[debug] Exception during dry-run batch.")
            raise

    # Train
    best_val = -1.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, g_train, fs_train, device, optim, criterion, debug=getattr(args, "debug", False))
        va_loss, va_acc = run_epoch(model, val_loader,   g_val,   fs_val,   device, None, None, debug=False)
        logging.info(f"[epoch {ep:02d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            os.makedirs("artifacts", exist_ok=True)
            torch.save(model.state_dict(), "artifacts/best_edge_sage.pt")

    logging.info(f"Best val acc: {best_val:.4f}")
    return {"best_val_acc": float(best_val), "edge_in": edge_in, "fanouts": fanouts}

# ---------------- Helper ----------------
def assert_loader_seed_alignment(g: dgl.DGLGraph, eids_local: torch.Tensor, split_dir: str, *, name: str):
    """
    g: graph used by the loader
    eids_local: the EXACT tensor of local edge IDs you passed to DataLoader
    split_dir: feature_store/<split> directory (has edge_indices.npy)
    name: a label for logs ('train' or 'val')
    """
    # 1) what the store says belongs to this split (GLOBAL EIDs)
    store_eids = np.load(os.path.join(split_dir, "edge_indices.npy")).astype(np.int64)

    # 2) what the loader will iterate (LOCAL -> GLOBAL)
    eids_local_np = eids_local.cpu().numpy().astype(np.int64)
    loader_global = g.edata[dgl.EID][torch.from_numpy(eids_local_np)].cpu().numpy().astype(np.int64)

    # 3) membership and shape sanity
    missing = np.setdiff1d(loader_global, store_eids, assume_unique=False)
    extra   = np.setdiff1d(store_eids, loader_global, assume_unique=False)

    logging.debug(f"[seed-check:{name}] seeds(local)={eids_local_np.size}  store={store_eids.size}")
    logging.debug(f"[seed-check:{name}] missing_in_store={missing.size}  extra_in_store={extra.size}")

    assert missing.size == 0, f"[{name}] Some loader seeds (GLOBAL) are not in {split_dir}. First: {missing[:10].tolist()}"

    # Optional: prove we're not accidentally using positional IDs (0..E-1) as GLOBAL IDs
    if np.array_equal(loader_global, eids_local_np):
        logging.debug(f"[seed-check:{name}] WARNING: loader 'GLOBAL' IDs equal local indices; "
                      "this would indicate you're not mapping local->global. Double-check.")

def assert_graph_equals_store(g, split_dir, split_name="train"):
    store = np.load(os.path.join(split_dir, "edge_indices.npy")).astype(np.int64)
    g_ids = g.edata[dgl.EID].cpu().numpy().astype(np.int64)
    logging.debug(f"[preflight:{split_name}] graph_edges={g_ids.size}  store_rows={store.size}")
    # Strong check: equal arrays
    if store.shape != g_ids.shape:
        raise AssertionError(f"[{split_name}] size mismatch: graph {g_ids.size} vs store {store.size}")
    if not np.array_equal(store, g_ids):
        # show a small diff to debug order
        logging.debug(" first graph EIDs   : %s", g_ids[:10].tolist())
        logging.debug(" first store EIDs   : %s", store[:10].tolist())
        raise AssertionError(f"[{split_name}] order mismatch: graph EIDs != store edge_indices")
    logging.debug(f"✅ [{split_name}] graph.edata[dgl.EID] is identical to {split_dir}/edge_indices.npy")


# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_store", default="feature_store", type=str)
    ap.add_argument("--graphs_dir",    default="graphs", type=str)
    ap.add_argument("--split_train",   default="train", type=str)
    ap.add_argument("--split_val",     default="val", type=str)
    ap.add_argument("--hidden", default=128, type=int)
    ap.add_argument("--layers", default=2, type=int)
    ap.add_argument("--aggregator", default="mean", choices=["mean","pool","lstm","gcn"])
    ap.add_argument("--edge_in", default=0, type=int)  # 0 => infer from a tiny batch
    ap.add_argument("--edge_mlp_hidden", default=128, type=int)
    ap.add_argument("--dropout", default=0.3, type=float)
    ap.add_argument("--fanouts", default="25,15", type=str)
    ap.add_argument("--batch_size", default=2048, type=int)
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--weight_decay", default=1e-4, type=float)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", default=0, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--debug", action="store_true", help="Print shapes for first batch and on failure")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
