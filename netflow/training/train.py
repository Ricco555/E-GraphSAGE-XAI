"""Training loop for EdgeGraphSAGE using the GraphBolt OnDiskDataset.

Replaces run_training() in train_edgecls_dbg.py.  Data comes from the
OnDiskDataset produced by scripts/prepare_data.py — no legacy feature_store/
or graphs/ directories required.

Key simplification over the legacy pipeline:
  All edges live in one flat graph (train+val+test concatenated).
  pair_graph.edata[dgl.EID] gives LOCAL edge IDs that ARE the OnDiskDataset
  feature indices.  No global/local remapping needed.

Usage:
  python training/train.py --config configs/train.yaml
  python training/train.py --config configs/train.yaml --epochs 5 --frac 0.1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy import sparse

HERE = Path(__file__).resolve().parent
NETFLOW_ROOT = HERE.parent
if str(NETFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(NETFLOW_ROOT))

import dgl
from datasets.nfunsw_nb15 import load_dataset, load_store_meta
from models.edge_graphsage import EdgeGraphSAGE
from models.gcn import GCNEdgeClassifier
from models.gat import GATEdgeClassifier
from training.samplers import build_dgl_graph, make_edge_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(debug: bool) -> None:
    lvl = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout, force=True)
    os.environ["PYTHONUNBUFFERED"] = "1"


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ---------------------------------------------------------------------------
# Feature fetching from OnDiskDataset
# ---------------------------------------------------------------------------

def fetch_edge_features(
    eids: np.ndarray,
    feature,           # gb.FeatureStore loaded by load_dataset()
    x_cat_csr: sparse.csr_matrix,
) -> torch.Tensor:
    """Return (B, d_num + d_cat) float32 feature tensor for the given edge IDs."""
    ids_t = torch.from_numpy(eids.astype(np.int64))
    x_num = feature.read("edge", None, "x_num", ids=ids_t)          # (B, d_num)
    x_cat_np = x_cat_csr[eids].toarray().astype(np.float32)         # (B, d_cat)
    x_cat = torch.from_numpy(x_cat_np)
    return torch.cat([x_num.float(), x_cat], dim=1)


# ---------------------------------------------------------------------------
# One epoch (train or eval)
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader,
    g: dgl.DGLGraph,
    feature,
    x_cat_csr: sparse.csr_matrix,
    device: torch.device,
    optim: torch.optim.Optimizer | None,
    criterion: nn.Module | None,
    debug: bool = False,
) -> tuple[float, float, float]:
    training = optim is not None
    model.train(training)
    total_loss = total_correct = total = 0
    t0 = time.time()

    for bidx, (input_nodes, pair_graph, blocks) in enumerate(loader):
        blocks = [b.to(device) for b in blocks]
        pair_graph = pair_graph.to(device)

        x_nodes = None  # no structural node features; EdgeGraphSAGE uses learned embedding

        # LOCAL edge IDs in the flat graph == OnDiskDataset feature indices.
        local_eids = pair_graph.edata[dgl.EID].cpu().numpy().astype(np.int64)

        e_feat = fetch_edge_features(local_eids, feature, x_cat_csr).to(device)
        y = g.edata["y"][torch.from_numpy(local_eids)].view(-1).to(device)

        if debug and bidx == 0:
            logging.debug(
                f"[DBG] batch=0 B={pair_graph.num_edges()} "
                f"e_feat={tuple(e_feat.shape)} y_classes={torch.unique(y).tolist()}"
            )

        if training:
            optim.zero_grad(set_to_none=True)

        logits = model(blocks, x_nodes, pair_graph, e_feat)
        loss = criterion(logits, y) if training else nn.functional.cross_entropy(logits, y)

        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total += bs

    elapsed = time.time() - t0
    return total_loss / max(total, 1), total_correct / max(total, 1), elapsed


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_model(m_cfg: dict, edge_in: int, num_classes: int) -> nn.Module:
    """Instantiate a model from the 'model' section of the config.

    Dispatches on m_cfg.get("type", "edge_graphsage").
    """
    model_type = m_cfg.get("type", "edge_graphsage")
    hidden = m_cfg["hidden"]
    dropout = m_cfg.get("dropout", 0.3)

    if model_type == "edge_graphsage":
        return EdgeGraphSAGE(
            in_node=0,
            hidden=hidden,
            num_layers=m_cfg.get("num_layers", 2),
            aggregator=m_cfg.get("aggregator", "mean"),
            edge_in=edge_in,
            edge_mlp_hidden=m_cfg.get("edge_mlp_hidden", hidden),
            num_classes=num_classes,
            dropout=dropout,
        )
    if model_type == "gcn":
        return GCNEdgeClassifier(
            node_in=0,
            edge_in=edge_in,
            hidden=hidden,
            num_classes=num_classes,
            dropout=dropout,
        )
    if model_type == "gat":
        return GATEdgeClassifier(
            in_node=0,
            hidden=hidden,
            num_heads=m_cfg.get("num_heads", 4),
            num_layers=m_cfg.get("num_layers", 2),
            edge_in=edge_in,
            edge_mlp_hidden=m_cfg.get("edge_mlp_hidden", hidden),
            num_classes=num_classes,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model type: {model_type!r}. "
                     "Choose from: edge_graphsage, gcn, gat")


# ---------------------------------------------------------------------------
# Main training runner
# ---------------------------------------------------------------------------

def run_training(cfg: dict) -> dict:
    ds_cfg  = cfg["dataset"]
    m_cfg   = cfg["model"]
    t_cfg   = cfg["training"]
    out_cfg = cfg["out"]
    debug   = cfg.get("debug", False)

    _setup_logging(debug)
    _seed_everything(t_cfg["seed"])

    device = _resolve_device(t_cfg["device"])
    logging.info(f"[train] device={device}")

    root = ds_cfg["root"]
    meta = load_store_meta(root)
    logging.info(f"[train] n_nodes={meta.n_nodes} n_edges={meta.n_edges} "
                 f"d_num={meta.d_num} d_cat={meta.d_cat} num_classes={meta.num_classes}")

    # Load feature store (x_num, y, ts; x_cat handled separately as CSR).
    ds = load_dataset(root)

    # Load CSR directly — avoids needing feature store to support sparse slicing.
    x_cat_csr = sparse.load_npz(os.path.join(root, "features", "x_cat.npz"))

    # Build a DGL graph from the flat edge list.
    g = build_dgl_graph(root, meta.n_nodes)

    # Attach labels onto the graph (indexed by LOCAL = OnDiskDataset edge ID).
    y_all = ds.feature.read("edge", None, "y").view(-1)  # (E,) int64 on CPU
    g.edata["y"] = y_all.long()

    # Split eids (LOCAL ids — they ARE the feature indices).
    train_eids = torch.from_numpy(
        np.load(os.path.join(root, "tasks", "train_eids.npy")).astype(np.int64)
    )
    val_eids = torch.from_numpy(
        np.load(os.path.join(root, "tasks", "val_eids.npy")).astype(np.int64)
    )

    # Optional frac subsampling (for hyperparameter sweeps).
    frac = cfg.get("frac", None)
    if frac is not None and frac < 1.0:
        from sklearn.model_selection import StratifiedShuffleSplit
        y_tr = y_all[train_eids].numpy()
        y_va = y_all[val_eids].numpy()
        sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=t_cfg["seed"])
        keep_tr, _ = next(sss.split(np.arange(len(train_eids)), y_tr))
        keep_va, _ = next(sss.split(np.arange(len(val_eids)), y_va))
        train_eids = train_eids[torch.from_numpy(np.sort(keep_tr))]
        val_eids   = val_eids[torch.from_numpy(np.sort(keep_va))]
        logging.info(f"[train] subsampled frac={frac}: train={len(train_eids)} val={len(val_eids)}")

    # Class weights from TRAIN edges.
    y_train = y_all[train_eids].view(-1).long()
    num_classes = int(meta.num_classes)
    binc = torch.bincount(y_train, minlength=num_classes).float()
    weights = (binc.sum() / (binc + 1e-8)) / num_classes
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # Model.
    fanouts = tuple(t_cfg["fanouts"])
    edge_in = meta.d_num + meta.d_cat
    model = _build_model(m_cfg, edge_in, num_classes).to(device)
    logging.info(f"[train] model params={sum(p.numel() for p in model.parameters()):,} "
                 f"edge_in={edge_in}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"]
    )

    # DataLoaders.
    train_loader = make_edge_loader(
        g, train_eids, fanouts=fanouts,
        batch_size=t_cfg["batch_size"], shuffle=True,
        num_workers=t_cfg["num_workers"], seed=t_cfg["seed"],
    )
    val_loader = make_edge_loader(
        g, val_eids, fanouts=fanouts,
        batch_size=t_cfg["batch_size"], shuffle=False,
        num_workers=t_cfg["num_workers"], seed=t_cfg["seed"],
    )

    # Training loop.
    best_val = -1.0
    history = []
    for ep in range(1, t_cfg["epochs"] + 1):
        tr_loss, tr_acc, tr_time = run_epoch(
            model, train_loader, g, ds.feature, x_cat_csr, device, optim, criterion, debug,
        )
        va_loss, va_acc, va_time = run_epoch(
            model, val_loader, g, ds.feature, x_cat_csr, device, None, None, False,
        )
        logging.info(
            f"[epoch {ep:02d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )
        history.append({
            "epoch": ep, "train_loss": float(tr_loss), "train_acc": float(tr_acc),
            "train_time": float(tr_time), "val_loss": float(va_loss),
            "val_acc": float(va_acc), "val_time": float(va_time),
        })

        if va_acc > best_val:
            best_val = va_acc
            os.makedirs(out_cfg["artifacts_dir"], exist_ok=True)
            torch.save(model.state_dict(), out_cfg["checkpoint"])

    logging.info(f"[train] best_val_acc={best_val:.4f}")
    os.makedirs(out_cfg["artifacts_dir"], exist_ok=True)
    with open(out_cfg["train_log"], "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {"best_val_acc": float(best_val), "edge_in": edge_in, "fanouts": fanouts}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    # Allow CLI overrides for key training settings.
    ap.add_argument("--epochs",       type=int,   default=None)
    ap.add_argument("--batch_size",   type=int,   default=None)
    ap.add_argument("--lr",           type=float, default=None)
    ap.add_argument("--hidden",       type=int,   default=None)
    ap.add_argument("--frac",         type=float, default=None,
                    help="Stratified subsample fraction of TRAIN/VAL (for tuning).")
    ap.add_argument("--seed",         type=int,   default=None)
    ap.add_argument("--device",       type=str,   default=None)
    ap.add_argument("--debug",        action="store_true")
    ap.add_argument("--num_layers",   type=int,   default=None)
    ap.add_argument("--aggregator",   type=str,   default=None,
                    choices=["mean", "pool", "lstm", "gcn"])
    ap.add_argument("--dropout",      type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--fanouts",      type=str,   default=None,
                    help="Comma-separated fanouts e.g. 25,15")
    return ap.parse_args()


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Write CLI overrides into the config dict (mutates in place)."""
    if args.epochs      is not None: cfg["training"]["epochs"]       = args.epochs
    if args.batch_size  is not None: cfg["training"]["batch_size"]   = args.batch_size
    if args.lr          is not None: cfg["training"]["lr"]           = args.lr
    if args.hidden      is not None: cfg["model"]["hidden"]          = args.hidden
    if args.seed        is not None: cfg["training"]["seed"]         = args.seed
    if args.device      is not None: cfg["training"]["device"]       = args.device
    if args.frac        is not None: cfg["frac"]                     = args.frac
    if args.debug:                   cfg["debug"]                    = True
    if args.num_layers  is not None: cfg["model"]["num_layers"]      = args.num_layers
    if args.aggregator  is not None: cfg["model"]["aggregator"]      = args.aggregator
    if args.dropout     is not None: cfg["model"]["dropout"]         = args.dropout
    if args.weight_decay is not None: cfg["training"]["weight_decay"] = args.weight_decay
    if args.fanouts     is not None:
        cfg["training"]["fanouts"] = [int(f) for f in args.fanouts.split(",")]
    return cfg


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _apply_overrides(cfg, args)
    result = run_training(cfg)
    print(json.dumps(result, indent=2))
