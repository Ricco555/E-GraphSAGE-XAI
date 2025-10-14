# train_edgecls.py
from __future__ import annotations
import os, argparse, json
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl.dataloading import NeighborSampler, as_edge_prediction_sampler, DataLoader

# === parts created in previous steps ===
from feature_store import fetch_edge_features
# If we already have a model.py, import it; otherwise the fallback class below is used.
try:
    from model import EdgeGraphSAGE   # expects forward(blocks, x_nodes, pair_graph, e_feat)
except Exception:
    EdgeGraphSAGE = None

# ---------- minimal fallback model (remove if you have your own) ----------
class _FallbackEdgeGraphSAGE(nn.Module):
    def __init__(self, in_node=0, hidden=128, num_layers=2, aggregator='mean',
                 edge_in=0, edge_mlp_hidden=128, num_classes=8, dropout=0.3):
        super().__init__()
        import dgl.nn as dglnn
        self.sage = nn.ModuleList()
        self.norms = nn.ModuleList()
        for li in range(num_layers):
            in_dim = in_node if li == 0 else hidden
            self.sage.append(dglnn.SAGEConv(in_dim, hidden, aggregator_type=aggregator))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.use_dummy_nodes = (in_node == 0)
        if self.use_dummy_nodes:
            self.node_embed = nn.Embedding(1, hidden)  # learned constant per src node
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + edge_in, edge_mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_mlp_hidden, num_classes),
        )

    def forward(self, blocks, x_nodes, pair_graph, e_feat):
        h = x_nodes
        if h.numel() == 0:
            h = self.node_embed.weight[0].expand(blocks[0].num_src_nodes(), -1)
        for conv, bn, block in zip(self.sage, self.norms, blocks):
            h_dst = h[:block.num_dst_nodes()]
            h = conv(block, (h, h_dst))
            h = bn(h)
            h = torch.relu(h)
        u, v = pair_graph.edges(form='uv')
        hu, hv = h[u], h[v]
        e_repr = torch.cat([hu, hv, e_feat], dim=1)
        return self.edge_mlp(e_repr)

# ---------- dataloader (edge mode) ----------
def make_edge_loader(g: dgl.DGLGraph, batch_size=2048, fanouts=(15, 10),
                     shuffle=True, num_workers=0, seed=42, device='cpu') -> DataLoader:
    base = NeighborSampler(list(fanouts))
    sampler = as_edge_prediction_sampler(base)  # yields (input_nodes, pair_graph, blocks)
    eids = torch.arange(g.num_edges(), dtype=torch.int64)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        g, eids, sampler,
        batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers, use_ddp=False, generator=gen, device=device,
        persistent_workers=False, pin_memory=False,
    )
    return loader

# ---------- one epoch ----------
def run_epoch(model, loader: DataLoader, g: dgl.DGLGraph, split_dir: str, device, optim=None, criterion=None):
    train = optim is not None
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for input_nodes, pair_graph, blocks in loader:
        # if device was set in DataLoader, blocks/pair_graph arrive on device already
        if blocks[0].device != torch.device(device):
            blocks = [b.to(device) for b in blocks]
            pair_graph = pair_graph.to(device)

        # node features (optional)
        x_nodes = g.ndata.get("x", torch.empty(blocks[0].num_src_nodes(), 0)).to(device)

        # edge features by GLOBAL EIDs
        global_eids = pair_graph.edata[dgl.EID].cpu().numpy()
        e_feat = torch.from_numpy(fetch_edge_features(global_eids, split_dir)).to(device)

        # labels aligned by GLOBAL EIDs
        y = g.edata["y"][pair_graph.edata[dgl.EID]].to(device)

        if train:
            optim.zero_grad(set_to_none=True)
        logits = model(blocks, x_nodes, pair_graph, e_feat)
        loss = (criterion(logits, y) if train else nn.functional.cross_entropy(logits, y))
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return (total_loss / max(total, 1), total_correct / max(total, 1))

# ---------- training runner ----------
def run_training(args: argparse.Namespace | SimpleNamespace):
    # paths
    fs_train = os.path.join(args.feature_store, args.split_train)
    fs_val   = os.path.join(args.feature_store, args.split_val)

    # load graphs you saved earlier
    g_train = dgl.load_graphs(os.path.join(args.graphs_dir, "train.bin"))[0][0]
    g_val   = dgl.load_graphs(os.path.join(args.graphs_dir, "val.bin"))[0][0]

    # infer edge feature dim if needed
    if args.edge_in == 0:
        sample_eid = g_train.edata[dgl.EID][:1].cpu().numpy()
        e_feat_sample = fetch_edge_features(sample_eid, split_dir=fs_train)
        edge_in = int(e_feat_sample.shape[1])
    else:
        edge_in = int(args.edge_in)

    # classes
    num_classes = int(np.load(os.path.join(fs_train, "y.npy")).max()) + 1

    # model
    device = torch.device(args.device)
    ModelClass = EdgeGraphSAGE if EdgeGraphSAGE is not None else _FallbackEdgeGraphSAGE
    model = ModelClass(
        in_node=0, hidden=args.hidden, num_layers=args.layers, aggregator=args.aggregator,
        edge_in=edge_in, edge_mlp_hidden=args.edge_mlp_hidden, num_classes=num_classes,
        dropout=args.dropout
    ).to(device)

    # loss with class weights from train labels
    y_train = torch.from_numpy(np.load(os.path.join(fs_train, "y.npy")))
    binc = torch.bincount(y_train, minlength=num_classes).float()
    weights = (binc.sum() / (binc + 1e-8)) / num_classes
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # loaders
    fanouts = tuple(int(x) for x in args.fanouts.split(","))
    train_loader = make_edge_loader(g_train, batch_size=args.batch_size, fanouts=fanouts,
                                    shuffle=True, num_workers=args.num_workers, seed=args.seed,
                                    device="cpu")  # keep sampling on CPU for stability
    val_loader   = make_edge_loader(g_val,   batch_size=args.batch_size, fanouts=fanouts,
                                    shuffle=False, num_workers=args.num_workers, seed=args.seed,
                                    device="cpu")

    best_val = -1.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, g_train, fs_train, device, optim, criterion)
        va_loss, va_acc = run_epoch(model, val_loader,   g_val,   fs_val,   device, None, None)
        print(f"[epoch {ep:02d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            os.makedirs("artifacts", exist_ok=True)
            torch.save(model.state_dict(), "artifacts/best_edge_sage.pt")

    print(f"Best val acc: {best_val:.4f}")
    return {"best_val_acc": float(best_val), "edge_in": edge_in, "fanouts": fanouts}

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_store", default="feature_store", type=str)
    ap.add_argument("--graphs_dir",    default="graphs", type=str)
    ap.add_argument("--split_train",   default="train", type=str)
    ap.add_argument("--split_val",     default="val", type=str)
    ap.add_argument("--hidden", default=128, type=int)
    ap.add_argument("--layers", default=2, type=int)
    ap.add_argument("--aggregator", default="mean", choices=["mean","pool","lstm","gcn"])
    ap.add_argument("--edge_in", default=0, type=int, help="0 = infer from sample")
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
    return ap.parse_args()

def main(args):
    run_training(args)

if __name__ == "__main__":
    args = parse_args()
    run_training(args)

"""
# How to use
From the command line:

python train_edgecls.py \
  --graphs_dir graphs \
  --feature_store feature_store \
  --fanouts 25,15 \
  --layers 2 \
  --hidden 128 \
  --batch_size 2048 \
  --epochs 10

# Run from Jupyter notebook:

  from types import SimpleNamespace
from train_edgecls import run_training

args = SimpleNamespace(
    feature_store="feature_store",
    graphs_dir="graphs",
    split_train="train",
    split_val="val",
    hidden=128,
    layers=2,
    aggregator="mean",
    edge_in=0,               # infer from sample
    edge_mlp_hidden=128,
    dropout=0.3,
    fanouts="25,15",
    batch_size=2048,
    epochs=10,
    lr=3e-4,
    weight_decay=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_workers=0,
    seed=42,
)
run_training(args)
"""