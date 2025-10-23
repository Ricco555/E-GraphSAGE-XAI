# eval_infer.py
import torch, numpy as np, dgl
from typing import Tuple, List, Dict

@torch.no_grad()
def infer_split(model, loader, g, device, fetch_edge_features) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        y_true: (N,) int
        y_pred: (N,) int
        y_prob: (N, C) float (softmax probabilities), needed for ROC
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    for input_nodes, pair_graph, blocks in loader:
        # ensure on device
        if blocks and blocks[0].device.type != torch.device(device).type:
            blocks = [b.to(device) for b in blocks]
        if pair_graph.device.type != torch.device(device).type:
            pair_graph = pair_graph.to(device)

        # LOCAL → GLOBAL mapping via parent graph g (robust)
        local_eids = pair_graph.edata[dgl.EID].cpu().numpy().astype(np.int64)
        global_eids = g.edata[dgl.EID][torch.from_numpy(local_eids)].cpu().numpy().astype(np.int64)

        # edge features (GLOBAL ids + split_dir)
        x_edge_np = fetch_edge_features(global_eids, loader.split_dir)
        x_edge = torch.from_numpy(x_edge_np).to(device)

        # node features (if your model needs them; else pass None)
        # For SAGE, you pass node features aligned with blocks[0].srcnodes()
        # If you're using degree features:
        #   x_nodes = x_nodes_full[blocks[0].srcnodes()].to(device)
        # else:
        x_nodes = None

        # forward
        logits = model(blocks, x_nodes, pair_graph, x_edge)  # shape (B, C)
        probs  = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred   = logits.argmax(dim=1).detach().cpu().numpy()
        y      = g.edata["y"][torch.from_numpy(local_eids)].cpu().numpy()

        # drop unknown labels if any slipped in (paranoia)
        keep = (y >= 0)
        if not np.all(keep):
            y    = y[keep]; pred = pred[keep]; probs = probs[keep]

        all_true.append(y)
        all_pred.append(pred)
        all_prob.append(probs)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)
    return y_true, y_pred, y_prob
