# eval_utils.py
import os, numpy as np, torch, dgl
from dgl.dataloading import DataLoader, NeighborSampler, as_edge_prediction_sampler

def map_store_global_to_graph_local(g: dgl.DGLGraph, split_dir: str) -> torch.Tensor:
    store_global = np.load(os.path.join(split_dir, "edge_indices.npy")).astype(np.int64)
    g_global = g.edata[dgl.EID].cpu().numpy().astype(np.int64)
    order = np.argsort(g_global, kind="mergesort")
    g_sorted = g_global[order]
    pos = np.searchsorted(g_sorted, store_global)
    ok = (pos < g_sorted.size) & (g_sorted[pos] == store_global)
    if not np.all(ok):
        miss = store_global[~ok][:10]
        raise RuntimeError(f"[align] split has EIDs not in graph. Example: {miss.tolist()}")
    local = order[pos].astype(np.int64)
    return torch.from_numpy(local)

def make_eval_loader(g: dgl.DGLGraph, split_dir: str, fanouts=(15,10), batch_size=4096):
    # map store GLOBAL → graph LOCAL edge ids
    eids_local = map_store_global_to_graph_local(g, split_dir)
    # drop edges with y < 0 (unknown labels) if present
    if "y" in g.edata:
        valid_mask = (g.edata["y"] >= 0).cpu().numpy()
        eids_local = eids_local[torch.from_numpy(valid_mask[eids_local.cpu().numpy()])]
    sampler = as_edge_prediction_sampler(NeighborSampler(list(fanouts)))
    loader = DataLoader(g, eids_local, sampler, batch_size=batch_size, shuffle=False, drop_last=False)
    # cache split_dir + store_eids for batch mapping
    loader.split_dir = split_dir
    loader.store_eids = np.load(os.path.join(split_dir, "edge_indices.npy")).astype(np.int64)
    return loader
