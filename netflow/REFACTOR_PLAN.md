# TE-G-SAGE-XAI — Refactor & Modernization Plan

**Author:** Phase-1 exploration deliverable
**Date:** 2026-05-03
**Target stack:** DGL (built from source, CUDA, GraphBolt enabled, no HugeCTR), PyTorch + CUDA, WSL.
**Scope:** Convert the notebook-driven research pipeline under `netflow/` into a clean, GPU-first DGL + GraphBolt project. **No code changes are made in Phase 1.** This document is for review.

---

## 1. Current Structure Summary

The codebase is a research pipeline for **edge classification on NetFlow records** (NF-UNSW-NB15-v3). Each NetFlow record becomes a directed edge between two endpoint nodes, where an endpoint is a `IP:port` string. The model is an **E-GraphSAGE**-style classifier: multi-layer `dgl.nn.SAGEConv` produces per-node embeddings, then a per-edge MLP consumes `[h_src ‖ h_dst ‖ e_feat]` and outputs a class logit. Edge features (numeric + one-hot categorical = ~601 dims) are stored **off-graph** in a per-split memmap+CSR feature store; only labels, timestamps, and global edge IDs (EIDs) live on the graph. A non-trivial alignment invariant — `g.edata[dgl.EID]` must exactly match `feature_store/<split>/edge_indices.npy` — is enforced by hand via assertions.

### 1.1 Notebook-by-notebook

**`01_E-GraphSAGE_NFNB15v3_mean_agg_multiclass.ipynb`** (28 cells, ~3.9 MB).
End-to-end driver. Imports many *unused* legacy libs at the top (`networkx`, `category_encoders`, `sklearn.LabelEncoder`, `train_test_split`, `PCA`, `seaborn`). Reads raw CSV → `clean_nfunsw_nb15` → concatenates `IP:port` to form host node identifiers → drops binary `Label` (multi-class only) → renames `Attack`→`label`. Then chronological 60/30/10 split, label fitting on TRAIN, numeric pipeline (log1p, scaler, zero-var prune, optional Spearman prune at 0.995) with correlation heatmap, OHE on numeric-coded categoricals (PROTOCOL, L7_PROTO, ICMP_TYPE, ICMP_IPV4_TYPE, DNS_QUERY_TYPE, DNS_QUERY_ID, FTP_COMMAND_RET_CODE), feature-store materialization, light DGL graph build (CPU-only), small 10% stratified subset for tuning, sanity checks (graph EIDs vs store EIDs), and finally a 5-axis grid search (hidden, aggregator, fanouts, dropout, batch_size) followed by a final retrain with the best params.

**`012_Complexity_test.ipynb`** (21 cells). Re-runs the same preprocessing, then trains the model on `FRAC ∈ {0.25, 0.5, 0.75}` of TRAIN+VAL to measure scaling; reads back per-epoch `train_time` / `val_time` from saved histories and plots time vs fraction. **Side effect warning** noted in markdown cell 1: this notebook overwrites `artifacts/best_edge_sage.pt`.

**`02_E-SAGE_metrics.ipynb`** (11 cells). Loads `best_edge_sage.pt` (rebuilds `_FallbackEdgeGraphSAGE` with hyperparams read back from `best_params.json`), runs inference on the **test** graph via `make_eval_loader`, computes per-class P/R/F1/FAR + macro, generates confusion matrix (counts and normalized), per-class FPR, ROC OvR, Precision-Recall curves, calibration plots (binary Attack-vs-Benign and per-class), and writes a CSV/LaTeX results table. Note: variable named `g_val` actually loads `graphs/test.bin` and `fs_val = "feature_store/test"` — the metrics are computed on the test split, but the file paths still say `*_val.json` etc., which is misleading.

**`03_E-SAGE_XAI.ipynb`** (6 cells). Loads the trained model and test graph, computes per-class **mean |SHAP|** via `aggregate_shap_per_class` (KernelSHAP over the edge MLP head with frozen `[h_src ‖ h_dst]`), produces top-k bar charts per class, spider/radar plots over the union of top features, beeswarm plots per class, signed (directional) SHAP, group-aggregated SHAP (collapsing one-hot columns back to their parent column), and a heatmap over the union of top features × classes. Also runs **structural XAI** via `visualize_neighbor_impacts`, which masks individual neighbor embeddings and measures `Δ P(class)`. Cell 5 contains heavy defensive shape-handling for SHAP outputs (3 different shape branches with multiple fallbacks), suggesting `local_shap_for_edge` returns inconsistent shapes depending on `target_class`.

**`04_baseline-xg-gcn.ipynb`** (19 cells). Two baselines on the **same temporal splits**: **(a)** XGBoost, fed only the tabular edge feature vector (40k stratified per class), trained with `tree_method="hist"` and `device="cuda"`; **(b)** a minimal **GCN** edge classifier — `dglnn.GraphConv` × 2 + the same edge MLP head — trained for 25 epochs with `Adam` and gradient clipping. Both write metrics JSON and ROC plots; a final cell assembles a comparison CSV/LaTeX table that pulls in the GraphSAGE numbers from `metrics_val.json`.

### 1.2 Helper modules

| File | Role | LoC | Key smells |
|---|---|---|---|
| `data_cleaning.py` | Drop NA IDs, ±inf→NA, fillna(0). | 185 | OK. CLI works. |
| `chronological_split.py` | Sort by start time, 60/30/10 with boundary-safe cuts at equal timestamps. | 195 | OK. |
| `label_mapping.py` | Fit string→int on TRAIN, transform val/test, class weights. | 150 | OK. |
| `feature_numeric.py` | log1p (where ≥0 ≥99.5%), StandardScaler, zero-var drop, Spearman prune at 0.995. | 269 | Spearman prune is O(d²) Python loop with column-wise rank; slow for d>500. |
| `categorical_encoding.py` | OHE on TRAIN with rare-collapse + optional IANA port bucketing; CSR float32. | 231 | OK. Has a hack to inject `__RARE__` into one row when no rare values present. |
| `feature_store.py` | Build per-split numeric memmap + CSR + EID/y/ts arrays + sorted EID lookup index. | 297 | `fetch_edge_features` re-opens memmap and CSR every call (no cache); `_load_numeric_memmap` re-derives shape every batch. |
| `graph_build.py` | IP→node-ID dict, build DGL directed multigraph, save with `dgl.save_graphs`. | 131 | `_make_ip_ids` is a **Python `for` loop over every row** — slow for millions of edges. |
| `train_edgecls_dbg.py` | The training loop, `_FallbackEdgeGraphSAGE` model, alignment assertions. | 479 | `np.load(edge_indices.npy)` **inside the per-batch loop** (lines 180, 191). Custom global↔local EID mapping done by hand 3× over (`_map_global_to_local`, `map_eids_to_rows`, the inline batch mapping). Re-maps labels on graph in-place. |
| `eval_utils.py` | Build a no-shuffle eval loader; map store-global EIDs to graph-local EIDs. | 31 | OK. |
| `eval_infer.py` | Inference loop: returns y_true, y_pred, y_prob. | 56 | OK. |
| `eval_metrics.py` | Per-class P/R/F1, FAR (one-vs-rest from CM). | 56 | OK. |
| `eval_roc.py` | ROC OvR plot, false-negative dict, confusion matrix plot. | 125 | OK. |
| `xai.py` | KernelSHAP feature-level explanations + neighbor-masking structural XAI + grouping/plotting. | 843 | Largest single file. Heavy shape-handling everywhere because `local_shap_for_edge` returns one of: `(1, d)` ndarray, list of `(1, d)` per class, etc. Globals leakage (`feature_names = globals().get(...)`). |
| `corr_visualisation.py` | Pre-pruning Spearman heatmap + top-k pairs CSV. | 147 | OK; Python loop in `_top_pairs` is O(d²) but only used once. |
| `debug_align.py` | Pre-flight sanity check: graph EIDs ⊂ store EIDs. | 32 | OK. |

### 1.3 Generated artifacts (gitignored)

- `data/` — raw CSVs.
- `feature_store/{train,val,test}/` — memmap + CSR + `edge_indices.npy` + `y.npy` + `ts.npy` + `eid_map.npz`.
- `graphs/{train,val,test}.bin` (+ `.ip2id.npz`).
- `artifacts/` — `best_edge_sage.pt`, `best_params.json`, `label_map.json`, history JSONs, plots.
- Hyper-tuning subset: `hyper_tune_small/`, `graphs_small/`.
- Complexity sweeps: `complexity_fs_frac{25,50,75}/`, `graphs_small_frac{25,50,75}/`.

---

## 2. Proposed New File / Folder Structure

```
netflow/
├── data/                       # raw CSVs + processed parquet (existing, gitignored)
├── datasets/
│   ├── __init__.py
│   ├── nfunsw_nb15.py          # NFUNSWNB15Dataset(dgl.data.DGLDataset) + GraphBolt OnDiskDataset config
│   ├── feature_pipeline.py     # numeric + OHE pipelines (was feature_numeric + categorical_encoding)
│   ├── chronological_split.py  # unchanged logic, kept here
│   ├── label_mapping.py        # unchanged logic, kept here
│   └── data_cleaning.py        # unchanged logic, kept here
├── models/
│   ├── __init__.py
│   ├── edge_graphsage.py       # EdgeGraphSAGE (canonical impl, no fallback fork)
│   ├── gcn.py                  # GCN_EdgeClassifier (was inline in nb 04)
│   ├── gat.py                  # GAT_EdgeClassifier (NEW; GATConv + same edge MLP head)
│   └── edge_head.py            # shared EdgeMLP module used by all 3 models
├── layers/
│   └── __init__.py             # only if a custom message-passing layer is needed (empty for now)
├── training/
│   ├── __init__.py
│   ├── train.py                # CLI entry; replaces train_edgecls_dbg.run_training
│   ├── evaluate.py             # CLI eval; replaces nb 02 cells 0,3,6,8,10
│   ├── samplers.py             # GraphBolt pipeline factory (NeighborSampler + FeatureFetcher + CopyTo)
│   └── hyperparam.py           # grid/random search loop (replaces nb 01 cells 26-27)
├── xai/
│   ├── __init__.py
│   ├── shap_explainer.py       # KernelSHAP wrapper, simplified return contract
│   ├── neighbor_masking.py     # structural XAI (was xai.neighbor_impact_approx + visualize_*)
│   ├── grouping.py             # group OHE columns back to parent (was xai.build_group_index, etc.)
│   └── plots.py                # all matplotlib plotting helpers
├── utils/
│   ├── __init__.py
│   ├── profiling.py            # torch.profiler wrapper + nvidia-smi sampler + DGL timer hooks
│   ├── io.py                   # load/save graphs, feature stores, label maps
│   ├── seeds.py                # deterministic seeding helpers
│   └── alignment.py            # the EID-store check, simplified after GraphBolt cuts most of this
├── configs/
│   ├── data.yaml               # paths, split ratios, columns
│   ├── model_sage.yaml         # E-GraphSAGE hyperparams
│   ├── model_gcn.yaml          # GCN baseline
│   ├── model_gat.yaml          # GAT baseline
│   └── train.yaml              # batch_size, fanouts, lr, epochs, device, profiling
├── scripts/
│   ├── prepare_data.py         # one-shot: clean + split + label + features + graph build
│   ├── train_sage.py           # CLI: python -m netflow.scripts.train_sage --config configs/train.yaml
│   ├── train_gcn.py
│   ├── train_gat.py
│   ├── eval_metrics.py         # full nb 02 in script form
│   ├── run_xai.py              # nb 03 in script form
│   ├── run_baselines.py        # nb 04 (XGB) in script form (GCN moves to train_gcn.py)
│   └── run_complexity.py       # nb 012 in script form
├── notebooks/                  # original notebooks moved here, kept for reference
│   ├── 01_E-GraphSAGE_NFNB15v3_mean_agg_multiclass.ipynb
│   ├── 012_Complexity_test.ipynb
│   ├── 02_E-SAGE_metrics.ipynb
│   ├── 03_E-SAGE_XAI.ipynb
│   └── 04_baseline-xg-gcn.ipynb
├── REFACTOR_PLAN.md            # this file
└── README.md                   # how to run each script + reproduce paper results
```

**Why this layout**
- Separates *data* (`datasets/`) from *models* (`models/`) from *training loops* (`training/`) — the standard PyTorch+DGL research layout.
- `configs/` lifts every magic number out of code (current code has `hidden=128`, `fanouts="25,15"`, `batch_size=2048`, etc. hardcoded in `parse_args()` defaults and notebook cells).
- `scripts/` mirrors the original notebooks 1:1, so reviewers can map old → new.
- The original `.ipynb` files move to `notebooks/` so the markdown intent is preserved verbatim and citable.

---

## 3. Local Re-implementations to Replace with DGL Native APIs

| # | Current code | Lives in | DGL / GraphBolt native replacement | Risk |
|---|---|---|---|---|
| 1 | Custom IP→node-ID Python loop (`_make_ip_ids`) | `graph_build.py:17-52` | `pd.factorize` (or `np.unique(return_inverse=True)`) on `np.concatenate([src, dst])`; this is data-prep, not a DGL replacement, but it's the single biggest CPU bottleneck in graph build. | **Low** — pure refactor, identical mapping with stable order. |
| 2 | Manual feature store (`feature_store.py`) — memmap + CSR + sorted EID lookup | whole file (297 LoC) | `dgl.graphbolt.OnDiskDataset` + `dgl.graphbolt.TorchBasedFeatureStore` (in-memory tensor) or `dgl.graphbolt.DiskBasedFeatureStore`. GraphBolt handles GLOBAL→LOCAL row mapping internally and supports GPU UVA prefetching. | **Medium** — schema migration. Existing `.dat`/`.npz` artifacts must be converted (script: `prepare_data.py`). Verify identical batches by comparing one full epoch's features against the legacy store. |
| 3 | Manual `np.load("edge_indices.npy")` *inside* the batch loop | `train_edgecls_dbg.py:180,191` | Eliminated by GraphBolt (feature fetcher handles it); in pure-DGL fallback, hoist to a single `np.load(..., mmap_mode='r')` outside the loop. | **Low** — cosmetic, large speedup. |
| 4 | Manual GLOBAL↔LOCAL EID mapping (3 separate impls) | `feature_store.py:36-65, 92-112`; `train_edgecls_dbg.py:175-192`; `eval_utils.py:5-16` | One helper in `utils/alignment.py`; or remove entirely once GraphBolt's `FeatureFetcher` is wired up. | **Low**. |
| 5 | `_FallbackEdgeGraphSAGE` per-block manual unrolling + BatchNorm + ReLU | `train_edgecls_dbg.py:30-124` | Already uses `dgl.nn.SAGEConv` natively (good). The custom edge head `[h_src ‖ h_dst ‖ e_feat] → MLP` is novel (E-GraphSAGE) — **keep** as a clean `nn.Module`. Move to `models/edge_graphsage.py` and split out `EdgeMLP` (`models/edge_head.py`) for reuse by GCN/GAT baselines. | **Low** — same numerics, just relocation. |
| 6 | `node_embed = nn.Embedding(1, hidden)` + `expand` | `train_edgecls_dbg.py:47-49,59-61` | Either keep (it's actually fine) or replace with `nn.Parameter(torch.zeros(1, hidden))` + broadcasting. Slightly cleaner; same numerics. | **Low**. |
| 7 | `encode()` / `predict_from_embeddings()` / `compute_pair_embedding(return_src=...)` plumbing for XAI | `train_edgecls_dbg.py:75-124`, `xai.py:142-199` | Replace shape-introspection with **explicit** `forward_node_embeddings()` returning `(h_src, h_dst)` tuple by contract. Also expose `forward_edge_logits(h_src, h_dst, e_feat)`. Then SHAP wrapper just calls these. | **Medium** — the existing branch logic exists because old call-sites can't be broken; with a refactor we set the new contract and update all call-sites in one shot. |
| 8 | Spearman prune column-by-column ranking | `feature_numeric.py:88-130` | `scipy.stats.rankdata(X, axis=0, method='average')` (vectorized); then `np.corrcoef`. Drop the pandas conversion. | **Low** — same math, ~10× faster on d≈600. |
| 9 | KernelSHAP wrapper with shape-fallback hell | `xai.py:283-337` | Pin the contract: always return `(d,)` ndarray for the single requested target class; never branch on `target_class is None`. Iterate classes in the caller. Removes ~150 LoC of defensive code. | **Medium** — XAI plots must be re-validated against existing PNGs to confirm identical bars. |
| 10 | Hand-rolled `make_eval_loader` (`eval_utils.py`) | 31 LoC | Replace with `dgl.graphbolt.DataLoader` configured for evaluation (no shuffle, no drop_last). Single source of truth shared with training. | **Low**. |
| 11 | Per-batch label `.cpu()` round-trip in `_FallbackEdgeGraphSAGE.forward` | `train_edgecls_dbg.py:175-203` | After GraphBolt: labels are part of the MiniBatch — fetched directly on the right device. | **Low**. |
| 12 | Manual class label remapping at training start | `train_edgecls_dbg.py:282-318` | Move into `prepare_data.py` once: write the remapped labels into the feature store directly. Training then never re-maps. | **Low**. |
| 13 | `_FallbackEdgeGraphSAGE` model imported from training script for inference (`02`, `03`, `04`) | `02_E-SAGE_metrics.ipynb:0`; `03_E-SAGE_XAI.ipynb:0`; `04_baseline-xg-gcn.ipynb`* | Move to `models/edge_graphsage.py`. The "fallback" naming is misleading — there is no non-fallback. | **Low**. |
| 14 | `os.remove("artifacts/best_edge_sage.pt")` between grid-search runs | `01_*.ipynb:cell 26` | Use unique paths per run; never overwrite. The current code is a footgun acknowledged in `012_*.ipynb`. | **Low**. |
| 15 | Manual `BatchNorm1d` after each `SAGEConv` | `train_edgecls_dbg.py:42-44` | Keep — this is a deliberate architectural choice. But document it as such; not all SAGE variants use BN. | **None** (keep). |

**What stays unchanged** (these are already DGL-native or correct):
- `dgl.nn.SAGEConv` (used in E-GraphSAGE).
- `dgl.nn.GraphConv` (used in GCN baseline).
- `dgl.dataloading.NeighborSampler` + `as_edge_prediction_sampler` — will be migrated to GraphBolt for perf, but the abstraction is already correct.
- `dgl.save_graphs` / `dgl.load_graphs`.
- `dgl.EID` for global edge IDs.

---

## 4. GraphBolt Integration Points

DGL is built with GraphBolt enabled (without HugeCTR `gpu_cache`). We integrate at three levels:

### 4.1 Storage layer — `dgl.graphbolt.OnDiskDataset`

Replace the hand-rolled `feature_store/` + `graphs/` pair with one **YAML-described OnDiskDataset**:

```yaml
# datasets/nfunsw_nb15/metadata.yaml
dataset_name: NF-UNSW-NB15-v3-EdgeCls
graph:
  nodes:
    - num: <N>      # filled by prepare_data.py
  edges:
    - format: numpy
      path: graph_edges.npy        # (2, E) src/dst
      type: directed
feature_data:
  - domain: edge
    type: null
    name: x_num
    format: numpy
    in_memory: false               # memmap
    path: edge_x_num.npy
  - domain: edge
    name: x_cat
    format: numpy
    path: edge_x_cat.npy           # densified OHE OR keep CSR via custom feature
    in_memory: false
  - domain: edge
    name: label
    format: numpy
    path: edge_y.npy
  - domain: edge
    name: ts
    format: numpy
    path: edge_ts.npy
tasks:
  - name: edge_classification
    num_classes: <K>
    train_set:
      - format: numpy
        path: train_eids.npy
    validation_set:
      - format: numpy
        path: val_eids.npy
    test_set:
      - format: numpy
        path: test_eids.npy
```

GraphBolt's `OnDiskDataset` then handles: graph loading, feature memmap, GLOBAL EID semantics, train/val/test ID sets, and per-split iteration. **This eliminates `feature_store.py`, `eval_utils.py:map_store_global_to_graph_local`, and the alignment assertions.**

**Open question**: the current categorical features are stored as `scipy.sparse.csr_matrix` to save memory (d_cat ≈ 580). GraphBolt expects dense numpy. Two options to evaluate during Phase 3:
- **(a)** Densify on disk (cost: ~2.5 GB at d_cat=580, n=11M, float32) — simple, GPU-friendly.
- **(b)** Register a `Feature` subclass that holds a CSR backend and slices to dense per batch — keeps disk small, custom code burden.

Recommendation: start with (a) on a 10% subset, measure peak RSS; switch to (b) if needed.

### 4.2 Sampling layer — `dgl.graphbolt.NeighborSampler`

```python
# training/samplers.py
from dgl import graphbolt as gb

def make_train_pipeline(dataset, fanouts, batch_size, device, num_workers=4):
    train_set = dataset.tasks[0].train_set
    return (
        gb.ItemSampler(train_set, batch_size=batch_size, shuffle=True)
          .copy_to(device)                          # H2D early
          .sample_neighbor(dataset.graph, fanouts)  # k-hop neighborhood
          .fetch_feature(dataset.feature, node_feature_keys=None,
                         edge_feature_keys={"_E": ["x_num", "x_cat", "label"]})
    )
```

`fetch_feature` performs the GLOBAL-EID → row lookup that the legacy code does by hand each batch. Pinned host memory + UVA prefetch are automatic with GraphBolt when `device="cuda"`.

### 4.3 DataLoader

```python
loader = gb.DataLoader(pipeline, num_workers=num_workers)
for minibatch in loader:
    # minibatch.blocks    : list[DGLBlock]   (already on GPU if copy_to was set)
    # minibatch.compacted_seeds  : per-edge LOCAL ids in pair_graph
    # minibatch.edge_features["_E"]["x_num"], ["x_cat"], ["label"]
    ...
```

**Throughput expectations** (from DGL benchmarks):
- Legacy `dgl.dataloading.DataLoader` w/ `num_workers=0`, no UVA: ~baseline.
- GraphBolt w/ `num_workers=4`, `copy_to(device)`, pinned: typically 1.8-3× per-epoch wall time on multi-class edge cls.
- Gain comes from (i) async H2D, (ii) eliminating the per-batch `np.load` + memmap reopen, (iii) overlapping sample/fetch/compute via the multi-stage pipeline.

---

## 5. GPU Training Pipeline Design

### 5.1 Device placement

| Tensor | Where in legacy code | Proposed |
|---|---|---|
| Graph (`g_train`, `g_val`, `g_test`) | CPU (always) | CPU; structure stays on host (millions of edges, low memory pressure). |
| Edge features (memmap + CSR) | CPU disk | `dgl.graphbolt.Feature` with **UVA pinning** (`device="cuda"`, `in_memory=true` once measured); fall back to CPU pinned for very large stores. |
| Sampled blocks per batch | `.to(device)` per batch (sync) | Async transfer in pipeline (`copy_to(device)` step), overlapping H2D with compute. |
| Model parameters | `.to(device)` once | Same. |
| Edge features per batch | CPU `np.ndarray` → `torch.from_numpy` → `.to(device)` (sync, 2 copies) | Single zero-copy via GraphBolt fetch directly to GPU. |
| Labels per batch | `.cpu()` round-trip in eval_infer | Direct from MiniBatch on GPU. |

### 5.2 Mixed precision

E-GraphSAGE forward is dominated by:
- `SAGEConv` matmuls (large hidden×hidden) — **excellent** AMP candidate.
- Concatenate + edge MLP (small) — also fine.
- BatchNorm — works in BF16/FP16 with PyTorch native scaler.

Use **`torch.cuda.amp.autocast(dtype=torch.bfloat16)`** if GPU is Ampere or newer (Ampere/Ada/Hopper); fall back to `float16` with `GradScaler` otherwise. Expected ~1.4× throughput, no accuracy regression for this task (validate on the 10% subset first).

### 5.3 Compilation

Wrap the model with `torch.compile(model, mode="reduce-overhead")` once stabilized. SAGEConv has no graph-breaking ops in DGL ≥ 2.x. Expected another 10-25% on top of AMP. **Measure both with and without** to keep an option for debugging.

### 5.4 Memory management

- The graph itself is small (a few hundred MB at most).
- Edge features at full size: `(11M × 601 × 4 bytes) ≈ 26 GB` if dense. **Keep CSR for x_cat or memmap on disk**; do not load into GPU RAM.
- Sampling fanouts `25,15` → per batch worst case `2048 × 25 × 15 = 768k` edges in the deepest block; comfortably fits.
- For hyperparameter tuning, the existing 10% subset usage stays — it's the right move for grid search.

### 5.5 Reproducibility

Centralize in `utils/seeds.py`: `torch.manual_seed`, `np.random.seed`, `dgl.seed`, `torch.cuda.manual_seed_all`, `torch.use_deterministic_algorithms(True)` (opt-in via config flag — disables some kernels for ~5-10% slowdown).

---

## 6. Recommended GNN Model Implementations

Three models, all sharing one **`EdgeMLP`** head (`models/edge_head.py`) so comparison stays apples-to-apples:

```python
# models/edge_head.py
class EdgeMLP(nn.Module):
    def __init__(self, hidden, edge_in, num_classes, dropout=0.3):
        ...
    def forward(self, h_src, h_dst, e_feat):
        return self.mlp(torch.cat([h_src, h_dst, e_feat], dim=-1))
```

### 6.1 `models/edge_graphsage.py` — primary model

- `dgl.nn.SAGEConv` × num_layers (default 2), `aggregator_type ∈ {mean, pool, lstm, gcn}`.
- `BatchNorm1d` + `ReLU` between layers.
- `EdgeMLP` head.
- Constant-node-embedding option (`in_node=0`) preserved.
- New explicit API: `forward_blocks(blocks, x_nodes) -> h_dst` and `forward_pair(h_dst, pair_graph, e_feat) -> logits`. XAI calls these directly; no shape introspection.

### 6.2 `models/gcn.py` — baseline

- `dgl.nn.GraphConv` × 2 with `norm="both"`, `allow_zero_in_degree=True`.
- Same `EdgeMLP` head.
- This is the existing nb 04 model, lifted into a module verbatim.

### 6.3 `models/gat.py` — new baseline

- `dgl.nn.GATConv` × 2 (4-8 heads), with `feat_drop` and `attn_drop`.
- Concat-then-mean head fusion: `h = mean(heads)` after the last layer.
- Same `EdgeMLP` head.
- Adds a third reference point to the comparison table — useful for the paper.

**No custom message-passing layers** are needed for the current scope. If a future ablation requires (e.g.) edge-feature-conditioned aggregation, we'd add it under `layers/` using `dgl.function.u_mul_e_sum` or `apply_edges`. **Do not write this until needed.**

---

## 7. Profiling and Debugging Hooks

### 7.1 `utils/profiling.py`

```python
@contextmanager
def profile(active_steps=20, warmup=3, output_dir="artifacts/profiles"):
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=warmup, active=active_steps),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True, profile_memory=True, with_stack=True,
    ) as prof:
        yield prof
```

Wired into `training/train.py` via `--profile` flag. Outputs `.pt.trace.json` viewable in TensorBoard or chrome://tracing.

### 7.2 GPU sampler (lightweight)

Background thread polling `nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv` every 200 ms and writing CSV. Useful when you suspect a host-side bottleneck (low GPU utilization despite long step time).

### 7.3 DGL built-in timers

DGL's `dgl.utils.Timer` and the GraphBolt `DataLoader` already expose per-stage timings. Add `--timing` flag that sets `DGL_LOG_LEVEL=DEBUG` and dumps per-stage histograms.

### 7.4 Alignment debug (kept, simplified)

`utils/alignment.py:check_split_alignment(dataset, split)` — single helper that asserts feature row count == ID set size. **Remove** the three-way custom EID mapping once GraphBolt owns the lookups.

---

## 8. Migration Order and Dependencies

We migrate **bottom-up**, validating each layer against legacy outputs before moving on.

```
[Step 1] Data prep → datasets/
   ↓ outputs identical feature tensors and graph as legacy
[Step 2] Models → models/
   ↓ outputs identical loss/acc on first batch with same seeds
[Step 3] Training pipeline → training/
   ↓ outputs identical val_acc on the 10% subset
[Step 4] Evaluation → scripts/eval_metrics.py
   ↓ identical metrics JSON to legacy nb 02
[Step 5] Baselines → train_gcn.py, train_gat.py, run_baselines.py (XGB)
[Step 6] XAI → xai/
[Step 7] Complexity sweep → run_complexity.py
```

### Step-by-step

**Step 1** — Convert preprocessing (notebooks 01 cells 0-23) into `scripts/prepare_data.py`. Outputs:
- legacy paths (`feature_store/`, `graphs/`) for back-compat,
- new GraphBolt `OnDiskDataset` under `datasets/nfunsw_nb15/`.

**Validation:** `feature_store/train/numeric.dat` byte-equal to legacy; `graph_edges.npy` matches `g_train.edges()`.

**Step 2** — Move `_FallbackEdgeGraphSAGE` → `models/edge_graphsage.py`. Move GCN inline class from nb 04 → `models/gcn.py`. Add `models/gat.py`. All import from `models/edge_head.py`.

**Validation:** Load `artifacts/best_edge_sage.pt` into the new `EdgeGraphSAGE`, run a single batch, compare logits to legacy.

**Step 3** — Build GraphBolt training pipeline in `training/samplers.py` and `training/train.py` with `argparse` (or hydra) consuming `configs/*.yaml`. Replace `run_training`.

**Validation:** Train for 5 epochs on the 10% subset with both legacy `train_edgecls_dbg.py` and new `training/train.py` using `seed=42`. Val acc must match within ±0.001.

**Step 4** — Move nb 02 to `scripts/eval_metrics.py`. The `_FallbackEdgeGraphSAGE` rebuild is replaced with import from `models/`.

**Step 5** — `scripts/train_gcn.py` and `scripts/train_gat.py`. XGBoost stays as `scripts/run_baselines.py` (it doesn't touch the graph; only feature-store reads).

**Step 6** — `xai/shap_explainer.py` with the simplified return contract; update notebook 03 cell 5's defensive code to vanish. Validate by re-generating the existing PNGs and diffing pixel-wise (allow small tolerance for KernelSHAP randomness — set `seed=0` consistently).

**Step 7** — `scripts/run_complexity.py` re-uses `training/train.py` with sub-fractions; add `--frac` flag to `prepare_data.py`.

### Cross-step dependencies

- Step 2 only needs Step 1's *legacy* outputs (so it can be done in parallel by a second contributor if desired).
- Step 3 needs Steps 1 (new) + 2.
- Step 4 needs Step 2 (model class) but reads legacy artifacts, so can land before Step 3.
- Step 6 (XAI) needs Step 2 (model API: `forward_node_embeddings` returning `(h_src, h_dst)`).
- The current XAI notebook's heavy shape-fallback code in cell 5 only goes away once Step 6 lands.

### Risk flag — domain logic that MUST be preserved bit-exactly

These are not safe to refactor "while we're here":

1. **Chronological cut at equal timestamps** (`chronological_split.py:_adjust_cut_at_equal_timestamps`) — this guarantees no temporal leakage. Tests must verify identical split indices vs. legacy.
2. **Train-fitted transforms reused for val/test** (numeric, categorical, label_map) — if val/test ever sees a category not in train, the artifact-based transforms set `__RARE__` / `-1` consistently. Any reshuffling of fit/transform order will break this.
3. **IP→node-ID mapping is shared across splits** (`graph_build.py`) — train's `ip2id` is reused by val and test. If we replace with `pd.factorize`, we must factorize on the **concatenation** of train+val+test src/dst, in the order train-then-val-then-test, to preserve node ID stability.
4. **Edge order in DGL graph matches feature_store row order** (`feature_store.py` writes `edge_indices.npy` from `df_split.index`; `graph_build.py` adds edges in `df_split` row order). Refactoring either side breaks alignment silently. The new GraphBolt dataset must write `graph_edges.npy` in the same row order as `edge_y.npy`.
5. **Class weights use TRAIN bincount of remapped labels** (`train_edgecls_dbg.py:344-346`). Pre-computing during data-prep and saving alongside `label_map.json` is cleaner; numerics must match.
6. **Spearman corr threshold = 0.995** is a paper number. The faster vectorized impl must produce the same `keep_corr` mask on the published dataset (test in `test_feature_pipeline.py`).

---

## 9. Decisions Locked (Phase 2)

User confirmed on 2026-05-03:

1. **GraphBolt categorical**: **custom CSR-backed Feature subclass**. GPU memory is constrained; future work targets larger datasets where dense densification would not fit. We pay the per-batch CSR→dense cost; budget it in profiling.
2. **Configuration**: **argparse + YAML** (light path; no Hydra dependency).
3. **GAT baseline**: **skip**. Comparison stays at GraphSAGE vs. GCN vs. XGBoost.
4. **AMP / `torch.compile`**: **opt-in flags only** (`--amp`, `--compile`). Never default-on.
5. **Notebooks**: **remodel into 4 milestone notebooks**, each a thin wrapper over the new modules:
    1. `01_data_ingestion.ipynb` — clean → split → label → features → graph → GraphBolt OnDiskDataset
    2. `02_gnn_training_and_evaluation.ipynb` — train E-GraphSAGE → checkpoint → metrics + plots
    3. `03_baseline_comparison.ipynb` — XGBoost + GCN baselines on the same splits + comparison table
    4. `04_shap_evaluation.ipynb` — feature-level SHAP + structural neighbor-masking XAI
   The legacy 5 notebooks under `notebooks/legacy/` for reference. The complexity sweep (legacy `012`) becomes `scripts/run_complexity.py` only — no notebook.
6. **Backward compatibility**: **none**. Cut over cleanly to the GraphBolt `OnDiskDataset` layout. Legacy `feature_store/{train,val,test}/*.dat` and `graphs/*.bin` are gitignored; they'll be regenerated by `prepare_data.py` on first run.
7. **`artifacts/best_edge_sage.pt`**: treated as **intermediate**. No overwrite-protection logic, no compatibility hooks. The training script writes to `artifacts/checkpoints/<run_id>/best.pt` with unique run IDs.

**Workflow**: git is used to commit at clean step boundaries (initial state, then one commit per migration step from §8). No `git push`.

---

## 10. Concise Summary of Findings

- **5 notebooks**, ~70k lines of source counted (most of it embedded notebook outputs); the actual logic is ~3.4k LoC across 14 Python helper files. Heart of it is `train_edgecls_dbg.py` (479 LoC) and `xai.py` (843 LoC).
- **Models**: E-GraphSAGE (primary, `dglnn.SAGEConv` + edge MLP head), GCN baseline (`dglnn.GraphConv` + edge MLP), XGBoost baseline. **No GAT yet** (proposed).
- **Hand-rolled abstractions that DGL/GraphBolt already provide**: feature store + GLOBAL/LOCAL EID mapping (replaceable with `dgl.graphbolt.OnDiskDataset` + `FeatureFetcher`), per-batch `np.load` (cache once or eliminate via GraphBolt), eval dataloader (use `gb.DataLoader`).
- **Performance hotspots**: `_make_ip_ids` Python loop, `np.load` inside training batch loop, `num_workers=0` everywhere, no AMP, no `torch.compile`, sync H2D transfers for blocks and edge features.
- **Domain logic that must stay byte-identical**: chronological split with equal-timestamp boundary handling, fit-on-train transforms, train's IP→ID map shared with val/test, EID order alignment, Spearman prune at 0.995.
- **Migration**: 7 steps, bottom-up, each validated against legacy outputs on the existing 10% hyper-tuning subset before scaling up.

---

**End of Phase 1.** Awaiting confirmation, priority adjustments, or flags on domain logic before starting Phase 3 implementation.
