# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TE-G-SAGE-XAI: A research pipeline for edge-aware, explainable GraphSAGE-based network intrusion detection. NetFlow records (NF-UNSW-NB15-v3) are processed into temporal graph datasets, used to train an **EdgeGraphSAGE** classifier, and analyzed with SHAP-based XAI. The paper is submitted for peer review.

## Running the Pipeline

All code lives under `netflow/`. Run notebooks in order (each step depends on artifacts from the previous):

```bash
cd netflow/
jupyter notebook
```

1. `01_E-GraphSAGE_NFNB15v3_mean_agg_multiclass.ipynb` — data cleaning, feature store, graph build, training, hyperparameter tuning
2. `02_E-SAGE_metrics.ipynb` — load checkpoint, compute precision/recall/F1/AUC/FAR, generate plots
3. `03_E-SAGE_XAI.ipynb` — SHAP feature explanations and structural neighbor-masking XAI
4. `04_baseline-xg-gcn.ipynb` — XGBoost and GCN baselines on the same splits

The training script can also be run directly:
```bash
cd netflow/
python train_edgecls_dbg.py \
  --feature_store feature_store \
  --graphs_dir graphs \
  --hidden 128 --layers 2 --aggregator mean \
  --fanouts 25,15 --batch_size 2048 --epochs 20 \
  --lr 3e-4 --weight_decay 1e-4 --debug
```

Data cleaning CLI:
```bash
python netflow/data_cleaning.py input.parquet output_clean.parquet
```

## Architecture

### Data Flow

```
Raw NetFlow CSV  →  data_cleaning.py       →  cleaned DataFrame
                 →  chronological_split.py →  60/30/10 temporal splits (no leakage)
                 →  label_mapping.py       →  y_train/val/test (int64 labels)
                 →  feature_numeric.py     →  numeric.dat (float32 memmap per split)
                 →  categorical_encoding.py→  categorical.npz (CSR OHE per split)
                 →  feature_store.py       →  feature_store/{train,val,test}/
                 →  graph_build.py         →  graphs/{train,val,test}.bin (DGL graphs)
                 →  train_edgecls_dbg.py   →  artifacts/best_edge_sage.pt
                 →  eval_infer.py          →  y_true, y_pred, y_prob
                 →  eval_metrics.py        →  metrics JSON + confusion matrix
                 →  xai.py                 →  SHAP plots, neighbor impact charts
```

### Model: EdgeGraphSAGE

- **Nodes**: unique IPv4 addresses (no structural node features; uses a learned constant embedding)
- **Edges**: NetFlow records; each edge carries the full feature vector (numeric + OHE)
- **Forward**: multi-layer `SAGEConv` → per-edge MLP `[h_src ‖ h_dst ‖ e_feat]` → class logits
- Class weights (inverse frequency) applied in `CrossEntropyLoss`
- The model exposes `encode()` and `predict_from_embeddings()` for XAI decoupling

### Feature Store Layout

```
feature_store/
  {train,val,test}/
    numeric.dat          # float32 memmap, shape (N, d_num)
    categorical.npz      # scipy CSR float32, shape (N, d_cat)
    edge_indices.npy     # int64 (N,)  — GLOBAL edge IDs; must match g.edata[dgl.EID]
    y.npy                # int64 (N,)  — class labels
    ts.npy               # float32 (N,) — flow start time in seconds
    eid_map.npz          # sorted index for fast global→local row lookup
```

### Key Invariant: Edge Alignment

`edge_indices.npy` and `g.edata[dgl.EID]` must contain identical arrays in identical order for each split. `assert_graph_equals_store()` and `assert_loader_seed_alignment()` in `train_edgecls_dbg.py` enforce this at training time. If you rebuild the graph or feature store, always rebuild both together from the same indices.

### XAI Strategy

Two complementary explanations in `xai.py`:
1. **Feature-level (KernelSHAP)**: freeze `[h_src ‖ h_dst]` pair embedding, explain `f(e_feat) → P(class)` for a single edge. `local_shap_for_edge()` → `aggregate_shap_per_class()` for global importance.
2. **Structural (neighbor masking)**: zero-out individual neighbor embeddings in `h_src`/`h_dst`, measure `Δ P(class)`. `neighbor_impact_approx()` + `visualize_neighbor_impacts()`.

## Module Responsibilities

| File | Role |
|------|------|
| `data_cleaning.py` | Drop rows missing IPs/ports, convert ±inf→NA, fill numeric NA→0 |
| `chronological_split.py` | Temporal 60/30/10 split; boundary-safe for equal timestamps; persists indices + meta |
| `label_mapping.py` | Fit string→int map on TRAIN; apply to val/test with configurable unknown policy |
| `feature_numeric.py` | log1p on non-negative cols, StandardScaler, zero-var drop, Spearman corr prune |
| `categorical_encoding.py` | Rare-collapse + OneHotEncoder for categorical/numeric-coded-categorical cols; optional port bucketing |
| `feature_store.py` | Orchestrate numeric+cat pipelines, write per-split memmaps + CSR, build EID lookup index |
| `graph_build.py` | Map IP strings → node IDs (reuse train's map for val/test), build DGL directed multigraph |
| `train_edgecls_dbg.py` | Training loop with `NeighborSampler`; alignment assertions; saves best checkpoint + history |
| `eval_infer.py` | Batched inference returning `y_true`, `y_pred`, `y_prob` |
| `eval_metrics.py` | Accuracy, precision, recall, F1, FAR per class and macro |
| `eval_roc.py` | ROC/AUC curves |
| `xai.py` | KernelSHAP feature explanations, neighbor masking structural XAI, group aggregation, plots |
| `debug_align.py` | Pre-flight check that graph EIDs match feature store rows |

## Generated Artifacts (gitignored)

- `data/` — raw NetFlow CSVs (place here before running notebook 01)
- `feature_store/` — per-split features and alignment indices
- `graphs/` — serialized DGL graphs (`train.bin`, `val.bin`, `test.bin`)
- `artifacts/` — trained model (`best_edge_sage.pt`), metrics JSON, eval plots, label/param JSONs
- `artifacts/xai/` — SHAP and structural XAI figures
- `artifacts/numeric/` and `artifacts/categorical/` — fitted transform objects (joblib)

## Dataset

NF-UNSW-NB15-v3, available at https://staff.itee.uq.edu.au/marius/NIDS_datasets/  
Place raw CSV files under `data/` and verify the path in Cell 3 of notebook 01.

## Dependencies

Core: `torch`, `dgl`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `shap`, `matplotlib`  
Data I/O: `pyarrow` (parquet support)
