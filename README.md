# TE-G-SAGE-XAI

Temporally edge-aware, explainable GraphSAGE for network intrusion detection.
NetFlow records (NF-UNSW-NB15-v3) are processed into a temporal graph, used to
train an **EdgeGraphSAGE** edge classifier, compared against GCN, GAT, and
XGBoost baselines, and explained with variable-level SHAP.

> **Published article:**
> Luša, R.; Pintar, D.; Vranić, M. TE-G-SAGE: Explainable Edge-Aware Graph Neural Networks for Network Intrusion Detection. *Modelling* 2025, *6*, 165.
> <https://doi.org/10.3390/modelling6040165> — [Full text at MDPI](https://www.mdpi.com/2673-3951/6/4/165)

## Citation

If you use this code in your research, please cite:

```bibtex
@Article{modelling6040165,
  author         = {Luša, Riko and Pintar, Damir and Vranić, Mihaela},
  title          = {TE-G-SAGE: Explainable Edge-Aware Graph Neural Networks for Network Intrusion Detection},
  journal        = {Modelling},
  year           = {2025},
  volume         = {6},
  number         = {4},
  article-number = {165},
  issn           = {2673-3951},
  doi            = {10.3390/modelling6040165},
  url            = {https://www.mdpi.com/2673-3951/6/4/165}
}
```

---

## Installation

```bash
pip install -r requirements.txt
```

DGL requires a separate index URL matched to your CUDA version — see
[dgl.ai/install](https://www.dgl.ai/pages/start.html) for the exact command.

---

## Dataset

Download **NF-UNSW-NB15-v3** from
<https://staff.itee.uq.edu.au/marius/NIDS_datasets/> and place the CSV under:

```
data/NF-UNSW-NB15-v3.csv
```

The path is configured in `netflow/configs/data.yaml` (`raw.csv_path`).

---

## Pipeline

All commands run from the `netflow/` directory.

```
data/NF-UNSW-NB15-v3.csv
        │
        ▼
scripts/prepare_data.py      →  datasets/nfunsw_nb15/   (GraphBolt OnDiskDataset)
                                artifacts/label_map.json
                                artifacts/numeric/
                                artifacts/categorical/
        │
        ▼
training/train.py            →  artifacts/best_edge_sage.pt
                                artifacts/train_log.json
        │
        ├──▶ scripts/train_gcn.py    →  artifacts/best_gcn.pt
        ├──▶ scripts/train_gat.py    →  artifacts/best_gat.pt
        └──▶ scripts/run_baselines.py →  artifacts/best_xgb.json
        │
        ▼
scripts/eval_metrics.py      →  artifacts/metrics_test.json        (GraphSAGE)
                                artifacts/metrics_test_gcn.json     (GCN)
                                artifacts/metrics_test_gat.json     (GAT)
                                artifacts/Results_comparison.csv    (after run_baselines)
        │
        ▼
scripts/run_complexity.py    →  artifacts/complexity_fracXXX/
                                artifacts/complexity_summary.json
```

### Step 1 — Data preparation

```bash
cd netflow/
python scripts/prepare_data.py --config configs/data.yaml
```

Runs cleaning, chronological 60/30/10 split, label encoding, numeric pipeline
(log1p + StandardScaler + correlation pruning), categorical OHE, and writes a
GraphBolt `OnDiskDataset` under `datasets/nfunsw_nb15/`.

**Key options** (override any `data.yaml` field on the command line):

| Flag | Default | Description |
|------|---------|-------------|
| `--frac F` | `1.0` | Subsample fraction for quick tuning runs |
| `--config PATH` | — | Path to data YAML (required) |

**`configs/data.yaml` highlights:**

```yaml
raw:
  csv_path: data/NF-UNSW-NB15-v3.csv
split:
  train_ratio: 0.60
  val_ratio:   0.30
  test_ratio:  0.10
numeric:
  scaler: standard          # standard | robust
  apply_corr_prune: true
  corr_threshold: 0.995
categorical:
  rare_min_freq: 50
out:
  root: datasets/nfunsw_nb15
```

---

### Step 2 — Train EdgeGraphSAGE

```bash
python training/train.py --config configs/train.yaml
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | `20` | Number of training epochs |
| `--lr F` | `3e-4` | Learning rate |
| `--hidden N` | `128` | Hidden dimension |
| `--num_layers N` | `2` | Number of SAGEConv layers |
| `--aggregator mean\|pool\|lstm\|gcn` | `mean` | Neighbourhood aggregator |
| `--dropout F` | `0.3` | Dropout rate |
| `--weight_decay F` | `1e-4` | L2 regularisation |
| `--fanouts "N,N,…"` | `"25,15"` | Per-layer fanout sizes (comma-separated) |
| `--batch_size N` | `2048` | Mini-batch size |
| `--frac F` | `1.0` | Subsample training set fraction |
| `--seed N` | `42` | Random seed |
| `--device cuda\|cpu` | `auto` | Force device |
| `--config PATH` | — | Path to training YAML (required) |

**`configs/train.yaml` highlights:**

```yaml
model:
  hidden: 128
  num_layers: 2
  aggregator: mean          # mean | pool | lstm | gcn
  dropout: 0.3
training:
  fanouts: [25, 15]         # per-layer neighbourhood sample sizes
  batch_size: 2048
  epochs: 20
  lr: 3.0e-4
  weight_decay: 1.0e-4
  device: auto
out:
  checkpoint: artifacts/best_edge_sage.pt
  train_log:  artifacts/train_log.json
```

---

### Step 3 — Baselines

Run any or all of the three baselines after Step 2.

#### GCN

```bash
python scripts/train_gcn.py --config configs/gcn.yaml
# options: --epochs N  --device cuda|cpu
```

#### GAT

```bash
python scripts/train_gat.py --config configs/gat.yaml
# options: --epochs N  --device cuda|cpu
```

#### XGBoost

```bash
python scripts/run_baselines.py --config configs/baselines.yaml [--plots]
# options: --n_estimators N  --per_class_sample N  --plots
```

`--plots` saves one-vs-rest ROC curves (`artifacts/roc_xgb.png`).

---

### Step 3b — Evaluate GNN models

Run after training to produce per-split metrics JSON files and plots.
`run_baselines.py` reads these files to build the comparison table, so run
this before `run_baselines.py` (or re-run it afterwards).

```bash
# EdgeGraphSAGE  →  metrics_test.json
python scripts/eval_metrics.py --config configs/train.yaml --split test

# GCN            →  metrics_test_gcn.json
python scripts/eval_metrics.py --config configs/gcn.yaml  --split test --model_tag gcn

# GAT            →  metrics_test_gat.json
python scripts/eval_metrics.py --config configs/gat.yaml  --split test --model_tag gat
```

| Flag | Default | Description |
|------|---------|-------------|
| `--split train\|val\|test` | `test` | Which split to evaluate |
| `--model_tag TAG` | `""` | Appended to output filenames, e.g. `gcn` → `metrics_test_gcn.json` |
| `--checkpoint PATH` | from config | Override checkpoint path |
| `--no_plots` | off | Skip matplotlib output |

After all baselines and eval runs complete, `artifacts/Results_comparison.csv`
and `artifacts/Results_comparison.tex` contain a side-by-side table of
Accuracy / Macro-F1 / FAR for all four models.

**`configs/baselines.yaml` highlights:**

```yaml
xgboost:
  n_estimators: 600
  max_depth: 8
  per_class_sample: 40000   # cap training rows per class; null = use all
  tree_method: hist         # hist | gpu_hist
```

---

### Step 4 — Complexity sweep

Trains EdgeGraphSAGE at multiple dataset fractions and records per-epoch
timing to mirror `012_Complexity_test.ipynb`.

```bash
python scripts/run_complexity.py --config configs/train.yaml \
    --fracs 0.25 0.5 0.75 1.0 [--epochs 5] [--plots]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--fracs F…` | `0.25 0.5 0.75 1.0` | Dataset fractions to sweep |
| `--epochs N` | value from config | Override epochs per sweep point |
| `--plots` | off | Save `complexity_scaling.png` |
| `--device` | auto | Force device |

Each fraction writes to its own sub-directory (`artifacts/complexity_fracXXX/`)
so the main checkpoint is never overwritten.  A summary JSON is written to
`artifacts/complexity_summary.json`.

---

### Step 5 — XAI

Runs variable-level grouped KernelSHAP (~45 dims: 38 numeric + 7 categorical
variables) and structural neighbour-masking XAI.  Each categorical variable's
entire OHE block is treated as one coalition member, making SHAP tractable and
producing directly interpretable attributions ("L7_PROTO contributed X").

```bash
python scripts/run_xai.py --config configs/xai.yaml
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--n_per_class N` | `50` | Edges sampled per class for SHAP aggregation |
| `--background_size N` | `50` | Background samples for KernelExplainer |
| `--classes C…` | all | Limit beeswarm to these class IDs |
| `--device cuda\|cpu` | auto | Force device |

**`configs/xai.yaml` highlights:**

```yaml
xai:
  n_per_class:    50    # higher = more stable, slower
  background_size: 50
  n_beeswarm:     50
  top_k:          10
  classes: null         # null = all classes
```

**Outputs** (under `artifacts/xai/`):

| File | Description |
|------|-------------|
| `shap_summary.json` | Mean \|SHAP\| per variable per class |
| `topk_<class>.png` | Top-k variable importance bar chart |
| `spider.png` | Spider chart across all classes |
| `signed_shap_<class>.png` | Signed mean SHAP bar chart |
| `beeswarm_<class>.png` | SHAP beeswarm dot plot |

For single-edge or custom explanations, use the Python API directly:

```python
from xai import local_shap_for_edge_grouped, build_variable_groups
```

Pass `variable_groups=None` to any aggregation function to fall back to the
raw 601-dim OHE-level SHAP.

---

## Notebooks (interactive / exploratory path)

The notebooks in `netflow/` provide an interactive path through the same
pipeline and are useful for exploration and visualisation.  Run them in order:

| Notebook | Content |
|----------|---------|
| `01_E-GraphSAGE_NFNB15v3_mean_agg_multiclass.ipynb` | Data cleaning, feature store, graph build, training, hyperparameter tuning |
| `02_E-SAGE_metrics.ipynb` | Load checkpoint, compute P/R/F1/AUC/FAR, generate plots |
| `03_E-SAGE_XAI.ipynb` | SHAP feature explanations and structural neighbour-masking XAI |
| `04_baseline-xg-gcn.ipynb` | XGBoost and GCN baselines on same splits |

```bash
cd netflow/
jupyter notebook
```

---

## Generated artifacts

| Path | Contents |
|------|----------|
| `datasets/nfunsw_nb15/` | GraphBolt OnDiskDataset (edge features, graph topology, split indices) |
| `artifacts/label_map.json` | String → integer class mapping |
| `artifacts/numeric/` | Fitted scaler + column list (joblib) |
| `artifacts/categorical/` | Fitted OHE encoder + column metadata (joblib / JSON) |
| `artifacts/best_edge_sage.pt` | Best EdgeGraphSAGE checkpoint |
| `artifacts/best_gcn.pt` | Best GCN checkpoint |
| `artifacts/best_gat.pt` | Best GAT checkpoint |
| `artifacts/best_xgb.json` | Best XGBoost model |
| `artifacts/train_log.json` | Per-epoch loss / accuracy / timing |
| `artifacts/Results_comparison.csv` | Side-by-side model comparison table |
| `artifacts/complexity_summary.json` | Timing vs dataset fraction |
| `artifacts/xai/` | SHAP plots, neighbour-impact charts |
| `artifacts/corr/` | Correlation heatmaps |

---

## Acknowledgements

Experiments performed using the Advanced Computing service provided by the
University of Zagreb University Computing Centre (SRCE).

## Release: v2

Full code refactor using [Claude Code](https://claude.ai/code) making it more streamlined without reliance on Jupyter notebooks.
