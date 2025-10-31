# TE-G-SAGE-XAI 
Chronologicaly splitted NetFlow processing pipline for temporaly edge-aware explainable GraphSAGE model classificator  

Short project description
- Pipeline to build and explain edge aware GraphSAGE models on temporaly processed NetFlow-style dataset.  
- Focus: preprocess NetFlow records into graph data, train/validate GNN, produce XAI artefacts for model interpretation.

## Instructions
Go to netflow folder. All files are located there.

# Requirements (you can run in jupyter notebook):
#choose dgl based on what you have available. This one is for support of nVidia GPU. Code does not have to be changed if you are on cpu only.
`!pip3 install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

!pip3 install torch==2.3.0

!pip3 install category_encoders

!pip install shap

!pip install xgboost`

## Dataset
- Source: raw NetFlow/flow-record exports from https://staff.itee.uq.edu.au/marius/NIDS_datasets/ (CSV) containing typical fields (timestamps, src/dst IP, src/dst port, protocol, bytes, packets, labels/alerts).  
- Layout: put raw netflow files under `data/`. Check Cell 3 in first jupyter notebook. Processed graph datasets are written to `artifacts` (formats: Parquet/CSV for tabular features, serialized graph objects).  
- Preprocessing steps: data cleaning, IP/port encoding, chronological splits, label encoding, numerical transformation, feature engineering, categorical encoding, mini-batch preparation, graph construction (node/edge creation, edge attributes, temporal slices).

## Run (Jupyter notebooks — run in order)
1. `01_E-GraphSAGE_NFNB15v3_mean_agg_multiclass.ipynb`  
    - Netflow data processing and model training and parameter tuning.  
2. `02_E-SAGE_metrics.ipynb`  
    - Load checkpoints, compute metrics (precision/recall/F1/AUC), generate evaluation plots and tables.  
3. `03_E-SAGE_XAI.ipynb`  
    - Run SHAP explainer on the model, draw plots per edge.  
4. `04_baseline-xg-gcn.ipynb`  
    - Run comparison models on same splits for dataset (XGBoost, GCN).  

Note: run notebooks sequentially because later notebooks depend on processed data, model checkpoints, and saved artifacts from earlier steps.

## Generated artefacts
- feature_store/ — cleaned flows, feature tables, serialized graph datasets (.pt, .pkl, .parquet).  
- graphs/ — generated graphs (.bin, .npz).  
- artifacts/ — evaluation metrics, CSV/JSON reports. label_map.json. best_params.json and best_edge_sage.pt (tuned model) 
- artifacts/xai — PNG figures for XAI results.  
- artifacts/corr — PNG figures for Correlation results.  

## Quick notes and thank you's
- This research was performed on NetFlow dataset NF-UNSW-NB15-v3 available at the https://staff.itee.uq.edu.au/marius/NIDS_datasets/
- This research was performed using the Advanced Computing service provided by University of Zagreb University Computing Centre - SRCE

- citation (tba): TE-G-SAGE: Explainable Edge-Aware Graph Neural Networks for Network Intrusion Detection [submitted for peer-review]
