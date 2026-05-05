"""XGBoost edge-classification baseline on NF-UNSW-NB15-v3.

Reads features directly from the GraphBolt OnDiskDataset (no graph structure
needed — XGBoost treats each flow record as an independent i.i.d. sample).

Outputs (under <out.artifacts_dir>):
  best_xgb.json                  — XGBoost model (JSON format)
  metrics_xgb.json               — accuracy, macro P/R/F1/FAR, per-class
  per_class_results_xgb.csv/.tex — per-class table
  roc_xgb.png                    — one-vs-rest ROC curves (if --plots)
  Results_comparison.csv/.tex    — comparison table vs saved GNN metrics

Training data note:
  The full NF-UNSW-NB15-v3 training set (~8M rows) may not fit comfortably in
  RAM.  `xgboost.per_class_sample` caps the training rows per class; set to
  null to use all rows.

Usage:
  python scripts/run_baselines.py --config configs/baselines.yaml
  python scripts/run_baselines.py --config configs/baselines.yaml --plots
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

HERE = Path(__file__).resolve().parent
NETFLOW_ROOT = HERE.parent
if str(NETFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(NETFLOW_ROOT))

from datasets.nfunsw_nb15 import load_dataset, load_store_meta
from eval_metrics import compute_metrics


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def _load_split_features(
    root: str,
    split: str,
    feature,
    x_cat_csr: sparse.csr_matrix,
    per_class_sample: int | None = None,
    y_all: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) float32/int32 arrays for the requested split.

    If per_class_sample is set and split=="train", applies stratified
    per-class capping to keep RAM manageable on large datasets.
    """
    import torch
    eids = np.load(os.path.join(root, "tasks", f"{split}_eids.npy")).astype(np.int64)

    if per_class_sample is not None and split == "train" and y_all is not None:
        rng = np.random.default_rng(seed)
        y_split = y_all[eids]
        keep = []
        for c in np.unique(y_split):
            idx_c = np.where(y_split == c)[0]
            take  = min(per_class_sample, idx_c.size)
            keep.append(rng.choice(idx_c, size=take, replace=False))
        eids = eids[np.sort(np.concatenate(keep))]
        logging.info(f"[baselines] {split}: sampled {len(eids)} rows "
                     f"(≤{per_class_sample} per class)")
    else:
        logging.info(f"[baselines] {split}: {len(eids)} rows")

    ids_t   = torch.from_numpy(eids)
    x_num   = feature.read("edge", None, "x_num", ids=ids_t).numpy().astype(np.float32)
    x_cat   = x_cat_csr[eids].toarray().astype(np.float32)
    X       = np.concatenate([x_num, x_cat], axis=1)
    y       = y_all[eids].astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Comparison table helper
# ---------------------------------------------------------------------------

def _load_metrics_safe(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "metrics" in raw:
            raw = raw["metrics"]
        return raw
    except Exception:
        return None


def _build_comparison_table(
    art_dir: str,
    xgb_metrics: dict,
    label2id: dict,
) -> None:
    import pandas as pd

    rows = []

    def _row(name: str, m: dict) -> dict:
        macro = m.get("macro", {})
        return {
            "Model":           name,
            "Accuracy":        m.get("accuracy"),
            "Macro Precision": macro.get("precision"),
            "Macro Recall":    macro.get("recall"),
            "Macro F1":        macro.get("f1"),
            "Macro FAR":       m.get("macro_FAR"),
        }

    rows.append(_row("XGBoost", xgb_metrics))
    for model_name, fname in [
        ("GCN",       "metrics_test_gcn.json"),
        ("GAT",       "metrics_test_gat.json"),
        ("GraphSAGE", "metrics_test.json"),
    ]:
        m = _load_metrics_safe(os.path.join(art_dir, fname))
        if m is not None:
            rows.append(_row(model_name, m))

    df = (
        __import__("pandas").DataFrame(rows)
        [["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Macro FAR"]]
    )
    csv_path = os.path.join(art_dir, "Results_comparison.csv")
    tex_path = os.path.join(art_dir, "Results_comparison.tex")
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))
    logging.info(f"[baselines] comparison table → {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baselines(cfg: dict, plots: bool = False) -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout, force=True,
    )

    ds_cfg  = cfg["dataset"]
    xgb_cfg = cfg["xgboost"]
    out_cfg = cfg["out"]

    root    = ds_cfg["root"]
    art_dir = out_cfg["artifacts_dir"]
    os.makedirs(art_dir, exist_ok=True)

    # Label map.
    with open(os.path.join(art_dir, "label_map.json"), "r", encoding="utf-8") as f:
        label2id: dict = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    classes  = [id2label[i] for i in range(len(id2label))]
    num_classes = len(classes)

    # Dataset.
    meta      = load_store_meta(root)
    ds        = load_dataset(root)
    x_cat_csr = sparse.load_npz(os.path.join(root, "features", "x_cat.npz"))
    y_all     = ds.feature.read("edge", None, "y").view(-1).numpy().astype(np.int32)

    per_class = xgb_cfg.get("per_class_sample", None)
    X_train, y_train = _load_split_features(
        root, "train", ds.feature, x_cat_csr, per_class, y_all, seed=42,
    )
    X_val, y_val = _load_split_features(
        root, "val", ds.feature, x_cat_csr, None, y_all, seed=42,
    )
    logging.info(f"[baselines] X_train={X_train.shape} X_val={X_val.shape}")

    # XGBoost.
    import xgboost as xgb
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg.get("n_estimators", 600),
        max_depth=xgb_cfg.get("max_depth", 8),
        learning_rate=xgb_cfg.get("learning_rate", 0.05),
        subsample=xgb_cfg.get("subsample", 0.9),
        colsample_bytree=xgb_cfg.get("colsample_bytree", 0.9),
        reg_lambda=xgb_cfg.get("reg_lambda", 1.0),
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method=xgb_cfg.get("tree_method", "hist"),
        device=device,
        n_jobs=max(1, (os.cpu_count() or 2) // 2),
        num_class=num_classes,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    elapsed = time.perf_counter() - t0
    logging.info(f"[baselines] XGBoost training: {elapsed:.1f}s "
                 f"({elapsed / model.n_estimators:.4f}s / tree)")

    ckpt_path = out_cfg["checkpoint"]
    model.save_model(ckpt_path)
    logging.info(f"[baselines] saved model → {ckpt_path}")

    # Evaluate on validation set.
    dval = xgb.DMatrix(X_val, device=device)
    y_prob = model.get_booster().predict(dval).reshape(-1, num_classes)
    y_pred = y_prob.argmax(axis=1)
    metrics = compute_metrics(y_val, y_pred, label2id)

    metrics_path = out_cfg["metrics"]
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"model": "XGBoost", "metrics": metrics}, f, indent=2, ensure_ascii=False)
    logging.info(
        f"[baselines] XGBoost: acc={metrics['accuracy']:.4f} "
        f"macro_F1={metrics['macro']['f1']:.4f} "
        f"FAR={metrics['macro_FAR']:.4f}"
    )

    # Per-class table.
    import pandas as pd
    rows = [
        {"Class": cls, "Precision": v["precision"], "Recall": v["recall"],
         "F1": v["f1"], "FAR": v["FAR"]}
        for cls, v in metrics["per_class"].items()
    ]
    df = pd.DataFrame(rows).round(4)
    df.to_csv(os.path.join(art_dir, "per_class_results_xgb.csv"), index=False)
    with open(os.path.join(art_dir, "per_class_results_xgb.tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False))

    # Optional plots.
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        from eval_roc import plot_roc_ovr
        plot_roc_ovr(y_val, y_prob, classes,
                     out_path=os.path.join(art_dir, "roc_xgb.png"))

    # Comparison table (collects any previously saved GNN metrics).
    _build_comparison_table(art_dir, metrics, label2id)

    return {"model": "XGBoost", "metrics": metrics, "train_time_s": elapsed}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="Path to baselines YAML config.")
    ap.add_argument("--plots",  action="store_true", help="Save ROC/PR plots.")
    ap.add_argument("--n_estimators", type=int,   default=None)
    ap.add_argument("--per_class_sample", type=int, default=None,
                    help="Override per-class training sample cap.")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.n_estimators is not None:
        cfg["xgboost"]["n_estimators"] = args.n_estimators
    if args.per_class_sample is not None:
        cfg["xgboost"]["per_class_sample"] = args.per_class_sample
    result = run_baselines(cfg, plots=args.plots)
    print(json.dumps({
        "accuracy":  result["metrics"]["accuracy"],
        "macro":     result["metrics"]["macro"],
        "macro_FAR": result["metrics"]["macro_FAR"],
        "train_s":   result["train_time_s"],
    }, indent=2))
