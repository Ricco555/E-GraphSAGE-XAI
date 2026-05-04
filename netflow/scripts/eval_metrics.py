"""Evaluate a trained EdgeGraphSAGE checkpoint: inference → metrics → plots.

Replaces notebooks/legacy/02_E-SAGE_metrics.ipynb.
Reads from the GraphBolt OnDiskDataset produced by scripts/prepare_data.py.

Outputs (under <out.artifacts_dir>):
  metrics_<split>.json          — accuracy, macro/weighted P/R/F1, per-class, FAR
  per_class_results_<split>.csv — per-class table
  per_class_results_<split>.tex — LaTeX table
  confusion_matrix_<split>.png  — normalized confusion matrix
  fpr_per_class_<split>.png     — false-positive-rate bar chart
  roc_<split>.png               — one-vs-rest ROC curves
  pr_<split>.png                — one-vs-rest precision-recall curves

Usage:
  python scripts/eval_metrics.py --config configs/train.yaml
  python scripts/eval_metrics.py --config configs/train.yaml --split val
  python scripts/eval_metrics.py --config configs/train.yaml --split test --no_plots
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

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
from training.train import _build_model
from training.samplers import build_dgl_graph, make_edge_loader
from eval_metrics import compute_metrics
from eval_roc import plot_roc_ovr, plot_confusion_matrix


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_split(
    model: nn.Module,
    loader,
    g: dgl.DGLGraph,
    feature,
    x_cat_csr: sparse.csr_matrix,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run batched inference; return (y_true, y_pred, y_prob).

    y_prob is softmax-probability matrix (N, C) — needed for ROC/PR curves.
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    for _input_nodes, pair_graph, blocks in loader:
        blocks     = [b.to(device) for b in blocks]
        pair_graph = pair_graph.to(device)

        local_eids = pair_graph.edata[dgl.EID].cpu().numpy().astype(np.int64)

        ids_t  = torch.from_numpy(local_eids)
        x_num  = feature.read("edge", None, "x_num", ids=ids_t).float()
        x_cat_np = x_cat_csr[local_eids].toarray().astype(np.float32)
        e_feat = torch.cat([x_num, torch.from_numpy(x_cat_np)], dim=1).to(device)

        logits = model(blocks, None, pair_graph, e_feat)          # (B, C)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        y      = g.edata["y"][ids_t].view(-1).numpy()

        keep = y >= 0
        all_true.append(y[keep])
        all_pred.append(preds[keep])
        all_prob.append(probs[keep])

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_fpr_bar(cm: np.ndarray, labels: list[str], out_path: str) -> None:
    import matplotlib.pyplot as plt
    C = cm.shape[0]
    fpr = np.zeros(C)
    for c in range(C):
        TP = cm[c, c]
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - cm[c, :].sum() - FP
        fpr[c] = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    x = np.arange(C)
    fig, ax = plt.subplots(figsize=(max(8, C), 4))
    ax.bar(x, fpr, color="dodgerblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("FPR")
    ax.set_title("False Positive Rate per class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[str],
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    C = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(C)))
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = mpl.colormaps["tab10"]
    for i in range(C):
        if y_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(rec, prec, label=f"{classes[i]} (AP={ap:.3f})", color=cmap(i % 10))
    prec_m, rec_m, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
    ap_m = average_precision_score(y_bin, y_prob, average="micro")
    ax.plot(rec_m, prec_m, "k--", lw=2, label=f"micro-avg (AP={ap_m:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (one-vs-rest)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(cfg: dict, split: str = "test", no_plots: bool = False, model_tag: str = "") -> dict:
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )

    ds_cfg  = cfg["dataset"]
    m_cfg   = cfg["model"]
    t_cfg   = cfg["training"]
    out_cfg = cfg["out"]

    root     = ds_cfg["root"]
    art_dir  = out_cfg["artifacts_dir"]
    ckpt     = cfg.get("checkpoint_override") or out_cfg["checkpoint"]
    os.makedirs(art_dir, exist_ok=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fanouts  = tuple(t_cfg["fanouts"])

    # Load label map.
    label_map_path = os.path.join(art_dir, "label_map.json")
    with open(label_map_path, "r", encoding="utf-8") as f:
        label2id: dict = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    classes  = [id2label[i] for i in range(len(id2label))]

    # Dataset.
    meta = load_store_meta(root)
    ds   = load_dataset(root)
    x_cat_csr = sparse.load_npz(os.path.join(root, "features", "x_cat.npz"))

    g = build_dgl_graph(root, meta.n_nodes)
    y_all = ds.feature.read("edge", None, "y").view(-1)
    g.edata["y"] = y_all.long()

    split_file = {"train": "train_eids.npy", "val": "val_eids.npy", "test": "test_eids.npy"}[split]
    eids = torch.from_numpy(
        np.load(os.path.join(root, "tasks", split_file)).astype(np.int64)
    )

    loader = make_edge_loader(
        g, eids, fanouts=fanouts,
        batch_size=t_cfg["batch_size"], shuffle=False,
        num_workers=t_cfg.get("num_workers", 0), seed=t_cfg["seed"],
    )

    # Model — dispatch on model.type so GCN/GAT/GraphSAGE all work.
    edge_in = meta.d_num + meta.d_cat
    model   = _build_model(m_cfg, edge_in, meta.num_classes)
    state   = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    logging.info(f"[eval] loaded checkpoint {ckpt}")

    # Inference.
    y_true, y_pred, y_prob = infer_split(model, loader, g, ds.feature, x_cat_csr, device)
    logging.info(f"[eval] {split}: N={len(y_true)}  classes={sorted(set(y_true.tolist()))}")

    # Metrics.
    metrics  = compute_metrics(y_true, y_pred, label2id)
    tag_part = f"_{model_tag}" if model_tag else ""
    out_json = os.path.join(art_dir, f"metrics_{split}{tag_part}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logging.info(
        f"[eval] macro P={metrics['macro']['precision']:.4f} "
        f"R={metrics['macro']['recall']:.4f} "
        f"F1={metrics['macro']['f1']:.4f} "
        f"FAR={metrics['macro_FAR']:.4f}"
    )

    # Per-class table.
    rows = []
    for cls_name, v in metrics["per_class"].items():
        rows.append({"Class": cls_name, "Precision": v["precision"],
                     "Recall": v["recall"], "F1": v["f1"], "FAR": v["FAR"]})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(os.path.join(art_dir, f"per_class_results_{split}{tag_part}.csv"), index=False)
    with open(os.path.join(art_dir, f"per_class_results_{split}{tag_part}.tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False))
    logging.info(f"[eval] saved per-class table to {art_dir}/per_class_results_{split}{tag_part}.*")

    if not no_plots:
        import matplotlib
        matplotlib.use("Agg")

        cm = np.array(metrics["confusion_matrix"])

        plot_confusion_matrix(
            y_true, y_pred, classes, normalize=True,
            title=f"Confusion Matrix — {split}",
            savepath=os.path.join(art_dir, f"confusion_matrix_{split}{tag_part}.png"),
        )
        _plot_fpr_bar(cm, classes, os.path.join(art_dir, f"fpr_per_class_{split}{tag_part}.png"))
        plot_roc_ovr(y_true, y_prob, classes,
                     out_path=os.path.join(art_dir, f"roc_{split}{tag_part}.png"))
        _plot_pr_curves(y_true, y_prob, classes,
                        os.path.join(art_dir, f"pr_{split}{tag_part}.png"))
        logging.info(f"[eval] plots saved to {art_dir}/")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config",     required=True,  help="Path to train YAML config.")
    ap.add_argument("--split",      default="test", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", default=None,   help="Override checkpoint path.")
    ap.add_argument("--model_tag",  default="",     help="Suffix for output files, e.g. 'gcn' → metrics_test_gcn.json.")
    ap.add_argument("--no_plots",   action="store_true", help="Skip matplotlib plots.")
    return ap.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.checkpoint:
        cfg["checkpoint_override"] = args.checkpoint
    metrics = run_eval(cfg, split=args.split, no_plots=args.no_plots, model_tag=args.model_tag)
    print(json.dumps({
        "accuracy":   metrics["accuracy"],
        "macro":      metrics["macro"],
        "macro_FAR":  metrics["macro_FAR"],
    }, indent=2))
