"""Variable-level SHAP XAI for the trained EdgeGraphSAGE model.

Runs grouped KernelSHAP in the ~45-dim variable space (38 numeric +
7 categorical variables) so each categorical variable is one coalition
member rather than up to 255 independent one-hot bits.

Outputs (under <out.save_dir>):
  topk_<class>.png          — top-k mean |SHAP| bar chart per class
  spider.png                — spider chart across all classes
  beeswarm_<class>.png      — SHAP beeswarm dot plot per class
  shap_summary.json         — mean |SHAP| per variable per class

Usage:
  python scripts/run_xai.py --config configs/xai.yaml
  python scripts/run_xai.py --config configs/xai.yaml --n_per_class 20 --device cpu
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
import yaml
from scipy import sparse

HERE = Path(__file__).resolve().parent
NETFLOW_ROOT = HERE.parent
if str(NETFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(NETFLOW_ROOT))

from datasets.nfunsw_nb15 import load_dataset, load_store_meta
from training.train import _build_model
from training.samplers import build_dgl_graph, make_edge_loader
from xai import (
    load_feature_names,
    load_cat_variable_names,
    build_variable_groups,
    aggregate_shap_per_class,
    aggregate_signed_shap_per_class,
    plot_topk_per_class,
    plot_spider,
    beeswarm_for_class,
    CategoryColorMap,
    plot_grouped_shap_bar,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_merged_config(xai_cfg_path: str) -> tuple[dict, dict]:
    """Load xai.yaml and merge with its base_config."""
    with open(xai_cfg_path, "r", encoding="utf-8") as f:
        xai_cfg = yaml.safe_load(f)
    base_path = xai_cfg.get("base_config", "configs/train.yaml")
    with open(base_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    return xai_cfg, base_cfg


def _sample_background(g, split_root: str, n: int, seed: int) -> np.ndarray:
    """Sample n edge IDs from the training split as SHAP background."""
    train_eids = np.load(os.path.join(split_root, "tasks", "train_eids.npy"))
    rng = np.random.default_rng(seed)
    return rng.choice(train_eids, size=min(n * 10, len(train_eids)), replace=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_xai(xai_cfg: dict, base_cfg: dict) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )

    x_cfg    = xai_cfg["xai"]
    out_cfg  = xai_cfg["out"]
    ds_cfg   = base_cfg["dataset"]
    m_cfg    = base_cfg["model"]
    t_cfg    = base_cfg["training"]
    out_base = base_cfg["out"]

    root     = ds_cfg["root"]
    art_dir  = out_base["artifacts_dir"]
    ckpt     = out_base["checkpoint"]
    save_dir = out_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[xai] device={device}")

    # Dataset.
    meta      = load_store_meta(root)
    ds        = load_dataset(root)
    x_cat_csr = sparse.load_npz(os.path.join(root, "features", "x_cat.npz"))

    g = build_dgl_graph(root, meta.n_nodes)
    y_all = ds.feature.read("edge", None, "y").view(-1)
    g.edata["y"] = y_all.long()

    # Label map.
    with open(os.path.join(art_dir, "label_map.json"), "r", encoding="utf-8") as f:
        label2id: dict = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    # Loader over val split (held-out; not used for SHAP background).
    val_eids = torch.from_numpy(
        np.load(os.path.join(root, "tasks", "val_eids.npy")).astype(np.int64)
    )
    loader = make_edge_loader(
        g, val_eids,
        fanouts=tuple(t_cfg["fanouts"]),
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        num_workers=t_cfg.get("num_workers", 0),
        seed=t_cfg["seed"],
    )

    # Model.
    edge_in = meta.d_num + meta.d_cat
    model   = _build_model(m_cfg, edge_in, meta.num_classes)
    state   = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    logging.info(f"[xai] loaded {ckpt}")

    # Variable groups (38 numeric + 7 categorical = 45 variables).
    feat_names, d_num, _ = load_feature_names(art_dir)
    cat_cols             = load_cat_variable_names(art_dir)
    variable_groups      = build_variable_groups(feat_names, d_num, cat_cols)
    var_names            = [grp["name"] for grp in variable_groups]
    logging.info(f"[xai] variable space: {len(variable_groups)} variables "
                 f"({d_num} numeric + {len(variable_groups) - d_num} categorical)")

    # Background edges from training split.
    bg_eids = _sample_background(g, root, x_cfg["background_size"], x_cfg["seed"])

    # Which classes to process.
    all_classes = sorted(id2label.keys())
    classes_to_run = x_cfg.get("classes") or all_classes

    # -----------------------------------------------------------------------
    # 1. Mean |SHAP| per variable per class.
    # -----------------------------------------------------------------------
    logging.info("[xai] running aggregate_shap_per_class (grouped)…")
    class2imp = aggregate_shap_per_class(
        model, g, loader, ds.feature, x_cat_csr, bg_eids,
        n_per_class=x_cfg["n_per_class"],
        background_size=x_cfg["background_size"],
        device=str(device),
        variable_groups=variable_groups,
    )
    logging.info(f"[xai] collected mean |SHAP| for {len(class2imp)} classes")

    # Save summary JSON.
    summary = {
        id2label[c]: {var_names[j]: float(v)
                      for j, v in enumerate(imp)}
        for c, imp in class2imp.items()
    }
    with open(os.path.join(save_dir, "shap_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"[xai] shap_summary.json → {save_dir}")

    # -----------------------------------------------------------------------
    # 2. Top-k bar charts per class.
    # -----------------------------------------------------------------------
    plot_topk_per_class(
        class2imp, var_names, id2label,
        k=x_cfg["top_k"],
        save_dir=save_dir,
        title="Top-k Variable Importances",
        signed=False,
    )
    logging.info("[xai] top-k bar charts saved")

    # -----------------------------------------------------------------------
    # 3. Spider chart (all classes, top-k union of variables).
    # -----------------------------------------------------------------------
    plot_spider(
        class2imp, var_names, id2label,
        k_union=x_cfg["top_k"],
        save_path=os.path.join(save_dir, "spider.png"),
    )
    logging.info("[xai] spider.png saved")

    # -----------------------------------------------------------------------
    # 4. Signed mean SHAP bar chart per class.
    # -----------------------------------------------------------------------
    logging.info("[xai] running aggregate_signed_shap_per_class (grouped)…")
    class2signed = aggregate_signed_shap_per_class(
        model, g, loader, ds.feature, x_cat_csr, bg_eids,
        n_per_class=x_cfg["n_per_class"],
        background_size=x_cfg["background_size"],
        device=str(device),
        variable_groups=variable_groups,
    )
    cmap = CategoryColorMap()
    for c, vals in class2signed.items():
        label = id2label.get(c, str(c))
        safe  = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in label).lower()
        plot_grouped_shap_bar(
            var_names, vals, cmap,
            top_k=x_cfg["top_k"],
            title=f"Signed SHAP — {label}",
            signed=True,
            savepath=os.path.join(save_dir, f"signed_shap_{safe}.png"),
        )
    logging.info("[xai] signed SHAP bar charts saved")

    # -----------------------------------------------------------------------
    # 5. Beeswarm per class.
    # -----------------------------------------------------------------------
    for c in classes_to_run:
        if c not in id2label:
            continue
        label = id2label[c]
        safe  = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in label).lower()
        logging.info(f"[xai] beeswarm class {c} ({label})…")
        try:
            beeswarm_for_class(
                model, g, loader, ds.feature, x_cat_csr, bg_eids,
                class_id=c,
                n_samples=x_cfg["n_beeswarm"],
                background_size=x_cfg["background_size"],
                top_k=x_cfg["top_k"],
                feature_names=var_names,
                device=str(device),
                savepath=os.path.join(save_dir, f"beeswarm_{safe}.png"),
                variable_groups=variable_groups,
            )
        except Exception as e:
            logging.warning(f"[xai] beeswarm class {c} failed: {e}")
    logging.info("[xai] beeswarm plots saved")

    logging.info(f"[xai] all outputs in {save_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config",         required=True, help="Path to xai.yaml config.")
    ap.add_argument("--n_per_class",    type=int,   default=None,
                    help="Override xai.n_per_class.")
    ap.add_argument("--background_size", type=int,  default=None,
                    help="Override xai.background_size.")
    ap.add_argument("--classes",        nargs="+",  type=int, default=None,
                    help="Limit to these class IDs.")
    ap.add_argument("--device",         type=str,   default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    xai_cfg, base_cfg = _load_merged_config(args.config)

    if args.n_per_class is not None:
        xai_cfg["xai"]["n_per_class"] = args.n_per_class
    if args.background_size is not None:
        xai_cfg["xai"]["background_size"] = args.background_size
    if args.classes is not None:
        xai_cfg["xai"]["classes"] = args.classes
    if args.device is not None:
        base_cfg["training"]["device"] = args.device

    run_xai(xai_cfg, base_cfg)
