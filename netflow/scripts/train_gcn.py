"""Train the GCN edge-classifier baseline on NF-UNSW-NB15-v3.

Thin wrapper around training.train.run_training with model.type=gcn.

Usage:
  python scripts/train_gcn.py --config configs/gcn.yaml
  python scripts/train_gcn.py --config configs/gcn.yaml --epochs 10 --frac 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
NETFLOW_ROOT = HERE.parent
if str(NETFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(NETFLOW_ROOT))

from training.train import run_training, _apply_overrides


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config",     required=True,  help="Path to YAML config (e.g. configs/gcn.yaml).")
    ap.add_argument("--epochs",     type=int,   default=None)
    ap.add_argument("--batch_size", type=int,   default=None)
    ap.add_argument("--lr",         type=float, default=None)
    ap.add_argument("--hidden",     type=int,   default=None)
    ap.add_argument("--frac",       type=float, default=None,
                    help="Stratified subsample fraction of TRAIN/VAL.")
    ap.add_argument("--seed",       type=int,   default=None)
    ap.add_argument("--device",     type=str,   default=None)
    ap.add_argument("--debug",      action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["type"] = "gcn"  # enforce model type regardless of config
    cfg = _apply_overrides(cfg, args)
    result = run_training(cfg)
    print(json.dumps(result, indent=2))
