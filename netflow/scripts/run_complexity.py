"""Complexity sweep: train time vs. dataset fraction.

Mirrors legacy notebook 012_Complexity_test.ipynb.  Re-uses training/train.py
for each fraction so results are directly comparable to the main training run.

Each fraction gets its own artifacts sub-directory so the main checkpoint is
NOT overwritten.

Usage:
  python scripts/run_complexity.py --config configs/train.yaml
  python scripts/run_complexity.py --config configs/train.yaml \\
      --fracs 0.25 0.5 0.75 1.0 --epochs 5 --plots
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
NETFLOW_ROOT = HERE.parent
if str(NETFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(NETFLOW_ROOT))

from training.train import run_training


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_complexity(
    base_cfg: dict,
    fracs: list[float],
    plots: bool = False,
) -> list[dict]:
    """Train once per fraction; return list of per-frac summary dicts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout, force=True,
    )

    art_base = Path(base_cfg["out"]["artifacts_dir"])
    results: list[dict] = []

    for frac in fracs:
        cfg = copy.deepcopy(base_cfg)
        cfg["frac"] = frac

        # Route outputs to a fraction-specific sub-directory.
        frac_tag  = f"frac{int(frac * 100):03d}"
        frac_dir  = art_base / f"complexity_{frac_tag}"
        frac_dir.mkdir(parents=True, exist_ok=True)
        cfg["out"]["artifacts_dir"] = str(frac_dir)
        cfg["out"]["checkpoint"]    = str(frac_dir / "best_edge_sage.pt")
        cfg["out"]["train_log"]     = str(frac_dir / "train_log.json")

        logging.info(f"[complexity] frac={frac:.2f}  output → {frac_dir}")
        t_wall_start = time.perf_counter()
        run_training(cfg)
        wall = time.perf_counter() - t_wall_start

        # Load per-epoch timings from train_log.
        log_path = frac_dir / "train_log.json"
        with open(log_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        mean_tr  = float(np.mean([e["train_time"] for e in history]))
        mean_val = float(np.mean([e["val_time"]   for e in history]))
        results.append({
            "frac":             frac,
            "mean_train_time":  mean_tr,
            "mean_val_time":    mean_val,
            "total_wall_s":     wall,
            "epochs":           len(history),
            "best_val_acc":     max(e["val_acc"] for e in history),
            "history":          history,
        })
        logging.info(
            f"[complexity] frac={frac:.2f}  "
            f"mean_train={mean_tr:.1f}s  mean_val={mean_val:.1f}s  "
            f"wall={wall:.1f}s  best_acc={results[-1]['best_val_acc']:.4f}"
        )

    # Save summary JSON.
    summary_path = art_base / "complexity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "history"} for r in results],
                  f, indent=2)
    logging.info(f"[complexity] summary → {summary_path}")

    if plots:
        _plot_complexity(results, art_base)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_complexity(results: list[dict], art_base: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs     = [r["frac"]            for r in results]
    tr_times  = [r["mean_train_time"] for r in results]
    val_times = [r["mean_val_time"]   for r in results]
    accs      = [r["best_val_acc"]    for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(fracs, tr_times,  "o-", label="train/epoch")
    ax.plot(fracs, val_times, "s--", label="val/epoch")
    ax.set_xlabel("Dataset fraction")
    ax.set_ylabel("Mean time per epoch (s)")
    ax.set_title("Training time vs dataset fraction")
    ax.legend()
    ax.set_xticks(fracs)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(fracs, accs, "^-", color="C2")
    ax2.set_xlabel("Dataset fraction")
    ax2.set_ylabel("Best val accuracy")
    ax2.set_title("Val accuracy vs dataset fraction")
    ax2.set_xticks(fracs)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = art_base / "complexity_scaling.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.getLogger(__name__).info(f"[complexity] plot → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config",  required=True, help="Path to base training YAML config.")
    ap.add_argument("--fracs",   nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0],
                    help="Dataset fractions to sweep (default: 0.25 0.5 0.75 1.0).")
    ap.add_argument("--epochs",  type=int,   default=None, help="Override epochs per sweep point.")
    ap.add_argument("--plots",   action="store_true", help="Save scaling plots.")
    ap.add_argument("--device",  type=str,   default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.device is not None:
        cfg["training"]["device"] = args.device

    results = run_complexity(cfg, sorted(set(args.fracs)), plots=args.plots)
    print(json.dumps(
        [{k: v for k, v in r.items() if k != "history"} for r in results],
        indent=2,
    ))
