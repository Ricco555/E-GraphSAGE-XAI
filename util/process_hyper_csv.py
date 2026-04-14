import re
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import argparse  # <--- new

# default path (if no CLI argument is given)
DEFAULT_LOG_PATH = Path(r"c:\Users\riko.lusa\src\phd-i4sec\E-GraphSAGE-XAI\util\Hyperparameter_tuning_results.txt")

# Provided max val accuracy for normalization
MAX_VAL_ACC = 0.95586

# Example run line:
# === RUN: hid64_aggmean_fan15-10_drop20_bs256 ===
run_re = re.compile(
    r"=== RUN: hid(?P<hidden>\d+)_agg(?P<agg>\w+)_fan(?P<fan_s>\d+)-(?P<fan_l>\d+)_drop(?P<drop>\d+)_bs(?P<bs>\d+)\s*==="
)

# Example epoch line (adapt if your format differs):
# 2025-11-30 13:56:00,604 INFO [epoch 1] train loss 0.9 train acc 0.5 | val loss 1.0 val acc 0.45
epoch_re = re.compile(
    r"\[epoch\s+(?P<epoch>\d+)]\s+"
    r"train loss\s+(?P<train_loss>[0-9.]+)\s+train acc\s+(?P<train_acc>[0-9.]+)\s+\|\s+"
    r"val loss\s+(?P<val_loss>[0-9.]+)\s+val acc\s+(?P<val_acc>[0-9.]+)"
)


def parse_log(log_path: Path):
    rows = []
    current_run = None

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 1) Detect start of a run
            m_run = run_re.match(line)
            if m_run:
                gd = m_run.groupdict()
                current_run = {
                    "run_name": line.split("=== RUN:")[-1].split("===")[0].strip(),
                    "hidden": int(gd["hidden"]),
                    "aggregator": gd["agg"],
                    "fanout_s": int(gd["fan_s"]),
                    "fanout_l": int(gd["fan_l"]),
                    "dropout": int(gd["drop"]) / 100.0,
                    "batch_size": int(gd["bs"]),
                }
                continue

            # 2) Epoch metrics within current run
            m_ep = epoch_re.search(line)
            if m_ep and current_run is not None:
                egd = m_ep.groupdict()
                epoch = int(egd["epoch"])
                train_loss = float(egd["train_loss"])
                train_acc = float(egd["train_acc"])
                val_loss = float(egd["val_loss"])
                val_acc = float(egd["val_acc"])

                # Normalize by provided max val acc
                norm_train_acc = train_acc
                norm_val_acc = val_acc

                row = {
                    **current_run,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "norm_train_acc": norm_train_acc,
                    "norm_val_acc": norm_val_acc,
                }
                rows.append(row)

    return rows


def write_csv(rows, csv_path: Path):
    if not rows:
        print("No rows parsed – CSV not written.")
        return

    fieldnames = [
        "run_name",
        "hidden",
        "aggregator",
        "fanout_s",
        "fanout_l",
        "dropout",
        "batch_size",
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "norm_train_acc",
        "norm_val_acc",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote CSV with {len(rows)} rows -> {csv_path}")


def plot_all_runs(rows, fig_path: Path):
    if not rows:
        print("No rows, skipping plot.")
        return

    from collections import defaultdict

    # 1) Filter: fanouts 25-15 and dropout 0.40
    filtered = [
        r for r in rows
        if r.get("fanout_s") == 25
        and r.get("fanout_l") == 15
        and abs(r.get("dropout", 0.0) - 0.40) < 1e-8
    ]
    if not filtered:
        print("No rows after filtering for fan25-15 & drop=0.40, skipping plot.")
        return

    # 2) Group by (hidden, batch_size)
    by_cfg = defaultdict(list)
    for r in filtered:
        key = (r["hidden"], r["batch_size"])
        by_cfg[key].append(r)

    plt.figure(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, ((hidden, bs), epochs) in enumerate(sorted(by_cfg.items())):
        epochs = sorted(epochs, key=lambda x: x["epoch"])
        # keep every 5th epoch only
        xs = [e["epoch"] for e in epochs]
        ys = [e["norm_train_acc"] for e in epochs] 

        if not xs:
            continue

        color = colors[i % len(colors)]
        label = f"hid={hidden}, bs={bs}"
        plt.plot(
            xs,
            ys,
            color=color,
            linestyle="-",
            linewidth=1.5,
            alpha=0.9,
            label=label,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    print(f"Saved plot -> {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Process hyperparameter tuning log to CSV + plot.")
    parser.add_argument(
        "log_file",
        nargs="?",
        help="Path to log .txt file (default: built-in Hyperparameter_tuning_results.txt)",
    )
    args = parser.parse_args()

    # decide which log path to use
    log_path = Path(args.log_file) if args.log_file else DEFAULT_LOG_PATH

    if not log_path.exists():
        raise FileNotFoundError(log_path)

    print(f"Parsing log: {log_path}")

    rows = parse_log(log_path)
    print(f"Parsed {len(rows)} epoch rows across {len(set(r['run_name'] for r in rows))} runs")

    out_csv = log_path.with_suffix(".csv")
    out_fig = log_path.with_name(log_path.stem + "_acc_vs_epoch.png")

    write_csv(rows, out_csv)
    plot_all_runs(rows, out_fig)


if __name__ == "__main__":
    main()