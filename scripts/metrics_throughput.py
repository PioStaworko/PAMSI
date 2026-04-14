# metrics_throughput.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def load_http_files(base_dir):
    base_dir = Path(base_dir)
    files = {
        "httpget": base_dir / "curr_httpget.csv",
        "httpgetmt": base_dir / "curr_httpgetmt.csv",
        "httppost": base_dir / "curr_httppost.csv",
        "httppostmt": base_dir / "curr_httppostmt.csv",
    }
    dfs = []
    for name, path in files.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["direction"] = "download" if "httpget" in name else "upload"
        df["mode"] = "multi" if "mt" in name else "single"
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True, sort=False)
    if "successes" in df_all.columns:
        df_all = df_all[(df_all["successes"] == 1) & (df_all["failures"] == 0)]
    return df_all

def compute_throughput_metrics(base_dir, year=None, out_dir="plots"):
    df = load_http_files(base_dir)
    df = df.dropna(subset=["bytes_sec"])
    df["bytes_sec"] = df["bytes_sec"].astype(float)
    results = {}
    for direction in ["download", "upload"]:
        sub = df[df["direction"] == direction]
        percentiles = np.percentile(sub["bytes_sec"], [1,5,10,25,50,75,90,95,99])
        p_index = [1,5,10,25,50,75,90,95,99]
        pct = pd.Series(percentiles, index=p_index)
        results[direction] = {
            "count": len(sub),
            "median": float(np.median(sub["bytes_sec"])),
            "mean": float(np.mean(sub["bytes_sec"])),
            "std": float(np.std(sub["bytes_sec"])),
            "percentiles": pct
        }

    # asymmetry index
    asym = results["download"]["median"] / (results["upload"]["median"] + 1e-9)
    results["asymmetry_index"] = asym

    # save simple plots
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plot_throughput_distributions(df, out_dir, year)
    return results

def plot_throughput_distributions(df, out_dir, year=None):
    out = Path(out_dir)
    plt.figure(figsize=(8,5))
    sns.kdeplot(data=df, x="bytes_sec", hue="direction", common_norm=False, log_scale=True)
    plt.title(f"Rozkład throughput (bytes/sec) - {year}" if year else "Rozkład throughput")
    plt.xlabel("bytes_sec (log scale)")
    plt.tight_layout()
    plt.savefig(out / f"throughput_density_{year}.png")
    plt.close()

    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="direction", y="bytes_sec", hue="mode")
    plt.yscale("log")
    plt.title(f"Throughput vs tryb (single/multi) - {year}" if year else "Throughput vs tryb")
    plt.tight_layout()
    plt.savefig(out / f"throughput_box_{year}.png")
    plt.close()
