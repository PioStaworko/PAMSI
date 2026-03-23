import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_http_data(base_dir):
    base = Path(base_dir)

    # download
    httpget = pd.read_csv(base / "curr_httpget.csv")
    httpgetmt = pd.read_csv(base / "curr_httpgetmt.csv")


    # upload
    httppost = pd.read_csv(base / "curr_httppost.csv")
    httppostmt = pd.read_csv(base / "curr_httppostmt.csv")

    httpget["direction"] = "download"
    httpget["mode"] = "single"
    httpgetmt["direction"] = "download"
    httpgetmt["mode"] = "multi"

    httppost["direction"] = "upload"
    httppost["mode"] = "single"
    httppostmt["direction"] = "upload"
    httppostmt["mode"] = "multi"

    df = pd.concat([httpget, httpgetmt, httppost, httppostmt], ignore_index=True)

    df = df[(df["successes"] == 1) & (df["failures"] == 0)]

    return df

def analyze_download_upload(base_dir, year, out_dir="plots"):
    df = load_http_data(base_dir)
    df["year"] = year

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8,5))
    sns.kdeplot(data=df, x="bytes_sec", hue="direction", common_norm=False, log_scale=True)
    plt.title(f"Rozkład throughput (bytes/sec) - {year}")
    plt.xlabel("bytes/sec (log scale)")
    plt.tight_layout()
    plt.savefig(out_path / f"throughput_{year}.png")
    plt.close()

    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="direction", y="bytes_sec", hue="mode")
    plt.yscale("log")
    plt.title(f"Throughput vs tryb (single/multi) - {year}")
    plt.tight_layout()
    plt.savefig(out_path / f"throughput_box_direction_mode{year}.png")
    plt.close()

    return df