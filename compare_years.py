# compare_years.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from download_upload_analysis import load_http_data

def compare_years(base_dir_2021, base_dir_2023, out_dir="plots"):
    df21 = load_http_data(base_dir_2021)
    df21["year"] = 2021

    df23 = load_http_data(base_dir_2023)
    df23["year"] = 2023

    df = pd.concat([df21, df23], ignore_index=True)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Porównanie throughput download dla 2021 vs 2023
    download = df[df["direction"] == "download"]

    plt.figure(figsize=(8,5))
    sns.kdeplot(data=download, x="bytes_sec", hue="year", common_norm=False, log_scale=True)
    plt.title("Download throughput – porównanie 2021 vs 2023")
    plt.xlabel("bytes_sec (log scale)")
    plt.tight_layout()
    plt.savefig(out_path / "download_throughput_2021_vs_2023.png")
    plt.close()

    # Porównanie upload
    upload = df[df["direction"] == "upload"]

    plt.figure(figsize=(8,5))
    sns.kdeplot(data=upload, x="bytes_sec", hue="year", common_norm=False, log_scale=True)
    plt.title("Upload throughput – porównanie 2021 vs 2023")
    plt.xlabel("bytes_sec (log scale)")
    plt.tight_layout()
    plt.savefig(out_path / "upload_throughput_2021_vs_2023.png")
    plt.close()

    # Możesz dodać porównanie median, boxploty itd.
    return df
