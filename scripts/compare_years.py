# compare_years.py

from scripts.metrics_throughput import compute_throughput_metrics, load_http_files
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def compare_and_plot_years(dir2021, dir2023, out_dir="plots/compare"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # throughput metrics
    t21 = compute_throughput_metrics(dir2021, year=2021, out_dir=out_dir)
    t23 = compute_throughput_metrics(dir2023, year=2023, out_dir=out_dir)

    # build median table
    med_table = pd.DataFrame({
        "2021_download_median": [t21["download"]["median"]],
        "2021_upload_median": [t21["upload"]["median"]],
        "2023_download_median": [t23["download"]["median"]],
        "2023_upload_median": [t23["upload"]["median"]],
    })

    # plot side-by-side KDE for download
    df21 = load_http_files(dir2021); df21["year"]=2021
    df23 = load_http_files(dir2023); df23["year"]=2023
    df = pd.concat([df21, df23], ignore_index=True, sort=False)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,5))
    sns.kdeplot(data=df[df["direction"]=="download"], x="bytes_sec", hue="year", log_scale=True)
    plt.title("Download throughput 2021 vs 2023")
    plt.savefig(Path(out_dir)/"download_kde_2021_2023.png")
    plt.close()
    plt.figure(figsize=(8,5))
    sns.kdeplot(data=df[df["direction"]=="upload"], x="bytes_sec", hue="year", log_scale=True)
    plt.title("Upload throughput 2021 vs 2023")
    plt.savefig(Path(out_dir)/"upload_kde_2021_2023.png")
    plt.close()
    return {"median_table": med_table, "t21": t21, "t23": t23}
