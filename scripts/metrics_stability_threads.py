# metrics_stability_threads.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def load_http_interval(base_dir, filename="curr_httpgetmt.csv"):
    path = Path(base_dir) / filename
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "successes" in df.columns and "failures" in df.columns:
        df = df[(df["successes"] == 1) & (df["failures"] == 0)]

    return df

def compute_stability_and_threads(base_dir, year=None, out_dir="plots"):
    # load multi-thread files
    df_mt = load_http_interval(base_dir, "curr_httpgetmt.csv")
    df_mt_post = load_http_interval(base_dir, "curr_httppostmt.csv")

    # Dodaj direction
    if not df_mt.empty:
        df_mt["direction"] = "download"
    if not df_mt_post.empty:
        df_mt_post["direction"] = "upload"

    df = pd.concat([df_mt, df_mt_post], ignore_index=True, sort=False)

    if df.empty:
        return {}

    df = df.dropna(subset=["bytes_sec_interval", "sequence"])
    df["bytes_sec_interval"] = df["bytes_sec_interval"].astype(float)

    group_cols = ["unit_id","direction"]
    stats = df.groupby(group_cols)["bytes_sec_interval"].agg(["mean","std","count"]).reset_index()
    stats["cv"] = stats["std"] / (stats["mean"] + 1e-9)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,5))
    sns.kdeplot(data=stats, x="cv", hue="direction", common_norm=False)
    plt.title(f"Coefficient of Variation (throughput intervals) - {year}")
    plt.xlabel("CV")
    plt.tight_layout()
    plt.savefig(Path(out_dir)/f"stability_cv_{year}.png")
    plt.close()

    agg = df.groupby(["direction","threads"])["bytes_sec_interval"].mean().reset_index()
    plt.figure(figsize=(8,5))
    sns.lineplot(data=agg, x="threads", y="bytes_sec_interval", hue="direction", marker="o")
    plt.yscale("log")
    plt.title(f"Throughput scaling with threads - {year}")
    plt.xlabel("threads")
    plt.ylabel("mean bytes_sec_interval")
    plt.tight_layout()
    plt.savefig(Path(out_dir)/f"threads_scaling_{year}.png")
    plt.close()

    return {"stability_stats": stats, "threads_agg": agg}
