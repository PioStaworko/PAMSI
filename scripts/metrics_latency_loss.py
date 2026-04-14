# metrics_latency_loss.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def load_ping(base_dir, filename):
    p = Path(base_dir) / filename
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

def compute_latency_loss_metrics(base_dir, year=None, out_dir="plots"):
    # load ping and udp files
    ping = load_ping(base_dir, "curr_ping.csv")
    dlping = load_ping(base_dir, "curr_dlping.csv")
    ulping = load_ping(base_dir, "curr_ulping.csv")
    udpj = load_ping(base_dir, "curr_udpjitter.csv")
    udplat = load_ping(base_dir, "curr_udplatency.csv")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = {}

    # RTT basic stats
    if not ping.empty:
        if "rtt_avg" in ping.columns:
            ping = ping[(ping["rtt_avg"] != 0)]

        ping["rtt_avg"] = ping["rtt_avg"].astype(float)
        rtt_stats = ping["rtt_avg"].describe(percentiles=[.1,.25,.5,.75,.9,.95]).to_dict()
        results["rtt_stats"] = rtt_stats

        # plot rtt distribution
        plt.figure(figsize=(8,5))
        sns.kdeplot(ping["rtt_avg"], log_scale=True)
        plt.title(f"RTT avg distribution - {year}")
        plt.xlabel("rtt_avg (microseconds)")
        plt.tight_layout()
        plt.savefig(Path(out_dir)/f"rtt_dist_{year}.png")
        plt.close()

    # latency under load: compare dlping/ulping vs ping
    if not dlping.empty and not ping.empty:
        # merge by unit_id,dtime,target if possible (approx)
        merged = dlping.merge(ping[["unit_id","dtime","rtt_avg"]], on=["unit_id","dtime"], how="left", suffixes=("_dl","_idle"))
        merged["delta"] = merged["rtt_avg_dl"] - merged["rtt_avg_idle"]
        results["dl_delta_mean"] = float(merged["delta"].mean())
    if not ulping.empty and not ping.empty:
        merged2 = ulping.merge(ping[["unit_id","dtime","rtt_avg"]], on=["unit_id","dtime"], how="left", suffixes=("_ul","_idle"))
        merged2["delta"] = merged2["rtt_avg_ul"] - merged2["rtt_avg_idle"]
        results["ul_delta_mean"] = float(merged2["delta"].mean())

    # jitter and packet loss
    if not udpj.empty:
        udpj["jitter_up"] = pd.to_numeric(udpj.get("jitter_up", pd.Series(dtype=float)), errors="coerce")
        udpj["jitter_down"] = pd.to_numeric(udpj.get("jitter_down", pd.Series(dtype=float)), errors="coerce")
        results["jitter_stats"] = {
            "up_median": float(udpj["jitter_up"].median()) if "jitter_up" in udpj.columns else None,
            "down_median": float(udpj["jitter_down"].median()) if "jitter_down" in udpj.columns else None
        }
    if not udplat.empty:
        udplat["rtt_avg"] = pd.to_numeric(udplat.get("rtt_avg", pd.Series(dtype=float)), errors="coerce")
        # packet loss estimate: failures/(successes+failures) if present
        if "successes" in udplat.columns and "failures" in udplat.columns:
            udplat["loss_rate"] = udplat["failures"] / (udplat["successes"] + udplat["failures"] + 1e-9)
            results["loss_median"] = float(udplat["loss_rate"].median())

    # try to load httpget and merge by unit_id,dtime,target
    try:
        http = pd.read_csv(Path(base_dir)/"curr_httpget.csv")
        if "bytes_sec" in http.columns and "rtt_avg" in ping.columns:
            merged_h = http.merge(ping[["unit_id","dtime","rtt_avg"]], on=["unit_id","dtime"], how="left")
            merged_h = merged_h.dropna(subset=["bytes_sec","rtt_avg"])
            corr = merged_h["bytes_sec"].corr(merged_h["rtt_avg"])
            results["throughput_rtt_corr"] = float(corr)
    except Exception:
        pass

    return results
