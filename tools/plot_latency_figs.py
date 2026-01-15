#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_latency_compare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize column names (your file uses these headers)
    # instr(prev), base_samples, base_mean_ns, base_p90_ns~, opt_samples, opt_mean_ns, opt_p90_ns~
    for col in ["base_samples", "base_mean_ns", "opt_samples", "opt_mean_ns"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["instr"] = df["instr(prev)"].astype(str)

    # Contribution proxy (PPT explanation: start->next-start accumulated)
    df["base_contrib"] = df["base_samples"] * df["base_mean_ns"]
    df["opt_contrib"]  = df["opt_samples"]  * df["opt_mean_ns"]

    base_total = df["base_contrib"].sum()
    opt_total  = df["opt_contrib"].sum()

    # Convert to percent to be scale-invariant / PPT-friendly
    df["base_pct"] = (df["base_contrib"] / base_total * 100.0) if base_total > 0 else 0.0
    df["opt_pct"]  = (df["opt_contrib"]  / opt_total  * 100.0) if opt_total  > 0 else 0.0

    df["delta_pct"] = df["opt_pct"] - df["base_pct"]
    df["max_pct"]   = np.maximum(df["base_pct"], df["opt_pct"])
    df["abs_delta"] = np.abs(df["delta_pct"])

    return df


def plot_top_contrib(df: pd.DataFrame, out_png: str, top_n: int = 10):
    # Pick Top-N by max contribution share (baseline or opt)
    top = df.sort_values("max_pct", ascending=False).head(top_n).copy()
    top = top.sort_values("max_pct", ascending=True)  # for nice barh order

    y = np.arange(len(top))
    h = 0.40

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)

    ax.barh(y - h/2, top["base_pct"], height=h, label="baseline")
    ax.barh(y + h/2, top["opt_pct"],  height=h, label="optimized")

    ax.set_yticks(y)
    ax.set_yticklabels(top["instr"])
    ax.set_xlabel("Contribution share (%)  ~  samples × mean_ns normalized")
    ax.set_title(f"Top-{top_n} instruction time contribution (baseline vs optimized)")

    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
    ax.legend()

    # Optional numeric labels
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r["base_pct"], y[i] - h/2, f" {r['base_pct']:.2f}%", va="center", fontsize=9)
        ax.text(r["opt_pct"],  y[i] + h/2, f" {r['opt_pct']:.2f}%",  va="center", fontsize=9)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_delta_tornado(df: pd.DataFrame, out_png: str, top_n: int = 10):
    # Pick Top-N by absolute delta contribution
    top = df.sort_values("abs_delta", ascending=False).head(top_n).copy()
    top = top.sort_values("delta_pct", ascending=True)  # negative first (saves), then positive

    y = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)

    ax.barh(y, top["delta_pct"])
    ax.set_yticks(y)
    ax.set_yticklabels(top["instr"])
    ax.set_xlabel("Δ contribution share (opt - baseline), percentage points")
    ax.set_title(f"Top-{top_n} contribution changes (tornado)")

    # zero line (no explicit color)
    ax.axvline(0, linewidth=1.0)

    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    # Labels
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r["delta_pct"], y[i], f" {r['delta_pct']:+.2f}pp", va="center", fontsize=9)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_bubble_scatter(df: pd.DataFrame, out_png: str, label_k: int = 8):
    """
    Scatter: x=samples (log), y=mean_ns (log), bubble size ~ sqrt(contrib).
    Overlay baseline and optimized with different markers.
    """
    # Avoid zeros in log scale
    base_x = df["base_samples"].replace(0, np.nan)
    base_y = df["base_mean_ns"].replace(0, np.nan)
    opt_x  = df["opt_samples"].replace(0, np.nan)
    opt_y  = df["opt_mean_ns"].replace(0, np.nan)

    # Bubble size scaling (sqrt to compress dynamic range)
    base_s = np.sqrt(df["base_contrib"].replace(0, np.nan))
    opt_s  = np.sqrt(df["opt_contrib"].replace(0, np.nan))

    # Normalize sizes to a reasonable range
    def norm_sizes(s):
        s = s.copy()
        mx = np.nanmax(s.values) if np.nanmax(s.values) else 1.0
        return (s / mx * 800.0).fillna(0.0)

    base_sizes = norm_sizes(base_s)
    opt_sizes  = norm_sizes(opt_s)

    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)

    ax.scatter(base_x, base_y, s=base_sizes, marker="o", alpha=0.7, label="baseline")
    ax.scatter(opt_x,  opt_y,  s=opt_sizes,  marker="x", alpha=0.9, label="optimized")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("samples (log)")
    ax.set_ylabel("mean_ns (log)")
    ax.set_title("Frequency vs latency (bubble size ~ sqrt(contribution))")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    # Label a few most informative points:
    # here we label the largest |delta| contributors
    lab = df.sort_values("abs_delta", ascending=False).head(label_k)
    for _, r in lab.iterrows():
        # baseline point label
        if r["base_samples"] > 0 and r["base_mean_ns"] > 0:
            ax.text(r["base_samples"], r["base_mean_ns"], f" {r['instr']}", fontsize=8)
        # optimized point label
        if r["opt_samples"] > 0 and r["opt_mean_ns"] > 0:
            ax.text(r["opt_samples"], r["opt_mean_ns"], f" {r['instr']}", fontsize=8)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="latency_compare.csv path")
    ap.add_argument("--outdir", default="plots", help="output directory")
    ap.add_argument("--topn", type=int, default=10, help="Top-N for bar/tornado")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_latency_compare(args.csv)

    plot_top_contrib(df, os.path.join(args.outdir, "top_contribution.png"), top_n=args.topn)
    plot_delta_tornado(df, os.path.join(args.outdir, "delta_tornado.png"), top_n=args.topn)
    plot_bubble_scatter(df, os.path.join(args.outdir, "bubble_scatter.png"), label_k=min(8, args.topn))

    # Print PPT-ready highlights (you can paste into speaker notes)
    top_save = df.sort_values("delta_pct").head(5)[["instr","delta_pct","base_pct","opt_pct"]]
    top_cost = df.sort_values("delta_pct", ascending=False).head(5)[["instr","delta_pct","base_pct","opt_pct"]]

    print("\n=== PPT-ready: Biggest SAVINGS (negative Δ) ===")
    print(top_save.to_string(index=False))
    print("\n=== PPT-ready: Biggest COSTS (positive Δ) ===")
    print(top_cost.to_string(index=False))
    print("\n[OK] Saved plots to:", args.outdir)


if __name__ == "__main__":
    main()
