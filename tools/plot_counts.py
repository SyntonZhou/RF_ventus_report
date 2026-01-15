#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_counts(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Flexible column mapping (compatible with your headers)
    colmap = {}
    for c in df.columns:
        c2 = c.strip()
        colmap[c] = c2
    df = df.rename(columns=colmap)

    # Required columns (try to infer)
    # instr, baseline_count, opt_count, delta(opt-base), ratio(opt/base)
    if "instr" not in df.columns:
        raise ValueError("CSV must contain column: instr")

    # Normalize numeric columns robustly
    num_cols = [c for c in df.columns if c != "instr"]
    for c in num_cols:
        # remove stray spaces like "0.00 "
        df[c] = df[c].astype(str).str.strip()
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Make standard names
    if "baseline_count" not in df.columns:
        # try common alternatives
        for alt in ["base_count", "baseline", "base"]:
            if alt in df.columns:
                df["baseline_count"] = df[alt]
                break
    if "opt_count" not in df.columns:
        for alt in ["optimized_count", "opt", "new_count"]:
            if alt in df.columns:
                df["opt_count"] = df[alt]
                break
    if "delta(opt-base)" not in df.columns:
        for alt in ["delta", "opt-base", "delta_count"]:
            if alt in df.columns:
                df["delta(opt-base)"] = df[alt]
                break
    if "ratio(opt/base)" not in df.columns:
        for alt in ["ratio", "opt_over_base"]:
            if alt in df.columns:
                df["ratio(opt/base)"] = df[alt]
                break

    # Compute missing delta/ratio if absent
    if "delta(opt-base)" not in df.columns:
        df["delta(opt-base)"] = df["opt_count"] - df["baseline_count"]

    if "ratio(opt/base)" not in df.columns:
        df["ratio(opt/base)"] = np.where(df["baseline_count"] > 0,
                                         df["opt_count"] / df["baseline_count"],
                                         np.inf)

    df["instr"] = df["instr"].astype(str)

    # Standardized helper columns
    df["delta"] = df["delta(opt-base)"].fillna(0).astype(float)
    df["ratio"] = df["ratio(opt/base)"].replace([np.inf, -np.inf], np.nan)

    # abs(delta) for sorting
    df["abs_delta"] = df["delta"].abs()

    # log10 ratio (only valid when base>0 and ratio>0)
    df["log10_ratio"] = np.nan
    m = (df["baseline_count"] > 0) & (df["ratio"] > 0) & (~df["ratio"].isna())
    df.loc[m, "log10_ratio"] = np.log10(df.loc[m, "ratio"])

    return df


def plot_top_delta(df: pd.DataFrame, out_png: str, top_k: int = 12):
    # Sort by absolute delta
    top = df.sort_values("abs_delta", ascending=False).head(top_k).copy()
    top = top.sort_values("delta", ascending=True)

    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)

    ax.barh(y, top["delta"])
    ax.set_yticks(y)
    ax.set_yticklabels(top["instr"])
    ax.set_xlabel("Δ count = opt - baseline (negative means reduced)")
    ax.set_title(f"Top-{top_k} instruction count changes (absolute Δ)")

    ax.axvline(0, linewidth=1.0)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r["delta"], y[i], f" {int(r['delta'])}", va="center", fontsize=9)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_top_log_ratio(df: pd.DataFrame, out_png: str, top_k: int = 12):
    sub = df.dropna(subset=["log10_ratio"]).copy()
    if len(sub) == 0:
        print("[WARN] No valid log10_ratio points (check ratio column). Skip plot:", out_png)
        return

    sub["abs_log"] = sub["log10_ratio"].abs()
    top = sub.sort_values("abs_log", ascending=False).head(top_k).copy()
    top = top.sort_values("log10_ratio", ascending=True)

    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)

    ax.barh(y, top["log10_ratio"])
    ax.set_yticks(y)
    ax.set_yticklabels(top["instr"])
    ax.set_xlabel("log10(ratio) where ratio = opt / baseline")
    ax.set_title(f"Top-{top_k} multiplicative changes (log scale)")

    ax.axvline(0, linewidth=1.0)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r["log10_ratio"], y[i], f"  ×{r['ratio']:.3g}", va="center", fontsize=9)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_grouped_share(df: pd.DataFrame, out_png: str):
    """
    Stacked bar: compute / memory / control share (count-based)
    Adjust the sets if you want a stricter ISA-based taxonomy.
    """
    compute = {"VFMADD","VMADD","VADD","VADD12","VMUL","MUL","ADD","ADDI","VREMU","VDIVU"}
    memory  = {"VLW12","VLW","VSW12","VSW","LW","SW"}

    def group_of(instr: str) -> str:
        if instr in compute:
            return "Compute"
        if instr in memory:
            return "Memory"
        return "Control"

    df2 = df.copy()
    df2["group"] = df2["instr"].apply(group_of)

    base_total = df2["baseline_count"].sum()
    opt_total  = df2["opt_count"].sum()

    base_g = df2.groupby("group")["baseline_count"].sum()
    opt_g  = df2.groupby("group")["opt_count"].sum()

    groups = ["Compute","Memory","Control"]
    base_pct = np.array([base_g.get(g,0)/base_total*100 if base_total>0 else 0 for g in groups])
    opt_pct  = np.array([opt_g.get(g,0)/opt_total*100  if opt_total>0 else 0 for g in groups])

    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)

    x = np.array([0,1])
    bottom = np.zeros(2)

    for i, g in enumerate(groups):
        vals = np.array([base_pct[i], opt_pct[i]])
        ax.bar(x, vals, bottom=bottom, label=g)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Optimized"])
    ax.set_ylabel("Share of instruction counts (%)")
    ax.set_title("Instruction mix shift (count-based)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.legend()

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="plots_counts")
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_counts(args.csv)

    plot_top_delta(df, os.path.join(args.outdir, "top_delta_counts.png"), top_k=args.topk)
    plot_top_log_ratio(df, os.path.join(args.outdir, "top_log_ratio.png"), top_k=args.topk)
    plot_grouped_share(df, os.path.join(args.outdir, "mix_share_stacked.png"))

    # PPT-ready highlights
    reduced = df[df["delta"] < 0].sort_values("delta").head(8)[["instr","baseline_count","opt_count","delta","ratio"]]
    increased = df[df["delta"] > 0].sort_values("delta", ascending=False).head(8)[["instr","baseline_count","opt_count","delta","ratio"]]

    print("\n=== Biggest reductions (by Δ) ===")
    print(reduced.to_string(index=False))
    print("\n=== Biggest increases (by Δ) ===")
    print(increased.to_string(index=False))

    newi = df[(df["baseline_count"]==0) & (df["opt_count"]>0)][["instr","opt_count"]]
    if len(newi):
        print("\n=== New instructions in opt (baseline=0) ===")
        print(newi.to_string(index=False))

    print("\n[OK] Saved plots to:", args.outdir)


if __name__ == "__main__":
    main()
