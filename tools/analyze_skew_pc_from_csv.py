#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, argparse
from collections import defaultdict

def read_warp_pc_counts(path):
    # warp_pc_counts.csv: warp,pc,count
    d = defaultdict(lambda: defaultdict(int))
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            w = int(row["warp"])
            pc = row["pc"].strip()
            cnt = int(row["count"])
            d[pc][w] += cnt
    return d

def read_warp_totals_from_summary(path):
    # summary.json 里有 warp_totals（若你的 summary.json 不含，可改为手动输入）
    import json
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    wt = js.get("warp_totals", {})
    # wt 可能是 {"0":123,...} 字符串 key
    return {int(k): int(v) for k, v in wt.items()}

def read_pc_inst_hint(path):
    # global_pc_counts.csv: pc,count,inst_type_hint
    hint = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            hint[row["pc"].strip()] = row.get("inst_type_hint", "").strip()
    return hint

def read_pc_skew(path):
    # pc_warp_skew.csv: pc,total_count,max_warp_share,argmax_warp
    skew = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pc = row["pc"].strip()
            skew[pc] = (int(row["total_count"]), float(row["max_warp_share"]), int(row["argmax_warp"]))
    return skew

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="out_512 目录")
    ap.add_argument("--pc", required=True, help="要分析的 PC，例如 0x800001b4")
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--find_others", action="store_true",
                    help="额外列出其它略偏的 PC（max_warp_share>1/warps+eps 且 count>阈值）")
    ap.add_argument("--eps", type=float, default=0.002, help="偏差阈值，例如 0.002 表示 >0.127 才算略偏")
    ap.add_argument("--min_count", type=int, default=1000, help="只列出 total_count>=该值的略偏 PC")
    args = ap.parse_args()

    out_dir = args.out
    pc = args.pc.strip()
    warps = args.warps

    warp_pc_counts = read_warp_pc_counts(os.path.join(out_dir, "warp_pc_counts.csv"))
    pc_hint = read_pc_inst_hint(os.path.join(out_dir, "global_pc_counts.csv"))
    pc_skew = read_pc_skew(os.path.join(out_dir, "pc_warp_skew.csv"))

    summary_path = os.path.join(out_dir, "summary.json")
    warp_totals = read_warp_totals_from_summary(summary_path) if os.path.exists(summary_path) else {}

    counts = [warp_pc_counts.get(pc, {}).get(w, 0) for w in range(warps)]
    tot = sum(counts)
    mean = tot / warps if warps else 0.0
    mx = max(counts) if counts else 0
    mn = min(counts) if counts else 0
    inst = pc_hint.get(pc, "")

    print(f"[PC] {pc}  inst_hint={inst!r}")
    if pc in pc_skew:
        tcnt, mshare, am = pc_skew[pc]
        print(f"[SkewCSV] total_count={tcnt:,}  max_warp_share={mshare:.6f}  argmax_warp={am}")
    print(f"[Counts] total={tot:,}  mean_per_warp={mean:.2f}  max={mx}  min={mn}  (max-min)={mx-mn}")

    print("\nPer-warp counts:")
    for w, c in enumerate(counts):
        print(f"  warp {w}: {c}")

    # 解释 warp 总数差异是否“主要来自该 PC”
    if warp_totals:
        wt = [warp_totals.get(w, 0) for w in range(warps)]
        wt_mean = sum(wt) / warps
        # 该PC对每个warp的“偏差”
        pc_dev = [counts[w] - mean for w in range(warps)]
        tot_dev = [wt[w] - wt_mean for w in range(warps)]
        print("\nWarp total deltas (warp_total - mean_total):")
        for w in range(warps):
            print(f"  warp {w}: total_delta={tot_dev[w]: .2f}   pc_delta={pc_dev[w]: .2f}")

        # 粗略贡献度：用L1范数衡量该PC解释了多少总体不均衡
        l1_pc = sum(abs(x) for x in pc_dev)
        l1_tot = sum(abs(x) for x in tot_dev) + 1e-9
        print(f"\n[Attribution] L1(pc_delta)/L1(total_delta) ≈ {l1_pc/l1_tot:.3f}")
        print("  （若接近 1，说明几乎都由该PC解释；若明显小于 1，说明还有其它PC也在贡献。）")
    else:
        print("\n[Note] summary.json 不存在或无 warp_totals，无法做归因比对。")

    # 额外找其它略偏PC
    if args.find_others:
        baseline = 1.0 / warps
        threshold = baseline + args.eps
        print(f"\nOther skew PCs (max_warp_share>{threshold:.6f}, total_count>={args.min_count}):")
        items = []
        for p, (tcnt, mshare, am) in pc_skew.items():
            if tcnt >= args.min_count and mshare > threshold:
                items.append((mshare, tcnt, p, am))
        items.sort(reverse=True)
        for mshare, tcnt, p, am in items[:50]:
            print(f"  pc={p} total={tcnt:,} max_share={mshare:.6f} argmax_warp={am} inst_hint={pc_hint.get(p,'')!r}")

if __name__ == "__main__":
    main()
