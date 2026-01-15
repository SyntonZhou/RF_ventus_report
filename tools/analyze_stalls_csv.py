#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stalls_csv", help="out_prefix.stalls.csv")
    ap.add_argument("--op", default="", help="filter by prev_op (e.g., LW, VSW12_V). empty = all")
    ap.add_argument("--top", type=int, default=20, help="top N PCs")
    ap.add_argument("--min-gap", type=int, default=0, help="ignore stalls smaller than this gap_ns")
    args = ap.parse_args()

    # pc -> stats
    cnt = defaultdict(int)
    sum_gap = defaultdict(int)
    max_gap = defaultdict(int)

    # keep a few examples for each pc
    examples = defaultdict(list)

    with open(args.stalls_csv, "r", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            prev_op = row["prev_op"]
            if args.op and prev_op != args.op:
                continue
            gap = int(float(row["gap_ns"]))
            if gap < args.min_gap:
                continue

            pc = row["prev_pc"]
            cnt[pc] += 1
            sum_gap[pc] += gap
            if gap > max_gap[pc]:
                max_gap[pc] = gap

            if len(examples[pc]) < 3:
                examples[pc].append({
                    "gap_ns": gap,
                    "sm": row["sm"],
                    "warp": row["warp"],
                    "time_ns": row["time_ns"],
                    "curr_pc": row["curr_pc"],
                    "curr_op": row["curr_op"],
                })

    items = sorted(cnt.keys(), key=lambda pc: sum_gap[pc], reverse=True)[:args.top]

    print(f"=== Top PCs by sum_gap for prev_op={args.op or 'ALL'} (min_gap={args.min_gap}) ===")
    print("rank\tprev_pc\tcount\tsum_gap_ns\tavg_gap_ns\tmax_gap_ns")
    for i, pc in enumerate(items, 1):
        c = cnt[pc]
        s = sum_gap[pc]
        m = max_gap[pc]
        avg = s / c if c else 0
        print(f"{i}\t{pc}\t{c}\t{s}\t{avg:.2f}\t{m}")

        for ex in examples[pc]:
            print(f"    ex: gap={ex['gap_ns']}ns sm={ex['sm']} warp={ex['warp']} t={ex['time_ns']} curr={ex['curr_pc']} {ex['curr_op']}")

if __name__ == "__main__":
    main()
