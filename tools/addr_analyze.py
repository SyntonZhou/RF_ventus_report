#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
import argparse
from collections import defaultdict, Counter

INSTR_RE = re.compile(
    r"SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+0x(?P<pc>[0-9a-fA-F]+)\s+"
    r"(?P<op>[A-Z0-9]+(?:_[A-Z0-9]+)?)_0x[0-9a-fA-F]+.*?@(?P<ns>\d+)ns"
)
MEMADDR_RE = re.compile(r"MEMADDR\s+(?P<addrs>[0-9a-fA-F ]+?)\s+@", re.IGNORECASE)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out-prefix", default="mem_out")
    ap.add_argument("--sm", type=int, default=None)
    ap.add_argument("--warp", type=int, default=None)
    ap.add_argument("--max-delta-samples", type=int, default=2000000,
                    help="每个(pc,warp)记录 base_delta 的样本上限，防止极端膨胀")
    return ap.parse_args()

def main():
    args = parse_args()

    # region: 用高 8bit 做粗聚类（你也可以改成 >> 20 / >> 24）
    region_cnt = Counter()

    # per (pc,op)
    stats = defaultdict(lambda: {
        "count": 0,
        "min": None,
        "max": None,
        "broadcast": 0,
        "contig4": 0,
    })

    # base stride: per (sm,warp,pc,op) 记录 base 地址的跨次增量
    last_base = {}
    base_delta = defaultdict(Counter)
    base_delta_samples = defaultdict(int)

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = INSTR_RE.search(line)
            if not m:
                continue

            sm = int(m.group("sm"))
            warp = int(m.group("warp"))
            if args.sm is not None and sm != args.sm:
                continue
            if args.warp is not None and warp != args.warp:
                continue

            pc = int(m.group("pc"), 16)
            op = m.group("op")

            mm = MEMADDR_RE.search(line)
            if not mm:
                continue

            # 解析地址列表
            toks = mm.group("addrs").split()
            addrs = [int(x, 16) for x in toks if x.strip()]
            if not addrs:
                continue

            st = stats[(pc, op)]
            st["count"] += 1
            mn = min(addrs)
            mx = max(addrs)
            st["min"] = mn if st["min"] is None else min(st["min"], mn)
            st["max"] = mx if st["max"] is None else max(st["max"], mx)

            # region 粗聚类
            for a in addrs:
                region_cnt[(a >> 24) & 0xFF] += 1

            # 广播：全部相等
            if all(a == addrs[0] for a in addrs):
                st["broadcast"] += 1

            # 连续：按 lane 递增 4（允许出现“重复半向量”的情况：你日志里经常前后 16 个重复）
            # 这里用“去重后是否连续4”做判据
            uniq = sorted(set(addrs))
            if len(uniq) >= 2:
                ok = True
                for i in range(1, len(uniq)):
                    if uniq[i] - uniq[i-1] != 4:
                        ok = False
                        break
                if ok:
                    st["contig4"] += 1

            # base delta：用最小地址作为 base（也可以换成 addrs[0]）
            key = (sm, warp, pc, op)
            if key in last_base:
                d = mn - last_base[key]
                if base_delta_samples[key] < args.max_delta_samples:
                    base_delta[key][d] += 1
                    base_delta_samples[key] += 1
            last_base[key] = mn

    # 输出 region
    region_txt = f"{args.out_prefix}_regions.txt"
    with open(region_txt, "w", encoding="utf-8") as w:
        w.write("region_hi8,count\n")
        for r, c in region_cnt.most_common():
            w.write(f"0x{r:02x},{c}\n")
    print(f"[OUT] {region_txt}")

    # 输出 per-pc 汇总
    pc_csv = f"{args.out_prefix}_by_pc.csv"
    with open(pc_csv, "w", newline="", encoding="utf-8") as w:
        cw = csv.writer(w)
        cw.writerow([
            "pc", "opcode", "count",
            "min_addr", "max_addr",
            "broadcast_ratio", "contig4_ratio"
        ])
        for (pc, op), st in sorted(stats.items(), key=lambda x: (-x[1]["count"], x[0][0])):
            cnt = st["count"]
            cw.writerow([
                f"0x{pc:08x}", op, cnt,
                f"0x{st['min']:08x}", f"0x{st['max']:08x}",
                st["broadcast"] / cnt,
                st["contig4"] / cnt
            ])
    print(f"[OUT] {pc_csv}")

    # 输出 base stride（挑每个 (sm,warp,pc,op) 的 top deltas）
    stride_csv = f"{args.out_prefix}_base_stride.csv"
    with open(stride_csv, "w", newline="", encoding="utf-8") as w:
        cw = csv.writer(w)
        cw.writerow(["sm", "warp", "pc", "opcode", "delta_bytes", "count"])
        for (sm, warp, pc, op), ctr in sorted(base_delta.items(), key=lambda x: (-sum(x[1].values()), x[0])):
            for d, c in ctr.most_common(5):
                cw.writerow([sm, warp, f"0x{pc:08x}", op, d, c])
    print(f"[OUT] {stride_csv}")

if __name__ == "__main__":
    main()
 # python addr_analyze.py inner_loop.log --out-prefix inner_mem --sm 1 --warp 0
