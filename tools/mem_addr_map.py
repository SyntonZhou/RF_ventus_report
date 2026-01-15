#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import csv
from collections import defaultdict, Counter

# 例：
# SM 1 warp 0 0x800002f8  VLW12_V_0x0007287b mask=... MEMADDR 70002680 70002684 ... @82315ns,1
RE_MEM_LINE = re.compile(
    r"\bSM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+(?P<pc>0x[0-9a-fA-F]+)\s+(?P<op>\S+).*?\bMEMADDR\s+(?P<addrs>.+?)\s+@"
)

# op token: MNEMONIC_0xENC
RE_OP_WITH_ENC = re.compile(r"^(.*)_(0x[0-9a-fA-F]+)$")

# MEMADDR 后面是无 0x 前缀的 hex（也可能混入其它 token，严格过滤）
RE_HEX_TOKEN = re.compile(r"^[0-9a-fA-F]{6,16}$")  # 兼容 24-bit/32-bit/64-bit 表达


def split_op_token(op_token: str):
    m = RE_OP_WITH_ENC.match(op_token)
    if not m:
        return op_token, None
    return m.group(1), m.group(2)


def is_store_mn(mn: str) -> bool:
    u = mn.upper()
    # 覆盖：SW, VSW_V, VSW12_V, etc.
    return ("SW" in u) and (not ("SLL" in u))  # 防止 VSLL 误判


def is_load_mn(mn: str) -> bool:
    u = mn.upper()
    return ("LW" in u) and (not ("SLL" in u))


def hi8(addr: int) -> int:
    return (addr >> 24) & 0xFF


def hi12(addr: int) -> int:
    return (addr >> 20) & 0xFFF


def contig4_score(addrs):
    """
    返回是否满足“按 4 字节连续”的常见向量访存形态。
    支持：
    - 全 32 lane 连续
    - 前 16 lane 连续且后 16 lane 连续（且模式相同或各自连续）
    """
    if len(addrs) < 4:
        return False

    def is_contig(seq):
        base = seq[0]
        for i, a in enumerate(seq):
            if a != base + 4 * i:
                return False
        return True

    n = len(addrs)
    if is_contig(addrs):
        return True
    if n == 32:
        first = addrs[:16]
        second = addrs[16:]
        if is_contig(first) and is_contig(second):
            return True
    return False


def parse_addrs(addr_str: str):
    toks = addr_str.strip().split()
    out = []
    for t in toks:
        t = t.strip()
        if RE_HEX_TOKEN.match(t):
            try:
                out.append(int(t, 16))
            except ValueError:
                pass
    return out


class PcAgg:
    __slots__ = (
        "pc", "mnemonic", "encs",
        "inst_count", "addr_count",
        "min_addr", "max_addr",
        "contig4_hits",
        "avg_unique_sum", "avg_len_sum",
        "region_hi8_counter", "region_hi12_counter",
        "stride_counter",
    )

    def __init__(self, pc, mnemonic):
        self.pc = pc
        self.mnemonic = mnemonic
        self.encs = set()

        self.inst_count = 0
        self.addr_count = 0
        self.min_addr = None
        self.max_addr = None

        self.contig4_hits = 0
        self.avg_unique_sum = 0
        self.avg_len_sum = 0

        self.region_hi8_counter = Counter()
        self.region_hi12_counter = Counter()
        self.stride_counter = Counter()

    def update(self, addrs, enc, stride_delta=None):
        self.inst_count += 1
        self.addr_count += len(addrs)

        if enc:
            self.encs.add(enc)

        if addrs:
            mn = min(addrs)
            mx = max(addrs)
            self.min_addr = mn if self.min_addr is None else min(self.min_addr, mn)
            self.max_addr = mx if self.max_addr is None else max(self.max_addr, mx)

            self.region_hi8_counter.update([hi8(a) for a in addrs])
            self.region_hi12_counter.update([hi12(a) for a in addrs])

            if contig4_score(addrs):
                self.contig4_hits += 1

            uniq = len(set(addrs))
            self.avg_unique_sum += uniq
            self.avg_len_sum += len(addrs)

        if stride_delta is not None:
            self.stride_counter[stride_delta] += 1


def write_txt_region_summary(path, region_stats):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Memory Region Summary (by hi8)\n")
        f.write("Format: hi8  load_addrs  store_addrs  load_bytes  store_bytes\n\n")
        for r in sorted(region_stats.keys()):
            la, sa, lb, sb = region_stats[r]
            f.write(f"0x{r:02x}  {la:>12d}  {sa:>12d}  {lb:>12d}  {sb:>12d}\n")


def write_txt_pc_summary(path, pc_aggs):
    pcs = sorted(pc_aggs.keys(), key=lambda x: int(x, 16))
    with open(path, "w", encoding="utf-8") as f:
        f.write("Memory Access Summary by PC\n")
        f.write("Columns:\n")
        f.write("PC  mnemonic  inst_count  addr_count  bytes  min_addr  max_addr  contig4_ratio  unique_ratio  top_stride\n\n")
        for pc in pcs:
            agg = pc_aggs[pc]
            bytes_rw = agg.addr_count * 4
            contig_ratio = (agg.contig4_hits / agg.inst_count) if agg.inst_count else 0.0
            # unique_ratio = avg_unique/avg_len（越接近1越“每lane不同地址”，越小越“广播/复用”）
            unique_ratio = (agg.avg_unique_sum / agg.avg_len_sum) if agg.avg_len_sum else 0.0

            top_stride = "-"
            if agg.stride_counter:
                s, c = agg.stride_counter.most_common(1)[0]
                top_stride = f"{s:+d}B({c})"

            min_addr = f"0x{agg.min_addr:08x}" if agg.min_addr is not None else "-"
            max_addr = f"0x{agg.max_addr:08x}" if agg.max_addr is not None else "-"

            f.write(
                f"{pc}  {agg.mnemonic:>10s}  {agg.inst_count:>10d}  {agg.addr_count:>10d}  {bytes_rw:>10d}  "
                f"{min_addr:>12s}  {max_addr:>12s}  {contig_ratio:>10.4f}  {unique_ratio:>10.4f}  {top_stride}\n"
            )


def write_txt_store_candidates(path, pc_aggs):
    # 只筛 store，按写字节排序
    stores = []
    for pc, agg in pc_aggs.items():
        if is_store_mn(agg.mnemonic):
            stores.append((agg.addr_count * 4, pc, agg))
    stores.sort(reverse=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("Store Candidates (likely C output is near the top)\n")
        f.write("Sorted by total store bytes.\n")
        f.write("Columns: rank  pc  mnemonic  store_bytes  inst_count  addr_count  min_addr  max_addr  contig4_ratio  unique_ratio  top_stride  top_hi8\n\n")

        for i, (b, pc, agg) in enumerate(stores[:200], 1):
            contig_ratio = (agg.contig4_hits / agg.inst_count) if agg.inst_count else 0.0
            unique_ratio = (agg.avg_unique_sum / agg.avg_len_sum) if agg.avg_len_sum else 0.0
            min_addr = f"0x{agg.min_addr:08x}" if agg.min_addr is not None else "-"
            max_addr = f"0x{agg.max_addr:08x}" if agg.max_addr is not None else "-"
            top_stride = "-"
            if agg.stride_counter:
                s, c = agg.stride_counter.most_common(1)[0]
                top_stride = f"{s:+d}B({c})"
            top_hi8 = "-"
            if agg.region_hi8_counter:
                r, rc = agg.region_hi8_counter.most_common(1)[0]
                top_hi8 = f"0x{r:02x}({rc})"

            f.write(
                f"{i:>4d}  {pc}  {agg.mnemonic:>10s}  {b:>12d}  {agg.inst_count:>10d}  {agg.addr_count:>10d}  "
                f"{min_addr:>12s}  {max_addr:>12s}  {contig_ratio:>10.4f}  {unique_ratio:>10.4f}  {top_stride:>14s}  {top_hi8}\n"
            )

        if not stores:
            f.write("No store instructions with MEMADDR were found in this log.\n")


def write_txt_hot_addresses(path, hot_load, hot_store, topn=200):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Hot Addresses (Top by access count)\n\n")
        f.write("[LOAD]\n")
        for addr, c in hot_load.most_common(topn):
            f.write(f"0x{addr:08x}  {c}\n")
        f.write("\n[STORE]\n")
        for addr, c in hot_store.most_common(topn):
            f.write(f"0x{addr:08x}  {c}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="raw log path")
    ap.add_argument("--out-prefix", default="mem", help="output file prefix (default: mem)")
    ap.add_argument("--progress-every", type=int, default=5_000_000, help="print progress every N lines, 0=disable")
    ap.add_argument("--track-hot-addr", action="store_true",
                    help="track hot individual addresses (may consume memory on extremely large logs)")
    ap.add_argument("--hot-limit", type=int, default=5_000_000,
                    help="cap unique hot addresses tracked (rough safety), default 5,000,000")
    args = ap.parse_args()

    # region_stats[hi8] = (load_addr_cnt, store_addr_cnt, load_bytes, store_bytes)
    region_stats = defaultdict(lambda: [0, 0, 0, 0])

    pc_aggs = {}  # pc(str) -> PcAgg

    # stride：按 (sm,warp,pc) 维护 last_base
    last_base = {}  # key=(sm,warp,pc) -> base_addr(int)

    hot_load = Counter()
    hot_store = Counter()
    unique_hot_seen = set()  # rough cap

    total_mem_insts = 0

    with open(args.logfile, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            m = RE_MEM_LINE.search(line)
            if not m:
                if args.progress_every and args.progress_every > 0 and (ln % args.progress_every == 0):
                    print(f"[progress] lines={ln:,}  mem_insts={total_mem_insts:,}  unique_pc={len(pc_aggs):,}")
                continue

            sm = int(m.group("sm"))
            warp = int(m.group("warp"))
            pc = m.group("pc").lower()
            op_token = m.group("op")

            mnemonic, enc = split_op_token(op_token)
            addrs = parse_addrs(m.group("addrs"))
            if not addrs:
                continue

            total_mem_insts += 1

            # stride 统计：用每次访问的“第一个 lane 地址”作为 base
            base = addrs[0]
            key = (sm, warp, pc)
            delta = None
            if key in last_base:
                delta = base - last_base[key]
            last_base[key] = base

            # 聚合到 pc
            if pc not in pc_aggs:
                pc_aggs[pc] = PcAgg(pc, mnemonic)
            pc_aggs[pc].update(addrs, enc, stride_delta=delta)

            # region 统计（按 hi8）
            bytes_rw = len(addrs) * 4
            if is_store_mn(mnemonic):
                for a in addrs:
                    r = hi8(a)
                    region_stats[r][1] += 1
                    region_stats[r][3] += 4
                if args.track_hot_addr:
                    for a in addrs:
                        if len(unique_hot_seen) < args.hot_limit or a in unique_hot_seen:
                            hot_store[a] += 1
                            unique_hot_seen.add(a)
            else:
                for a in addrs:
                    r = hi8(a)
                    region_stats[r][0] += 1
                    region_stats[r][2] += 4
                if args.track_hot_addr:
                    for a in addrs:
                        if len(unique_hot_seen) < args.hot_limit or a in unique_hot_seen:
                            hot_load[a] += 1
                            unique_hot_seen.add(a)

            if args.progress_every and args.progress_every > 0 and (ln % args.progress_every == 0):
                print(f"[progress] lines={ln:,}  mem_insts={total_mem_insts:,}  unique_pc={len(pc_aggs):,}")

    # 输出 TXT
    region_txt = f"{args.out_prefix}_region_summary.txt"
    pc_txt = f"{args.out_prefix}_by_pc.txt"
    store_txt = f"{args.out_prefix}_store_candidates.txt"
    hot_txt = f"{args.out_prefix}_hot_addresses.txt"

    write_txt_region_summary(region_txt, region_stats)
    write_txt_pc_summary(pc_txt, pc_aggs)
    write_txt_store_candidates(store_txt, pc_aggs)
    if args.track_hot_addr:
        write_txt_hot_addresses(hot_txt, hot_load, hot_store, topn=200)

    # 输出 CSV（可选后处理）
    csv_path = f"{args.out_prefix}_by_pc.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow([
            "pc", "mnemonic", "inst_count", "addr_count", "bytes",
            "min_addr", "max_addr",
            "contig4_ratio", "unique_ratio",
            "top_stride_bytes", "top_stride_count",
            "top_hi8", "top_hi8_count",
            "enc_set"
        ])
        for pc in sorted(pc_aggs.keys(), key=lambda x: int(x, 16)):
            agg = pc_aggs[pc]
            bytes_rw = agg.addr_count * 4
            contig_ratio = (agg.contig4_hits / agg.inst_count) if agg.inst_count else 0.0
            unique_ratio = (agg.avg_unique_sum / agg.avg_len_sum) if agg.avg_len_sum else 0.0

            top_stride_b, top_stride_c = "", ""
            if agg.stride_counter:
                s, c = agg.stride_counter.most_common(1)[0]
                top_stride_b, top_stride_c = str(s), str(c)

            top_hi8_v, top_hi8_c = "", ""
            if agg.region_hi8_counter:
                r, rc = agg.region_hi8_counter.most_common(1)[0]
                top_hi8_v, top_hi8_c = f"0x{r:02x}", str(rc)

            min_addr = f"0x{agg.min_addr:08x}" if agg.min_addr is not None else ""
            max_addr = f"0x{agg.max_addr:08x}" if agg.max_addr is not None else ""

            w.writerow([
                pc, agg.mnemonic, agg.inst_count, agg.addr_count, bytes_rw,
                min_addr, max_addr,
                f"{contig_ratio:.6f}", f"{unique_ratio:.6f}",
                top_stride_b, top_stride_c,
                top_hi8_v, top_hi8_c,
                "|".join(sorted(agg.encs))
            ])

    print("Done.")
    print("TXT outputs:")
    print(" ", region_txt)
    print(" ", pc_txt)
    print(" ", store_txt)
    if args.track_hot_addr:
        print(" ", hot_txt)
    print("CSV output:")
    print(" ", csv_path)


if __name__ == "__main__":
    main()
