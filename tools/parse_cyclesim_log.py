#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse Ventus cyclesim trace log.
- Extract SM/warp/PC/opcode/time(ns)
- Compute per-warp inter-instruction gaps and attribute the gap to the previous opcode (your assumption)
- Summarize opcode counts/time shares + top-K largest stalls

Usage:
  python3 parse_cyclesim_log.py log.txt --out out --cycle-ns 10 --topk 80
"""

import argparse
import csv
import gzip
import io
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
import heapq

# ------------------------
# Regex patterns
# ------------------------
# Example:
# SM 0 warp 7 0x800002d8         VFMADD_VV_0xa2b61157 ... @405485ns,1 [trace ...]
INSTR_RE = re.compile(
    r"^SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+"
    r"(?P<pc>0x[0-9a-fA-F]+)\s+"
    r"(?P<instr>\S+).*?"
    r"@(?P<time>\d+)ns"
)

# Example:
# SM 1 warp 1 receive kernel 0 ... block 0 warp 0 @15ns,1 [...]
RECV_RE = re.compile(
    r"^SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+receive\s+kernel.*?"
    r"block\s+(?P<block>\d+)\s+warp\s+(?P<wib>\d+)\s+@(?P<time>\d+)ns"
)

# Extract opcode from token like "VFMADD_VV_0xa2b61157"
OPCODE_RE = re.compile(r"^(?P<op>.+?)_0x[0-9a-fA-F]+$")


def open_maybe_gz(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def opcode_from_instr(instr_token: str) -> str:
    m = OPCODE_RE.match(instr_token)
    if m:
        return m.group("op")
    # Fallback: keep full token
    return instr_token


def classify_opcode(op: str) -> str:
    """
    Coarse categories for reporting. You can extend this mapping if needed.
    """
    # vector load/store / scalar load/store
    if op.startswith("VLW") or op.startswith("VSW") or op in ("LW", "SW", "LD", "SD"):
        return "mem"
    # main compute (rough)
    if op.startswith("VFM") or op.startswith("VF") or op.startswith("VMUL") or op.startswith("VADD") or op.startswith("VSUB"):
        return "compute"
    # control / branch / rpc / simt
    if op in ("AUIPC", "JAL", "JALR", "BEQ", "BNE", "BLT", "BGE", "BGT", "BLE", "RET"):
        return "control"
    if op.startswith("VBEQ") or op.startswith("SETRPC") or op.startswith("JUMP") or op.startswith("REGEXT") or op.startswith("VSETVLI"):
        return "control"
    # address-gen-ish
    if op.startswith("ADDI") or op.startswith("ADD") or op.startswith("SUB"):
        return "addr"
    return "other"


def log2_bin(x: int) -> int:
    """Log2 bin for gap values (ns)."""
    if x <= 0:
        return 0
    return int(math.log2(x + 1))


def approx_quantile_from_log2_hist(hist: dict, q: float) -> int:
    """
    Approximate quantile value from log2 histogram.
    Returns approximate ns value at quantile q.
    """
    total = sum(hist.values())
    if total == 0:
        return 0
    target = int(math.ceil(total * q))
    c = 0
    for b in sorted(hist.keys()):
        c += hist[b]
        if c >= target:
            # approximate representative: 2^b
            return (1 << b)
    return (1 << max(hist.keys()))


@dataclass
class StallEvent:
    gap_ns: int
    sm: int
    warp: int
    time_ns: int
    prev_pc: str
    prev_op: str
    curr_pc: str
    curr_op: str


class OpStats:
    __slots__ = ("count", "sum_gap_ns", "max_gap_ns", "hist_log2")

    def __init__(self):
        self.count = 0
        self.sum_gap_ns = 0
        self.max_gap_ns = 0
        self.hist_log2 = defaultdict(int)

    def add_gap(self, gap_ns: int):
        self.count += 1
        self.sum_gap_ns += gap_ns
        if gap_ns > self.max_gap_ns:
            self.max_gap_ns = gap_ns
        self.hist_log2[log2_bin(gap_ns)] += 1

    def mean(self) -> float:
        return self.sum_gap_ns / self.count if self.count else 0.0

    def p50(self) -> int:
        return approx_quantile_from_log2_hist(self.hist_log2, 0.50)

    def p95(self) -> int:
        return approx_quantile_from_log2_hist(self.hist_log2, 0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="cyclesim log file path (.txt or .gz)")
    ap.add_argument("--out", default="cyclesim", help="output prefix")
    ap.add_argument("--cycle-ns", type=int, default=10, help="ns per cycle (for convenience display)")
    ap.add_argument("--topk", type=int, default=50, help="top-K largest stall events to keep")
    ap.add_argument("--min-gap-ns", type=int, default=0, help="ignore gaps smaller than this when accumulating time (default 0)")
    args = ap.parse_args()

    # Per (sm,warp) last state
    last_time = {}      # (sm,warp) -> time_ns
    last_op = {}        # (sm,warp) -> opcode
    last_pc = {}        # (sm,warp) -> pc

    # Per opcode stats (gap attributed to previous opcode)
    opstats = defaultdict(OpStats)
    cat_sum_gap = defaultdict(int)
    cat_count = defaultdict(int)

    # For coverage info
    sm_instr_cnt = defaultdict(int)
    warp_instr_cnt = defaultdict(int)
    sm_first_time = {}
    sm_last_time = {}

    # Top-K largest stalls (min-heap)
    topk_heap = []  # (gap_ns, idx, StallEvent)
    idx_counter = 0

    # Track global time range of parsed instructions
    global_first = None
    global_last = None

    # Optional: map receive lines (block info) if you want to extend later
    recv_map = {}  # (sm,warp) -> (block, warp_in_block)

    total_instr_lines = 0
    total_recv_lines = 0

    with open_maybe_gz(args.logfile) as f:
        for line in f:
            line = line.rstrip("\n")

            m_recv = RECV_RE.match(line)
            if m_recv:
                total_recv_lines += 1
                sm = int(m_recv.group("sm"))
                warp = int(m_recv.group("warp"))
                block = int(m_recv.group("block"))
                wib = int(m_recv.group("wib"))
                recv_map[(sm, warp)] = (block, wib)
                continue

            m = INSTR_RE.match(line)
            if not m:
                continue

            total_instr_lines += 1
            sm = int(m.group("sm"))
            warp = int(m.group("warp"))
            pc = m.group("pc")
            instr_token = m.group("instr")
            t = int(m.group("time"))
            op = opcode_from_instr(instr_token)

            # Update time range
            if global_first is None or t < global_first:
                global_first = t
            if global_last is None or t > global_last:
                global_last = t

            # Per SM time range
            if sm not in sm_first_time:
                sm_first_time[sm] = t
            sm_last_time[sm] = t

            sm_instr_cnt[sm] += 1
            warp_instr_cnt[(sm, warp)] += 1

            key = (sm, warp)
            if key in last_time:
                gap = t - last_time[key]
                # Attribute gap to previous opcode (same as your existing assumption)
                prev_op = last_op.get(key, "UNKNOWN")
                prev_pc = last_pc.get(key, "0x0")
                curr_op = op
                curr_pc = pc

                if gap >= args.min_gap_ns:
                    st = opstats[prev_op]
                    st.add_gap(gap)
                    cat = classify_opcode(prev_op)
                    cat_sum_gap[cat] += gap
                    cat_count[cat] += 1

                    # Maintain top-K largest stalls
                    ev = StallEvent(
                        gap_ns=gap, sm=sm, warp=warp, time_ns=t,
                        prev_pc=prev_pc, prev_op=prev_op, curr_pc=curr_pc, curr_op=curr_op
                    )
                    idx_counter += 1
                    if len(topk_heap) < args.topk:
                        heapq.heappush(topk_heap, (gap, idx_counter, ev))
                    else:
                        if gap > topk_heap[0][0]:
                            heapq.heapreplace(topk_heap, (gap, idx_counter, ev))

            # Update last state
            last_time[key] = t
            last_op[key] = op
            last_pc[key] = pc

    # Summaries
    elapsed_ns = (global_last - global_first) if (global_first is not None and global_last is not None) else 0
    total_gap_ns = sum(s.sum_gap_ns for s in opstats.values())
    cycle_ns = args.cycle_ns if args.cycle_ns > 0 else 1

    # Write opcode CSV
    out_csv = args.out + ".opcodes.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow([
            "opcode", "category",
            "count(gap-attrib)", "sum_gap_ns", "time_share(%)",
            "mean_gap_ns", "p50_gap_ns~", "p95_gap_ns~", "max_gap_ns",
            "mean_cycles", "max_cycles"
        ])
        # sort by sum_gap_ns desc
        for op, st in sorted(opstats.items(), key=lambda kv: kv[1].sum_gap_ns, reverse=True):
            share = (100.0 * st.sum_gap_ns / total_gap_ns) if total_gap_ns else 0.0
            cat = classify_opcode(op)
            w.writerow([
                op, cat,
                st.count, st.sum_gap_ns, f"{share:.4f}",
                f"{st.mean():.3f}", st.p50(), st.p95(), st.max_gap_ns,
                f"{st.mean()/cycle_ns:.3f}", f"{st.max_gap_ns/cycle_ns:.3f}"
            ])

    # Write top stalls CSV
    out_stalls = args.out + ".stalls.csv"
    top_events = sorted(topk_heap, key=lambda x: x[0], reverse=True)
    with open(out_stalls, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["rank", "gap_ns", "gap_cycles", "sm", "warp", "time_ns", "prev_pc", "prev_op", "curr_pc", "curr_op", "block", "warp_in_block"])
        for i, (gap, _, ev) in enumerate(top_events, start=1):
            block, wib = recv_map.get((ev.sm, ev.warp), (-1, -1))
            w.writerow([i, ev.gap_ns, f"{ev.gap_ns/cycle_ns:.3f}", ev.sm, ev.warp, ev.time_ns,
                        ev.prev_pc, ev.prev_op, ev.curr_pc, ev.curr_op, block, wib])

    # Write summary JSON
    out_json = args.out + ".summary.json"
    summary = {
        "logfile": os.path.abspath(args.logfile),
        "parsed_instr_lines": total_instr_lines,
        "parsed_receive_lines": total_recv_lines,
        "global_first_time_ns": global_first,
        "global_last_time_ns": global_last,
        "elapsed_ns_between_first_last_instr": elapsed_ns,
        "elapsed_cycles_between_first_last_instr": (elapsed_ns / cycle_ns) if cycle_ns else None,
        "total_gap_ns_attributed": total_gap_ns,
        "unique_opcodes": len(opstats),
        "sm_instr_count": dict(sm_instr_cnt),
        "sm_time_range_ns": {str(sm): [sm_first_time[sm], sm_last_time[sm]] for sm in sm_first_time},
        "num_warps_seen": len(warp_instr_cnt),
        "top_categories_by_time": sorted(
            [{"category": c, "sum_gap_ns": s, "time_share(%)": (100.0*s/total_gap_ns if total_gap_ns else 0.0)} for c, s in cat_sum_gap.items()],
            key=lambda x: x["sum_gap_ns"], reverse=True
        ),
    }
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    # Print a compact console report
    print("=== cyclesim trace parsing done ===")
    print(f"logfile: {args.logfile}")
    print(f"parsed instruction lines: {total_instr_lines}")
    print(f"parsed receive lines    : {total_recv_lines}")
    print(f"time window (instr): {global_first} ns -> {global_last} ns  (Δ={elapsed_ns} ns ≈ {elapsed_ns/cycle_ns:.1f} cycles @ {cycle_ns}ns/cycle)")
    print(f"attributed gap total: {total_gap_ns} ns")
    print(f"unique opcodes: {len(opstats)}")
    print(f"output: {out_csv}")
    print(f"output: {out_stalls}")
    print(f"output: {out_json}")
    print("\nTop-10 opcodes by attributed time (sum_gap_ns):")
    for op, st in list(sorted(opstats.items(), key=lambda kv: kv[1].sum_gap_ns, reverse=True))[:10]:
        share = (100.0 * st.sum_gap_ns / total_gap_ns) if total_gap_ns else 0.0
        print(f"  {op:20s}  cat={classify_opcode(op):7s}  sum={st.sum_gap_ns:10d} ns  share={share:7.3f}%  mean={st.mean():8.2f} ns  p95~={st.p95():6d} ns  max={st.max_gap_ns:8d} ns")

    print("\nTop categories by attributed time:")
    for item in summary["top_categories_by_time"]:
        print(f"  {item['category']:7s}  sum={item['sum_gap_ns']:10d} ns  share={item['time_share(%)']:.3f}%")

if __name__ == "__main__":
    main()
