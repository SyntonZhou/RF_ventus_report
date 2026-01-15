#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare two Ventus cyclesim logs (1wg vs multiwg) and pinpoint where extra time comes from.

Outputs:
- <out>.report.txt         : human-readable diff report
- <out>.opcode_diff.csv    : opcode-level diff (count/time/share)
- <out>.pc_diff.csv        : hotspot PC diff (by opcode+pc)
"""

import argparse
import csv
import json
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# -------- parsing regex --------
# example instr line:
# SM 0 warp 7 0x800002d8         VFMADD_VV_0xa2b61157 ... @405485ns,1 [trace regfile.cpp:109]
INSTR_RE = re.compile(
    r"^SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+(?P<pc>0x[0-9a-fA-F]+)\s+(?P<op>[A-Za-z0-9_.]+).*?@(?P<tns>\d+)ns",
    re.IGNORECASE
)

# example receive line:
# SM 1 warp 1 receive kernel 0 sgemm_one_wg block 0 warp 0 @15ns,1 [trace BASE.cpp:244]
RECV_RE = re.compile(
    r"^SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+receive\s+kernel\s+(?P<kid>\d+)\s+(?P<kname>\S+)\s+block\s+(?P<block>\d+)\s+warp\s+(?P<bwarp>\d+)\s+@(?P<tns>\d+)ns",
    re.IGNORECASE
)

# kernel init line:
# kernel 0 sgemm_one_wg initialized, size={2,2,1} @0ns,0 ...
KINIT_RE = re.compile(
    r"^kernel\s+(?P<kid>\d+)\s+(?P<kname>\S+)\s+initialized,\s+size=\{(?P<sx>\d+),(?P<sy>\d+),(?P<sz>\d+)\}",
    re.IGNORECASE
)

def norm_op(op: str) -> str:
    """Normalize opcode token, strip suffixes like _0x..., keep main mnemonic."""
    op = op.strip()
    # common pattern: VFMADD_VV_0xa2b61157 => VFMADD_VV
    op = re.sub(r"_0x[0-9a-fA-F]+$", "", op)
    # barrier sometimes appears as 'barrier' or 'BARRIER_0x...'
    op = op.replace("barrier", "BARRIER").replace("Barrier", "BARRIER").replace("BARRIER", "BARRIER")
    return op

def category(op: str) -> str:
    opu = op.upper()
    if opu in ("LW", "SW") or opu.startswith("VLW") or opu.startswith("VSW"):
        return "mem"
    if opu in ("JAL", "J", "JR", "RET") or "BEQ" in opu or "BNE" in opu or opu.startswith("VBEQ") or opu.startswith("JUMP") or opu.startswith("SETRPC"):
        return "control"
    if opu in ("ADDI","ADD","SUB","MUL","SLL","SRL","SRA","AND","OR","XOR","AUIPC","LUI"):
        return "addr"
    if opu.startswith("VF") or opu.startswith("VFM") or opu.startswith("VADD") or opu.startswith("VMUL") or opu.startswith("VMA") or opu.startswith("F"):
        return "compute"
    if opu.startswith("REGEXT") or opu.startswith("JOIN") or opu.startswith("VSETVLI") or opu.startswith("VMV") or opu.startswith("VAND") or opu.startswith("VMSLT") or opu.startswith("VOR") or opu.startswith("VXOR"):
        return "other"
    if opu.startswith("BARRIER"):
        return "sync"
    return "other"

@dataclass
class Instr:
    sm: int
    warp: int
    pc: str
    op: str
    tns: int

@dataclass
class RunProfile:
    logfile: str
    instr_count: int
    recv_count: int
    kernel_grid: Optional[Tuple[int,int,int]]
    blocks: int
    warps_launched: int
    t0: int
    t1: int
    window_ns: int
    # aggregated stats
    op_count: Counter
    op_time: Dict[str,int]               # attributed gap time per opcode
    cat_time: Dict[str,int]
    pcop_time: Dict[Tuple[str,str],int]  # (pc,op) -> sum_gap_ns
    pcop_count: Dict[Tuple[str,str],int] # (pc,op) -> count
    op_gap_max: Dict[str,int]            # opcode -> max gap
    op_gap_p95: Dict[str,int]            # opcode -> approx p95 from histogram
    # gap histogram per opcode (log2 buckets) for p95 estimation
    _op_hist: Dict[str, Counter]

def bucket_ns(x: int) -> int:
    """Log2-ish bucket to approximate percentiles without storing all gaps."""
    if x <= 0:
        return 0
    b = 1
    while b < x and b < (1<<26):
        b <<= 1
    return b

def hist_p95(hist: Counter) -> int:
    if not hist:
        return 0
    total = sum(hist.values())
    target = int(total * 0.95 + 0.999999)
    acc = 0
    for b in sorted(hist.keys()):
        acc += hist[b]
        if acc >= target:
            return b
    return max(hist.keys())

def parse_log(path: str, cycle_ns: int) -> RunProfile:
    last: Dict[Tuple[int,int], Instr] = {}
    op_count = Counter()
    op_time = defaultdict(int)
    cat_time = defaultdict(int)
    pcop_time = defaultdict(int)
    pcop_count = defaultdict(int)
    op_gap_max = defaultdict(int)
    op_hist = defaultdict(Counter)

    recv_count = 0
    blocks_set = set()
    warps_launched = 0
    kernel_grid = None
    t0 = None
    t1 = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = KINIT_RE.match(line)
            if m and kernel_grid is None:
                kernel_grid = (int(m.group("sx")), int(m.group("sy")), int(m.group("sz")))
                continue

            m = RECV_RE.match(line)
            if m:
                recv_count += 1
                blocks_set.add(int(m.group("block")))
                warps_launched += 1
                continue

            m = INSTR_RE.match(line)
            if not m:
                continue

            sm = int(m.group("sm"))
            warp = int(m.group("warp"))
            pc = m.group("pc").lower()
            op = norm_op(m.group("op"))
            tns = int(m.group("tns"))

            if t0 is None:
                t0 = tns
            t1 = tns

            op_count[op] += 1

            key = (sm, warp)
            cur = Instr(sm, warp, pc, op, tns)

            if key in last:
                prev = last[key]
                gap = cur.tns - prev.tns
                # attribute gap to prev instruction (same as your current assumption)
                pop = prev.op
                op_time[pop] += gap
                cat_time[category(pop)] += gap
                pcop_time[(prev.pc, pop)] += gap
                pcop_count[(prev.pc, pop)] += 1
                if gap > op_gap_max[pop]:
                    op_gap_max[pop] = gap
                op_hist[pop][bucket_ns(gap)] += 1

            last[key] = cur

    if t0 is None or t1 is None:
        t0, t1 = 0, 0
    window = t1 - t0

    # p95 approx
    op_gap_p95 = {}
    for op, hist in op_hist.items():
        op_gap_p95[op] = hist_p95(hist)

    prof = RunProfile(
        logfile=path,
        instr_count=sum(op_count.values()),
        recv_count=recv_count,
        kernel_grid=kernel_grid,
        blocks=len(blocks_set),
        warps_launched=warps_launched,
        t0=t0,
        t1=t1,
        window_ns=window,
        op_count=op_count,
        op_time=dict(op_time),
        cat_time=dict(cat_time),
        pcop_time=dict(pcop_time),
        pcop_count=dict(pcop_count),
        op_gap_max=dict(op_gap_max),
        op_gap_p95=op_gap_p95,
        _op_hist=op_hist
    )
    return prof

def write_opcode_diff(out_csv: str, a: RunProfile, b: RunProfile):
    ops = set(a.op_count.keys()) | set(b.op_count.keys()) | set(a.op_time.keys()) | set(b.op_time.keys())
    total_a = sum(a.op_time.values()) if a.op_time else 1
    total_b = sum(b.op_time.values()) if b.op_time else 1

    rows = []
    for op in ops:
        ta = a.op_time.get(op, 0)
        tb = b.op_time.get(op, 0)
        ca = a.op_count.get(op, 0)
        cb = b.op_count.get(op, 0)
        sa = ta / total_a
        sb = tb / total_b
        rows.append((
            op,
            category(op),
            ca, cb, cb - ca,
            ta, tb, tb - ta,
            sa, sb, sb - sa,
            a.op_gap_p95.get(op, 0), b.op_gap_p95.get(op, 0),
            a.op_gap_max.get(op, 0), b.op_gap_max.get(op, 0),
        ))

    rows.sort(key=lambda x: x[7], reverse=True)  # delta_time desc

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "opcode","category",
            "count_A","count_B","delta_count",
            "time_ns_A","time_ns_B","delta_time_ns",
            "share_A","share_B","delta_share",
            "p95_gap_ns_A","p95_gap_ns_B",
            "max_gap_ns_A","max_gap_ns_B"
        ])
        for r in rows:
            w.writerow(r)

def write_pc_diff(out_csv: str, a: RunProfile, b: RunProfile, topk: int = 200):
    keys = set(a.pcop_time.keys()) | set(b.pcop_time.keys())
    rows = []
    for (pc, op) in keys:
        ta = a.pcop_time.get((pc, op), 0)
        tb = b.pcop_time.get((pc, op), 0)
        ca = a.pcop_count.get((pc, op), 0)
        cb = b.pcop_count.get((pc, op), 0)
        rows.append((pc, op, category(op), ca, cb, cb-ca, ta, tb, tb-ta))

    rows.sort(key=lambda x: x[8], reverse=True)
    rows = rows[:topk]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pc","opcode","category","count_A","count_B","delta_count","time_ns_A","time_ns_B","delta_time_ns"])
        for r in rows:
            w.writerow(r)

def fmt_ns(ns: int) -> str:
    if ns >= 1_000_000_000:
        return f"{ns/1e9:.3f}s"
    if ns >= 1_000_000:
        return f"{ns/1e6:.3f}ms"
    if ns >= 1_000:
        return f"{ns/1e3:.3f}us"
    return f"{ns}ns"

def top_items(d: Dict[str,int], k: int = 10) -> List[Tuple[str,int]]:
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

def write_report(out_txt: str, a: RunProfile, b: RunProfile):
    total_a = sum(a.op_time.values()) if a.op_time else 1
    total_b = sum(b.op_time.values()) if b.op_time else 1

    def op_line(op: str) -> str:
        ta = a.op_time.get(op, 0)
        tb = b.op_time.get(op, 0)
        da = ta/total_a
        db = tb/total_b
        return (f"{op:12s} cat={category(op):7s} "
                f"timeA={fmt_ns(ta):>10s} shareA={da:6.2%} "
                f"timeB={fmt_ns(tb):>10s} shareB={db:6.2%} "
                f"Δtime={fmt_ns(tb-ta):>10s} "
                f"p95(A/B)={fmt_ns(a.op_gap_p95.get(op,0))}/{fmt_ns(b.op_gap_p95.get(op,0))} "
                f"max(A/B)={fmt_ns(a.op_gap_max.get(op,0))}/{fmt_ns(b.op_gap_max.get(op,0))}")

    # compute-instruction presence check
    compute_ops = [op for op in set(a.op_count)|set(b.op_count) if category(op)=="compute"]
    key_compute = ["VFMADD_VV","VFMADD","VFMA","FMADD","FMA","VADD_VV","VMUL_VV"]
    present = [op for op in key_compute if (op in a.op_count or op in b.op_count)]

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=== Ventus cyclesim log diff report ===\n\n")
        f.write(f"[A] {a.logfile}\n")
        f.write(f"  instr_lines : {a.instr_count}\n")
        f.write(f"  recv_lines  : {a.recv_count} | blocks={a.blocks} warps_launched={a.warps_launched}\n")
        f.write(f"  kernel_grid : {a.kernel_grid}\n")
        f.write(f"  time_window : {fmt_ns(a.window_ns)} (t0={a.t0}ns, t1={a.t1}ns)\n")
        f.write(f"  attributed  : {fmt_ns(sum(a.op_time.values()))}\n\n")

        f.write(f"[B] {b.logfile}\n")
        f.write(f"  instr_lines : {b.instr_count}\n")
        f.write(f"  recv_lines  : {b.recv_count} | blocks={b.blocks} warps_launched={b.warps_launched}\n")
        f.write(f"  kernel_grid : {b.kernel_grid}\n")
        f.write(f"  time_window : {fmt_ns(b.window_ns)} (t0={b.t0}ns, t1={b.t1}ns)\n")
        f.write(f"  attributed  : {fmt_ns(sum(b.op_time.values()))}\n\n")

        f.write("=== Category time (attributed gap) ===\n")
        cats = sorted(set(a.cat_time) | set(b.cat_time))
        for c in cats:
            ta = a.cat_time.get(c, 0)
            tb = b.cat_time.get(c, 0)
            f.write(f"  {c:7s} A={fmt_ns(ta):>10s} ({ta/total_a:6.2%}) | "
                    f"B={fmt_ns(tb):>10s} ({tb/total_b:6.2%}) | Δ={fmt_ns(tb-ta)}\n")
        f.write("\n")

        f.write("=== Top opcodes by extra time in B (Δtime desc) ===\n")
        # compute delta per opcode
        deltas = {}
        for op in set(a.op_time) | set(b.op_time) | set(a.op_count) | set(b.op_count):
            deltas[op] = b.op_time.get(op, 0) - a.op_time.get(op, 0)
        for op, dt in sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:20]:
            if dt <= 0:
                continue
            f.write("  " + op_line(op) + "\n")
        f.write("\n")

        f.write("=== Top opcodes that got faster in B (Δtime asc) ===\n")
        for op, dt in sorted(deltas.items(), key=lambda x: x[1])[:20]:
            if dt >= 0:
                continue
            f.write("  " + op_line(op) + "\n")
        f.write("\n")

        f.write("=== Compute-path sanity check ===\n")
        if present:
            for op in present:
                ca, cb = a.op_count.get(op, 0), b.op_count.get(op, 0)
                ta, tb = a.op_time.get(op, 0), b.op_time.get(op, 0)
                f.write(f"  {op:10s} count A/B={ca}/{cb} | time A/B={fmt_ns(ta)}/{fmt_ns(tb)}\n")
        else:
            f.write("  (no obvious compute mnemonics from the preset list; check opcode_diff.csv)\n")
        f.write("\n")

        f.write("=== Quick diagnosis hints (rules of thumb) ===\n")
        f.write("  1) If VFMADD/VF* counts drop a lot in B, compute path probably de-vectorized.\n")
        f.write("  2) If BARRIER/JOIN/REGEXT time explodes in B, bottleneck is sync/runtime, not math.\n")
        f.write("  3) If LW hotspots cluster near kernel entry PCs (args loads), meta/arg block contention is likely.\n")
        f.write("  4) If VSW12 hotspots sit just before barrier, your attribution may be charging barrier wait to VSW12.\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logA", help="baseline log, e.g. 1wg.log")
    ap.add_argument("logB", help="comparison log, e.g. 2wg/multiwg.log")
    ap.add_argument("--cycle-ns", type=int, default=10, help="ns per cycle (only for display; gap uses ns timestamps)")
    ap.add_argument("--out", default="diff", help="output prefix")
    ap.add_argument("--topk-pc", type=int, default=200, help="topk hotspot pc-op rows to export")
    args = ap.parse_args()

    A = parse_log(args.logA, args.cycle_ns)
    B = parse_log(args.logB, args.cycle_ns)

    write_opcode_diff(args.out + ".opcode_diff.csv", A, B)
    write_pc_diff(args.out + ".pc_diff.csv", A, B, topk=args.topk_pc)
    write_report(args.out + ".report.txt", A, B)

    print("=== diff done ===")
    print("report :", args.out + ".report.txt")
    print("opcode :", args.out + ".opcode_diff.csv")
    print("pcdiff :", args.out + ".pc_diff.csv")

if __name__ == "__main__":
    main()
