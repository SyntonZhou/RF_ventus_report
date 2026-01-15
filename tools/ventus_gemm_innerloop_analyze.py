#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import csv
import math
import argparse
from collections import defaultdict, deque

# ---------------- Regex ----------------
# Normal instruction line (with PC/opcode/@time)
# Example:
# SM 1 warp 0 0x800002ec VADD_VX_0x02b047d7 ... @82205ns,1
INSTR_RE = re.compile(
    r"SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+0x(?P<pc>[0-9a-fA-F]+)\s+"
    r"(?P<op>[A-Z0-9]+(?:_[A-Z0-9]+)*)_0x[0-9a-fA-F]+.*?@(?P<ns>\d+)ns"
)

# JUMP-to trace line (no PC)
# Example:
# SM 1 warp 0 JUMP to 0x800002ec @82795ns,1
JUMP_RE = re.compile(
    r"SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+JUMP\s+to\s+0x(?P<tgt>[0-9a-fA-F]+)\s+@(?P<ns>\d+)ns"
)

# Extract jumpTO in JAL line:
# ... JAL_... JUMP=true, jumpTO 0x800002ec @...
JAL_JUMPTO_RE = re.compile(r"jumpTO\s+0x(?P<tgt>[0-9a-fA-F]+)")

# MEMADDR list:
# ... MEMADDR 700026c0 700026c4 ...
MEMADDR_RE = re.compile(r"\bMEMADDR\b\s+(?P<addrs>[0-9a-fA-F\s]+)")

# ---------------- Utilities ----------------
def parse_hex_int(s: str) -> int:
    return int(s, 16)

def gcd_update(g, x):
    if x <= 0:
        return g
    return math.gcd(g, x) if g else x

class RunningStats:
    """Welford mean/std + min/max + count + total."""
    __slots__ = ("n", "mean", "m2", "minv", "maxv", "total")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minv = None
        self.maxv = None
        self.total = 0

    def add(self, x: int):
        self.n += 1
        self.total += x
        if self.minv is None or x < self.minv:
            self.minv = x
        if self.maxv is None or x > self.maxv:
            self.maxv = x
        # Welford
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0

def is_event_line(op: str, line: str) -> bool:
    """
    Decide whether this line is the 'completion/retire' moment for timing.
    - Prefer WB lines.
    - Branch family has no WB; use 'join mask' line as the event.
    - JOIN sometimes appears without WB; treat it as event (rare but important).
    """
    if " WB " in line:
        return True
    # SIMT branch/join events (no WB)
    if op in ("VBEQ", "VBNE", "VBLT", "VBLTU", "VBG", "BGE", "BEQ", "BNE"):
        # your trace has: "join mask=..."
        if "join mask" in line:
            return True
    if op == "JOIN":
        return True
    return False

def in_loop_pc(pc: int, loop_min: int, loop_max: int) -> bool:
    return loop_min <= pc <= loop_max

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="Ventus trace log file (huge ok; streaming parse)")
    ap.add_argument("--out", default="out_loop", help="output directory")
    ap.add_argument("--sm", type=int, default=None, help="filter SM id")
    ap.add_argument("--warps", default="0-7", help="warp list, e.g. '0-7' or '0,2,5'")

    ap.add_argument("--header-pc", default="0x800002ec", help="loop header PC (iteration start)")
    ap.add_argument("--loop-min", default="0x800002ec", help="min PC considered as loop body")
    ap.add_argument("--loop-max", default="0x8000031c", help="max PC considered as loop body")
    ap.add_argument("--max-iters", type=int, default=0, help="max iterations per warp (0=unlimited)")

    ap.add_argument("--a-load-pc", default="0x800002f4", help="A load PC (VLW12_V)")
    ap.add_argument("--b-load-pc", default="0x800002f8", help="B load PC (VLW12_V)")

    ap.add_argument("--ns-per-cycle", type=int, default=0, help="if 0, auto gcd-detect from per-warp deltas")
    ap.add_argument("--outlier-cycles", type=int, default=500, help="flag dt>=this cycles as outlier (needs ns-per-cycle)")
    args = ap.parse_args()

    # parse warps
    warps = set()
    if "-" in args.warps and "," not in args.warps:
        a, b = args.warps.split("-")
        for w in range(int(a), int(b) + 1):
            warps.add(w)
    else:
        for x in args.warps.split(","):
            x = x.strip()
            if x:
                warps.add(int(x))

    header_pc = parse_hex_int(args.header_pc.replace("0x", ""))
    loop_min = parse_hex_int(args.loop_min.replace("0x", ""))
    loop_max = parse_hex_int(args.loop_max.replace("0x", ""))
    a_load_pc = parse_hex_int(args.a_load_pc.replace("0x", ""))
    b_load_pc = parse_hex_int(args.b_load_pc.replace("0x", ""))

    os.makedirs(args.out, exist_ok=True)

    iters_path = os.path.join(args.out, "iters.csv")
    stats_path = os.path.join(args.out, "opcode_stats.csv")
    load_lat_path = os.path.join(args.out, "load_lat.csv")
    outliers_path = os.path.join(args.out, "outliers.csv")
    summary_path = os.path.join(args.out, "summary.txt")

    # per (sm,warp) states
    class WarpState:
        __slots__ = (
            "in_region", "iter_id", "iter_start_ns",
            "a0", "a1", "bbase",
            "a_issue_ns", "b_issue_ns",
            "last_ns", "last_op", "last_pc", "last_inloop",
            "last_prev1", "last_prev2",
            "iters_done"
        )
        def __init__(self):
            self.in_region = False
            self.iter_id = -1
            self.iter_start_ns = None
            self.a0 = None
            self.a1 = None
            self.bbase = None
            self.a_issue_ns = None
            self.b_issue_ns = None

            self.last_ns = None
            self.last_op = None
            self.last_pc = None
            self.last_inloop = False

            # keep 2-step context for outlier dump
            self.last_prev1 = None  # (pc, op, ns)
            self.last_prev2 = None
            self.iters_done = 0

    ws = defaultdict(WarpState)

    # running stats: opcode -> dt_ns stats
    op_dt_stats = defaultdict(RunningStats)
    total_loop_dt_ns = 0

    # load latency stats (issue->WB): keyed by "A_LOAD" / "B_LOAD"
    load_lat_stats = defaultdict(RunningStats)

    # gcd detector for ns_per_cycle (on per-warp deltas within loop)
    gcd_ns = 0

    # write CSV headers
    with open(iters_path, "w", newline="", encoding="utf-8") as f_it, \
         open(outliers_path, "w", newline="", encoding="utf-8") as f_ol:
        itw = csv.writer(f_it)
        itw.writerow(["sm","warp","iter_id","start_ns","end_ns","iter_ns","a0_addr","a1_addr","b_base"])

        olw = csv.writer(f_ol)
        olw.writerow(["sm","warp","iter_id","dt_ns","prev2_pc","prev2_op","prev1_pc","prev1_op","cur_pc","cur_op","cur_ns"])

        def flush_iter(sm, warp, st: WarpState, end_ns: int):
            if st.iter_start_ns is None:
                return
            itw.writerow([
                sm, warp, st.iter_id, st.iter_start_ns, end_ns,
                end_ns - st.iter_start_ns,
                f"0x{st.a0:08x}" if st.a0 is not None else "",
                f"0x{st.a1:08x}" if st.a1 is not None else "",
                f"0x{st.bbase:08x}" if st.bbase is not None else "",
            ])
            st.iter_start_ns = None
            st.a0 = st.a1 = st.bbase = None
            st.a_issue_ns = st.b_issue_ns = None

        # stream parse
        with open(args.log, "r", encoding="utf-8", errors="ignore") as fin:
            for line in fin:
                # JUMP-to line
                mj = JUMP_RE.search(line)
                if mj:
                    sm = int(mj.group("sm"))
                    warp = int(mj.group("warp"))
                    if args.sm is not None and sm != args.sm:
                        continue
                    if warp not in warps:
                        continue

                    ns = int(mj.group("ns"))
                    tgt = int(mj.group("tgt"), 16)
                    # 在 JUMP handler 里，解析完 tgt/ns 后加：
                    if tgt == header_pc and st.in_region and st.iter_start_ns is not None:
                        flush_iter(sm, warp, st, ns)

                    st = ws[(sm, warp)]

                    # treat as an event inside loop region if it jumps within loop range
                    cur_inloop = in_loop_pc(tgt, loop_min, loop_max) or (tgt == header_pc)

                    # duration attribution (per-warp)
                    if st.last_ns is not None and st.last_inloop and cur_inloop:
                        dt = ns - st.last_ns
                        if dt >= 0:
                            op_dt_stats[st.last_op].add(dt)
                            nonlocal_total = dt  # local var
                            # update global
                            nonlocal_total_loop = dt
                            # python scoping hack not needed; just add:
                            pass

                    # update gcd using dt
                    if st.last_ns is not None and st.last_inloop and cur_inloop:
                        dt = ns - st.last_ns
                        if dt > 0:
                            nonlocal_g = dt

                    # we can't "nonlocal" in this scope; do direct updates:
                    if st.last_ns is not None and st.last_inloop and cur_inloop:
                        dt = ns - st.last_ns
                        if dt >= 0:
                            op_dt_stats[st.last_op].add(dt)
                            total_loop_dt_ns += dt
                            if dt > 0:
                                gcd_ns = gcd_update(gcd_ns, dt)

                            # outlier (ns) later after ns_per_cycle known; we dump now in ns only
                            # if user provided ns_per_cycle already, we can check now
                            if args.ns_per_cycle:
                                if dt >= args.outlier_cycles * args.ns_per_cycle:
                                    p2 = st.last_prev2
                                    p1 = st.last_prev1
                                    olw.writerow([
                                        sm, warp, st.iter_id, dt,
                                        f"0x{p2[0]:08x}" if p2 else "", p2[1] if p2 else "",
                                        f"0x{p1[0]:08x}" if p1 else "", p1[1] if p1 else "",
                                        "", "JUMP_TO", ns
                                    ])

                    # update context as an event
                    st.last_prev2 = st.last_prev1
                    st.last_prev1 = (st.last_pc if st.last_pc is not None else 0, st.last_op if st.last_op else "", st.last_ns if st.last_ns else 0)

                    st.last_ns = ns
                    st.last_pc = None
                    st.last_op = "JUMP_TO"
                    st.last_inloop = cur_inloop
                    continue

                # instruction line
                mi = INSTR_RE.search(line)
                if not mi:
                    continue

                sm = int(mi.group("sm"))
                warp = int(mi.group("warp"))
                if args.sm is not None and sm != args.sm:
                    continue
                if warp not in warps:
                    continue

                pc = int(mi.group("pc"), 16)
                op = mi.group("op")
                ns = int(mi.group("ns"))

                st = ws[(sm, warp)]

                # If we are inside loop region, capture MEMADDR issue for A/B loads
                if in_loop_pc(pc, loop_min, loop_max):
                    mm = MEMADDR_RE.search(line)
                    if mm:
                        addrs = mm.group("addrs").strip().split()
                        if addrs:
                            # parse first and 17th (two half-warp groups), plus base for B
                            if pc == a_load_pc:
                                # issue time
                                st.a_issue_ns = ns
                                # A0/A1 addr
                                a0 = int(addrs[0], 16)
                                a1 = int(addrs[16], 16) if len(addrs) > 16 else a0
                                st.a0, st.a1 = a0, a1
                            elif pc == b_load_pc:
                                st.b_issue_ns = ns
                                b0 = int(addrs[0], 16)
                                st.bbase = b0

                # Decide if this is the event time for this instruction
                if not is_event_line(op, line):
                    continue

                cur_inloop = in_loop_pc(pc, loop_min, loop_max)

                # iteration start detection: header pc
                if pc == header_pc and cur_inloop:
                    # respect max-iters per warp
                    if args.max_iters and st.iters_done >= args.max_iters:
                        # stop tracking this warp (but keep parsing others)
                        st.in_region = False
                    else:
                        st.in_region = True
                        st.iter_id += 1
                        st.iters_done += 1
                        st.iter_start_ns = ns
                        st.a0 = st.a1 = st.bbase = None
                        st.a_issue_ns = st.b_issue_ns = None

                # attribute duration to previous event (per-warp timeline)
                if st.last_ns is not None and st.last_inloop and cur_inloop:
                    dt = ns - st.last_ns
                    if dt >= 0:
                        op_dt_stats[st.last_op].add(dt)
                        total_loop_dt_ns += dt
                        if dt > 0:
                            gcd_ns = gcd_update(gcd_ns, dt)

                        if args.ns_per_cycle:
                            if dt >= args.outlier_cycles * args.ns_per_cycle:
                                p2 = st.last_prev2
                                p1 = st.last_prev1
                                olw.writerow([
                                    sm, warp, st.iter_id, dt,
                                    f"0x{p2[0]:08x}" if p2 else "", p2[1] if p2 else "",
                                    f"0x{p1[0]:08x}" if p1 else "", p1[1] if p1 else "",
                                    f"0x{pc:08x}", op, ns
                                ])

                # load latency capture on WB for A/B load PCs
                if pc == a_load_pc and " WB " in line and st.a_issue_ns is not None:
                    lat = ns - st.a_issue_ns
                    if lat >= 0:
                        load_lat_stats["A_LOAD(issue->WB)_ns"].add(lat)
                if pc == b_load_pc and " WB " in line and st.b_issue_ns is not None:
                    lat = ns - st.b_issue_ns
                    if lat >= 0:
                        load_lat_stats["B_LOAD(issue->WB)_ns"].add(lat)

                # iteration end detection: JAL jumpTO header OR leaving loop range
                end_iter = False
                if op == "JAL":
                    mto = JAL_JUMPTO_RE.search(line)
                    if mto:
                        tgt = int(mto.group("tgt"), 16)
                        if tgt == header_pc and st.in_region:
                            end_iter = True

                if end_iter:
                    flush_iter(sm, warp, st, ns)
                    # region continues; next header starts next iter
                    st.in_region = True

                # update event context
                st.last_prev2 = st.last_prev1
                st.last_prev1 = (pc, op, ns)

                st.last_ns = ns
                st.last_pc = pc
                st.last_op = op
                st.last_inloop = cur_inloop

    # decide ns_per_cycle
    ns_per_cycle = args.ns_per_cycle if args.ns_per_cycle else (gcd_ns if gcd_ns else 10)

    # write opcode_stats
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "opcode","count","total_ns","share_loop_time",
            "mean_ns","std_ns","min_ns","max_ns",
            "mean_cycles","std_cycles","min_cycles","max_cycles","total_cycles",
            "ns_per_cycle"
        ])
        for op in sorted(op_dt_stats.keys(), key=lambda k: op_dt_stats[k].total, reverse=True):
            st = op_dt_stats[op]
            share = (st.total / total_loop_dt_ns) if total_loop_dt_ns else 0.0
            w.writerow([
                op, st.n, st.total, f"{share:.6f}",
                f"{st.mean:.6f}", f"{st.std():.6f}", st.minv, st.maxv,
                f"{(st.mean/ns_per_cycle):.6f}", f"{(st.std()/ns_per_cycle):.6f}",
                f"{(st.minv/ns_per_cycle):.6f}" if st.minv is not None else "",
                f"{(st.maxv/ns_per_cycle):.6f}" if st.maxv is not None else "",
                f"{(st.total/ns_per_cycle):.6f}",
                ns_per_cycle
            ])

    # write load_lat stats
    with open(load_lat_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric","count","mean_ns","std_ns","min_ns","max_ns","mean_cycles","std_cycles","min_cycles","max_cycles","ns_per_cycle"])
        for k in sorted(load_lat_stats.keys()):
            st = load_lat_stats[k]
            w.writerow([
                k, st.n, f"{st.mean:.6f}", f"{st.std():.6f}", st.minv, st.maxv,
                f"{(st.mean/ns_per_cycle):.6f}", f"{(st.std()/ns_per_cycle):.6f}",
                f"{(st.minv/ns_per_cycle):.6f}" if st.minv is not None else "",
                f"{(st.maxv/ns_per_cycle):.6f}" if st.maxv is not None else "",
                ns_per_cycle
            ])

    # summary.txt
    top_ops = sorted(op_dt_stats.items(), key=lambda kv: kv[1].total, reverse=True)[:15]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"ns_per_cycle = {ns_per_cycle}\n")
        f.write(f"total_loop_dt_ns = {total_loop_dt_ns}\n\n")
        f.write("Top opcodes by total time:\n")
        for op, st in top_ops:
            share = (st.total / total_loop_dt_ns) if total_loop_dt_ns else 0.0
            f.write(f"  {op:<12} total_ns={st.total:<12} share={share:.4f} count={st.n:<10} mean_ns={st.mean:.3f} max_ns={st.maxv}\n")

    print("[DONE] outputs:")
    print(" ", iters_path)
    print(" ", stats_path)
    print(" ", load_lat_path)
    print(" ", outliers_path)
    print(" ", summary_path)
    print(f"[INFO] ns_per_cycle = {ns_per_cycle} (gcd_ns={gcd_ns})")

if __name__ == "__main__":
    main()
