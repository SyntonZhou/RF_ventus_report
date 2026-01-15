#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ventus trace: per-warp adjacent-instruction delta timing analysis.

Idea:
  For each (sm, warp), when we see a new instruction event at time t,
  we treat delta = t - last_t as the "effective time" spent on last_opcode
  (includes stalls + unscheduled time for that warp).
Aggregate deltas by opcode to answer: where time goes.

Works in streaming mode for huge logs (tens of GB).
Median is approximated via reservoir sampling per opcode.
"""

import os
import re
import csv
import math
import random
import argparse
from collections import defaultdict, Counter

# ---- Regex (bytes) ----
# Standard instruction line (has PC + OPCODE_0x... + @xxxns)
INSTR_RE = re.compile(
    rb"SM\s*(\d+)\s+warp\s+(\d+)\s+0x([0-9a-fA-F]+)\s+"
    rb"([A-Z0-9]+(?:_[A-Z0-9]+)*)_0x[0-9a-fA-F]+.*?@(\d+)ns"
)

# Special "JUMP to ..." line (no PC token)
JUMP_TO_RE = re.compile(
    rb"SM\s*(\d+)\s+warp\s+(\d+)\s+JUMP\s+to\s+0x([0-9a-fA-F]+)\s+@(\d+)ns"
)

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def median_of_list(xs):
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])

class OpStats:
    """Welford running variance + min/max + reservoir sample + sum."""
    __slots__ = ("count", "sum_ns", "min_ns", "max_ns", "mean", "M2",
                 "seen", "sample", "k")

    def __init__(self, k=50000):
        self.count = 0
        self.sum_ns = 0
        self.min_ns = None
        self.max_ns = None
        self.mean = 0.0
        self.M2 = 0.0
        self.seen = 0
        self.sample = []
        self.k = k

    def update(self, x_ns: int):
        self.count += 1
        self.sum_ns += x_ns
        self.min_ns = x_ns if self.min_ns is None else min(self.min_ns, x_ns)
        self.max_ns = x_ns if self.max_ns is None else max(self.max_ns, x_ns)

        # Welford
        delta = x_ns - self.mean
        self.mean += delta / self.count
        delta2 = x_ns - self.mean
        self.M2 += delta * delta2

        # Reservoir sampling
        self.seen += 1
        if len(self.sample) < self.k:
            self.sample.append(x_ns)
        else:
            j = random.randint(1, self.seen)
            if j <= self.k:
                self.sample[j - 1] = x_ns

    def finalize(self):
        if self.count <= 1:
            var = 0.0
        else:
            var = self.M2 / (self.count - 1)
        std = math.sqrt(var)
        med = median_of_list(self.sample)
        return {
            "count": self.count,
            "sum_ns": self.sum_ns,
            "mean_ns": self.mean,
            "std_ns": std,
            "min_ns": 0 if self.min_ns is None else self.min_ns,
            "max_ns": 0 if self.max_ns is None else self.max_ns,
            "median_ns_approx": med if med is not None else 0.0,
        }

def gcd_list(nums):
    g = 0
    for x in nums:
        g = math.gcd(g, x)
    return g

def infer_ns_per_cycle_from_small_diffs(small_diff_counter: Counter, fallback=10):
    """
    Robust-ish inference:
      take most common small diffs (<=200ns),
      compute gcd over top N values.
    """
    if not small_diff_counter:
        return fallback
    top = [d for d, _ in small_diff_counter.most_common(30)]
    g = gcd_list(top)
    return g if g > 0 else fallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="trace log file path")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--sm", type=int, default=None, help="filter SM id (optional)")
    ap.add_argument("--warp", type=int, default=None, help="filter warp id (optional)")
    ap.add_argument("--include-jump-to", action="store_true",
                    help="treat 'JUMP to ...' lines as pseudo opcode JUMP_TO (default: off)")
    ap.add_argument("--sample-k", type=int, default=50000,
                    help="reservoir sample size per opcode for median (default 50000)")
    ap.add_argument("--ns-per-cycle", type=int, default=None,
                    help="manual set ns per cycle; if not set, infer from small diffs")
    ap.add_argument("--time-min-ns", type=int, default=None,
                    help="ignore events earlier than this ns (optional)")
    ap.add_argument("--time-max-ns", type=int, default=None,
                    help="ignore events later than this ns (optional)")
    args = ap.parse_args()

    safe_mkdir(args.out)

    # Progress bar (optional)
    try:
        from tqdm import tqdm
        use_tqdm = True
    except Exception:
        use_tqdm = False

    file_size = os.path.getsize(args.log)
    pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Parsing") if use_tqdm else None

    # last_event[(sm,warp)] = (last_ns, last_opcode)
    last_event = {}

    # per-opcode stats (delta in ns)
    op_stats = defaultdict(lambda: OpStats(k=args.sample_k))

    # per-warp summaries (8 warps is tiny; safe)
    warp_span = {}  # (sm,warp) -> (first_ns, last_ns)
    warp_time_ns = Counter()  # (sm,warp) -> sum of deltas (ns)  [note: equals last-first if continuous]
    warp_opcode_time_ns = Counter()  # ((sm,warp), opcode) -> sum delta (ns)

    # for ns-per-cycle inference
    small_diff_counter = Counter()
    small_diff_cap = 2_000_000  # collect at most this many small diffs occurrences (by counting, not list)

    # counters
    parsed_events = 0
    ignored_negative = 0
    ignored_zero = 0

    with open(args.log, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if pbar is not None:
                pbar.update(len(line))

            m = INSTR_RE.search(line)
            opcode = None
            sm = None
            warp = None
            ns = None

            if m:
                sm = int(m.group(1))
                warp = int(m.group(2))
                opcode = m.group(4).decode("ascii", errors="ignore")
                ns = int(m.group(5))
            else:
                if args.include_jump_to:
                    jm = JUMP_TO_RE.search(line)
                    if jm:
                        sm = int(jm.group(1))
                        warp = int(jm.group(2))
                        opcode = "JUMP_TO"
                        ns = int(jm.group(4))

            if opcode is None:
                continue

            if args.sm is not None and sm != args.sm:
                continue
            if args.warp is not None and warp != args.warp:
                continue
            if args.time_min_ns is not None and ns < args.time_min_ns:
                continue
            if args.time_max_ns is not None and ns > args.time_max_ns:
                continue

            parsed_events += 1
            key = (sm, warp)

            # warp span
            if key not in warp_span:
                warp_span[key] = (ns, ns)
            else:
                warp_span[key] = (warp_span[key][0], ns)

            if key in last_event:
                last_ns, last_op = last_event[key]
                d = ns - last_ns
                if d < 0:
                    ignored_negative += 1
                else:
                    if d == 0:
                        ignored_zero += 1
                    # attribute delta to previous opcode
                    op_stats[last_op].update(d)
                    warp_time_ns[key] += d
                    warp_opcode_time_ns[(key, last_op)] += d

                    # collect small diffs for ns-per-cycle inference
                    if args.ns_per_cycle is None and d > 0 and d <= 200 and sum(small_diff_counter.values()) < small_diff_cap:
                        small_diff_counter[d] += 1

            # update last event to current opcode/time
            last_event[key] = (ns, opcode)

    if pbar is not None:
        pbar.close()

    if parsed_events < 2:
        print("No enough events parsed. Check filters / regex.")
        return

    # infer ns_per_cycle
    if args.ns_per_cycle is not None:
        ns_per_cycle = args.ns_per_cycle
    else:
        ns_per_cycle = infer_ns_per_cycle_from_small_diffs(small_diff_counter, fallback=10)

    # finalize opcode stats
    rows = []
    total_warp_time_ns = sum(st.sum_ns for st in op_stats.values())  # sum of deltas across warps (warp-time)
    if total_warp_time_ns == 0:
        total_warp_time_ns = 1

    for op, st in op_stats.items():
        fin = st.finalize()
        share = fin["sum_ns"] / total_warp_time_ns
        rows.append({
            "opcode": op,
            "count": fin["count"],
            "total_ns": fin["sum_ns"],
            "share_warp_time": share,
            "mean_ns": fin["mean_ns"],
            "median_ns_approx": fin["median_ns_approx"],
            "std_ns": fin["std_ns"],
            "min_ns": fin["min_ns"],
            "max_ns": fin["max_ns"],
            "mean_cycles": fin["mean_ns"] / ns_per_cycle,
            "median_cycles_approx": fin["median_ns_approx"] / ns_per_cycle,
            "std_cycles": fin["std_ns"] / ns_per_cycle,
            "min_cycles": fin["min_ns"] / ns_per_cycle,
            "max_cycles": fin["max_ns"] / ns_per_cycle,
            "total_cycles": fin["sum_ns"] / ns_per_cycle,
        })

    # sort for reporting: by total time desc
    rows.sort(key=lambda r: r["total_ns"], reverse=True)

    # write CSV
    csv_path = os.path.join(args.out, "opcode_time_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as wf:
        w = csv.DictWriter(wf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # write per-warp summary
    warp_rows = []
    for key, (t0, t1) in sorted(warp_span.items()):
        sm, warp = key
        span_ns = t1 - t0
        warp_rows.append({
            "sm": sm,
            "warp": warp,
            "first_ns": t0,
            "last_ns": t1,
            "span_ns": span_ns,
            "span_cycles": span_ns / ns_per_cycle,
            "sum_delta_ns": warp_time_ns[key],
            "sum_delta_cycles": warp_time_ns[key] / ns_per_cycle,
        })
    warp_csv = os.path.join(args.out, "warp_summary.csv")
    with open(warp_csv, "w", newline="", encoding="utf-8") as wf:
        w = csv.DictWriter(wf, fieldnames=list(warp_rows[0].keys()))
        w.writeheader()
        w.writerows(warp_rows)

    # write a TXT summary (per your preference)
    txt_path = os.path.join(args.out, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as wf:
        wf.write(f"[Parsed events] {parsed_events}\n")
        wf.write(f"[Ignored] negative_deltas={ignored_negative}, zero_deltas={ignored_zero}\n")
        wf.write(f"[ns_per_cycle] {ns_per_cycle} ns\n")
        wf.write(f"[Total warp-time] {total_warp_time_ns} ns  ({total_warp_time_ns/ns_per_cycle:.2f} cycles)\n\n")
        wf.write("Top 20 opcodes by total warp-time:\n")
        wf.write(f"{'OPCODE':<16} {'TOTAL(ns)':>14} {'SHARE':>8} {'MEAN(cyc)':>12} {'MED(cyc)':>12} {'MAX(cyc)':>12}\n")
        for r in rows[:20]:
            wf.write(f"{r['opcode']:<16} {r['total_ns']:>14} {r['share_warp_time']*100:>7.2f}% "
                     f"{r['mean_cycles']:>12.2f} {r['median_cycles_approx']:>12.2f} {r['max_cycles']:>12.2f}\n")
        wf.write("\nPer-warp span (wall time on that warp timeline):\n")
        for wr in warp_rows:
            wf.write(f"SM{wr['sm']} warp{wr['warp']}: span={wr['span_cycles']:.2f} cycles\n")

    # console summary
    print(f"[DONE] parsed_events={parsed_events}")
    print(f"[ns_per_cycle] {ns_per_cycle} ns")
    print(f"[OUT] {csv_path}")
    print(f"[OUT] {warp_csv}")
    print(f"[OUT] {txt_path}")
    print("\nTop 10 opcodes by total warp-time:")
    for r in rows[:10]:
        print(f"  {r['opcode']:<12} share={r['share_warp_time']*100:>6.2f}%  "
              f"mean={r['mean_cycles']:.2f} cyc  med≈{r['median_cycles_approx']:.2f} cyc  max={r['max_cycles']:.2f} cyc")

if __name__ == "__main__":
    main()
