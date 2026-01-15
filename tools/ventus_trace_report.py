#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ventus trace (25GB+) streaming analyzer for PPT-grade evidence.

It produces three categories of results (baseline vs optimized):
(1) Per-SM / per-warp / per-instruction counts  -> parallelism & load balance
(2) Per-instruction "start->next start" latency (same SM+warp) -> control overhead evidence
(3) Instruction total comparison (counts + %) -> control-instruction reduction evidence

Usage:
  python ventus_trace_report.py \
    --baseline  /path/to/baseline.log \
    --optimized /path/to/optimized.log \
    --outdir    ./out

Outputs (CSV):
  out/summary_baseline.csv
  out/summary_optimized.csv
  out/sm_warp_instr_baseline.csv
  out/sm_warp_instr_optimized.csv
  out/instr_compare.csv
  out/latency_baseline.csv
  out/latency_optimized.csv
  out/latency_compare.csv
"""

import argparse
import csv
import gzip
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Iterable, Any


# ----------------------------- parsing helpers -----------------------------

def open_text(path: str):
    """Open plain text or .gz transparently, in text mode with large buffer."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore", buffering=1024 * 1024)


def canonical_instr(token: str) -> str:
    """
    Normalize instruction token to a compact 'opcode family' name.

    Examples:
      "AUIPC_0x00004197" -> "AUIPC"
      "VFMADD_VV"        -> "VFMADD"
      "VLW12_V"          -> "VLW12"
      "VADD12_VI"        -> "VADD12"
    """
    if not token:
        return "UNKNOWN"
    base = token.split("_", 1)[0]
    # remove non-alnum tail if any
    base = "".join(ch for ch in base if ch.isalnum())
    return base or "UNKNOWN"


def parse_trace_line(line: str) -> Optional[Tuple[int, int, str, int]]:
    """
    Parse one Ventus trace line into (sm, warp, instr, time_ns).
    Returns None if the line isn't an instruction line.

    Typical instruction line:
      SM 1 warp 2 0x80000000 AUIPC_0x00004197 ... @75ns,1 ...
    JUMP event line:
      SM 1 warp 2 JUMP to 0x80000064 @1355ns,1 ...
    Non-instruction lines ("receive kernel", alloc logs, etc.) are ignored.
    """
    if not line.startswith("SM "):
        return None

    toks = line.split()
    # Minimal sanity: "SM <id> warp <id> ..."
    if len(toks) < 6 or toks[0] != "SM" or toks[2] != "warp":
        return None

    try:
        sm = int(toks[1])
        warp = int(toks[3])
    except ValueError:
        return None

    # Filter out "receive kernel" events etc.
    # instruction token is:
    #   - toks[5] if toks[4] is address "0x..."
    #   - toks[4] if it is "JUMP"/"JOIN"/etc
    if toks[4].startswith("0x"):
        instr_tok = toks[5]
    else:
        instr_tok = toks[4]
        # Heuristic: ignore "receive"
        if instr_tok == "receive":
            return None

    # Find time token like "@1355ns,1"
    time_ns = None
    for t in toks:
        if t.startswith("@") and "ns" in t:
            # strip "@", keep digits before "ns"
            try:
                time_ns = int(t[1:t.index("ns")])
            except Exception:
                time_ns = None
            break
    if time_ns is None:
        return None

    instr = canonical_instr(instr_tok)
    return sm, warp, instr, time_ns


# ----------------------------- latency stats ------------------------------

@dataclass
class LatStat:
    """Streaming latency stats with log2 histogram (tiny memory, PPT-friendly)."""
    count: int = 0
    s: int = 0
    mn: int = field(default_factory=lambda: 1 << 62)
    mx: int = 0
    # 0..63 bins for delta_ns in [1, 2^63]
    hist: list = field(default_factory=lambda: [0] * 64)

    def add(self, delta_ns: int):
        if delta_ns <= 0:
            # ignore non-positive (can happen if logs reorder or duplicated timestamps)
            return
        self.count += 1
        self.s += delta_ns
        if delta_ns < self.mn:
            self.mn = delta_ns
        if delta_ns > self.mx:
            self.mx = delta_ns
        b = int(math.log2(delta_ns)) if delta_ns > 0 else 0
        if b < 0:
            b = 0
        if b > 63:
            b = 63
        self.hist[b] += 1

    def mean(self) -> float:
        return (self.s / self.count) if self.count else 0.0

    def approx_percentile(self, q: float) -> int:
        """
        Approximate percentile from log2 histogram.
        Returns a representative ns value (midpoint of the selected bin).
        """
        if self.count == 0:
            return 0
        target = int(math.ceil(q * self.count))
        c = 0
        for b, n in enumerate(self.hist):
            c += n
            if c >= target:
                # midpoint of [2^b, 2^(b+1))
                lo = 1 << b
                hi = (1 << (b + 1)) if b < 63 else self.mx
                return (lo + hi) // 2
        return self.mx


# ----------------------------- main analyzer ------------------------------

@dataclass
class LogStats:
    path: str
    instr_total: Counter = field(default_factory=Counter)  # instr -> count
    sm_total: Counter = field(default_factory=Counter)     # sm -> instr count
    warp_total: Counter = field(default_factory=Counter)   # (sm,warp) -> instr count
    sm_warp_instr: Dict[Tuple[int, int], Counter] = field(default_factory=lambda: defaultdict(Counter))
    # timing
    first_ts_sm: Dict[int, int] = field(default_factory=dict)  # sm -> first time
    last_ts_sm: Dict[int, int] = field(default_factory=dict)   # sm -> last time
    first_ts_global: Optional[int] = None
    last_ts_global: Optional[int] = None
    # latency stats per instruction family (based on previous instruction in the same SM+warp)
    lat_by_instr: Dict[str, LatStat] = field(default_factory=lambda: defaultdict(LatStat))


def analyze_log(path: str) -> LogStats:
    st = LogStats(path=path)
    # For latency: per (sm,warp) keep last (instr, time)
    last_instr_time: Dict[Tuple[int, int], Tuple[str, int]] = {}

    with open_text(path) as f:
        for line in f:
            parsed = parse_trace_line(line)
            if parsed is None:
                continue
            sm, warp, instr, tns = parsed

            # counts
            st.instr_total[instr] += 1
            st.sm_total[sm] += 1
            st.warp_total[(sm, warp)] += 1
            st.sm_warp_instr[(sm, warp)][instr] += 1

            # timing per SM / global
            if sm not in st.first_ts_sm:
                st.first_ts_sm[sm] = tns
            st.last_ts_sm[sm] = tns

            if st.first_ts_global is None or tns < st.first_ts_global:
                st.first_ts_global = tns
            if st.last_ts_global is None or tns > st.last_ts_global:
                st.last_ts_global = tns

            # latency: attribute delta to *previous* instruction on this SM+warp
            key = (sm, warp)
            if key in last_instr_time:
                prev_instr, prev_t = last_instr_time[key]
                st.lat_by_instr[prev_instr].add(tns - prev_t)
            last_instr_time[key] = (instr, tns)

    return st


# ----------------------------- CSV writers --------------------------------

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def write_summary_csv(st: LogStats, out_csv: str):
    """SM-level summary and global summary in one file (easy to paste into PPT)."""
    active_sms = sorted(st.sm_total.keys())
    total_instr = sum(st.instr_total.values())
    dur_ns = (st.last_ts_global - st.first_ts_global) if (st.first_ts_global is not None and st.last_ts_global is not None) else 0
    instr_per_ns = (total_instr / dur_ns) if dur_ns > 0 else 0.0

    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["log", st.path])
        w.writerow(["total_instr", total_instr])
        w.writerow(["duration_ns(approx)", dur_ns])
        w.writerow(["instr_per_ns(approx)", f"{instr_per_ns:.6f}"])
        w.writerow(["active_sms", len(active_sms)])
        w.writerow([])

        w.writerow(["SM", "instr", "sm_dur_ns(approx)", "instr_per_ns(approx)"])
        for sm in active_sms:
            sm_instr = st.sm_total[sm]
            sm_dur = st.last_ts_sm[sm] - st.first_ts_sm[sm]
            sm_ipn = (sm_instr / sm_dur) if sm_dur > 0 else 0.0
            w.writerow([sm, sm_instr, sm_dur, f"{sm_ipn:.6f}"])

        w.writerow([])
        w.writerow(["TopInstructions", "count", "pct_of_total"])
        for instr, c in st.instr_total.most_common(20):
            pct = (100.0 * c / total_instr) if total_instr else 0.0
            w.writerow([instr, c, f"{pct:.3f}"])


def write_sm_warp_instr_csv(st: LogStats, out_csv: str):
    """Per (SM,warp,instr) counts; only ~SMs*warps*instr_types rows -> PPT-friendly."""
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["SM", "warp", "instr", "count"])
        for (sm, warp), ctr in sorted(st.sm_warp_instr.items()):
            for instr, c in ctr.most_common():
                w.writerow([sm, warp, instr, c])


def write_latency_csv(st: LogStats, out_csv: str):
    """Per-instruction latency stats (approx percentiles)."""
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["instr(prev)", "samples", "mean_ns", "min_ns", "p50_ns~", "p90_ns~", "p99_ns~", "max_ns"])
        for instr, ls in sorted(st.lat_by_instr.items(), key=lambda x: (-x[1].count, x[0])):
            if ls.count == 0:
                continue
            w.writerow([
                instr,
                ls.count,
                f"{ls.mean():.3f}",
                (ls.mn if ls.mn < (1 << 62) else 0),
                ls.approx_percentile(0.50),
                ls.approx_percentile(0.90),
                ls.approx_percentile(0.99),
                ls.mx
            ])


def write_instr_compare_csv(base: LogStats, opt: LogStats, out_csv: str):
    """Instruction totals: baseline vs optimized counts + pct + ratio."""
    base_total = sum(base.instr_total.values())
    opt_total = sum(opt.instr_total.values())
    instrs = sorted(set(base.instr_total.keys()) | set(opt.instr_total.keys()))

    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow([
            "instr",
            "baseline_count", "baseline_pct",
            "opt_count", "opt_pct",
            "delta(opt-base)", "ratio(opt/base)"
        ])
        for instr in instrs:
            b = base.instr_total.get(instr, 0)
            o = opt.instr_total.get(instr, 0)
            bp = (100.0 * b / base_total) if base_total else 0.0
            op = (100.0 * o / opt_total) if opt_total else 0.0
            ratio = (o / b) if b else ("inf" if o else 1.0)
            w.writerow([instr, b, f"{bp:.3f}", o, f"{op:.3f}", (o - b), ratio])


def write_latency_compare_csv(base: LogStats, opt: LogStats, out_csv: str):
    """Compare mean/p90 latency per instruction (approx)."""
    instrs = sorted(set(base.lat_by_instr.keys()) | set(opt.lat_by_instr.keys()))

    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow([
            "instr(prev)",
            "base_samples", "base_mean_ns", "base_p90_ns~",
            "opt_samples", "opt_mean_ns", "opt_p90_ns~"
        ])
        for instr in instrs:
            b = base.lat_by_instr.get(instr, LatStat())
            o = opt.lat_by_instr.get(instr, LatStat())
            if b.count == 0 and o.count == 0:
                continue
            w.writerow([
                instr,
                b.count, f"{b.mean():.3f}", b.approx_percentile(0.90),
                o.count, f"{o.mean():.3f}", o.approx_percentile(0.90),
            ])


# ----------------------------- entrypoint ---------------------------------

def main():
    ap = argparse.ArgumentParser(description="Ventus trace analyzer (streaming, PPT-friendly).")
    ap.add_argument("--baseline", required=True, help="Path to baseline log (.log or .gz).")
    ap.add_argument("--optimized", required=True, help="Path to optimized log (.log or .gz).")
    ap.add_argument("--outdir", required=True, help="Output directory for CSV reports.")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    print(f"[1/2] Analyzing baseline:  {args.baseline}")
    base = analyze_log(args.baseline)
    print(f"[2/2] Analyzing optimized: {args.optimized}")
    opt = analyze_log(args.optimized)

    # write CSVs
    write_summary_csv(base, os.path.join(args.outdir, "summary_baseline.csv"))
    write_summary_csv(opt,  os.path.join(args.outdir, "summary_optimized.csv"))

    write_sm_warp_instr_csv(base, os.path.join(args.outdir, "sm_warp_instr_baseline.csv"))
    write_sm_warp_instr_csv(opt,  os.path.join(args.outdir, "sm_warp_instr_optimized.csv"))

    write_instr_compare_csv(base, opt, os.path.join(args.outdir, "instr_compare.csv"))

    write_latency_csv(base, os.path.join(args.outdir, "latency_baseline.csv"))
    write_latency_csv(opt,  os.path.join(args.outdir, "latency_optimized.csv"))
    write_latency_compare_csv(base, opt, os.path.join(args.outdir, "latency_compare.csv"))

    # Print a short PPT-ready console summary
    def quick(st: LogStats) -> Dict[str, Any]:
        total = sum(st.instr_total.values())
        dur = (st.last_ts_global - st.first_ts_global) if (st.first_ts_global is not None and st.last_ts_global is not None) else 0
        return {
            "active_sms": len(st.sm_total),
            "total_instr": total,
            "dur_ns": dur,
            "instr_per_ns": (total / dur) if dur > 0 else 0.0
        }

    qb = quick(base)
    qo = quick(opt)
    print("\n=== PPT-ready highlights (approx from trace timestamps) ===")
    print(f"Baseline : active SMs={qb['active_sms']}, total_instr={qb['total_instr']}, dur_ns={qb['dur_ns']}, instr/ns={qb['instr_per_ns']:.6f}")
    print(f"Optimized: active SMs={qo['active_sms']}, total_instr={qo['total_instr']}, dur_ns={qo['dur_ns']}, instr/ns={qo['instr_per_ns']:.6f}")
    print(f"Instr reduction ratio (opt/base): {qo['total_instr']/qb['total_instr'] if qb['total_instr'] else 0:.4f}")
    print(f"Duration ratio (opt/base):        {qo['dur_ns']/qb['dur_ns'] if qb['dur_ns'] else 0:.4f}")
    print("\n[Done] CSV reports written to:", args.outdir)


if __name__ == "__main__":
    main()
