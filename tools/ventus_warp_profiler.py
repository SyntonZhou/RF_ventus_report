#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ventus cyclesim log profiler (streaming, 25GB-friendly).

Features
- Streaming parse with tqdm progress bar (byte-based).
- Understands three SM-lines:
    1) receive lines:
       SM 1 warp 1 receive kernel 0 sgemm_one_wg block 0 warp 0 @15ns,1 ...
    2) instruction lines:
       SM 1 warp 2 0x80000000 AUIPC_0x... ... @75ns,1 ...
    3) jump lines:
       SM 1 warp 2 JUMP to 0x80000064 @1355ns,1 ...
- Tracks warp "slot" -> (block_id, warp_in_block) mapping via receive lines,
  so instruction lines can be attributed to the correct logical warp.
- Produces per-SM, per-(SM,block,warp) stats:
  - executed instruction counts by opcode
  - "stall proxy" = delta_t between two consecutive events of the same logical warp,
    attributed to the *previous* opcode/pc (consistent with your earlier prev_op methodology)
- Outputs JSON + CSV + TXT report.
- Can compare two logs and output delta tables.

Usage examples
1) Analyze one log:
   python ventus_warp_profiler.py analyze 512512512.log --out A512 --mnk 512 512 512
   python ventus_warp_profiler.py analyze 512.log        --out B512 --mnk 512 512 512

2) Compare two logs directly (will analyze both, then diff):
   python ventus_warp_profiler.py compare 512512512.log 512.log --out diff512 --mnk 512 512 512

Notes
- "stall proxy" is per-warp inter-event delta; sums across warps can exceed wall time (overlap).
- For multi-SM, wall time is computed from global min/max timestamp across all parsed SM events.
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List, Any

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ----------------------------
# Helpers: opcode normalization and categorization
# ----------------------------

def normalize_opcode(tok: str) -> str:
    """
    Convert tokens like "AUIPC_0x00004197" -> "AUIPC"
                 "VSW_V_0xfe102c2b" -> "VSW_V"
                 "<unknown>" -> "UNKNOWN"
    Keep already-clean names.
    """
    if not tok:
        return "UNKNOWN"
    if tok == "<unknown>":
        return "UNKNOWN"
    # Some traces show "<unknown>" without angle? be safe
    if "unknown" in tok.lower():
        return "UNKNOWN"

    # Many opcodes come as OP_0xXXXXXXXX
    if "_" in tok:
        # Split once: base + tail
        base, tail = tok.split("_", 1)
        # If tail looks like hex immediate-ish, strip it.
        # Examples: 0x..., fe102c2b, 80818193, 0d0efed7
        t = tail.lower()
        is_hexish = (t.startswith("0x") and all(c in "0123456789abcdef" for c in t[2:])) or \
                    (all(c in "0123456789abcdef" for c in t) and len(t) >= 6)
        if is_hexish:
            return base
        # Some vector ops include additional semantic segments, e.g., VSW12_V (no hex tail)
        # In that case keep full token.
        return tok

    return tok


def opcode_category(op: str) -> str:
    """
    Heuristic categorization to support higher-level summary.
    """
    op = op.upper()

    # Control / sync-ish
    if op in {"J", "JAL", "JALR", "RET", "BEQ", "BNE", "BLT", "BGE", "BLE", "BGT",
              "VBEQ", "VBLT", "VBGE", "VBNE", "JUMP"}:
        return "control"
    if op in {"BARRIER", "JOIN", "SETRPC", "ENDPRG", "REGEXT", "CSRRS", "CSRR", "CSRRC", "CSRRW"}:
        return "other"

    # Address-ish
    if op in {"AUIPC", "LUI", "ADDI", "ADD", "SUB", "SLLI", "SRLI", "SRAI", "ANDI", "ORI", "XORI",
              "MUL"}:
        return "addr"

    # Memory-ish (scalar + vector)
    if op in {"LW", "SW", "LB", "LH", "LBU", "LHU", "SB", "SH"}:
        return "mem"
    if op.startswith("VL") or op.startswith("VS"):
        # e.g., VLW_V, VSW_V, VLW12_V, VSW12_V, VSE, VLE, etc.
        return "mem"

    # Compute-ish: vector float/int arithmetic
    if op.startswith("VF") or op.startswith("VM") or op.startswith("VFM") or op.startswith("VADD") or op.startswith("VSUB"):
        return "compute"
    if op.startswith("F") and op not in {"FENCE"}:
        return "compute"

    return "other"


# ----------------------------
# Streaming percentile histogram (for opcode stall proxy)
# ----------------------------

class Log2Hist:
    """
    Very small streaming histogram to approximate p95 of stall gaps.
    Bucket i stores gaps in [2^i, 2^(i+1)) ns, with i in [0..MAXB-1].
    Bucket 0 includes gap=0..1
    """
    MAXB = 64

    def __init__(self):
        self.bins = [0] * self.MAXB
        self.count = 0

    def add(self, x: int) -> None:
        if x <= 1:
            i = 0
        else:
            i = min(self.MAXB - 1, int(math.log2(x)))
        self.bins[i] += 1
        self.count += 1

    def p(self, q: float) -> int:
        """
        Approximate quantile (e.g. q=0.95) returns a representative upper bound of bucket.
        """
        if self.count == 0:
            return 0
        target = int(math.ceil(self.count * q))
        s = 0
        for i, c in enumerate(self.bins):
            s += c
            if s >= target:
                # representative value: upper bound of bucket
                if i == 0:
                    return 1
                return 1 << (i + 1)
        return 1 << self.MAXB


@dataclass
class StallStats:
    count: int = 0
    sum_gap_ns: int = 0
    max_gap_ns: int = 0

    # histogram for p95 approximation
    hist: Optional[Log2Hist] = None

    def add(self, gap_ns: int, with_hist: bool = False) -> None:
        self.count += 1
        self.sum_gap_ns += gap_ns
        if gap_ns > self.max_gap_ns:
            self.max_gap_ns = gap_ns
        if with_hist:
            if self.hist is None:
                self.hist = Log2Hist()
            self.hist.add(gap_ns)

    def mean(self) -> float:
        return (self.sum_gap_ns / self.count) if self.count else 0.0

    def p95(self) -> int:
        return self.hist.p(0.95) if self.hist is not None else 0


@dataclass
class ExecStats:
    count: int = 0

    def add(self) -> None:
        self.count += 1


# ----------------------------
# Core parser / aggregator
# ----------------------------

WarpKey = Tuple[int, int, int]  # (sm, block, warp_in_block)


@dataclass
class WarpRuntime:
    first_t: Optional[int] = None
    last_t: Optional[int] = None
    exec_total: int = 0
    stall_total_gap_ns: int = 0
    stall_total_count: int = 0


class VentusProfiler:
    def __init__(self, with_p95_hist: bool = True):
        # receive mapping: (sm, warp_slot) -> (block, warp_in_block)
        self.slot_map: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # per logical warp key state
        self.last_time: Dict[WarpKey, int] = {}
        self.last_op: Dict[WarpKey, str] = {}
        self.last_pc: Dict[WarpKey, int] = {}

        # runtime summary per warp
        self.warp_rt: Dict[WarpKey, WarpRuntime] = {}

        # exec counts
        self.exec_by_op: Dict[str, ExecStats] = {}
        self.exec_by_op_sm: Dict[Tuple[int, str], ExecStats] = {}
        self.exec_by_op_warp: Dict[Tuple[WarpKey, str], ExecStats] = {}

        self.exec_by_pc: Dict[int, ExecStats] = {}
        self.exec_by_pc_sm: Dict[Tuple[int, int], ExecStats] = {}  # (sm, pc)

        # stall proxy (attributed to previous op/pc)
        self.stall_by_op: Dict[str, StallStats] = {}
        self.stall_by_op_sm: Dict[Tuple[int, str], StallStats] = {}
        self.stall_by_op_warp: Dict[Tuple[WarpKey, str], StallStats] = {}

        self.stall_by_pc: Dict[int, StallStats] = {}
        self.stall_by_pc_sm: Dict[Tuple[int, int], StallStats] = {}

        # recv/jump counters
        self.recv_lines = 0
        self.jump_lines = 0
        self.instr_lines = 0
        self.other_lines = 0
        self.parse_errors = 0

        # global time window (SM events only)
        self.t_min: Optional[int] = None
        self.t_max: Optional[int] = None

        # per SM time window
        self.sm_tmin: Dict[int, int] = {}
        self.sm_tmax: Dict[int, int] = {}

        # config
        self.with_p95_hist = with_p95_hist

    def _update_time_window(self, sm: int, t: int) -> None:
        if self.t_min is None or t < self.t_min:
            self.t_min = t
        if self.t_max is None or t > self.t_max:
            self.t_max = t
        if sm not in self.sm_tmin or t < self.sm_tmin[sm]:
            self.sm_tmin[sm] = t
        if sm not in self.sm_tmax or t > self.sm_tmax[sm]:
            self.sm_tmax[sm] = t

    def _get_warp_key(self, sm: int, warp_slot: int) -> WarpKey:
        # If never received, use block=-1, warp=warp_slot as fallback
        block, wib = self.slot_map.get((sm, warp_slot), (-1, warp_slot))
        return (sm, block, wib)

    def _touch_warp_rt(self, key: WarpKey, t: int) -> None:
        rt = self.warp_rt.get(key)
        if rt is None:
            rt = WarpRuntime(first_t=t, last_t=t, exec_total=0, stall_total_gap_ns=0, stall_total_count=0)
            self.warp_rt[key] = rt
        else:
            if rt.first_t is None or t < rt.first_t:
                rt.first_t = t
            if rt.last_t is None or t > rt.last_t:
                rt.last_t = t

    @staticmethod
    def _parse_time_ns(line: str) -> Optional[int]:
        # find last "@", then "ns"
        at = line.rfind("@")
        if at < 0:
            return None
        ns = line.find("ns", at)
        if ns < 0:
            return None
        num = line[at + 1:ns].strip()
        # sometimes format like "@75ns,1"
        # we already cut before "ns"
        if not num.isdigit():
            # allow accidental spaces
            try:
                return int(num)
            except Exception:
                return None
        return int(num)

    @staticmethod
    def _safe_int(x: str, base: int = 10) -> Optional[int]:
        try:
            return int(x, base)
        except Exception:
            return None

    def _add_exec_op(self, sm: int, key: WarpKey, op: str) -> None:
        self.exec_by_op.setdefault(op, ExecStats()).add()
        self.exec_by_op_sm.setdefault((sm, op), ExecStats()).add()
        self.exec_by_op_warp.setdefault((key, op), ExecStats()).add()

    def _add_exec_pc(self, sm: int, pc: int) -> None:
        self.exec_by_pc.setdefault(pc, ExecStats()).add()
        self.exec_by_pc_sm.setdefault((sm, pc), ExecStats()).add()

    def _add_stall_op(self, sm: int, key: WarpKey, prev_op: str, gap_ns: int) -> None:
        self.stall_by_op.setdefault(prev_op, StallStats()).add(gap_ns, with_hist=self.with_p95_hist)
        self.stall_by_op_sm.setdefault((sm, prev_op), StallStats()).add(gap_ns, with_hist=self.with_p95_hist)
        self.stall_by_op_warp.setdefault((key, prev_op), StallStats()).add(gap_ns, with_hist=False)

    def _add_stall_pc(self, sm: int, prev_pc: int, gap_ns: int) -> None:
        self.stall_by_pc.setdefault(prev_pc, StallStats()).add(gap_ns, with_hist=False)
        self.stall_by_pc_sm.setdefault((sm, prev_pc), StallStats()).add(gap_ns, with_hist=False)

    def _account_stall(self, key: WarpKey, sm: int, t: int) -> None:
        # stall proxy = time between consecutive events for this warp key
        if key in self.last_time:
            gap = t - self.last_time[key]
            if gap < 0:
                # out-of-order timestamps (rare); ignore
                return
            prev_op = self.last_op.get(key, "UNKNOWN")
            prev_pc = self.last_pc.get(key, -1)
            self._add_stall_op(sm, key, prev_op, gap)
            if prev_pc != -1:
                self._add_stall_pc(sm, prev_pc, gap)

            # warp runtime accounting
            rt = self.warp_rt.get(key)
            if rt:
                rt.stall_total_gap_ns += gap
                rt.stall_total_count += 1

    def _finish_event(self, key: WarpKey, sm: int, t: int, op: str, pc: int) -> None:
        self._update_time_window(sm, t)
        self._touch_warp_rt(key, t)

        # 1) attribute stall to previous event of the same logical warp
        self._account_stall(key, sm, t)

        # 2) count this exec
        self._add_exec_op(sm, key, op)
        if pc != -1:
            self._add_exec_pc(sm, pc)

        # 3) update per-warp runtime totals
        rt = self.warp_rt.get(key)
        if rt:
            rt.exec_total += 1

        # 4) update last state
        self.last_time[key] = t
        self.last_op[key] = op
        self.last_pc[key] = pc

    def parse_line(self, line: str) -> None:
        # Only care lines that start with "SM "
        if not line.startswith("SM "):
            self.other_lines += 1
            return

        # Fast tokenization
        parts = line.split()
        # Minimal sanity check
        if len(parts) < 6 or parts[0] != "SM" or parts[2] != "warp":
            self.parse_errors += 1
            return

        sm = self._safe_int(parts[1])
        warp_slot = self._safe_int(parts[3])
        if sm is None or warp_slot is None:
            self.parse_errors += 1
            return

        t = self._parse_time_ns(line)
        if t is None:
            self.parse_errors += 1
            return

        # Case A: receive
        # SM 1 warp 1 receive kernel 0 ... block 0 warp 0 @15ns,1 ...
        if parts[4] == "receive":
            self.recv_lines += 1
            # find "block" and the last "warp" after it
            block_id = -1
            warp_in_block = -1
            try:
                idx_block = parts.index("block")
                block_id = int(parts[idx_block + 1])
                # after idx_block, find the next "warp"
                idx_warp2 = parts.index("warp", idx_block + 1)
                warp_in_block = int(parts[idx_warp2 + 1])
            except Exception:
                # if parse fails, still record mapping fallback
                block_id = -1
                warp_in_block = warp_slot

            self.slot_map[(sm, warp_slot)] = (block_id, warp_in_block)

            # treat receive as an event (optional); here we record as control-like event
            key = self._get_warp_key(sm, warp_slot)
            self._finish_event(key, sm, t, op="RECEIVE", pc=-1)
            return

        # Case B: JUMP
        # SM 1 warp 2 JUMP to 0x80000064 @1355ns,1 ...
        if parts[4] == "JUMP":
            self.jump_lines += 1
            dest_pc = -1
            if len(parts) >= 7 and parts[5] == "to":
                v = self._safe_int(parts[6], 16) if parts[6].startswith("0x") else None
                if v is not None:
                    dest_pc = v
            key = self._get_warp_key(sm, warp_slot)
            self._finish_event(key, sm, t, op="JUMP", pc=dest_pc)
            return

        # Case C: instruction line
        # SM 1 warp 2 0x80000000 AUIPC_0x... ... @75ns,1 ...
        # parts[4] is PC
        pc = -1
        if parts[4].startswith("0x"):
            v = self._safe_int(parts[4], 16)
            if v is None:
                self.parse_errors += 1
                return
            pc = v
            # opcode token usually at parts[5]
            op_tok = parts[5] if len(parts) > 5 else "UNKNOWN"
            op = normalize_opcode(op_tok)
            self.instr_lines += 1
            key = self._get_warp_key(sm, warp_slot)
            self._finish_event(key, sm, t, op=op, pc=pc)
            return

        # Some rare variant: no PC but still an event; treat as unknown
        self.parse_errors += 1

    def summary(self, mnk: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        # global wall time (SM-event window)
        t0 = self.t_min if self.t_min is not None else 0
        t1 = self.t_max if self.t_max is not None else 0
        wall_ns = max(0, t1 - t0)

        # MNK normalization
        norm = {}
        if mnk:
            M, N, K = mnk
            out_elems = M * N
            flops = 2 * M * N * K
            norm = {
                "M": M, "N": N, "K": K,
                "output_elements": out_elems,
                "flops": flops,
                "stall_ns_per_out": (self._sum_stall_all() / out_elems) if out_elems else 0.0,
                "stall_ns_per_flop": (self._sum_stall_all() / flops) if flops else 0.0,
                "exec_per_out": (self._sum_exec_all() / out_elems) if out_elems else 0.0,
            }

        # SM summaries
        sm_stats = {}
        for sm in sorted(self.sm_tmin.keys()):
            sm_wall = max(0, self.sm_tmax.get(sm, 0) - self.sm_tmin.get(sm, 0))
            sm_stats[sm] = {
                "t_min_ns": self.sm_tmin.get(sm, 0),
                "t_max_ns": self.sm_tmax.get(sm, 0),
                "wall_ns": sm_wall,
                "exec_total": self._sum_exec_sm(sm),
                "stall_sum_gap_ns": self._sum_stall_sm(sm),
            }

        return {
            "parsed": {
                "instr_lines": self.instr_lines,
                "recv_lines": self.recv_lines,
                "jump_lines": self.jump_lines,
                "other_lines": self.other_lines,
                "parse_errors": self.parse_errors,
            },
            "time_window": {
                "t_min_ns": t0,
                "t_max_ns": t1,
                "wall_ns": wall_ns,
            },
            "normalization": norm,
            "sm_stats": sm_stats,
            "totals": {
                "exec_total": self._sum_exec_all(),
                "stall_sum_gap_ns": self._sum_stall_all(),
            }
        }

    def _sum_exec_all(self) -> int:
        return sum(v.count for v in self.exec_by_op.values())

    def _sum_stall_all(self) -> int:
        return sum(v.sum_gap_ns for v in self.stall_by_op.values())

    def _sum_exec_sm(self, sm: int) -> int:
        return sum(v.count for (s, _), v in self.exec_by_op_sm.items() if s == sm)

    def _sum_stall_sm(self, sm: int) -> int:
        return sum(v.sum_gap_ns for (s, _), v in self.stall_by_op_sm.items() if s == sm)


# ----------------------------
# I/O: progress iterator + writers
# ----------------------------

def iter_lines_with_progress(path: str, show_progress: bool = True):
    total = os.path.getsize(path)
    if show_progress and tqdm is None:
        show_progress = False

    if show_progress:
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(path), dynamic_ncols=True)
    else:
        bar = None

    with open(path, "rb", buffering=1024 * 1024) as f:
        for raw in f:
            if bar is not None:
                bar.update(len(raw))
            yield raw.decode("utf-8", errors="replace")
    if bar is not None:
        bar.close()


def write_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def topk_from_map_stall(stall_map: Dict[Any, StallStats], k: int, key_fn=None):
    items = []
    for kk, st in stall_map.items():
        items.append((st.sum_gap_ns, kk, st))
    items.sort(reverse=True, key=lambda x: x[0])
    if key_fn:
        items = [(s, key_fn(kk), st) for (s, kk, st) in items[:k]]
    else:
        items = items[:k]
    return items


def topk_from_map_exec(exec_map: Dict[Any, ExecStats], k: int, key_fn=None):
    items = []
    for kk, st in exec_map.items():
        items.append((st.count, kk, st))
    items.sort(reverse=True, key=lambda x: x[0])
    if key_fn:
        items = [(c, key_fn(kk), st) for (c, kk, st) in items[:k]]
    else:
        items = items[:k]
    return items


def dump_analysis(prof: VentusProfiler, out_prefix: str, mnk: Optional[Tuple[int, int, int]], topk: int = 50) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    summ = prof.summary(mnk=mnk)
    summ_path = out_prefix + ".summary.json"
    with open(summ_path, "w", encoding="utf-8") as fp:
        json.dump(summ, fp, indent=2, ensure_ascii=False)

    # SM opcode table
    rows = []
    for (sm, op), ex in prof.exec_by_op_sm.items():
        st = prof.stall_by_op_sm.get((sm, op), StallStats())
        rows.append([sm, op, opcode_category(op), ex.count, st.count, st.sum_gap_ns, st.mean(), st.max_gap_ns, st.p95()])
    rows.sort(key=lambda r: (r[0], -r[5], -r[3]))
    write_csv(out_prefix + ".sm_opcode.csv",
              ["sm", "opcode", "category", "exec_count", "stall_count", "stall_sum_gap_ns", "stall_mean_ns", "stall_max_ns", "stall_p95_approx_ns"],
              rows)

    # Warp overview
    wrows = []
    for (sm, block, wib), rt in prof.warp_rt.items():
        wall = (rt.last_t - rt.first_t) if (rt.first_t is not None and rt.last_t is not None) else 0
        wrows.append([sm, block, wib, rt.first_t or 0, rt.last_t or 0, wall, rt.exec_total, rt.stall_total_count, rt.stall_total_gap_ns,
                      (rt.stall_total_gap_ns / rt.stall_total_count) if rt.stall_total_count else 0.0])
    wrows.sort(key=lambda r: (-r[8], -r[6]))
    write_csv(out_prefix + ".warp_overview.csv",
              ["sm", "block", "warp_in_block", "t_first_ns", "t_last_ns", "warp_wall_ns", "exec_total", "stall_count", "stall_sum_gap_ns", "stall_mean_ns"],
              wrows)

    # Warp opcode table (can be large, but number of warps is small; still manageable)
    worows = []
    for (key, op), ex in prof.exec_by_op_warp.items():
        st = prof.stall_by_op_warp.get((key, op), StallStats())
        sm, block, wib = key
        worows.append([sm, block, wib, op, opcode_category(op), ex.count, st.count, st.sum_gap_ns, st.mean(), st.max_gap_ns])
    worows.sort(key=lambda r: (-r[7], -r[5]))
    write_csv(out_prefix + ".warp_opcode.csv",
              ["sm", "block", "warp_in_block", "opcode", "category", "exec_count", "stall_count", "stall_sum_gap_ns", "stall_mean_ns", "stall_max_ns"],
              worows)

    # PC hotspots (global)
    pc_rows = []
    for pc, ex in prof.exec_by_pc.items():
        st = prof.stall_by_pc.get(pc, StallStats())
        pc_rows.append([f"0x{pc:08x}", ex.count, st.count, st.sum_gap_ns, st.mean(), st.max_gap_ns])
    pc_rows.sort(key=lambda r: -r[3])
    write_csv(out_prefix + ".pc_hotspots.csv",
              ["pc", "exec_count", "stall_count", "stall_sum_gap_ns", "stall_mean_ns", "stall_max_ns"],
              pc_rows[: max(2000, topk * 20)])  # keep more rows, still small

    # TXT report (topk)
    rpt_path = out_prefix + ".report.txt"
    with open(rpt_path, "w", encoding="utf-8") as fp:
        fp.write("=== Ventus cyclesim SM/warp profile ===\n")
        fp.write(f"output prefix : {out_prefix}\n")
        fp.write(f"instr_lines   : {prof.instr_lines}\n")
        fp.write(f"recv_lines    : {prof.recv_lines}\n")
        fp.write(f"jump_lines    : {prof.jump_lines}\n")
        fp.write(f"parse_errors  : {prof.parse_errors}\n")
        t0 = summ["time_window"]["t_min_ns"]
        t1 = summ["time_window"]["t_max_ns"]
        fp.write(f"time window   : {t0} ns -> {t1} ns (wall {summ['time_window']['wall_ns']} ns)\n\n")

        if mnk:
            nrm = summ["normalization"]
            fp.write(f"MNK           : M={nrm['M']} N={nrm['N']} K={nrm['K']}\n")
            fp.write(f"outputs       : {nrm['output_elements']}\n")
            fp.write(f"flops         : {nrm['flops']}\n")
            fp.write(f"stall_ns/out  : {nrm['stall_ns_per_out']:.3f}\n")
            fp.write(f"stall_ns/flop : {nrm['stall_ns_per_flop']:.6e}\n")
            fp.write(f"exec/out      : {nrm['exec_per_out']:.3f}\n\n")

        fp.write("=== SM summary ===\n")
        for sm in sorted(summ["sm_stats"].keys(), key=int):
            s = summ["sm_stats"][sm]
            fp.write(f"SM{sm}: wall={s['wall_ns']} ns | exec={s['exec_total']} | stall_sum={s['stall_sum_gap_ns']} ns\n")
        fp.write("\n")

        fp.write(f"=== Top-{topk} opcodes by stall_sum_gap (global, attributed to prev op) ===\n")
        top_ops = topk_from_map_stall(prof.stall_by_op, topk)
        for rank, (sumgap, op, st) in enumerate(top_ops, 1):
            fp.write(f"{rank:3d}. {op:12s} cat={opcode_category(op):7s} stall_sum={sumgap:12d} ns  "
                     f"count={st.count:10d} mean={st.mean():8.2f} ns p95~={st.p95():6d} ns max={st.max_gap_ns:10d} ns\n")
        fp.write("\n")

        fp.write(f"=== Top-{topk} PCs by stall_sum_gap (global, attributed to prev pc) ===\n")
        top_pcs = topk_from_map_stall(prof.stall_by_pc, topk, key_fn=lambda pc: f"0x{pc:08x}")
        for rank, (sumgap, pc_str, st) in enumerate(top_pcs, 1):
            fp.write(f"{rank:3d}. {pc_str:12s} stall_sum={sumgap:12d} ns  count={st.count:10d} mean={st.mean():8.2f} ns max={st.max_gap_ns:10d} ns\n")
        fp.write("\n")

        fp.write(f"=== Top-{topk} logical warps by stall_sum_gap ===\n")
        # sort warps by stall_total_gap_ns
        wlist = []
        for key, rt in prof.warp_rt.items():
            wlist.append((rt.stall_total_gap_ns, key, rt))
        wlist.sort(reverse=True, key=lambda x: x[0])
        for rank, (sumgap, key, rt) in enumerate(wlist[:topk], 1):
            sm, block, wib = key
            wall = (rt.last_t - rt.first_t) if (rt.first_t is not None and rt.last_t is not None) else 0
            fp.write(f"{rank:3d}. SM{sm} block={block} warp={wib}  stall_sum={sumgap:12d} ns  "
                     f"exec={rt.exec_total:10d} warp_wall={wall:12d} ns stall_mean={(sumgap/rt.stall_total_count) if rt.stall_total_count else 0.0:8.2f} ns\n")

    return {
        "summary_json": summ_path,
        "sm_opcode_csv": out_prefix + ".sm_opcode.csv",
        "warp_overview_csv": out_prefix + ".warp_overview.csv",
        "warp_opcode_csv": out_prefix + ".warp_opcode.csv",
        "pc_hotspots_csv": out_prefix + ".pc_hotspots.csv",
        "report_txt": rpt_path,
    }


def analyze_one(log_path: str, out_prefix: str, mnk: Optional[Tuple[int, int, int]], topk: int, show_progress: bool) -> Dict[str, Any]:
    prof = VentusProfiler(with_p95_hist=True)
    for line in iter_lines_with_progress(log_path, show_progress=show_progress):
        prof.parse_line(line)
    outputs = dump_analysis(prof, out_prefix, mnk=mnk, topk=topk)
    outputs["logfile"] = log_path
    return outputs


def read_summary_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def compare_two_summaries(sumA: Dict[str, Any], sumB: Dict[str, Any]) -> Dict[str, Any]:
    # minimal compare for wall time and normalization metrics
    out = {
        "A": sumA.get("time_window", {}),
        "B": sumB.get("time_window", {}),
        "A_sm": sumA.get("sm_stats", {}),
        "B_sm": sumB.get("sm_stats", {}),
        "A_norm": sumA.get("normalization", {}),
        "B_norm": sumB.get("normalization", {}),
    }
    return out


def diff_csv_sm_opcode(pathA: str, pathB: str, out_path: str) -> None:
    """
    Diff two *.sm_opcode.csv files and produce a delta table by (sm, opcode).
    """
    def load(path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
        m = {}
        with open(path, "r", encoding="utf-8") as fp:
            r = csv.DictReader(fp)
            for row in r:
                k = (row["sm"], row["opcode"])
                m[k] = {
                    "category": row["category"],
                    "exec_count": int(row["exec_count"]),
                    "stall_sum_gap_ns": int(row["stall_sum_gap_ns"]),
                    "stall_count": int(row["stall_count"]),
                    "stall_mean_ns": float(row["stall_mean_ns"]),
                    "stall_max_ns": int(row["stall_max_ns"]),
                    "stall_p95_approx_ns": int(row["stall_p95_approx_ns"]) if row.get("stall_p95_approx_ns") else 0,
                }
        return m

    A = load(pathA)
    B = load(pathB)

    keys = set(A.keys()) | set(B.keys())
    rows = []
    for k in keys:
        sm, op = k
        a = A.get(k, None)
        b = B.get(k, None)
        cat = (b or a)["category"] if (a or b) else "other"
        a_exec = a["exec_count"] if a else 0
        b_exec = b["exec_count"] if b else 0
        a_st = a["stall_sum_gap_ns"] if a else 0
        b_st = b["stall_sum_gap_ns"] if b else 0
        rows.append([sm, op, cat, a_exec, b_exec, b_exec - a_exec, a_st, b_st, b_st - a_st])

    rows.sort(key=lambda r: -r[8])
    write_csv(out_path,
              ["sm", "opcode", "category",
               "exec_A", "exec_B", "delta_exec",
               "stall_sum_A_ns", "stall_sum_B_ns", "delta_stall_sum_ns"],
              rows)


def main():
    ap = argparse.ArgumentParser(description="Ventus cyclesim SM/warp profiler (streaming, tqdm, diff).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_a = sub.add_parser("analyze", help="Analyze one log and output stats.")
    ap_a.add_argument("log", help="cyclesim log file path")
    ap_a.add_argument("--out", required=True, help="output prefix (e.g., out/A512)")
    ap_a.add_argument("--mnk", nargs=3, type=int, default=None, metavar=("M", "N", "K"),
                      help="optional: provide M N K for normalization metrics")
    ap_a.add_argument("--topk", type=int, default=60, help="topK entries in TXT report (default 60)")
    ap_a.add_argument("--no-progress", action="store_true", help="disable tqdm progress bar")

    ap_c = sub.add_parser("compare", help="Analyze two logs then diff key tables.")
    ap_c.add_argument("logA", help="log A (e.g., single-wg)")
    ap_c.add_argument("logB", help="log B (e.g., multi-wg / optimized)")
    ap_c.add_argument("--out", required=True, help="output prefix (e.g., out/diff512)")
    ap_c.add_argument("--mnk", nargs=3, type=int, default=None, metavar=("M", "N", "K"),
                      help="optional: provide M N K for normalization metrics")
    ap_c.add_argument("--topk", type=int, default=80, help="topK entries in TXT reports (default 80)")
    ap_c.add_argument("--no-progress", action="store_true", help="disable tqdm progress bar")

    args = ap.parse_args()
    show_progress = (not args.no_progress)

    if args.cmd == "analyze":
        mnk = tuple(args.mnk) if args.mnk else None
        outs = analyze_one(args.log, args.out, mnk=mnk, topk=args.topk, show_progress=show_progress)
        print("=== analyze done ===")
        for k, v in outs.items():
            if k != "logfile":
                print(f"output: {v}")
        return

    if args.cmd == "compare":
        mnk = tuple(args.mnk) if args.mnk else None
        outA = args.out + ".A"
        outB = args.out + ".B"

        outsA = analyze_one(args.logA, outA, mnk=mnk, topk=args.topk, show_progress=show_progress)
        outsB = analyze_one(args.logB, outB, mnk=mnk, topk=args.topk, show_progress=show_progress)

        sumA = read_summary_json(outA + ".summary.json")
        sumB = read_summary_json(outB + ".summary.json")

        # write a high-level diff txt
        diff_txt = args.out + ".diff_report.txt"
        with open(diff_txt, "w", encoding="utf-8") as fp:
            fp.write("=== Ventus compare report (A vs B) ===\n\n")
            fp.write(f"[A] {args.logA}\n")
            fp.write(f"  instr_lines : {sumA['parsed']['instr_lines']}\n")
            fp.write(f"  recv_lines  : {sumA['parsed']['recv_lines']}\n")
            fp.write(f"  jump_lines  : {sumA['parsed']['jump_lines']}\n")
            fp.write(f"  time_window : {sumA['time_window']['wall_ns']} ns (t0={sumA['time_window']['t_min_ns']} ns, t1={sumA['time_window']['t_max_ns']} ns)\n\n")

            fp.write(f"[B] {args.logB}\n")
            fp.write(f"  instr_lines : {sumB['parsed']['instr_lines']}\n")
            fp.write(f"  recv_lines  : {sumB['parsed']['recv_lines']}\n")
            fp.write(f"  jump_lines  : {sumB['parsed']['jump_lines']}\n")
            fp.write(f"  time_window : {sumB['time_window']['wall_ns']} ns (t0={sumB['time_window']['t_min_ns']} ns, t1={sumB['time_window']['t_max_ns']} ns)\n\n")

            if mnk:
                fp.write("=== Normalization (if provided MNK) ===\n")
                fp.write(f"A stall_ns/out  : {sumA['normalization'].get('stall_ns_per_out', 0.0):.3f}\n")
                fp.write(f"B stall_ns/out  : {sumB['normalization'].get('stall_ns_per_out', 0.0):.3f}\n")
                fp.write(f"Δ stall_ns/out  : {sumB['normalization'].get('stall_ns_per_out', 0.0) - sumA['normalization'].get('stall_ns_per_out', 0.0):.3f}\n\n")

            fp.write("=== SM summary ===\n")
            all_sms = set(sumA.get("sm_stats", {}).keys()) | set(sumB.get("sm_stats", {}).keys())
            for sm in sorted(all_sms, key=int):
                a = sumA.get("sm_stats", {}).get(sm, {"wall_ns": 0, "exec_total": 0, "stall_sum_gap_ns": 0})
                b = sumB.get("sm_stats", {}).get(sm, {"wall_ns": 0, "exec_total": 0, "stall_sum_gap_ns": 0})
                fp.write(f"SM{sm}: wall A/B={a['wall_ns']}/{b['wall_ns']} ns | exec A/B={a['exec_total']}/{b['exec_total']} | stall_sum A/B={a['stall_sum_gap_ns']}/{b['stall_sum_gap_ns']} ns\n")

            fp.write("\n")
            fp.write("Next: check *.diff_sm_opcode.csv for opcode-level stall deltas per SM.\n")

        # diff SM-opcode table
        diff_smop = args.out + ".diff_sm_opcode.csv"
        diff_csv_sm_opcode(outA + ".sm_opcode.csv", outB + ".sm_opcode.csv", diff_smop)

        print("=== compare done ===")
        print(f"output: {diff_txt}")
        print(f"output: {diff_smop}")
        print(f"output: {outA}.report.txt  (A detailed)")
        print(f"output: {outB}.report.txt  (B detailed)")
        print(f"output: {outA}.warp_overview.csv / {outB}.warp_overview.csv")
        print(f"output: {outA}.warp_opcode.csv   / {outB}.warp_opcode.csv")
        print(f"output: {outA}.pc_hotspots.csv   / {outB}.pc_hotspots.csv")
        return


if __name__ == "__main__":
    main()
