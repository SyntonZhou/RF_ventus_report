#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
import argparse
from collections import defaultdict, deque
from statistics import mean, median

# 匹配“真正的指令行”（带 PC + 指令token + @xxns）
# 示例：
# SM 1 warp 1 0x80000000 AUIPC_0x00004197 WB ... @75ns,1
INSTR_RE = re.compile(
    r"SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+0x(?P<pc>[0-9a-fA-F]+)\s+"
    r"(?P<tok>[A-Z0-9]+(?:_[A-Z0-9]+)?)_0x[0-9a-fA-F]+.*?@(?P<ns>\d+)ns"
)

def gcd_list(nums):
    g = 0
    for x in nums:
        g = math.gcd(g, x)
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="trace log file")
    ap.add_argument("--warp", type=int, default=None, help="只统计指定 warp id")
    ap.add_argument("--sm", type=int, default=None, help="只统计指定 SM id")
    ap.add_argument("--ns-per-cycle", type=int, default=None, help="手动指定 1cycle=多少ns；不指定则自动探测")
    ap.add_argument("--mode", choices=["throughput", "latency", "both"], default="both",
                    help="throughput=按完成间隔算CPI；latency=issue->WB延迟；both=都输出")
    args = ap.parse_args()

    # 对“需要成对匹配 issue->WB”的指令，先记录 pending
    # key: (sm, warp, pc, opcode) -> queue of records
    pending = defaultdict(deque)

    # 已完成的“指令实例”列表（按时间顺序）
    # 每条：dict(opcode, sm, warp, issue_ns, done_ns)
    completed = []

    # 逐行扫描
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
            opcode = m.group("tok")
            ns = int(m.group("ns"))

            has_wb = (" WB " in line) or line.strip().endswith("WB") or (" WB" in line)
            is_store_or_branch_like = (" ADDR " in line) or ("JUMP=" in line) or ("SIMTSTK" in line)

            key = (sm, warp, pc, opcode)

            if has_wb:
                # 优先匹配一个之前的 issue
                if pending[key]:
                    rec = pending[key].popleft()
                    rec["done_ns"] = ns
                    completed.append(rec)
                else:
                    # 没看到 issue 行（或 issue 行被截断），就当 issue=done
                    completed.append({"sm": sm, "warp": warp, "pc": pc, "opcode": opcode, "issue_ns": ns, "done_ns": ns})
            else:
                if is_store_or_branch_like:
                    # 这类通常没有 WB，用该行时间作为 done
                    completed.append({"sm": sm, "warp": warp, "pc": pc, "opcode": opcode, "issue_ns": ns, "done_ns": ns})
                else:
                    # 可能是 load/CSR 等的 issue（后续会出现 WB）
                    pending[key].append({"sm": sm, "warp": warp, "pc": pc, "opcode": opcode, "issue_ns": ns, "done_ns": None})

    # 丢弃没等到 done 的（日志截断时会发生）
    completed = [r for r in completed if r["done_ns"] is not None]
    completed.sort(key=lambda r: r["done_ns"])

    if len(completed) < 2:
        print("有效完成指令太少（可能日志被截断或过滤太强），无法统计。")
        return

    # 自动探测 ns_per_cycle
    if args.ns_per_cycle is not None:
        ns_per_cycle = args.ns_per_cycle
    else:
        diffs = []
        for i in range(1, len(completed)):
            d = completed[i]["done_ns"] - completed[i-1]["done_ns"]
            if d > 0:
                diffs.append(d)
        ns_per_cycle = gcd_list(diffs) or 10  # 兜底10ns
    # ---- Throughput CPI：done间隔/周期 ----
    if args.mode in ("throughput", "both"):
        cpi_by_op = defaultdict(list)
        for i in range(1, len(completed)):
            d_ns = completed[i]["done_ns"] - completed[i-1]["done_ns"]
            cpi = d_ns / ns_per_cycle
            cpi_by_op[completed[i]["opcode"]].append(cpi)

        # 打印
        print(f"[Throughput CPI] ns_per_cycle={ns_per_cycle} ns")
        print(f"{'OPCODE':<16} {'COUNT':>8} {'AVG_CPI':>10} {'MED_CPI':>10} {'MIN':>8} {'MAX':>8}")
        for op in sorted(cpi_by_op.keys()):
            xs = cpi_by_op[op]
            print(f"{op:<16} {len(xs):>8} {mean(xs):>10.2f} {median(xs):>10.2f} {min(xs):>8.2f} {max(xs):>8.2f}")
        print()

    # ---- Latency：issue->done（主要对 load/长延迟指令有意义） ----
    if args.mode in ("latency", "both"):
        lat_by_op = defaultdict(list)
        for r in completed:
            lat_ns = r["done_ns"] - r["issue_ns"]
            if lat_ns > 0:
                lat_by_op[r["opcode"]].append(lat_ns / ns_per_cycle)

        print(f"[Latency issue->done] ns_per_cycle={ns_per_cycle} ns (仅统计 latency>0 的指令)")
        print(f"{'OPCODE':<16} {'COUNT':>8} {'AVG_LAT':>10} {'MED_LAT':>10} {'MIN':>8} {'MAX':>8}")
        for op in sorted(lat_by_op.keys()):
            xs = lat_by_op[op]
            print(f"{op:<16} {len(xs):>8} {mean(xs):>10.2f} {median(xs):>10.2f} {min(xs):>8.2f} {max(xs):>8.2f}")

if __name__ == "__main__":
    main()
