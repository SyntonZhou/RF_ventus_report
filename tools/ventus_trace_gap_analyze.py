#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import math
import argparse
from collections import defaultdict
import heapq

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# 动态指令行（带 PC + opcode_0x... + @xxns）
INSTR_RE = re.compile(
    r"^SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+0x(?P<pc>[0-9a-fA-F]+)\s+"
    r"(?P<op>[A-Z0-9]+(?:_[A-Z0-9]+)*)_0x[0-9a-fA-F]+.*?@(?P<ns>\d+)ns"
)

def has_wb(line: str) -> bool:
    # 你的日志里 WB 通常会出现 " WB " 或 " WB x["
    return (" WB " in line) or (" WB x[" in line) or line.rstrip().endswith("WB")

def welford_update(state, x):
    # state: (n, mean, m2, min, max)
    n, mu, m2, mn, mx = state
    n += 1
    if x < mn: mn = x
    if x > mx: mx = x
    delta = x - mu
    mu += delta / n
    m2 += delta * (x - mu)
    return (n, mu, m2, mn, mx)

def welford_init():
    return (0, 0.0, 0.0, float("inf"), float("-inf"))

def welford_finalize(state):
    n, mu, m2, mn, mx = state
    if n <= 1:
        std = 0.0
    else:
        std = math.sqrt(m2 / (n - 1))
    if mn == float("inf"):
        mn = 0.0
    if mx == float("-inf"):
        mx = 0.0
    return n, mu, std, mn, mx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="trace log file")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--sm", type=int, default=None)
    ap.add_argument("--ns-per-cycle", type=int, default=10)
    ap.add_argument("--stall-cycles", type=int, default=50,
                    help="把 >= 该阈值的间隔视为 stall gap（用于分解/TopK）")
    ap.add_argument("--merge-wb-window-cycles", type=int, default=8,
                    help="同一(PC,op)在该窗口内出现非WB->WB，视为同一条指令，取WB时间")
    ap.add_argument("--focus-op", default="VAND_VV",
                    help="重点分析的 opcode（例如 VAND_VV）")
    ap.add_argument("--topk", type=int, default=200,
                    help="记录最大的 TopK stall 事件（带上下文）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    merge_window_ns = args.merge_wb_window_cycles * args.ns_per_cycle
    stall_ns = args.stall_cycles * args.ns_per_cycle

    # 每个 warp 的“待定事件”（可能会被后续 WB 合并）
    open_evt = [None] * args.warps   # (pc, op, ns, has_wb)
    # 每个 warp 的“已确认上一条事件”
    last_final = [None] * args.warps # (pc, op, ns)

    # opcode -> Welford stats on dt_ns（按 prev_op 归因）
    op_dt = defaultdict(welford_init)
    op_total_ns = defaultdict(int)

    # (prev_op -> next_op) 对 stall 的统计
    trans_stall_total = defaultdict(int)
    trans_stall_cnt = defaultdict(int)

    # (prev_pc, prev_op) 的 stall 热点
    pc_stall_total = defaultdict(int)
    pc_stall_cnt = defaultdict(int)

    # focus_op 的 next_op 分解（总 dt + stall dt）
    focus_next_total = defaultdict(int)
    focus_next_cnt = defaultdict(int)
    focus_next_stall_total = defaultdict(int)
    focus_next_stall_cnt = defaultdict(int)

    # TopK stall gaps（最大 dt）
    # 元素：(dt_ns, warp, prev_pc, prev_op, prev_ns, next_pc, next_op, next_ns)
    topk = []
    seen_lines = 0
    used_lines = 0

    fsize = os.path.getsize(args.log)
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=fsize, unit="B", unit_scale=True, desc="Scanning")

    def finalize_open(w):
        """把 open_evt[w] 确认为一个事件，并用它与 last_final[w] 形成一个 dt 样本"""
        nonlocal used_lines
        evt = open_evt[w]
        if evt is None:
            return
        pc, op, ns, _ = evt

        prev = last_final[w]
        if prev is not None:
            prev_pc, prev_op, prev_ns = prev
            dt = ns - prev_ns
            if dt >= 0:
                # 归因给 prev_op：prev_op 的“推进间隔/有效耗时”
                op_dt[prev_op] = welford_update(op_dt[prev_op], dt)
                op_total_ns[prev_op] += dt

                # stall 分解：只对长间隔做上下文统计
                if dt >= stall_ns:
                    key_trans = (prev_op, op)
                    trans_stall_total[key_trans] += dt
                    trans_stall_cnt[key_trans] += 1

                    key_pc = (hex(prev_pc), prev_op)
                    pc_stall_total[key_pc] += dt
                    pc_stall_cnt[key_pc] += 1

                    if prev_op == args.focus_op:
                        focus_next_stall_total[op] += dt
                        focus_next_stall_cnt[op] += 1

                    # TopK 记录
                    item = (dt, w, hex(prev_pc), prev_op, prev_ns, hex(pc), op, ns)
                    if len(topk) < args.topk:
                        heapq.heappush(topk, item)
                    else:
                        if dt > topk[0][0]:
                            heapq.heapreplace(topk, item)

                # focus_op 的 next 分解（不论是否 stall）
                if prev_op == args.focus_op:
                    focus_next_total[op] += dt
                    focus_next_cnt[op] += 1

        # 更新 last_final
        last_final[w] = (pc, op, ns)
        open_evt[w] = None
        used_lines += 1

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            seen_lines += 1
            if pbar is not None:
                pbar.update(len(line.encode("utf-8", errors="ignore")))
            m = INSTR_RE.match(line)
            if not m:
                continue

            sm = int(m.group("sm"))
            if args.sm is not None and sm != args.sm:
                continue

            w = int(m.group("warp"))
            if w < 0 or w >= args.warps:
                continue

            pc = int(m.group("pc"), 16)
            op = m.group("op")
            ns = int(m.group("ns"))
            wb = has_wb(line)

            cur = (pc, op, ns, wb)
            oe = open_evt[w]
            if oe is None:
                open_evt[w] = cur
                continue

            # 尝试合并：同一条指令的非WB行 -> WB行（例如 CSR 的两段打印）
            o_pc, o_op, o_ns, o_wb = oe
            if (pc == o_pc and op == o_op and (not o_wb) and wb and (ns - o_ns) <= merge_window_ns):
                # 合并到 open_evt：取 WB 时间作为该“同一条指令”的最终时间
                open_evt[w] = (o_pc, o_op, ns, True)
                continue

            # 否则：先 finalize 旧 open，再把 cur 设为新的 open
            finalize_open(w)
            open_evt[w] = cur

    if pbar is not None:
        pbar.close()

    # 文件结束：把所有 warp 的 open_evt flush
    for w in range(args.warps):
        finalize_open(w)

    # -------- 输出：opcode 时间统计（按 prev_op 的 dt） --------
    out_opcode = os.path.join(args.out, "opcode_dt_stats.csv")
    total_time_all = sum(op_total_ns.values()) or 1

    with open(out_opcode, "w", newline="", encoding="utf-8") as fw:
        wr = csv.writer(fw)
        wr.writerow(["opcode", "samples", "total_ns", "share", "mean_ns", "std_ns", "min_ns", "max_ns",
                     "mean_cycles", "std_cycles", "min_cycles", "max_cycles"])
        for op, st in sorted(op_dt.items(), key=lambda kv: op_total_ns[kv[0]], reverse=True):
            n, mu, sd, mn, mx = welford_finalize(st)
            tot = op_total_ns[op]
            wr.writerow([
                op, n, tot, tot / total_time_all,
                mu, sd, mn, mx,
                mu / args.ns_per_cycle, sd / args.ns_per_cycle, mn / args.ns_per_cycle, mx / args.ns_per_cycle
            ])

    # -------- 输出：focus_op 的 next-op 分解 --------
    out_focus = os.path.join(args.out, f"{args.focus_op}_next_breakdown.csv")
    with open(out_focus, "w", newline="", encoding="utf-8") as fw:
        wr = csv.writer(fw)
        wr.writerow(["focus_op", "next_op",
                     "total_ns", "count", "mean_ns",
                     "stall_total_ns", "stall_count", "stall_mean_ns"])
        for nop in sorted(focus_next_total.keys(), key=lambda x: focus_next_total[x], reverse=True):
            tot = focus_next_total[nop]
            cnt = focus_next_cnt[nop]
            stot = focus_next_stall_total.get(nop, 0)
            scnt = focus_next_stall_cnt.get(nop, 0)
            wr.writerow([
                args.focus_op, nop,
                tot, cnt, (tot / cnt if cnt else 0.0),
                stot, scnt, (stot / scnt if scnt else 0.0)
            ])

    # -------- 输出：TopK stall 事件（带上下文） --------
    out_topk = os.path.join(args.out, "top_stall_events.csv")
    topk_sorted = sorted(topk, key=lambda x: x[0], reverse=True)
    with open(out_topk, "w", newline="", encoding="utf-8") as fw:
        wr = csv.writer(fw)
        wr.writerow(["dt_ns", "dt_cycles", "warp",
                     "prev_pc", "prev_op", "prev_time_ns",
                     "next_pc", "next_op", "next_time_ns"])
        for dt, w, ppc, pop, pns, npc, nop, nns in topk_sorted:
            wr.writerow([dt, dt / args.ns_per_cycle, w, ppc, pop, pns, npc, nop, nns])

    # -------- 输出：stall 热点（prev_pc, prev_op） --------
    out_pc = os.path.join(args.out, "pc_stall_hotspots.csv")
    with open(out_pc, "w", newline="", encoding="utf-8") as fw:
        wr = csv.writer(fw)
        wr.writerow(["prev_pc", "prev_op", "stall_total_ns", "stall_count", "stall_mean_ns"])
        for (ppc, pop) in sorted(pc_stall_total.keys(), key=lambda k: pc_stall_total[k], reverse=True):
            tot = pc_stall_total[(ppc, pop)]
            cnt = pc_stall_cnt[(ppc, pop)]
            wr.writerow([ppc, pop, tot, cnt, tot / cnt if cnt else 0.0])

    # -------- 输出：stall 转移（prev_op -> next_op） --------
    out_tr = os.path.join(args.out, "stall_transitions.csv")
    with open(out_tr, "w", newline="", encoding="utf-8") as fw:
        wr = csv.writer(fw)
        wr.writerow(["prev_op", "next_op", "stall_total_ns", "stall_count", "stall_mean_ns"])
        for (p, n) in sorted(trans_stall_total.keys(), key=lambda k: trans_stall_total[k], reverse=True):
            tot = trans_stall_total[(p, n)]
            cnt = trans_stall_cnt[(p, n)]
            wr.writerow([p, n, tot, cnt, tot / cnt if cnt else 0.0])

    print(f"[DONE] scanned_lines={seen_lines}, used_events={used_lines}")
    print(f"[OUT] {out_opcode}")
    print(f"[OUT] {out_focus}")
    print(f"[OUT] {out_topk}")
    print(f"[OUT] {out_pc}")
    print(f"[OUT] {out_tr}")

if __name__ == "__main__":
    main()
