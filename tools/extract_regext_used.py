#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
from collections import defaultdict, Counter

SM_PC_LINE = re.compile(
    r'^SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+(?P<pc>0x[0-9a-fA-F]+)\s+'
    r'(?P<rest>.+?)\s+@(?P<time>\d+)ns'
)

def parse_op_from_rest(rest: str) -> str:
    """
    rest 示例：
      "VADD_VV_0x02500357 mask=FFFFFFFF WB v[006]={...}"
      "VBEQ_0xfad78adb from VALU, current_mask=..."
      "REGEXT_0x0010200b REGEXT(s3,s2,s1,d)=..."
    """
    tok = rest.strip().split()[0]  # 取第一个 token，如 VADD_VV_0x..., REGEXT_0x...
    op = tok.split('_0x', 1)[0]    # 去掉 _0x... 编码后缀
    return op

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="输入日志文件路径（超大也可以）")
    ap.add_argument("--out", default="regext_out", help="输出目录")
    ap.add_argument("--limit_txt", type=int, default=0,
                    help="每类 txt 最多写多少条事件（0=不限制，避免文件太大可设如 20000）")
    args = ap.parse_args()

    ensure_dir(args.out)

    used_csv = os.path.join(args.out, "regext_used_events.csv")
    unused_csv = os.path.join(args.out, "regext_unused_events.csv")
    used_txt = os.path.join(args.out, "regext_used_context.txt")
    unused_txt = os.path.join(args.out, "regext_unused_context.txt")
    used_pair_csv = os.path.join(args.out, "regext_used_prev_next_pairs.csv")
    unused_pair_csv = os.path.join(args.out, "regext_unused_prev_next_pairs.csv")
    summary_txt = os.path.join(args.out, "regext_summary.txt")

    # 每个 (sm,warp) 维护：上一条“带pc的指令行”
    last_instr = {}  # (sm,warp) -> dict(line, pc, op, time)
    # 每个 (sm,warp) 维护：一个“悬而未决”的 REGEXT（等待下一条指令判断 used/unused）
    pending_regext = {}  # (sm,warp) -> dict(prev..., regext...)

    # 聚合统计：used/unused 下 (prev_op, next_op)
    used_pairs = Counter()
    unused_pairs = Counter()

    # 事件计数
    used_cnt = 0
    unused_cnt = 0

    # 写出器
    used_fields = [
        "event_id", "sm", "warp",
        "regext_time_ns", "regext_pc", "regext_op", "regext_line",
        "prev_time_ns", "prev_pc", "prev_op", "prev_line",
        "next_time_ns", "next_pc", "next_op", "next_line",
        "next_has_is_used"
    ]

    def open_csv_writer(path):
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=used_fields)
        w.writeheader()
        return f, w

    f_used, w_used = open_csv_writer(used_csv)
    f_unused, w_unused = open_csv_writer(unused_csv)

    f_used_txt = open(used_txt, "w", encoding="utf-8")
    f_unused_txt = open(unused_txt, "w", encoding="utf-8")

    event_id = 0
    used_txt_written = 0
    unused_txt_written = 0

    def flush_event(kind: str, key, reg, nxt):
        nonlocal event_id, used_cnt, unused_cnt, used_txt_written, unused_txt_written

        event_id += 1

        prev = reg.get("prev", {})
        reg_line = reg["line"]
        reg_pc = reg["pc"]
        reg_time = reg["time"]
        reg_op = reg["op"]

        next_line = nxt["line"]
        next_pc = nxt["pc"]
        next_time = nxt["time"]
        next_op = nxt["op"]

        next_has_is_used = 1 if nxt.get("has_is_used", False) else 0

        row = {
            "event_id": event_id,
            "sm": key[0],
            "warp": key[1],
            "regext_time_ns": reg_time,
            "regext_pc": reg_pc,
            "regext_op": reg_op,
            "regext_line": reg_line,
            "prev_time_ns": prev.get("time", ""),
            "prev_pc": prev.get("pc", ""),
            "prev_op": prev.get("op", ""),
            "prev_line": prev.get("line", ""),
            "next_time_ns": next_time,
            "next_pc": next_pc,
            "next_op": next_op,
            "next_line": next_line,
            "next_has_is_used": next_has_is_used
        }

        pair = (prev.get("op", ""), next_op)

        if kind == "used":
            used_cnt += 1
            used_pairs[pair] += 1
            w_used.writerow(row)
            if args.limit_txt == 0 or used_txt_written < args.limit_txt:
                f_used_txt.write("==========================================================================================\n")
                f_used_txt.write(f"[USED] EVENT#{event_id}  SM={key[0]} warp={key[1]}  REGEXT_pc={reg_pc} @ {reg_time}ns\n")
                if prev:
                    f_used_txt.write(f"  prev: pc={prev.get('pc','')} op={prev.get('op','')} @ {prev.get('time','')}ns\n")
                f_used_txt.write(f"  regext: {reg_line}\n")
                f_used_txt.write(f"  next: pc={next_pc} op={next_op} @ {next_time}ns  has_is_used={next_has_is_used}\n")
                f_used_txt.write("\n-- PREV LINE --\n" + prev.get("line","") + "\n")
                f_used_txt.write("\n-- REGEXT LINE --\n" + reg_line + "\n")
                f_used_txt.write("\n-- NEXT LINE --\n" + next_line + "\n\n")
                used_txt_written += 1
        else:
            unused_cnt += 1
            unused_pairs[pair] += 1
            w_unused.writerow(row)
            if args.limit_txt == 0 or unused_txt_written < args.limit_txt:
                f_unused_txt.write("==========================================================================================\n")
                f_unused_txt.write(f"[UNUSED] EVENT#{event_id}  SM={key[0]} warp={key[1]}  REGEXT_pc={reg_pc} @ {reg_time}ns\n")
                if prev:
                    f_unused_txt.write(f"  prev: pc={prev.get('pc','')} op={prev.get('op','')} @ {prev.get('time','')}ns\n")
                f_unused_txt.write(f"  regext: {reg_line}\n")
                f_unused_txt.write(f"  next: pc={next_pc} op={next_op} @ {next_time}ns  has_is_used={next_has_is_used}\n")
                f_unused_txt.write("\n-- PREV LINE --\n" + prev.get("line","") + "\n")
                f_unused_txt.write("\n-- REGEXT LINE --\n" + reg_line + "\n")
                f_unused_txt.write("\n-- NEXT LINE --\n" + next_line + "\n\n")
                unused_txt_written += 1

    def finalize_pending_at_eof():
        """文件结束时仍悬而未决的 REGEXT：记为 unused（没有 next 指令行）"""
        nonlocal event_id, unused_cnt
        for key, reg in list(pending_regext.items()):
            # next 为空，写一个占位
            nxt = {"line": "", "pc": "", "op": "", "time": "", "has_is_used": False}
            flush_event("unused", key, reg, nxt)
        pending_regext.clear()

    # 主循环：逐行读取，O(1) 内存，适合超大文件
    with open(args.log, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = SM_PC_LINE.match(line)
            if not m:
                continue  # 忽略没有 pc 的行（如 "SM 1 warp 0 JUMP to ..."）

            sm = int(m.group("sm"))
            warp = int(m.group("warp"))
            pc = m.group("pc")
            rest = m.group("rest")
            time_ns = int(m.group("time"))
            op = parse_op_from_rest(rest)

            key = (sm, warp)
            curr = {
                "line": line,
                "pc": pc,
                "op": op,
                "time": time_ns,
                "has_is_used": ("is used" in line)
            }

            # 如果当前是 REGEXT：挂起，等待同 warp 的下一条指令来判定 used/unused
            if op == "REGEXT":
                # 若之前还有 pending 没被结算（理论上不该发生），先把旧的按 unknown-next 记为 unused
                if key in pending_regext:
                    old = pending_regext.pop(key)
                    nxt_placeholder = {"line": "", "pc": "", "op": "", "time": "", "has_is_used": False}
                    flush_event("unused", key, old, nxt_placeholder)

                pending_regext[key] = {
                    "line": line,
                    "pc": pc,
                    "op": op,
                    "time": time_ns,
                    "prev": last_instr.get(key, {})
                }
                continue

            # 当前不是 REGEXT：如果存在 pending REGEXT，就用“当前指令”作为 next 来结算
            if key in pending_regext:
                reg = pending_regext.pop(key)
                kind = "used" if curr["has_is_used"] else "unused"
                flush_event(kind, key, reg, curr)

            # 更新 last_instr（只记录“带 pc 的真实指令行”）
            last_instr[key] = curr

    # EOF：仍未结算的 REGEXT
    finalize_pending_at_eof()

    # 输出 pair 统计
    def write_pairs(path, pairs: Counter):
        with open(path, "w", newline="", encoding="utf-8") as pf:
            w = csv.writer(pf)
            w.writerow(["prev_op", "next_op", "count"])
            for (p, n), c in pairs.most_common():
                w.writerow([p, n, c])

    write_pairs(used_pair_csv, used_pairs)
    write_pairs(unused_pair_csv, unused_pairs)

    # summary
    with open(summary_txt, "w", encoding="utf-8") as sf:
        sf.write(f"TOTAL REGEXT used   = {used_cnt}\n")
        sf.write(f"TOTAL REGEXT unused = {unused_cnt}\n")
        sf.write("\nTop used (prev_op -> next_op):\n")
        for (p, n), c in used_pairs.most_common(30):
            sf.write(f"  {p} -> {n}: {c}\n")
        sf.write("\nTop unused (prev_op -> next_op):\n")
        for (p, n), c in unused_pairs.most_common(30):
            sf.write(f"  {p} -> {n}: {c}\n")

    # 关闭文件
    f_used.close()
    f_unused.close()
    f_used_txt.close()
    f_unused_txt.close()

    print(f"[OK] out_dir = {args.out}")
    print(f"[OK] used={used_cnt}, unused={unused_cnt}")
    print(f"[OK] {used_csv}")
    print(f"[OK] {unused_csv}")
    print(f"[OK] {used_txt}")
    print(f"[OK] {unused_txt}")
    print(f"[OK] {used_pair_csv}")
    print(f"[OK] {unused_pair_csv}")
    print(f"[OK] {summary_txt}")

if __name__ == "__main__":
    main()
