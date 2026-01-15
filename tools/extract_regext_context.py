#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import csv
import argparse
import gzip
from collections import defaultdict, deque

# 只匹配“真正的指令行”（包含：SM/warp/PC/OPCODE/_0x.../@xxns）
# 例：
# SM 1 warp 0 0x8000031c JAL_0xfd1ff06f ... @82785ns,1 ...
INSTR_RE = re.compile(
    r"^SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+0x(?P<pc>[0-9a-fA-F]+)\s+"
    r"(?P<op>[A-Z0-9]+)(?:_[A-Z0-9]+)?_0x[0-9a-fA-F]+.*?@(?P<ns>\d+)ns"
)

def open_maybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def hex_int(s: str) -> int:
    s = s.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    return int(s, 16)

def classify_mode(prev_pc, prev_op, next_pc, next_op, jal_pc):
    """
    mode1: prev=BGE@*  -> REGEXT@pc -> next=JAL@jal_pc
    mode2: prev=JAL@jal_pc -> REGEXT@pc -> next=JAL@jal_pc
    else: other
    """
    if prev_op == "BGE" and next_op == "JAL" and next_pc == jal_pc:
        return "mode1"
    if prev_op == "JAL" and next_op == "JAL" and prev_pc == jal_pc and next_pc == jal_pc:
        return "mode2"
    return "other"

def fmt_instr(rec):
    # rec: dict(sm, warp, pc, op, ns, line)
    return f"{rec['ns']:>12}ns  pc=0x{rec['pc']:08x}  {rec['op']}\n  {rec['line']}\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="trace log file (.log or .log.gz)")
    ap.add_argument("--outdir", default="regext_ctx_out", help="output directory")
    ap.add_argument("--regext-pc", default="0x800001b4", help="REGEXT PC to track, default 0x800001b4")
    ap.add_argument("--jal-pc", default="0x80000188", help="JAL PC used in mode判定, default 0x80000188")
    ap.add_argument("--before", type=int, default=40, help="how many previous instructions (same sm/warp)")
    ap.add_argument("--after", type=int, default=40, help="how many following instructions (same sm/warp)")
    ap.add_argument("--only", default="mode1,mode2",
                    help="comma-separated modes to dump: mode1,mode2,other or all. default mode1,mode2")
    args = ap.parse_args()

    regext_pc = hex_int(args.regext_pc)
    jal_pc = hex_int(args.jal_pc)

    want = set(x.strip() for x in args.only.split(",") if x.strip())
    if "all" in want:
        want = {"mode1", "mode2", "other"}

    os.makedirs(args.outdir, exist_ok=True)

    # per (sm,warp): last N instructions buffer
    hist = defaultdict(lambda: deque(maxlen=args.before))

    # per (sm,warp): queue of pending regext events waiting to collect "after" instructions
    pending = defaultdict(deque)

    # output files
    f_mode1 = open(os.path.join(args.outdir, "mode1_regext_context.txt"), "w", encoding="utf-8")
    f_mode2 = open(os.path.join(args.outdir, "mode2_regext_context.txt"), "w", encoding="utf-8")
    f_other = open(os.path.join(args.outdir, "other_regext_context.txt"), "w", encoding="utf-8")

    csv_path = os.path.join(args.outdir, "regext_events.csv")
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "event_id", "mode", "sm", "warp",
        "regext_time_ns", "regext_pc", "regext_op",
        "prev_time_ns", "prev_pc", "prev_op",
        "next_time_ns", "next_pc", "next_op"
    ])

    def dump_event(ev, mode):
        # select output handle
        if mode == "mode1":
            out = f_mode1
        elif mode == "mode2":
            out = f_mode2
        else:
            out = f_other

        if mode not in want:
            return

        out.write("=" * 90 + "\n")
        out.write(f"EVENT #{ev['event_id']}  mode={mode}  sm={ev['sm']}  warp={ev['warp']}  "
                  f"regext_pc=0x{ev['pc']:08x}  time={ev['ns']}ns\n")
        out.write(f"prev: time={ev['prev'].get('ns','?')}ns  pc=0x{ev['prev'].get('pc',0):08x}  op={ev['prev'].get('op','?')}\n")
        out.write(f"next: time={ev['next'].get('ns','?')}ns  pc=0x{ev['next'].get('pc',0):08x}  op={ev['next'].get('op','?')}\n")
        out.write("-" * 90 + "\n")
        out.write("[BEFORE instructions (same sm/warp)]\n")
        for r in ev["before"]:
            out.write(fmt_instr(r))
        out.write("[REGEXT line]\n")
        out.write(fmt_instr(ev["self"]))
        out.write("[AFTER instructions (same sm/warp)]\n")
        for r in ev["after_list"]:
            out.write(fmt_instr(r))
        out.write("\n")

    event_id = 0
    parsed_instr = 0
    regext_seen = 0

    with open_maybe_gz(args.log) as f:
        for line in f:
            m = INSTR_RE.match(line)
            if not m:
                continue

            parsed_instr += 1
            sm = int(m.group("sm"))
            warp = int(m.group("warp"))
            pc = int(m.group("pc"), 16)
            op = m.group("op")
            ns = int(m.group("ns"))
            key = (sm, warp)

            rec = {"sm": sm, "warp": warp, "pc": pc, "op": op, "ns": ns, "line": line.rstrip("\n")}

            # 1) 如果该 warp 有 pending regext 事件：当前指令作为其 AFTER 收集对象
            if pending[key]:
                ev0 = pending[key][0]
                if ev0["next"] is None:
                    ev0["next"] = rec  # 第一条 after 指令用于判定模式
                if len(ev0["after_list"]) < args.after:
                    ev0["after_list"].append(rec)

                # 收集够 after 条数就 finalize
                if len(ev0["after_list"]) >= args.after:
                    pending[key].popleft()
                    prev_rec = ev0["prev"] if ev0["prev"] is not None else {}
                    next_rec = ev0["next"] if ev0["next"] is not None else {}
                    mode = classify_mode(
                        prev_rec.get("pc", -1), prev_rec.get("op", ""),
                        next_rec.get("pc", -1), next_rec.get("op", ""),
                        jal_pc
                    )

                    # write CSV index
                    csv_w.writerow([
                        ev0["event_id"], mode, sm, warp,
                        ev0["ns"], f"0x{ev0['pc']:08x}", ev0["op"],
                        prev_rec.get("ns", ""), f"0x{prev_rec.get('pc',0):08x}", prev_rec.get("op", ""),
                        next_rec.get("ns", ""), f"0x{next_rec.get('pc',0):08x}", next_rec.get("op", ""),
                    ])

                    # dump context
                    dump_event(ev0, mode)

            # 2) 处理当前指令：如果是目标 REGEXT，则创建事件并进入 pending
            # 注意：prev 是 hist[key] 的最后一条（即 REGEXT 之前那条）
            if pc == regext_pc and op.startswith("REGEXT"):
                regext_seen += 1
                before_list = list(hist[key])  # snapshot
                prev_rec = before_list[-1] if before_list else None

                ev = {
                    "event_id": event_id,
                    "sm": sm, "warp": warp,
                    "pc": pc, "op": op, "ns": ns,
                    "before": before_list[-args.before:],  # 保证长度
                    "self": rec,
                    "prev": prev_rec,
                    "next": None,
                    "after_list": [],
                }
                pending[key].append(ev)
                event_id += 1

            # 3) 更新历史：把当前指令放入该 warp 的 hist
            hist[key].append(rec)

    # 结束时：可能有未收集够 after 的事件，也导出（next/after可能不足）
    def finalize_leftovers():
        for key, q in pending.items():
            for ev in q:
                prev_rec = ev["prev"] if ev["prev"] is not None else {}
                next_rec = ev["next"] if ev["next"] is not None else {}
                mode = classify_mode(
                    prev_rec.get("pc", -1), prev_rec.get("op", ""),
                    next_rec.get("pc", -1), next_rec.get("op", ""),
                    jal_pc
                )

                csv_w.writerow([
                    ev["event_id"], mode, ev["sm"], ev["warp"],
                    ev["ns"], f"0x{ev['pc']:08x}", ev["op"],
                    prev_rec.get("ns", ""), f"0x{prev_rec.get('pc',0):08x}", prev_rec.get("op", ""),
                    next_rec.get("ns", ""), f"0x{next_rec.get('pc',0):08x}", next_rec.get("op", ""),
                ])
                dump_event(ev, mode)

    finalize_leftovers()

    # close files
    f_mode1.close()
    f_mode2.close()
    f_other.close()
    csv_f.close()

    print(f"[DONE] parsed_instr_lines={parsed_instr} regext_seen={regext_seen}")
    print(f"[OUT] {args.outdir}")
    print(f"  - mode1_regext_context.txt")
    print(f"  - mode2_regext_context.txt")
    print(f"  - other_regext_context.txt")
    print(f"  - regext_events.csv")

if __name__ == "__main__":
    main()
