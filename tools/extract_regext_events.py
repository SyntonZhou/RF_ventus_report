#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, csv, argparse
from collections import deque

PC_RE = re.compile(rb"\b0x[0-9a-fA-F]+\b")
TS_RE = re.compile(rb"@(\d+)ns,")

def parse_exec_line(b):
    # 只匹配标准 EXEC 行："... warp <w> 0x<pc> <inst> ... @<t>ns,"
    b = b.strip()
    if not b.startswith(b"SM") or b" warp " not in b:
        return None
    parts = b.split()
    # 期望形如: SM 1 warp 2 0x8000.... INST...
    # 或: SM1 warp 2 0x....
    try:
        wi = parts.index(b"warp")
        w = int(parts[wi+1])
    except Exception:
        return None
    # warp 后紧跟的 token 必须是 PC
    if wi+2 >= len(parts) or not parts[wi+2].startswith(b"0x"):
        return None
    pc = parts[wi+2].decode("ascii", "ignore")
    inst = parts[wi+3].decode("ascii", "ignore") if wi+3 < len(parts) else ""
    m = TS_RE.search(b)
    t = int(m.group(1)) if m else -1
    return w, pc, inst, t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="regext_events.csv")
    ap.add_argument("--pc", default="0x800001b4")
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=6, help="每个warp保留多少条历史PC作为上下文")
    args = ap.parse_args()

    target_pc = args.pc.lower()
    warps = args.warps

    # 每个warp保留最近ctx条EXEC (pc,inst,t)
    hist = [deque(maxlen=args.ctx) for _ in range(warps)]
    # 记录“刚看到REGEXT，等待下一条EXEC补齐next_pc”
    pending = [None] * warps  # (time, pc, inst, prev_pc, prev_inst)

    rows = []
    with open(args.log, "rb", buffering=1024*1024) as f:
        for line in f:
            rec = parse_exec_line(line)
            if rec is None:
                continue
            w, pc, inst, t = rec
            if not (0 <= w < warps):
                continue

            # 若该warp上一个事件在等待next_pc，用当前EXEC作为next_pc补齐并落盘
            if pending[w] is not None:
                (t0, pc0, inst0, prev_pc, prev_inst) = pending[w]
                rows.append([w, t0, pc0, inst0, prev_pc, prev_inst, pc, inst, t])
                pending[w] = None

            prev_pc, prev_inst = ("", "")
            if len(hist[w]) > 0:
                prev_pc, prev_inst, _ = hist[w][-1]

            # 如果命中目标PC，记录并等待next
            if pc.lower() == target_pc:
                pending[w] = (t, pc, inst, prev_pc, prev_inst)

            hist[w].append((pc, inst, t))

    # 输出
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["warp", "time_ns", "pc", "inst", "prev_pc", "prev_inst",
                    "next_pc", "next_inst", "next_time_ns"])
        w.writerows(rows)

    print(f"[DONE] events={len(rows)} saved to {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
