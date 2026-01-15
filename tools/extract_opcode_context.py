#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract context lines around a target opcode from huge Ventus cyclesim logs.

Usage example:
  python extract_opcode_context.py 512.log --op LW --context 10 --out LW_ctx.csv
  python extract_opcode_context.py 512.log --op VSW12_V --context 10 --every 1000 --max-hits 2000
"""

import os
import re
import csv
import argparse
from collections import deque
from tqdm import tqdm

# "SM 1 warp 2 ...."
RE_SM_WARP = re.compile(r'^SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+(?P<body>.*)$')
# "... @75ns,1"
RE_TIME = re.compile(r'@(?P<ns>\d+)ns,(?P<tag>\d+)')
# "0x80000000  AUIPC_0x00004197 ...."
RE_PC_OP = re.compile(r'(?P<pc>0x[0-9a-fA-F]+)\s+(?P<op>\S+)\s*(?P<rest>.*)$')
# "JUMP to 0x80000064 ..."
RE_JUMP = re.compile(r'JUMP\s+to\s+(?P<to>0x[0-9a-fA-F]+)')

def normalize_op(op_token: str) -> str:
    """
    Strip suffix like _0x1234abcd and trailing punctuation like ':'.
    """
    if op_token is None:
        return ""
    op = op_token.strip()
    op = op.rstrip(':')
    if "_0x" in op:
        op = op.split("_0x", 1)[0]
    return op

def parse_line(line: str):
    """
    Parse a log line into structured fields.
    Returns dict with keys:
      sm, warp, pc, op_raw, op, detail, time_ns, time_tag, time_str, raw, kind
    kind: INSN / JUMP / RECEIVE / OTHER
    """
    raw = line.rstrip("\n")
    out = {
        "sm": "", "warp": "", "pc": "",
        "op_raw": "", "op": "",
        "detail": "", "time_ns": "", "time_tag": "", "time_str": "",
        "raw": raw, "kind": "OTHER"
    }

    # time
    tm = RE_TIME.search(raw)
    if tm:
        out["time_ns"] = tm.group("ns")
        out["time_tag"] = tm.group("tag")
        out["time_str"] = f'{tm.group("ns")}ns,{tm.group("tag")}'

    m = RE_SM_WARP.match(raw)
    if not m:
        # Not an SM/warp line, keep as OTHER
        out["detail"] = raw
        return out

    out["sm"] = m.group("sm")
    out["warp"] = m.group("warp")
    body = m.group("body")

    # JUMP event line
    if "JUMP to" in body:
        out["kind"] = "JUMP"
        out["op_raw"] = "JUMP"
        out["op"] = "JUMP"
        out["detail"] = body
        return out

    # receive line
    if "receive kernel" in body:
        out["kind"] = "RECEIVE"
        out["op_raw"] = "RECEIVE"
        out["op"] = "RECEIVE"
        out["detail"] = body
        return out

    # normal instruction line: body begins with PC
    pm = RE_PC_OP.match(body)
    if pm:
        out["kind"] = "INSN"
        out["pc"] = pm.group("pc")
        op_raw = pm.group("op")
        out["op_raw"] = op_raw.rstrip(':')
        out["op"] = normalize_op(op_raw)
        out["detail"] = pm.group("rest")
        return out

    # fallback
    out["detail"] = body
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="Ventus cyclesim log file")
    ap.add_argument("--op", required=True, help="Target opcode name, e.g. LW / VSW12_V / VFMADD_VV")
    ap.add_argument("--context", type=int, default=10, help="Lines before/after each hit (default 10)")
    ap.add_argument("--out", default=None, help="Output CSV file path")
    ap.add_argument("--max-hits", type=int, default=0,
                    help="Stop after this many hits (0 = no limit). Strongly recommended for frequent ops.")
    ap.add_argument("--every", type=int, default=1,
                    help="Keep every k-th hit (default 1 = keep all). Useful for sampling.")
    ap.add_argument("--sm", type=int, default=None, help="Only consider hits from this SM (optional)")
    ap.add_argument("--warp", type=int, default=None, help="Only consider hits from this warp (optional)")
    args = ap.parse_args()

    target_op = args.op.strip()
    ctx = max(0, args.context)

    if args.out is None:
        base = os.path.basename(args.log)
        args.out = f"{base}.{target_op}.ctx{ctx}.csv"

    fsz = os.path.getsize(args.log)

    # ring buffer for previous lines (parsed)
    prev_buf = deque(maxlen=ctx)

    # Active contexts: each hit starts a "next-lines" capture for ctx lines after hit.
    # We allow overlaps by keeping multiple active contexts.
    active = []
    ctx_id = 0
    hit_total = 0  # total hits encountered (for --every sampling)
    hit_kept = 0   # hits actually kept (written)

    # CSV output
    fieldnames = [
        "ctx_id", "hit_index", "offset", "is_hit_line",
        "sm", "warp", "pc",
        "op", "op_raw", "kind",
        "time_ns", "time_tag", "time_str",
        "detail", "raw"
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as fw:
        w = csv.DictWriter(fw, fieldnames=fieldnames)
        w.writeheader()

        with open(args.log, "r", errors="ignore") as f, tqdm(total=fsz, unit="B", unit_scale=True) as pbar:
            for line in f:
                # progress: ASCII log, len(line) is good enough approximation
                pbar.update(len(line))

                parsed = parse_line(line)

                # 1) feed this line to all active contexts (as "after" lines)
                if active:
                    still = []
                    for c in active:
                        # write this line as next context line
                        row = {
                            "ctx_id": c["id"],
                            "hit_index": c["hit_index"],
                            "offset": c["next_offset"],
                            "is_hit_line": 0,
                            **{k: parsed.get(k, "") for k in ["sm","warp","pc","op","op_raw","kind","time_ns","time_tag","time_str","detail","raw"]}
                        }
                        w.writerow(row)

                        c["next_offset"] += 1
                        c["remain"] -= 1
                        if c["remain"] > 0:
                            still.append(c)
                    active = still

                # 2) check if this line is a hit (matches target opcode)
                is_sm_ok = (args.sm is None or (parsed["sm"] != "" and int(parsed["sm"]) == args.sm))
                is_warp_ok = (args.warp is None or (parsed["warp"] != "" and int(parsed["warp"]) == args.warp))

                is_hit = (parsed["op"] == target_op) and is_sm_ok and is_warp_ok

                if is_hit:
                    hit_total += 1

                    # sampling: keep every k-th hit
                    if args.every <= 1 or (hit_total % args.every == 0):
                        ctx_id += 1
                        hit_kept += 1

                        # write previous lines from ring buffer with negative offsets
                        prev_list = list(prev_buf)
                        # offsets: -len(prev_list) ... -1
                        for i, pl in enumerate(prev_list):
                            off = i - len(prev_list)
                            row = {
                                "ctx_id": ctx_id,
                                "hit_index": hit_kept,
                                "offset": off,
                                "is_hit_line": 0,
                                **{k: pl.get(k, "") for k in ["sm","warp","pc","op","op_raw","kind","time_ns","time_tag","time_str","detail","raw"]}
                            }
                            w.writerow(row)

                        # write current hit line with offset 0
                        row = {
                            "ctx_id": ctx_id,
                            "hit_index": hit_kept,
                            "offset": 0,
                            "is_hit_line": 1,
                            **{k: parsed.get(k, "") for k in ["sm","warp","pc","op","op_raw","kind","time_ns","time_tag","time_str","detail","raw"]}
                        }
                        w.writerow(row)

                        # activate capture of next ctx lines
                        if ctx > 0:
                            active.append({
                                "id": ctx_id,
                                "hit_index": hit_kept,
                                "remain": ctx,
                                "next_offset": 1
                            })

                        # stop if reached max hits
                        if args.max_hits and hit_kept >= args.max_hits:
                            break

                # 3) push current line into previous ring buffer (after hit handling)
                prev_buf.append(parsed)

    print(f"[DONE] target_op={target_op} context={ctx}")
    print(f"[DONE] hits_total_seen={hit_total} hits_kept={hit_kept}")
    print(f"[DONE] output_csv={args.out}")

if __name__ == "__main__":
    main()
