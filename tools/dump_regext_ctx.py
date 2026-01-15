#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# 典型行：
# SM 1 warp 2 0x80000184               BGE_0x02905e63 ... @51035ns,1 [trace ...]
INSTR_RE = re.compile(
    r"SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+0x(?P<pc>[0-9a-fA-F]+)\s+"
    r"(?P<tok>[A-Z0-9]+(?:_[A-Z0-9]+)?_0x[0-9a-fA-F]+)"
    r".*?@(?P<ns>\d+)ns"
)

def parse_tok(tok: str) -> Tuple[str, Optional[int]]:
    """
    tok: e.g. 'BGE_0x02905e63', 'VADD_VV_0x024685d7'
    return: (opcode, enc_int_or_None)
    """
    # 按最后一次 '_0x' 分割最稳
    if "_0x" not in tok:
        return tok, None
    op, enc = tok.rsplit("_0x", 1)
    try:
        return op, int(enc, 16)
    except ValueError:
        return op, None

@dataclass
class InstMeta:
    pc: int
    op: str
    enc: Optional[int]
    ns: int
    raw: str

@dataclass
class RegextEvent:
    eid: int
    sm: int
    warp: int
    regext_pc: int
    regext_ns: int
    regext_raw: str
    pre_lines: List[str]
    prev_inst: Optional[InstMeta]
    post_lines: List[str] = field(default_factory=list)
    next_inst: Optional[InstMeta] = None
    remaining_post: int = 0

def is_mode1(prev: Optional[InstMeta], nxt: Optional[InstMeta],
             mode1_prev_pc: int, mode1_next_pc: int) -> bool:
    if not prev or not nxt:
        return False
    return (prev.op == "BGE" and prev.pc == mode1_prev_pc and
            nxt.op == "JAL" and nxt.pc == mode1_next_pc)

def is_mode2(prev: Optional[InstMeta], nxt: Optional[InstMeta],
             jal_pc: int, jal_enc: int) -> bool:
    if not prev or not nxt:
        return False
    return (prev.op == "JAL" and prev.pc == jal_pc and prev.enc == jal_enc and
            nxt.op == "JAL" and nxt.pc == jal_pc and nxt.enc == jal_enc)

def dump_event(fh, tag: str, ev: RegextEvent):
    fh.write(f"\n{'='*90}\n")
    fh.write(f"[{tag}] EVENT#{ev.eid}  SM={ev.sm} warp={ev.warp}  "
             f"REGEXT_pc=0x{ev.regext_pc:08x} @ {ev.regext_ns}ns\n")
    if ev.prev_inst:
        fh.write(f"  prev: pc=0x{ev.prev_inst.pc:08x} op={ev.prev_inst.op} "
                 f"enc={('0x%08x' % ev.prev_inst.enc) if ev.prev_inst.enc is not None else 'NA'} "
                 f"@{ev.prev_inst.ns}ns\n")
    else:
        fh.write("  prev: NA\n")
    if ev.next_inst:
        fh.write(f"  next: pc=0x{ev.next_inst.pc:08x} op={ev.next_inst.op} "
                 f"enc={('0x%08x' % ev.next_inst.enc) if ev.next_inst.enc is not None else 'NA'} "
                 f"@{ev.next_inst.ns}ns\n")
    else:
        fh.write("  next: NA\n")

    fh.write("\n-- PRE CONTEXT --\n")
    for ln in ev.pre_lines:
        fh.write(ln)

    fh.write("\n-- REGEXT LINE --\n")
    fh.write(ev.regext_raw)

    fh.write("\n-- POST CONTEXT --\n")
    for ln in ev.post_lines:
        fh.write(ln)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="trace log file")
    ap.add_argument("--sm", type=int, default=None, help="只导出指定 SM")
    ap.add_argument("--warp", type=int, default=None, help="只导出指定 warp")
    ap.add_argument("--pre", type=int, default=30, help="REGEXT 前保留多少行")
    ap.add_argument("--post", type=int, default=60, help="REGEXT 后保留多少行")
    # mode1/mode2 默认按你当前统计的定义
    ap.add_argument("--mode1-prev-pc", type=lambda x: int(x, 16), default=0x80000184)
    ap.add_argument("--mode1-next-pc", type=lambda x: int(x, 16), default=0x80000188)
    ap.add_argument("--mode2-jal-pc", type=lambda x: int(x, 16), default=0x80000188)
    ap.add_argument("--mode2-jal-enc", type=lambda x: int(x, 16), default=0x0040006f)
    ap.add_argument("--out-mode1", default="mode1_regext_context.txt")
    ap.add_argument("--out-mode2", default="mode2_regext_context.txt")
    args = ap.parse_args()

    # 每个 (sm,warp) 的前文环形缓冲
    prebuf = defaultdict(lambda: deque(maxlen=args.pre))
    last_inst = {}  # (sm,warp) -> InstMeta
    pending = defaultdict(deque)  # (sm,warp) -> deque[RegextEvent]

    eid = 0

    f_mode1 = open(args.out_mode1, "w", encoding="utf-8")
    f_mode2 = open(args.out_mode2, "w", encoding="utf-8")

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            m = INSTR_RE.search(raw)
            # 不是标准指令行：如果该 warp 正在抓 post，也要把这些行当语境写进去
            if not m:
                # 尝试粗略提取 SM/warp（有些非标准行如 "SM 1 warp 0 JUMP to ..." 也可能要进语境）
                # 这里用一个轻量匹配，避免太贵
                m2 = re.search(r"SM\s+(\d+)\s+warp\s+(\d+)", raw)
                if m2:
                    sm2, w2 = int(m2.group(1)), int(m2.group(2))
                    if (args.sm is not None and sm2 != args.sm) or (args.warp is not None and w2 != args.warp):
                        continue
                    key2 = (sm2, w2)
                    if pending[key2]:
                        for ev in list(pending[key2]):
                            if ev.remaining_post > 0:
                                ev.post_lines.append(raw)
                                ev.remaining_post -= 1
                                if ev.remaining_post == 0:
                                    # 没 next_inst 也照样结束，但一般会有
                                    if is_mode1(ev.prev_inst, ev.next_inst, args.mode1_prev_pc, args.mode1_next_pc):
                                        dump_event(f_mode1, "MODE1", ev)
                                    if is_mode2(ev.prev_inst, ev.next_inst, args.mode2_jal_pc, args.mode2_jal_enc):
                                        dump_event(f_mode2, "MODE2", ev)
                                    pending[key2].popleft()
                continue

            sm = int(m.group("sm"))
            warp = int(m.group("warp"))
            if args.sm is not None and sm != args.sm:
                continue
            if args.warp is not None and warp != args.warp:
                continue

            pc = int(m.group("pc"), 16)
            tok = m.group("tok")
            ns = int(m.group("ns"))
            op, enc = parse_tok(tok)

            key = (sm, warp)

            # 如果有 pending 事件，在“post”里追加当前行
            if pending[key]:
                for ev in list(pending[key]):
                    if ev.remaining_post > 0:
                        ev.post_lines.append(raw)
                        ev.remaining_post -= 1
                        # 记录 next_inst：REGEXT 之后第一条“非 REGEXT”的指令
                        if ev.next_inst is None and op != "REGEXT":
                            ev.next_inst = InstMeta(pc=pc, op=op, enc=enc, ns=ns, raw=raw)
                        if ev.remaining_post == 0:
                            if is_mode1(ev.prev_inst, ev.next_inst, args.mode1_prev_pc, args.mode1_next_pc):
                                dump_event(f_mode1, "MODE1", ev)
                            if is_mode2(ev.prev_inst, ev.next_inst, args.mode2_jal_pc, args.mode2_jal_enc):
                                dump_event(f_mode2, "MODE2", ev)
                            pending[key].popleft()

            # REGEXT：启动一个事件
            if op == "REGEXT":
                eid += 1
                ev = RegextEvent(
                    eid=eid,
                    sm=sm,
                    warp=warp,
                    regext_pc=pc,
                    regext_ns=ns,
                    regext_raw=raw,
                    pre_lines=list(prebuf[key]),
                    prev_inst=last_inst.get(key, None),
                    post_lines=[],
                    next_inst=None,
                    remaining_post=args.post
                )
                pending[key].append(ev)

            # 更新 last_inst（注意：REGEXT 也算一条“指令行”，但我们通常希望 prev_inst 是 REGEXT 前的那条非 REGEXT）
            # 所以这里选择：只有非 REGEXT 才更新 last_inst
            if op != "REGEXT":
                last_inst[key] = InstMeta(pc=pc, op=op, enc=enc, ns=ns, raw=raw)

            # 更新 prebuf：把当前行放入“前文缓冲”
            prebuf[key].append(raw)

    f_mode1.close()
    f_mode2.close()

    print(f"Done. mode1 -> {args.out_mode1}, mode2 -> {args.out_mode2}")

if __name__ == "__main__":
    main()
