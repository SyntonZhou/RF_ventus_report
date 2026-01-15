#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import argparse
from collections import defaultdict, deque

INSTR_RE = re.compile(r"SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+0x(?P<pc>[0-9a-fA-F]+)\s+(?P<rest>.*)")
IS_USED_RE = re.compile(r"is used to set")
OP_RE = re.compile(r"^\s*(?P<op>[A-Z0-9]+)(?:_[A-Z0-9]+)?_0x(?P<enc>[0-9a-fA-F]+)")

def get_op_enc(rest: str):
    m = OP_RE.match(rest.strip())
    if not m:
        return None, None
    return m.group("op"), int(m.group("enc"), 16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--sm", type=int, default=1)
    ap.add_argument("--lookahead", type=int, default=4, help="REGEXT后向前看多少行来判断是否used")
    args = ap.parse_args()

    # (warp) -> counts
    used = defaultdict(int)
    phantom = defaultdict(int)
    used_mode1 = defaultdict(int)
    phantom_mode1 = defaultdict(int)

    # 为判断 mode1：记录每个warp最近一次“已执行”的关键控制流
    last_bge184 = defaultdict(bool)   # 最近是否看到 BGE@0x80000184
    last_jal188 = defaultdict(bool)   # 最近是否看到 JAL@0x80000188
    last_bge_jump_false = defaultdict(bool)

    # 每个warp：最近 N 行缓存，方便遇到REGEXT时向后lookahead
    pending = defaultdict(deque)  # warp -> deque of (remain, is_mode1)

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = INSTR_RE.search(line)
            if not m:
                # 处理 pending 的 lookahead 消耗
                for w in list(pending.keys()):
                    if not pending[w]:
                        continue
                    # 如果这行属于该warp，才消耗
                    if f"warp {w} " in line and IS_USED_RE.search(line):
                        # 最近的那个 REGEXT 事件判为 used
                        remain, is_m1 = pending[w][0]
                        used[w] += 1
                        if is_m1:
                            used_mode1[w] += 1
                        pending[w].popleft()
                    elif f"warp {w} " in line:
                        remain, is_m1 = pending[w][0]
                        remain -= 1
                        pending[w][0] = (remain, is_m1)
                        if remain <= 0:
                            phantom[w] += 1
                            if is_m1:
                                phantom_mode1[w] += 1
                            pending[w].popleft()
                continue

            sm = int(m.group("sm"))
            if sm != args.sm:
                continue
            w = int(m.group("warp"))
            pc = int(m.group("pc"), 16)
            rest = m.group("rest")
            op, enc = get_op_enc(rest)

            # 先更新 mode1 语境状态（只用你关心的关键点）
            # 你给的mode1定义：prev=BGE@0x80000184，next=JAL@0x80000188，且BGE显示JUMP=false
            if pc == 0x80000184 and op == "BGE":
                last_bge184[w] = True
                last_jal188[w] = False
                last_bge_jump_false[w] = ("JUMP=false" in rest)
            elif pc == 0x80000188 and op == "JAL":
                last_jal188[w] = True

            # 处理 pending lookahead 消耗（当前行也算lookahead的一部分）
            if pending[w]:
                remain, is_m1 = pending[w][0]
                if IS_USED_RE.search(line):
                    used[w] += 1
                    if is_m1:
                        used_mode1[w] += 1
                    pending[w].popleft()
                else:
                    remain -= 1
                    pending[w][0] = (remain, is_m1)
                    if remain <= 0:
                        phantom[w] += 1
                        if is_m1:
                            phantom_mode1[w] += 1
                        pending[w].popleft()

            # 捕获 REGEXT
            if pc == 0x800001b4 and op == "REGEXT":
                is_mode1_ctx = bool(last_bge184[w] and last_jal188[w] and last_bge_jump_false[w])
                pending[w].append((args.lookahead, is_mode1_ctx))

    # 输出
    warps = sorted(set(list(used.keys()) + list(phantom.keys())))
    print("warp,used,phantom,used_mode1,phantom_mode1")
    for w in warps:
        print(f"{w},{used[w]},{phantom[w]},{used_mode1[w]},{phantom_mode1[w]}")

if __name__ == "__main__":
    main()
