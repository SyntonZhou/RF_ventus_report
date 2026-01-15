#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import csv
from collections import defaultdict

# 匹配“执行轨迹行”里的 PC 与 op-token
# 例：SM 1 warp 0 0x800002ec           VADD_VX_0x02b047d7 ...
RE_SM_PC_OP = re.compile(r"\bSM\s+\d+\s+warp\s+\d+\s+(0x[0-9a-fA-F]+)\s+(\S+)")

# 某些行是 “SM .. warp .. JUMP to 0x8000....”，这种不是“PC=该地址的指令执行”
# 如果你想把 jump target 的出现也统计，可单独加；本脚本默认只统计“PC列”的地址。
RE_JUMP_TO = re.compile(r"\bJUMP\s+to\s+(0x[0-9a-fA-F]+)\b")

# op-token 常见形态：MNEMONIC_0xENC
RE_OP_WITH_ENC = re.compile(r"^(.*)_(0x[0-9a-fA-F]+)$")


def split_op_token(op_token: str):
    """
    将如 'VADD_VX_0x02b047d7' 拆成 (mnemonic, enc)
    若不符合该形态，则 enc=None，mnemonic=op_token
    """
    m = RE_OP_WITH_ENC.match(op_token)
    if not m:
        return op_token, None
    return m.group(1), m.group(2)


def classify_mnemonic(mn: str) -> str:
    """
    可选：给 mnemonic 粗分类，方便你做“指令存储表”的阅读。
    """
    u = mn.upper()
    if u.startswith("V") and ("LW" in u or "SW" in u):
        return "VLSU"
    if u in {"LW", "SW", "LH", "SH", "LB", "SB"} or u.endswith("LW") or u.endswith("SW"):
        return "LSU"
    if u.startswith("V") and ("FMA" in u or "FMADD" in u or "MUL" in u or "ADD" in u or "SLL" in u):
        return "VALU"
    if u in {"JAL", "JALR"} or u.startswith("J") or u.endswith("JUMP"):
        return "CTRL"
    if u.startswith("B") or u.startswith("VBEQ") or u.startswith("BEQ") or u.startswith("BGE"):
        return "BR"
    if "CSR" in u or u in {"SETRPC", "CSRRW", "CSRRWI", "CSRRS", "CSRRC"}:
        return "CSR"
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Path to the raw .log")
    ap.add_argument("--prefix", default="0x80", help="Only count PCs that start with this prefix (default: 0x80)")
    ap.add_argument("--csv", default="pc_map.csv", help="Output CSV path (default: pc_map.csv)")
    ap.add_argument("--txt", default="pc_map.txt", help="Output TXT path (default: pc_map.txt)")
    ap.add_argument("--progress-every", type=int, default=5_000_000,
                    help="Print progress every N lines (default: 5,000,000; set 0 to disable)")
    ap.add_argument("--also-count-jump-target", action="store_true",
                    help="Additionally count 'JUMP to 0x80...' occurrences into a separate section")
    args = ap.parse_args()

    prefix = args.prefix.lower()

    # pc_info[pc] = dict(count=..., mnemonics=set(...), encs=set(...), samples=[...])
    pc_count = defaultdict(int)
    pc_mn_set = defaultdict(set)
    pc_enc_set = defaultdict(set)
    pc_sample = {}  # store first sample line
    total_pc_hits = 0

    # 可选：统计 jump target 的出现次数（不是“该PC执行了多少次”，只是“被当作目标提到多少次”）
    jt_count = defaultdict(int)

    filesize = os.path.getsize(args.logfile)

    with open(args.logfile, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            m = RE_SM_PC_OP.search(line)
            if m:
                pc = m.group(1).lower()
                op_token = m.group(2)

                if pc.startswith(prefix):
                    total_pc_hits += 1
                    pc_count[pc] += 1

                    mnemonic, enc = split_op_token(op_token)
                    pc_mn_set[pc].add(mnemonic)
                    if enc:
                        pc_enc_set[pc].add(enc)

                    if pc not in pc_sample:
                        pc_sample[pc] = line.strip()

            if args.also_count_jump_target:
                mj = RE_JUMP_TO.search(line)
                if mj:
                    tgt = mj.group(1).lower()
                    if tgt.startswith(prefix):
                        jt_count[tgt] += 1

            if args.progress_every and (i % args.progress_every == 0):
                # 粗略进度（按行不精确，但足够用）
                # 真要精确可以用 f.tell()，Windows/编码下会稍微复杂，这里不做。
                print(f"[progress] lines={i:,}  unique_pc={len(pc_count):,}  total_pc_hits={total_pc_hits:,}")

    # 整理并排序
    pcs_sorted = sorted(pc_count.keys(), key=lambda x: int(x, 16))
    if not pcs_sorted:
        print("No PC entries matched. Check --prefix or log format.")
        return

    # 生成 CSV
    with open(args.csv, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["pc", "count", "share", "mnemonic", "mnemonic_set", "enc_set", "class", "sample_line"])
        for pc in pcs_sorted:
            cnt = pc_count[pc]
            share = cnt / total_pc_hits if total_pc_hits else 0.0
            mns = sorted(pc_mn_set[pc])
            encs = sorted(pc_enc_set[pc])
            # 理论上同一 pc 只有一个 mnemonic；若>1，说明你的日志里 op_token 解析或记录存在异常/混入
            mnemonic_main = mns[0] if mns else ""
            cls = classify_mnemonic(mnemonic_main)
            w.writerow([
                pc,
                cnt,
                f"{share:.8f}",
                mnemonic_main,
                "|".join(mns),
                "|".join(encs),
                cls,
                pc_sample.get(pc, "")
            ])

    # 生成 TXT（更像“指令存储位置表”）
    with open(args.txt, "w", encoding="utf-8") as tf:
        tf.write("Instruction Storage Map (PC -> instruction)\n")
        tf.write(f"prefix={args.prefix}  total_pc_hits={total_pc_hits}  unique_pc={len(pc_count)}\n")
        tf.write("Format: PC : count  share  class  mnemonic  enc(s)\n\n")
        for pc in pcs_sorted:
            cnt = pc_count[pc]
            share = cnt / total_pc_hits if total_pc_hits else 0.0
            mns = sorted(pc_mn_set[pc])
            encs = sorted(pc_enc_set[pc])
            mnemonic_main = mns[0] if mns else ""
            cls = classify_mnemonic(mnemonic_main)
            enc_str = ",".join(encs) if encs else "-"
            # 若 mnemonic 不唯一，标注冲突
            conflict = " [MNEMONIC_CONFLICT]" if len(mns) > 1 else ""
            tf.write(f"{pc} : {cnt:>10d}  {share:>8.4%}  {cls:>5s}  {mnemonic_main:>12s}  {enc_str}{conflict}\n")

        if args.also_count_jump_target:
            tf.write("\n\nJump-target mentions (NOT execution count):\n")
            for pc in sorted(jt_count.keys(), key=lambda x: int(x, 16)):
                tf.write(f"{pc} : {jt_count[pc]}\n")

    print(f"Done.\n  CSV: {args.csv}\n  TXT: {args.txt}\n  total_pc_hits={total_pc_hits:,}  unique_pc={len(pc_count):,}")


if __name__ == "__main__":
    main()
