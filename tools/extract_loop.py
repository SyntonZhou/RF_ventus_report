#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
from collections import defaultdict

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# 指令行（带 PC + opcode_0x... + @xxns）
INSTR_RE = re.compile(
    r"^SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+0x(?P<pc>[0-9a-fA-F]+)\s+"
    r"(?P<op>[A-Z0-9]+(?:_[A-Z0-9]+)*)_0x[0-9a-fA-F]+.*?@(?P<ns>\d+)ns"
)

# 控制流行：两种常见格式
# 1) "... JUMP to 0x80000064 @..."
# 2) "... jump=true, jumpTO 0x80000044 @..."
CTRL_RE = re.compile(
    r"^SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+).*?"
    r"(?:JUMP\s+to\s+0x(?P<t1>[0-9a-fA-F]+)|jumpTO\s+0x(?P<t2>[0-9a-fA-F]+)).*?@(?P<ns>\d+)ns"
)

def hex0(x: int) -> str:
    return "0x%08x" % x

def parse_hex(s: str) -> int:
    s = s.strip()
    if s.startswith("0x") or s.startswith("0X"):
        return int(s, 16)
    return int(s, 16)

def first_pass_find_headers(log_path: str, warps: int, sm_filter=None):
    """统计所有 warp 的 JUMP/jumpTO 目标次数，作为 loop header 候选。"""
    tgt_cnt = defaultdict(int)

    fsize = os.path.getsize(log_path)
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=fsize, unit="B", unit_scale=True, desc="Pass1: targets")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if pbar is not None:
                pbar.update(len(line.encode("utf-8", errors="ignore")))
            m = CTRL_RE.match(line)
            if not m:
                continue
            sm = int(m.group("sm"))
            if sm_filter is not None and sm != sm_filter:
                continue
            w = int(m.group("warp"))
            if w < 0 or w >= warps:
                continue
            t = m.group("t1") or m.group("t2")
            if t is None:
                continue
            tgt = int(t, 16)
            tgt_cnt[tgt] += 1

    if pbar is not None:
        pbar.close()
    return tgt_cnt

def second_pass_extract(
    log_path: str,
    out_dir: str,
    warps: int,
    header_pc: int,
    pick_warp: int,
    iters: int,
    skip_boundaries: int,
    sm_filter=None,
    include_ctrl=True,
):
    """
    以“跳转到 header 并到达 header”的事件作为迭代边界，
    抽取 pick_warp 的 iters 次迭代（默认 1 次）。
    """
    os.makedirs(out_dir, exist_ok=True)

    out_txt = os.path.join(out_dir, f"loop_iter_w{pick_warp}.txt")
    out_skel = os.path.join(out_dir, f"loop_skeleton_w{pick_warp}.csv")

    # 对每个 warp：记录最近一次“控制流目标”，用于判定下一条 PC 是否是“跳转到达”
    pending_target = [None] * warps  # int or None

    # 边界计数：跳回 header 的次数（按 pick_warp）
    boundary_seen = 0
    capturing = False
    captured_iters = 0

    # 循环骨架（只记录指令行）：RLE 压缩的 (pc, op, run_len)
    rle = []
    last_key = None
    last_run = 0

    def rle_push(pc, op):
        nonlocal last_key, last_run
        key = (pc, op)
        if last_key is None:
            last_key = key
            last_run = 1
        elif key == last_key:
            last_run += 1
        else:
            rle.append((last_key[0], last_key[1], last_run))
            last_key = key
            last_run = 1

    fsize = os.path.getsize(log_path)
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=fsize, unit="B", unit_scale=True, desc="Pass2: extract")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f, \
         open(out_txt, "w", encoding="utf-8") as fw:

        for line in f:
            if pbar is not None:
                pbar.update(len(line.encode("utf-8", errors="ignore")))

            # 先尝试控制流行
            mctrl = CTRL_RE.match(line)
            if mctrl:
                sm = int(mctrl.group("sm"))
                if sm_filter is not None and sm != sm_filter:
                    continue
                w = int(mctrl.group("warp"))
                if w < 0 or w >= warps:
                    continue
                t = mctrl.group("t1") or mctrl.group("t2")
                if t is None:
                    continue
                tgt = int(t, 16)
                pending_target[w] = tgt

                # 若正在捕获，只输出目标 warp 的控制流行
                if capturing and include_ctrl and w == pick_warp:
                    fw.write(line)
                continue

            # 再尝试指令行
            minst = INSTR_RE.match(line)
            if not minst:
                continue

            sm = int(minst.group("sm"))
            if sm_filter is not None and sm != sm_filter:
                continue

            w = int(minst.group("warp"))
            if w < 0 or w >= warps:
                continue

            pc = int(minst.group("pc"), 16)
            op = minst.group("op")

            jumped_here = (pending_target[w] == pc) if pending_target[w] is not None else False
            if jumped_here:
                pending_target[w] = None

            # 只用 pick_warp 的 header 回跳作为边界
            is_boundary = (w == pick_warp and pc == header_pc and jumped_here)

            if is_boundary:
                boundary_seen += 1
                if not capturing:
                    # 跳过前几次回跳（避开 warmup / 不稳定段）
                    if boundary_seen <= skip_boundaries:
                        continue
                    capturing = True
                    captured_iters = 0
                    fw.write(f"=== ITER_START #{captured_iters} at boundary_seen={boundary_seen} header={hex0(header_pc)} ===\n")
                    fw.write(line)  # 把 header 的第一条指令写进去
                    rle_push(pc, op)
                    continue
                else:
                    # 结束当前迭代，判断是否达到 iters
                    captured_iters += 1
                    if captured_iters >= iters:
                        fw.write(f"=== ITER_END #{captured_iters-1} at boundary_seen={boundary_seen} ===\n")
                        break
                    else:
                        fw.write(f"=== ITER_END #{captured_iters-1} at boundary_seen={boundary_seen} ===\n")
                        fw.write(f"=== ITER_START #{captured_iters} at boundary_seen={boundary_seen} header={hex0(header_pc)} ===\n")
                        fw.write(line)
                        rle_push(pc, op)
                        continue

            # 普通行：只在捕获期间输出目标 warp 的指令行
            if capturing and w == pick_warp:
                fw.write(line)
                rle_push(pc, op)

    if pbar is not None:
        pbar.close()

    # flush rle
    if last_key is not None:
        rle.append((last_key[0], last_key[1], last_run))

    # 输出 skeleton
    with open(out_skel, "w", newline="", encoding="utf-8") as fw:
        wr = csv.writer(fw)
        wr.writerow(["seq_idx", "pc", "opcode", "run_len"])
        for i, (pc, op, run_len) in enumerate(rle):
            wr.writerow([i, hex0(pc), op, run_len])

    return out_txt, out_skel, boundary_seen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--sm", type=int, default=None)

    ap.add_argument("--header", default=None,
                    help="手动指定 loop header PC，例如 0x800002ec；不指定则自动用回跳次数最多的目标")
    ap.add_argument("--warp", type=int, default=0, help="提取哪个 warp 的循环（建议 0）")
    ap.add_argument("--iters", type=int, default=1, help="提取多少次迭代（默认 1）")
    ap.add_argument("--skip", type=int, default=5, help="跳过前多少次回跳边界（避开 warmup）")
    ap.add_argument("--topk", type=int, default=20, help="输出 topK header 候选（回跳目标）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Pass1：找 header 候选
    tgt_cnt = first_pass_find_headers(args.log, args.warps, sm_filter=args.sm)
    cand_path = os.path.join(args.out, "loop_header_candidates.csv")
    top = sorted(tgt_cnt.items(), key=lambda kv: kv[1], reverse=True)[:args.topk]
    with open(cand_path, "w", newline="", encoding="utf-8") as fw:
        wr = csv.writer(fw)
        wr.writerow(["rank", "target_pc", "count"])
        for i, (tgt, cnt) in enumerate(top):
            wr.writerow([i, hex0(tgt), cnt])

    if args.header is None:
        if not top:
            raise SystemExit("未在日志中找到任何 JUMP/jumpTO 目标，无法自动推断 loop header。请手动 --header 0x....")
        header_pc = top[0][0]
    else:
        header_pc = parse_hex(args.header)

    # Pass2：抽取迭代 + skeleton
    out_txt, out_skel, boundary_seen = second_pass_extract(
        args.log, args.out, args.warps, header_pc, args.warp, args.iters, args.skip, sm_filter=args.sm
    )

    print(f"[HEADER] {hex0(header_pc)}")
    print(f"[CAND]   {cand_path}")
    print(f"[ITER]   {out_txt}")
    print(f"[SKEL]   {out_skel}")
    print(f"[INFO]   boundaries_seen_total_for_warp{args.warp}={boundary_seen}")

if __name__ == "__main__":
    main()
