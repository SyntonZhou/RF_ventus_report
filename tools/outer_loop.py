#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
import argparse
from collections import defaultdict
from statistics import mean, median

# 指令行（带 PC + opcode + @ns）
INSTR_RE = re.compile(
    r"SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+0x(?P<pc>[0-9a-fA-F]+)\s+"
    r"(?P<op>[A-Z0-9]+(?:_[A-Z0-9]+)?)_0x[0-9a-fA-F]+.*?@(?P<ns>\d+)ns"
)

# 指令行里带的 jumpTO（例如 JAL ... jumpTO 0x800002ec）
JUMPTO_INSTR_RE = re.compile(r"jumpTO\s+0x(?P<tgt>[0-9a-fA-F]+)", re.IGNORECASE)

# 独立的 “JUMP to ...” 行（你之前遇到的特殊格式）
JUMP_LINE_RE = re.compile(
    r"SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+JUMP\s+to\s+0x(?P<tgt>[0-9a-fA-F]+)\s+@(?P<ns>\d+)ns",
    re.IGNORECASE
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="trace log file")
    ap.add_argument("--out-prefix", default="loop_out", help="输出文件名前缀")
    ap.add_argument("--sm", type=int, default=None)
    ap.add_argument("--warp", type=int, default=None)

    ap.add_argument("--topk", type=int, default=20, help="输出 topK header（按出现次数）")
    ap.add_argument("--dump-iter", action="store_true", help="导出 header 迭代统计（CSV）")
    ap.add_argument("--header", type=lambda x: int(x, 16), default=None,
                    help="指定一个 header PC（十六进制，例如 0x800002ec）做更深入分析")
    ap.add_argument("--extract-iter", type=int, default=None,
                    help="导出指定 header 的第 N 次迭代（需要同时给 --header --sm --warp）")
    ap.add_argument("--extract-file", default="iter_slice.log",
                    help="迭代切片输出文件名")
    return ap.parse_args()

def main():
    args = parse_args()

    # -------- Pass 1: 统计 backward jump target，得到 header 候选 --------
    header_counts = defaultdict(int)

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = INSTR_RE.search(line)
            if m:
                sm = int(m.group("sm"))
                warp = int(m.group("warp"))
                if args.sm is not None and sm != args.sm:
                    continue
                if args.warp is not None and warp != args.warp:
                    continue

                pc = int(m.group("pc"), 16)
                j = JUMPTO_INSTR_RE.search(line)
                if j:
                    tgt = int(j.group("tgt"), 16)
                    # 只把 backward jump 当 loop header
                    if tgt < pc:
                        header_counts[tgt] += 1
                continue

            mj = JUMP_LINE_RE.search(line)
            if mj:
                sm = int(mj.group("sm"))
                warp = int(mj.group("warp"))
                if args.sm is not None and sm != args.sm:
                    continue
                if args.warp is not None and warp != args.warp:
                    continue
                tgt = int(mj.group("tgt"), 16)
                # 这类行没有 pc，无法判断 forward/backward；但在你日志里主要是回跳 header
                header_counts[tgt] += 1

    headers_sorted = sorted(header_counts.items(), key=lambda x: x[1], reverse=True)
    print("[PASS1] top loop headers by backward-jump target count:")
    for i, (h, c) in enumerate(headers_sorted[:args.topk]):
        print(f"  #{i:02d} header=0x{h:08x}  count={c}")

    # 保存 header 列表（txt）
    headers_txt = f"{args.out_prefix}_headers.txt"
    with open(headers_txt, "w", encoding="utf-8") as w:
        for h, c in headers_sorted:
            w.write(f"0x{h:08x}\t{c}\n")
    print(f"[OUT] {headers_txt}")

    if not args.dump_iter and args.header is None and args.extract_iter is None:
        return

    # -------- Pass 2: 针对 header 做迭代区间统计 --------
    # 若用户没指定 header，就对 topK header 都做
    target_headers = set()
    if args.header is not None:
        target_headers.add(args.header)
    else:
        target_headers = set(h for h, _ in headers_sorted[:args.topk])

    last_t = {}  # key=(sm,warp,header)->last_time_ns
    iter_rows = []  # (sm,warp,header,iter_id,start_ns,end_ns,iter_ns)

    # 用于导出某个迭代的切片
    extracting = False
    extract_header = args.header
    extract_sm = args.sm
    extract_warp = args.warp
    want_iter = args.extract_iter
    current_iter = -1
    slice_writer = None

    if want_iter is not None:
        if extract_header is None or extract_sm is None or extract_warp is None:
            raise SystemExit("要用 --extract-iter，必须同时给 --header --sm --warp")
        slice_writer = open(args.extract_file, "w", encoding="utf-8")

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
            ns = int(m.group("ns"))

            # 迭代切片：只对指定 header 的指定 warp
            if want_iter is not None and sm == extract_sm and warp == extract_warp:
                if pc == extract_header:
                    # 命中 header：一次新迭代开始
                    current_iter += 1
                    if current_iter == want_iter:
                        extracting = True
                    elif current_iter == want_iter + 1:
                        # 下一次迭代开始 -> 结束切片
                        extracting = False
                        break

                if extracting:
                    slice_writer.write(line)

            # 迭代统计
            if pc in target_headers:
                key = (sm, warp, pc)
                if key in last_t:
                    start = last_t[key]
                    end = ns
                    iter_id = None  # 这里不给全局 iter_id，只做区间记录；需要的话可扩展
                    iter_rows.append((sm, warp, pc, start, end, end - start))
                last_t[key] = ns

    if slice_writer:
        slice_writer.close()
        print(f"[OUT] iteration slice saved: {args.extract_file}")

    # 输出 CSV
    if args.dump_iter:
        iters_csv = f"{args.out_prefix}_iters.csv"
        with open(iters_csv, "w", newline="", encoding="utf-8") as w:
            cw = csv.writer(w)
            cw.writerow(["sm", "warp", "header", "start_ns", "end_ns", "iter_ns"])
            for sm, warp, h, s, e, d in iter_rows:
                cw.writerow([sm, warp, f"0x{h:08x}", s, e, d])
        print(f"[OUT] {iters_csv}")

    # 打印摘要（按 header/warp）
    by_hw = defaultdict(list)
    for sm, warp, h, s, e, d in iter_rows:
        by_hw[(sm, warp, h)].append(d)

    print("\n[PASS2] iteration time stats (ns):")
    for (sm, warp, h), ds in sorted(by_hw.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        print(f"  sm={sm} warp={warp} header=0x{h:08x}  n={len(ds)}  "
              f"mean={mean(ds):.2f}  med={median(ds):.2f}  min={min(ds)}  max={max(ds)}")

if __name__ == "__main__":
    main()
