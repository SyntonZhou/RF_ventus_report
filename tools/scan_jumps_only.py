#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, argparse
from collections import defaultdict

def try_import_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except Exception:
        return None

def parse_sm_warp_prefix(b):
    # 解析 SM / warp（兼容 "SM1" 和 "SM 1"）
    b = b.lstrip()
    if not b.startswith(b"SM"):
        return None
    n = len(b)
    i = 2
    # SM id
    if i < n and 48 <= b[i] <= 57:
        sm_start = i
        i += 1
        while i < n and 48 <= b[i] <= 57:
            i += 1
        sm = int(b[sm_start:i])
    else:
        while i < n and b[i] in b" \t":
            i += 1
        sm_start = i
        while i < n and 48 <= b[i] <= 57:
            i += 1
        if i == sm_start:
            return None
        sm = int(b[sm_start:i])

    j = b.find(b"warp", i)
    if j == -1:
        return None
    i = j + 4
    while i < n and b[i] in b" \t":
        i += 1
    w_start = i
    while i < n and 48 <= b[i] <= 57:
        i += 1
    if i == w_start:
        return None
    warp = int(b[w_start:i])
    return sm, warp

def extract_hex_after(b, key):
    # 找到 key 后的 0xHEX
    k = b.find(key)
    if k == -1:
        return None
    k += len(key)
    n = len(b)
    while k < n and b[k] in b" \t":
        k += 1
    if k + 1 >= n or b[k] != ord('0') or b[k+1] != ord('x'):
        return None
    start = k
    k += 2
    while k < n and (
        (48 <= b[k] <= 57) or (65 <= b[k] <= 70) or (97 <= b[k] <= 102)
    ):
        k += 1
    return b[start:k]

def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="jump_out")
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--include_jumpTO", action="store_true",
                    help="同时统计 'jumpTO 0x...'（条件跳转打印）")
    ap.add_argument("--progress_step_mb", type=int, default=16)
    args = ap.parse_args()

    log_path = args.log
    out_dir = args.out
    warps = args.warps
    topk = args.topk

    total_size = os.path.getsize(log_path)
    tqdm = try_import_tqdm()

    # 统计
    jump_to = defaultdict(int)
    warp_jump_to = [defaultdict(int) for _ in range(warps)]

    jumpTO = defaultdict(int)
    warp_jumpTO = [defaultdict(int) for _ in range(warps)]

    bytes_read = 0
    bytes_since = 0
    step = max(1, args.progress_step_mb) * 1024 * 1024

    if tqdm:
        pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="ScanJUMP")
    else:
        pbar = None
        print(f"[INFO] Scanning JUMP lines from {log_path} ({total_size/1024/1024:.2f} MB)")

    with open(log_path, "rb", buffering=1024 * 1024) as f:
        for line in f:
            bytes_read += len(line)
            bytes_since += len(line)

            # 快速过滤：必须含 "warp" 才可能是相关行
            if b"warp" not in line:
                pass
            else:
                prefix = parse_sm_warp_prefix(line)
                if prefix:
                    sm, w = prefix
                    # 1) JUMP to 0x....
                    if b" JUMP to " in line:
                        tgt = extract_hex_after(line, b"to")
                        if tgt and 0 <= w < warps:
                            jump_to[tgt] += 1
                            warp_jump_to[w][tgt] += 1

                    # 2) 可选：jumpTO 0x....
                    if args.include_jumpTO and b"jumpTO" in line:
                        tgt2 = extract_hex_after(line, b"jumpTO")
                        if tgt2 and 0 <= w < warps:
                            jumpTO[tgt2] += 1
                            warp_jumpTO[w][tgt2] += 1

            if pbar and bytes_since >= step:
                pbar.update(bytes_since)
                bytes_since = 0
            elif (not pbar) and bytes_since >= step:
                pct = 100.0 * bytes_read / total_size
                print(f"[INFO] progress {pct:.2f}%")
                bytes_since = 0

    if pbar:
        if bytes_since:
            pbar.update(bytes_since)
        pbar.close()

    # 输出 CSV：全局 target 分布
    jt_sorted = sorted(jump_to.items(), key=lambda x: x[1], reverse=True)
    write_csv(os.path.join(out_dir, "jump_to_targets.csv"),
              ["target", "count"],
              [(k.decode("ascii","ignore"), v) for k, v in jt_sorted])

    # 输出 CSV：按 warp 的 target 分布
    rows = []
    for w in range(warps):
        for tgt, cnt in warp_jump_to[w].items():
            rows.append([w, tgt.decode("ascii","ignore"), cnt])
    rows.sort(key=lambda x: x[2], reverse=True)
    write_csv(os.path.join(out_dir, "warp_jump_to_targets.csv"),
              ["warp", "target", "count"], rows)

    print(f"[DONE] JUMP to lines targets={len(jump_to)}; top{topk}:")
    for tgt, cnt in jt_sorted[:topk]:
        print(f"  {tgt.decode('ascii','ignore')}  {cnt:,}")

    if args.include_jumpTO:
        j2_sorted = sorted(jumpTO.items(), key=lambda x: x[1], reverse=True)
        write_csv(os.path.join(out_dir, "jumpTO_targets.csv"),
                  ["target", "count"],
                  [(k.decode("ascii","ignore"), v) for k, v in j2_sorted])
        rows2 = []
        for w in range(warps):
            for tgt, cnt in warp_jumpTO[w].items():
                rows2.append([w, tgt.decode("ascii","ignore"), cnt])
        rows2.sort(key=lambda x: x[2], reverse=True)
        write_csv(os.path.join(out_dir, "warp_jumpTO_targets.csv"),
                  ["warp", "target", "count"], rows2)

        print(f"[DONE] jumpTO targets={len(jumpTO)}; top{topk}:")
        for tgt, cnt in j2_sorted[:topk]:
            print(f"  {tgt.decode('ascii','ignore')}  {cnt:,}")

    print(f"[OUT] saved to {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()
