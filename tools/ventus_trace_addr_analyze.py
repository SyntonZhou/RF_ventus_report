import os
import csv
import json
import argparse
from collections import defaultdict

def try_import_tqdm():
    from tqdm import tqdm
    return tqdm

def parse_line(bline):
    b = bline.lstrip()
    if not b.startswith(b"SM"):
        return None

    n = len(b)
    i = 2

    if i < n and (48 <= b[i] <= 57):
        sm_start = i
        i += 1
        while i < n and (48 <= b[i] <= 57):
            i += 1
        sm = int(b[sm_start:i])
    else:
        while i < n and b[i] in b" \t":
            i += 1
        sm_start = i
        while i < n and (48 <= b[i] <= 57):
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
    while i < n and (48 <= b[i] <= 57):
        i += 1
    if i == w_start:
        return None
    warp = int(b[w_start:i])

    while i < n and b[i] in b" \t":
        i += 1
    if i >= n:
        return None

    if i + 1 < n and b[i] == ord('0') and b[i+1] == ord('x'):
        pc_start = i
        i += 2
        while i < n and (
            (48 <= b[i] <= 57) or (65 <= b[i] <= 70) or (97 <= b[i] <= 102)
        ):
            i += 1
        pc = b[pc_start:i]

        while i < n and b[i] in b" \t":
            i += 1
        t_start = i
        while i < n and b[i] not in b" \t\r\n":
            i += 1
        if i == t_start:
            return None
        token = b[t_start:i]

        cut = token.find(b"_0x")
        inst = token[:cut] if cut != -1 else token

        if not inst or inst.startswith(b"@"):
            return None

        return ("EXEC", sm, warp, pc, inst)

    t_start = i
    while i < n and b[i] not in b" \t\r\n":
        i += 1
    op = b[t_start:i]

    if op == b"JUMP":
        k = b.find(b"to", i)
        if k == -1:
            return None
        k += 2
        while k < n and b[k] in b" \t":
            k += 1
        if k + 1 >= n or b[k] != ord('0') or b[k+1] != ord('x'):
            return None
        pc_start = k
        k += 2
        while k < n and (
            (48 <= b[k] <= 57) or (65 <= b[k] <= 70) or (97 <= b[k] <= 102)
        ):
            k += 1
        target = b[pc_start:k]
        return ("JUMP_TO", sm, warp, target, b"JUMP")

    return None


def parse_sm_warp_pc_inst(bline):
    b = bline.lstrip()
    if not b.startswith(b"SM"):
        return None

    n = len(b)
    i = 2

    if i < n and (48 <= b[i] <= 57):
        sm_start = i
        i += 1
        while i < n and (48 <= b[i] <= 57):
            i += 1
        sm = int(b[sm_start:i])
    else:
        while i < n and b[i] in b" \t":
            i += 1
        sm_start = i
        while i < n and (48 <= b[i] <= 57):
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
    while i < n and (48 <= b[i] <= 57):
        i += 1
    if i == w_start:
        return None
    warp = int(b[w_start:i])

    k = b.find(b"0x", i)
    if k == -1:
        return None
    pc_start = k
    i = k + 2
    while i < n and (
        (48 <= b[i] <= 57) or (65 <= b[i] <= 70) or (97 <= b[i] <= 102)
    ):
        i += 1
    pc = b[pc_start:i]

    while i < n and b[i] in b" \t":
        i += 1
    t_start = i
    while i < n and b[i] not in b" \t\r\n":
        i += 1
    if i == t_start:
        return None
    token = b[t_start:i]

    cut = token.find(b"_0x")
    if cut != -1:
        inst = token[:cut]
    else:
        inst = token

    return sm, warp, pc, inst

def write_csv_kv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="trace日志路径（txt）")
    ap.add_argument("--out", default="trace_out", help="输出目录")
    ap.add_argument("--warps", type=int, default=8, help="warp数量（默认8：0..7）")
    ap.add_argument("--topk", type=int, default=50, help="输出TopK热点PC/指令")
    ap.add_argument("--progress_step_mb", type=int, default=8,
                    help="进度条更新粒度（MB），降低tqdm开销；默认8MB")
    args = ap.parse_args()

    log_path = args.log
    out_dir = args.out
    warps = args.warps
    topk = args.topk

    total_size = os.path.getsize(log_path)
    tqdm = try_import_tqdm()

    global_pc = defaultdict(int)
    global_inst = defaultdict(int)
    warp_pc = [defaultdict(int) for _ in range(warps)]
    warp_inst = [defaultdict(int) for _ in range(warps)]
    warp_total = [0] * warps
    jump_target = defaultdict(int)
    warp_jump_target = [defaultdict(int) for _ in range(warps)]

    pc2inst = {}

    bytes_read = 0
    bytes_since_update = 0
    step_bytes = max(1, args.progress_step_mb) * 1024 * 1024

    pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Parsing")

    with open(log_path, "rb", buffering=1024 * 1024) as f:
        for line in f:
            bytes_read += len(line)
            bytes_since_update += len(line)

            rec = parse_line(line)
            if rec is None:
                if pbar and bytes_since_update >= step_bytes:
                    pbar.update(bytes_since_update)
                    bytes_since_update = 0
                elif (not pbar) and bytes_since_update >= step_bytes:
                    pct = 100.0 * bytes_read / total_size
                    print(f"[INFO] progress: {pct:.2f}% ({bytes_read/1024/1024:.1f} MB read)")
                    bytes_since_update = 0
                continue

            kind = rec[0]
            if kind == "EXEC":
                _, sm, w, pc, inst = rec
                global_pc[pc] += 1
                global_inst[inst] += 1
                warp_pc[w][pc] += 1
                warp_inst[w][inst] += 1
                warp_total[w] += 1

            elif kind == "JUMP_TO":
                _, sm, w, target, inst = rec
                jump_target[target] += 1
                warp_jump_target[w][target] += 1

            if pc not in pc2inst:
                pc2inst[pc] = inst

            if pbar and bytes_since_update >= step_bytes:
                pbar.update(bytes_since_update)
                bytes_since_update = 0
            elif (not pbar) and bytes_since_update >= step_bytes:
                pct = 100.0 * bytes_read / total_size
                print(f"[INFO] progress: {pct:.2f}% ({bytes_read/1024/1024:.1f} MB read)")
                bytes_since_update = 0

    if pbar:
        if bytes_since_update > 0:
            pbar.update(bytes_since_update)
        pbar.close()

    total_dyn = sum(global_pc.values())
    print(f"[DONE] dynamic inst lines parsed = {total_dyn:,}")

    global_pc_sorted = sorted(global_pc.items(), key=lambda x: x[1], reverse=True)
    rows = []
    for pc, cnt in global_pc_sorted:
        inst = pc2inst.get(pc, b"")
        rows.append([pc.decode("ascii", "ignore"), cnt, inst.decode("ascii", "ignore")])
    write_csv_kv(os.path.join(out_dir, "global_pc_counts.csv"),
                 ["pc", "count", "inst_type_hint"], rows)

    rows = []
    for w in range(warps):
        for pc, cnt in warp_pc[w].items():
            rows.append([w, pc.decode("ascii", "ignore"), cnt])
    write_csv_kv(os.path.join(out_dir, "warp_pc_counts.csv"),
                 ["warp", "pc", "count"], rows)

    rows = []
    for w in range(warps):
        for inst, cnt in warp_inst[w].items():
            rows.append([w, inst.decode("ascii", "ignore"), cnt])
    write_csv_kv(os.path.join(out_dir, "warp_inst_counts.csv"),
                 ["warp", "inst_type", "count"], rows)

    global_total = sum(global_inst.values())
    global_ratio = {k: v / global_total for k, v in global_inst.items()}

    pref_rows = []
    for w in range(warps):
        if warp_total[w] == 0:
            continue
        for inst, cnt in warp_inst[w].items():
            wr = cnt / warp_total[w]
            gr = global_ratio.get(inst, 1e-18)
            pref = wr / gr if gr > 0 else 0.0
            pref_rows.append([w,
                              inst.decode("ascii", "ignore"),
                              cnt,
                              warp_total[w],
                              wr,
                              gr,
                              pref])
    pref_rows.sort(key=lambda x: x[-1], reverse=True)
    write_csv_kv(os.path.join(out_dir, "warp_inst_preference.csv"),
                 ["warp", "inst_type", "count", "warp_total", "warp_ratio", "global_ratio", "preference"],
                 pref_rows)

    skew_rows = []
    for pc, tot in global_pc.items():
        if tot <= 0:
            continue
        counts = [warp_pc[w].get(pc, 0) for w in range(warps)]
        m = max(counts)
        max_share = m / tot
        skew_rows.append([pc.decode("ascii", "ignore"),
                          tot,
                          max_share,
                          counts.index(m)])  # 哪个warp占比最大
    skew_rows.sort(key=lambda x: (x[2], x[1]), reverse=True)
    write_csv_kv(os.path.join(out_dir, "pc_warp_skew.csv"),
                 ["pc", "total_count", "max_warp_share", "argmax_warp"],
                 skew_rows)

    summary = {
        "log": os.path.abspath(log_path),
        "total_size_bytes": total_size,
        "dynamic_inst_lines": total_dyn,
        "warps": warps,
        "topk": topk,
        "top_global_pcs": [
            {"pc": pc.decode("ascii", "ignore"),
             "count": cnt,
             "inst_hint": pc2inst.get(pc, b"").decode("ascii", "ignore")}
            for pc, cnt in global_pc_sorted[:topk]
        ],
        "top_global_insts": [
            {"inst": inst.decode("ascii", "ignore"), "count": cnt}
            for inst, cnt in sorted(global_inst.items(), key=lambda x: x[1], reverse=True)[:topk]
        ],
        "warp_totals": {str(w): warp_total[w] for w in range(warps)},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[SUMMARY] per-warp top PCs:")
    for w in range(warps):
        items = sorted(warp_pc[w].items(), key=lambda x: x[1], reverse=True)[:10]
        if not items:
            continue
        print(f"  warp {w}: total={warp_total[w]:,}")
        for pc, cnt in items:
            print(f"    {pc.decode('ascii','ignore')}  {cnt:,}  inst={pc2inst.get(pc,b'').decode('ascii','ignore')}")

    print("\n[SUMMARY] per-warp top instruction preferences (top 5):")
    for w in range(warps):
        # 找出该warp的指令偏好Top5
        cand = [r for r in pref_rows if r[0] == w]
        cand = sorted(cand, key=lambda x: x[-1], reverse=True)[:5]
        if not cand:
            continue
        print(f"  warp {w}:")
        for r in cand:
            _, inst, cnt, wtot, wr, gr, pref = r
            print(f"    {inst:12s} pref={pref:8.3f}  warp_ratio={wr:.4f}  global_ratio={gr:.4f}  count={cnt:,}")

    print(f"\n[OUT] results saved to: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()
