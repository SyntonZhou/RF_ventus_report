# mmu_fault_probe.py
import re, argparse

RE_ALLOC = re.compile(r"vt_buf_alloc:.*vaddr_allocated=([0-9a-fA-F]+), size=0x([0-9a-fA-F]+)")
RE_MMU   = re.compile(r"MMU translate failed: SM(\d+)\s+warp(\d+)\s+ptroot=0x([0-9a-fA-F]+)\s+vaddr=0x([0-9a-fA-F]+)")
RE_INSTR = re.compile(r"^SM\s+(\d+)\s+warp\s+(\d+)\s+(0x[0-9a-fA-F]+)\s+([A-Za-z0-9_]+).*@(\d+)ns")

def find_region(vaddr, regions):
    for base, size, tag in regions:
        if base <= vaddr < base + size:
            return (tag, base, size, vaddr - base)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--max", type=int, default=20, help="report first N faults")
    args = ap.parse_args()

    regions = []  # (base, size, tag)
    last_instr = {}  # (sm,warp) -> (t_ns, pc, op, line)

    faults = []
    alloc_id = 0

    with open(args.log, "r", errors="ignore") as f:
        for line in f:
            m = RE_ALLOC.search(line)
            if m:
                base = int(m.group(1), 16)
                size = int(m.group(2), 16)
                alloc_id += 1
                tag = f"alloc#{alloc_id}"
                regions.append((base, size, tag))
                continue

            m = RE_INSTR.match(line)
            if m:
                sm = int(m.group(1)); wp = int(m.group(2))
                pc = int(m.group(3), 16)
                op = m.group(4)
                t  = int(m.group(5))
                last_instr[(sm,wp)] = (t, pc, op, line.strip())
                continue

            m = RE_MMU.search(line)
            if m:
                sm = int(m.group(1)); wp = int(m.group(2))
                ptroot = int(m.group(3), 16)
                vaddr  = int(m.group(4), 16)
                li = last_instr.get((sm,wp))
                region = find_region(vaddr, regions)
                faults.append((sm, wp, ptroot, vaddr, li, region))
                if len(faults) >= args.max:
                    break

    print(f"parsed alloc regions: {len(regions)}")
    if not faults:
        print("no MMU translate failed lines found.")
        return

    print(f"found {len(faults)} fault(s) (showing first {len(faults)})")
    for i,(sm,wp,ptroot,vaddr,li,region) in enumerate(faults,1):
        print(f"\n[{i}] SM{sm} warp{wp} ptroot=0x{ptroot:x} vaddr=0x{vaddr:x}")
        if region:
            tag, base, size, off = region
            print(f"  in region {tag}: base=0x{base:x} size=0x{size:x} offset=0x{off:x} ({off} bytes)")
        else:
            print("  vaddr not in any vt_buf_alloc region (likely runtime/other mapping).")

        if li:
            t, pc, op, lstr = li
            print(f"  last instr before fault: t={t}ns pc=0x{pc:x} op={op}")
            print(f"    {lstr}")
        else:
            print("  no preceding SM/warp instr captured (log ordering may differ).")

if __name__ == "__main__":
    main()
