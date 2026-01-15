#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
import argparse
from collections import defaultdict, deque
import heapq
from math import ceil

# 匹配"真正的指令行"（带 PC + 指令token + @xxns）
INSTR_RE = re.compile(
    r"SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+0x(?P<pc>[0-9a-fA-F]+)\s+"
    r"(?P<tok>[A-Z0-9]+(?:_[A-Z0-9]+)?)_0x[0-9a-fA-F]+.*?@(?P<ns>\d+)ns"
)

class StreamStatistics:
    """流式统计，用于处理大规模数据"""
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.min = float('inf')
        self.max = float('-inf')
        self._mean = 0.0
        self._m2 = 0.0  # 用于计算方差的中间值
        self.distribution = defaultdict(int)  # 直方图分布
        self.samples = []  # 用于中位数的抽样
        
    def add(self, value):
        """流式添加数据"""
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        
        # 在线更新均值和方差
        delta = value - self._mean
        self._mean += delta / self.count
        delta2 = value - self._mean
        self._m2 += delta * delta2
        
        # 抽样用于中位数估计（保持最多10000个样本）
        if len(self.samples) < 10000:
            heapq.heappush(self.samples, value)
        else:
            # 随机替换策略
            if random.random() < 0.01:  # 1%概率替换
                heapq.heappushpop(self.samples, value)
    
    def add_to_distribution(self, value, bin_width=1.0):
        """添加到分布直方图"""
        if bin_width > 0:
            bin_idx = int(value / bin_width)
            self.distribution[bin_idx] += 1
    
    def mean(self):
        return self._mean
    
    def variance(self):
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)
    
    def median(self):
        """返回抽样中位数"""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        n = len(sorted_samples)
        if n % 2 == 1:
            return sorted_samples[n // 2]
        else:
            return (sorted_samples[n // 2 - 1] + sorted_samples[n // 2]) / 2

def gcd_list(nums):
    """计算列表的最大公约数"""
    g = 0
    for x in nums:
        g = math.gcd(g, x)
    return g

def process_file_in_chunks(file_path, chunk_size=1024*1024*100):
    """分块读取大文件"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        buffer = ''
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                if buffer:
                    yield buffer
                break
            buffer += chunk
            lines = buffer.split('\n')
            buffer = lines.pop()
            for line in lines:
                yield line

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="trace log file")
    ap.add_argument("--warp", type=int, default=None, help="只统计指定 warp id")
    ap.add_argument("--sm", type=int, default=None, help="只统计指定 SM id")
    ap.add_argument("--ns-per-cycle", type=int, default=None, help="手动指定 1cycle=多少ns；不指定则自动探测")
    ap.add_argument("--mode", choices=["throughput", "latency", "both"], default="both")
    ap.add_argument("--show-distribution", action="store_true", help="显示每个指令的时间分布统计")
    ap.add_argument("--distribution-bins", type=int, default=None, help="分布统计的区间数量")
    ap.add_argument("--top-n", type=int, default=100, help="只显示出现次数最多的前N个指令的详细分布")
    ap.add_argument("--sample-rate", type=float, default=1.0, 
                   help="采样率(0.0-1.0)，用于减少内存使用")
    args = ap.parse_args()

    import random  # 用于采样
    
    # 数据结构优化：使用更紧凑的数据结构
    pending = {}  # key: (sm, warp, pc, opcode) -> deque
    throughput_stats = defaultdict(StreamStatistics)  # opcode -> StreamStatistics
    latency_stats = defaultdict(StreamStatistics)    # opcode -> StreamStatistics
    
    # 用于自动探测ns_per_cycle
    time_diffs = []
    last_done_ns = None
    sample_counter = 0
    
    # 流式处理文件
    total_lines = 0
    processed_lines = 0
    
    # 首先，如果需要自动探测ns_per_cycle，先读取一部分数据
    if args.ns_per_cycle is None:
        print("正在探测ns_per_cycle...")
        probe_lines = 0
        for line in process_file_in_chunks(args.log, chunk_size=1024*1024*10):
            m = INSTR_RE.search(line)
            if m:
                probe_lines += 1
                if probe_lines > 100000:  # 只探测前10万行
                    break
                    
                ns = int(m.group("ns"))
                if last_done_ns is not None:
                    diff = ns - last_done_ns
                    if diff > 0:
                        time_diffs.append(diff)
                last_done_ns = ns
        
        if time_diffs:
            ns_per_cycle = gcd_list(time_diffs[:1000]) or 10  # 只取前1000个计算
        else:
            ns_per_cycle = 10
        print(f"探测到的ns_per_cycle: {ns_per_cycle}")
        
        # 重置以进行正式处理
        last_done_ns = None
    else:
        ns_per_cycle = args.ns_per_cycle
    
    print("开始处理主文件...")
    
    # 主处理循环
    for line in process_file_in_chunks(args.log, chunk_size=1024*1024*100):
        total_lines += 1
        
        # 采样以减少内存使用
        if args.sample_rate < 1.0 and random.random() > args.sample_rate:
            continue
            
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
        opcode = m.group("tok")
        ns = int(m.group("ns"))

        has_wb = (" WB " in line) or line.strip().endswith("WB") or (" WB" in line)
        is_store_or_branch_like = (" ADDR " in line) or ("JUMP=" in line) or ("SIMTSTK" in line)

        key = (sm, warp, pc, opcode)
        
        # Throughput统计：记录完成时间间隔
        if args.mode in ("throughput", "both"):
            if last_done_ns is not None:
                d_ns = ns - last_done_ns
                if d_ns > 0:
                    cpi = d_ns / ns_per_cycle
                    throughput_stats[opcode].add(cpi)
                    
                    if args.show_distribution:
                        # 使用动态bin宽度
                        bin_width = max(0.1, cpi / 10)  # 自适应bin宽度
                        throughput_stats[opcode].add_to_distribution(cpi, bin_width)
        
        # 更新上一个完成时间
        last_done_ns = ns
        
        # 延迟统计
        if args.mode in ("latency", "both"):
            if has_wb:
                # 优先匹配一个之前的issue
                if key in pending and pending[key]:
                    issue_ns = pending[key].popleft()
                    lat_ns = ns - issue_ns
                    if lat_ns > 0:
                        lat_cycles = lat_ns / ns_per_cycle
                        latency_stats[opcode].add(lat_cycles)
                        
                        if args.show_distribution:
                            bin_width = max(0.1, lat_cycles / 10)
                            latency_stats[opcode].add_to_distribution(lat_cycles, bin_width)
                else:
                    # 没有issue记录，当作即时完成
                    pass
            else:
                if is_store_or_branch_like:
                    # 这类指令通常即时完成
                    pass
                else:
                    # 记录issue时间
                    if key not in pending:
                        pending[key] = deque(maxlen=100)  # 限制队列长度
                    pending[key].append(ns)
        
        processed_lines += 1
        if processed_lines % 1000000 == 0:
            print(f"已处理 {processed_lines} 行...")
            # 定期清理pending队列（防止内存泄漏）
            if len(pending) > 100000:
                # 移除最老的条目
                keys_to_remove = list(pending.keys())[:50000]
                for k in keys_to_remove:
                    if not pending[k]:  # 只移除空队列
                        del pending[k]
    
    print(f"处理完成！总共处理了 {processed_lines} 行有效数据")
    
    # ---- 输出Throughput统计 ----
    if args.mode in ("throughput", "both") and throughput_stats:
        print(f"\n[Throughput CPI] ns_per_cycle={ns_per_cycle} ns")
        print(f"{'OPCODE':<16} {'COUNT':>8} {'AVG_CPI':>10} {'EST_MED':>10} {'MIN':>8} {'MAX':>8} {'VAR':>10}")
        print("-" * 80)
        
        # 按计数排序
        sorted_ops = sorted(throughput_stats.items(), key=lambda x: x[1].count, reverse=True)
        
        for op, stats in sorted_ops:
            if stats.count > 0:
                print(f"{op:<16} {stats.count:>8} {stats.mean():>10.2f} {stats.median():>10.2f} "
                      f"{stats.min:>8.2f} {stats.max:>8.2f} {stats.variance():>10.4f}")
        
        # 分布统计
        if args.show_distribution:
            print_distribution_summary(throughput_stats, "Throughput CPI", args.top_n)
    
    # ---- 输出Latency统计 ----
    if args.mode in ("latency", "both") and latency_stats:
        print(f"\n[Latency issue->done] ns_per_cycle={ns_per_cycle} ns")
        print(f"{'OPCODE':<16} {'COUNT':>8} {'AVG_LAT':>10} {'EST_MED':>10} {'MIN':>8} {'MAX':>8} {'VAR':>10}")
        print("-" * 80)
        
        sorted_ops = sorted(latency_stats.items(), key=lambda x: x[1].count, reverse=True)
        
        for op, stats in sorted_ops:
            if stats.count > 0:
                print(f"{op:<16} {stats.count:>8} {stats.mean():>10.2f} {stats.median():>10.2f} "
                      f"{stats.min:>8.2f} {stats.max:>8.2f} {stats.variance():>10.4f}")
        
        # 分布统计
        if args.show_distribution:
            print_distribution_summary(latency_stats, "Latency", args.top_n)

def print_distribution_summary(stats_dict, title, top_n):
    """打印分布统计摘要"""
    print(f"\n{title} Distribution Summary (Top {top_n}):")
    print("=" * 80)
    
    # 按计数排序取top N
    sorted_items = sorted(stats_dict.items(), key=lambda x: x[1].count, reverse=True)[:top_n]
    
    for op, stats in sorted_items:
        if not stats.distribution:
            continue
            
        print(f"\n{op} (Total: {stats.count}):")
        print("-" * 60)
        
        # 合并和格式化分布数据
        dist_items = sorted(stats.distribution.items())
        bins = []
        counts = []
        
        for bin_idx, count in dist_items:
            if count > 0:
                bins.append(bin_idx)
                counts.append(count)
        
        # 如果数据太多，合并显示
        if len(bins) > 20:
            merged_bins = []
            merged_counts = []
            merge_factor = ceil(len(bins) / 20)
            
            for i in range(0, len(bins), merge_factor):
                chunk_bins = bins[i:i+merge_factor]
                chunk_counts = counts[i:i+merge_factor]
                if chunk_bins:
                    merged_bins.append(f"{chunk_bins[0]}-{chunk_bins[-1]}")
                    merged_counts.append(sum(chunk_counts))
            
            bins = merged_bins
            counts = merged_counts
        
        # 打印表格
        for bin_label, count in zip(bins, counts):
            percentage = (count / stats.count) * 100
            print(f"  Bin {bin_label:<12}: {count:>8} ({percentage:>6.2f}%)")
    
    print("=" * 80)

if __name__ == "__main__":
    import random
    random.seed(42)  # 为了可重复的采样
    main()