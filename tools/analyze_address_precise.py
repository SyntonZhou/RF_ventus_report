#!/usr/bin/env python3
"""
Ventus指令地址偏好分析器 - 精确版
使用准确的指令提取方法和完整指令统计
"""

import re
import sys
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
import json
import math

class ProgressTracker:
    """进度跟踪器"""
    def __init__(self, total_bytes, desc="处理进度"):
        self.total_bytes = total_bytes
        self.desc = desc
        self.start_time = time.time()
        self.processed_bytes = 0
        self.last_update = 0
        self.update_interval = 2  # 每2秒更新一次
        self.line_count = 0
        
    def update(self, bytes_read, line_count=0):
        self.processed_bytes = bytes_read
        if line_count:
            self.line_count = line_count
        current_time = time.time()
        
        # 控制更新频率
        if current_time - self.last_update >= self.update_interval:
            self._display()
            self.last_update = current_time
    
    def _display(self):
        elapsed = time.time() - self.start_time
        percent = (self.processed_bytes / self.total_bytes * 100) if self.total_bytes > 0 else 0
        
        # 计算速度
        speed = self.processed_bytes / elapsed if elapsed > 0 else 0
        speed_mb = speed / (1024 * 1024)
        
        # 估算剩余时间
        remaining_bytes = self.total_bytes - self.processed_bytes
        eta = remaining_bytes / speed if speed > 0 else 0
        
        # 格式化为易读的时间
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta)
        
        # 显示进度
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        sys.stderr.write(f"\r{self.desc} |{bar}| {percent:.1f}% "
                         f"({self.processed_bytes/(1024**3):.1f}GB/{self.total_bytes/(1024**3):.1f}GB) "
                         f"行数: {self.line_count:,} "
                         f"速度: {speed_mb:.1f} MB/s 已用: {elapsed_str} 剩余: {eta_str}")
        sys.stderr.flush()
    
    def _format_time(self, seconds):
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def close(self):
        self._display()
        sys.stderr.write("\n")
        sys.stderr.flush()

def extract_instruction_info_precise(line):
    """
    精确提取指令信息
    基于您提供的提取方法
    """
    if not line.startswith('SM'):
        return None
    
    # 检查是否为JUMP指令
    if 'JUMP to' in line:
        # 提取JUMP指令的地址
        jump_match = re.search(r'(0x[0-9a-fA-F]{8})\s+JUMP\s+to', line)
        if jump_match:
            return {
                'address': jump_match.group(1),
                'instruction': 'JUMP',
                'raw_line': line[:100]  # 只保存前100个字符以节省内存
            }
        return None
    
    # 常规指令：匹配地址后的指令名_操作码格式
    # 例如：0x80000000 AUIPC_0x00004197
    pattern = re.compile(r'(0x[0-9a-fA-F]{8})\s+([A-Z][A-Z0-9_]+?)_[0-9a-fA-Fx]+')
    match = pattern.search(line)
    
    if not match:
        return None
    
    addr, instr = match.groups()
    
    # 过滤掉非指令的误匹配
    if instr.startswith(('MEMADDR', 'ADDR', 'DATA', 'mask')):
        return None
    
    # 提取时间信息
    time_match = re.search(r'@(\d+)ns,(\d+)', line)
    time_ns = int(time_match.group(1)) if time_match else 0
    
    # 提取warp信息
    warp_match = re.search(r'warp\s+(\d+)', line)
    warp_num = int(warp_match.group(1)) if warp_match else -1
    
    # 提取SM信息
    sm_match = re.search(r'SM\s*(\d+)', line)
    sm_num = int(sm_match.group(1)) if sm_match else -1
    
    return {
        'address': addr,
        'address_int': int(addr, 16),
        'instruction': instr,
        'time_ns': time_ns,
        'warp': warp_num,
        'sm': sm_num,
        'raw_line': line[:100]  # 只保存前100个字符
    }

def analyze_log_file_precise(log_file, max_lines=None, chunk_size=64*1024*1024):
    """
    精确分析日志文件，使用准确指令提取
    """
    print(f"开始精确分析文件: {log_file}", file=sys.stderr)
    print(f"文件大小: {os.path.getsize(log_file) / (1024**3):.2f} GB", file=sys.stderr)
    
    # 完整的指令列表（基于您提供的数据）
    known_instructions = {
        'VLW12_V', 'SETPRC', 'JAL', 'VBEQ', 'VMV_V_X', 'AUIPC', 'VADD_VV', 
        'VADD_VX', 'JUMP', 'VADD12_VI', 'ADDI', 'VFMADD_VV', 'VADD_VI', 
        'LW', 'VMSLT_VX', 'JOIN', 'VBLT', 'LUI', 'VSLL_VI', 'VAND_VV', 
        'VSW12_V', 'VMADD_VX', 'VBNE', 'BGE', 'REGEXT', 'VMUL_VX', 
        'CSRRS', 'SW', 'JALR', 'VLW_V', 'MUL', 'VSW_V', 'ADD', 'CSRRW', 
        'VID_V', 'VREMU_VX', 'VSETVLI', 'BEQ', 'VDIVU_VX', 'ENDPRG'
    }
    
    # 统计信息
    stats = {
        'total_bytes': os.path.getsize(log_file),
        'total_lines': 0,
        'instruction_lines': 0,
        'addresses_counter': Counter(),
        'instruction_counter': Counter(),
        'sm_warp_counter': defaultdict(Counter),
        'address_instr_counter': defaultdict(Counter),  # 地址 -> 指令计数
        'instr_address_counter': defaultdict(Counter),  # 指令 -> 地址计数
        'warp_address_counter': defaultdict(lambda: defaultdict(Counter)),  # warp -> 地址 -> 计数
        'address_range': [float('inf'), float('-inf')],
        'start_time': time.time(),
        'unknown_instructions': set(),
        'known_instructions_matched': set()
    }
    
    try:
        # 创建进度跟踪器
        progress = ProgressTracker(stats['total_bytes'], "精确分析")
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            buffer = ""
            bytes_read = 0
            last_progress_lines = 0
            
            while True:
                # 读取一块数据
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                bytes_read += len(chunk)
                buffer += chunk
                
                # 按行处理缓冲区
                lines = buffer.split('\n')
                buffer = lines[-1]  # 保留最后一行（可能不完整）
                
                for line in lines[:-1]:
                    stats['total_lines'] += 1
                    
                    # 更新进度（每10万行）
                    if stats['total_lines'] - last_progress_lines >= 100000:
                        progress.update(bytes_read, stats['total_lines'])
                        last_progress_lines = stats['total_lines']
                    
                    # 限制最大处理行数
                    if max_lines and stats['total_lines'] > max_lines:
                        progress.close()
                        print(f"\n已达到最大行数限制: {max_lines}", file=sys.stderr)
                        return stats
                    
                    # 提取指令信息
                    instr_info = extract_instruction_info_precise(line)
                    if not instr_info:
                        continue
                    
                    stats['instruction_lines'] += 1
                    
                    addr = instr_info['address']
                    addr_int = instr_info['address_int']
                    instr = instr_info['instruction']
                    warp = instr_info['warp']
                    
                    # 更新地址统计
                    stats['addresses_counter'][addr] += 1
                    
                    # 更新地址范围
                    if addr_int < stats['address_range'][0]:
                        stats['address_range'][0] = addr_int
                    if addr_int > stats['address_range'][1]:
                        stats['address_range'][1] = addr_int
                    
                    # 更新指令统计
                    stats['instruction_counter'][instr] += 1
                    
                    # 记录指令是否在已知列表中
                    if instr in known_instructions:
                        stats['known_instructions_matched'].add(instr)
                    else:
                        stats['unknown_instructions'].add(instr)
                    
                    # 更新地址-指令关联统计
                    stats['address_instr_counter'][addr][instr] += 1
                    stats['instr_address_counter'][instr][addr] += 1
                    
                    # 更新warp统计
                    if warp >= 0:
                        stats['sm_warp_counter'][instr_info['sm']][warp] += 1
                        stats['warp_address_counter'][warp][addr][instr] += 1
            
            # 处理缓冲区剩余内容
            if buffer:
                stats['total_lines'] += 1
                instr_info = extract_instruction_info_precise(buffer)
                if instr_info:
                    stats['instruction_lines'] += 1
                    stats['addresses_counter'][instr_info['address']] += 1
                    stats['instruction_counter'][instr_info['instruction']] += 1
        
        progress.close()
        
        # 计算处理时间
        stats['elapsed_time'] = time.time() - stats['start_time']
        
        print(f"\n分析完成!", file=sys.stderr)
        print(f"处理了 {stats['total_lines']:,} 行日志", file=sys.stderr)
        print(f"其中 {stats['instruction_lines']:,} 行是指令行", file=sys.stderr)
        print(f"处理时间: {stats['elapsed_time']:.1f} 秒", file=sys.stderr)
        print(f"平均速度: {stats['total_lines']/stats['elapsed_time']:,.0f} 行/秒", file=sys.stderr)
        
        # 指令匹配统计
        print(f"\n指令匹配统计:", file=sys.stderr)
        print(f"  已知指令匹配数: {len(stats['known_instructions_matched'])}", file=sys.stderr)
        print(f"  未知指令数: {len(stats['unknown_instructions'])}", file=sys.stderr)
        
        if stats['unknown_instructions']:
            print(f"  未知指令示例: {list(stats['unknown_instructions'])[:10]}", file=sys.stderr)
        
    except KeyboardInterrupt:
        print(f"\n分析被用户中断，已处理 {stats['total_lines']:,} 行", file=sys.stderr)
        if 'progress' in locals():
            progress.close()
    except Exception as e:
        print(f"\n分析过程中出错: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None
    
    return stats

def analyze_instruction_address_preferences(stats):
    """
    分析指令地址偏好
    """
    print(f"\n🔍 指令地址偏好分析")
    print(f"{'='*80}")
    
    # 1. 每个地址的指令分布
    print(f"\n📍 每个地址的指令类型分布:")
    
    # 按访问次数排序地址
    sorted_addresses = sorted(
        stats['addresses_counter'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for addr, total_count in sorted_addresses[:15]:  # 显示前15个地址
        instr_dist = stats['address_instr_counter'][addr]
        
        print(f"\n  {addr} (总执行{total_count:,}次):")
        
        # 按指令出现次数排序
        sorted_instrs = sorted(instr_dist.items(), key=lambda x: x[1], reverse=True)
        
        for instr, count in sorted_instrs[:5]:  # 显示前5种指令
            percentage = count / total_count * 100
            print(f"    {instr:15s}: {count:8,} ({percentage:6.1f}%)")
        
        if len(sorted_instrs) > 5:
            print(f"    ... 还有 {len(sorted_instrs) - 5} 种其他指令")
    
    # 2. 每个指令的地址分布
    print(f"\n📝 每个指令类型的地址分布:")
    
    # 按指令出现次数排序
    sorted_instructions = sorted(
        stats['instruction_counter'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for instr, total_count in sorted_instructions[:15]:  # 显示前15种指令
        addr_dist = stats['instr_address_counter'][instr]
        
        print(f"\n  {instr:15s} (总执行{total_count:,}次):")
        
        # 按地址出现次数排序
        sorted_addrs = sorted(addr_dist.items(), key=lambda x: x[1], reverse=True)
        
        for addr, count in sorted_addrs[:3]:  # 显示前3个地址
            percentage = count / total_count * 100
            print(f"    {addr}: {count:8,} ({percentage:6.1f}%)")
        
        # 计算地址集中度
        if sorted_addrs:
            top_addr_count = sorted_addrs[0][1]
            concentration = top_addr_count / total_count * 100
            print(f"    地址集中度: {concentration:.1f}% 的指令在单个地址执行")
            
            # 计算地址多样性
            addr_diversity = len(sorted_addrs)
            print(f"    地址多样性: {addr_diversity} 个不同地址")
    
    # 3. 特殊指令模式分析
    print(f"\n🧮 特殊指令模式分析:")
    
    # 查找只在单一地址执行的指令
    single_address_instrs = []
    for instr, addr_dist in stats['instr_address_counter'].items():
        if len(addr_dist) == 1:
            addr, count = list(addr_dist.items())[0]
            single_address_instrs.append((instr, addr, count))
    
    if single_address_instrs:
        print(f"  只在单一地址执行的指令 ({len(single_address_instrs)} 种):")
        for instr, addr, count in sorted(single_address_instrs, key=lambda x: x[2], reverse=True)[:10]:
            print(f"    {instr:15s}: {addr} ({count:,} 次)")
    else:
        print(f"  没有只在单一地址执行的指令")
    
    # 查找在多个地址执行的指令
    multi_address_instrs = []
    for instr, addr_dist in stats['instr_address_counter'].items():
        if len(addr_dist) > 3:  # 在超过3个地址执行
            multi_address_instrs.append((instr, len(addr_dist), sum(addr_dist.values())))
    
    if multi_address_instrs:
        print(f"\n  在多个地址执行的指令 (超过3个地址):")
        for instr, addr_count, total_count in sorted(multi_address_instrs, key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {instr:15s}: {addr_count} 个不同地址，总执行{total_count:,}次")
    
    # 4. 地址热点模式分析
    print(f"\n🔥 地址热点模式分析:")
    
    # 按地址执行次数排序
    hot_addresses = sorted_addresses[:20]
    
    for rank, (addr, total_count) in enumerate(hot_addresses, 1):
        # 获取该地址的主要指令
        instr_dist = stats['address_instr_counter'][addr]
        main_instrs = sorted(instr_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        
        main_instr_str = ", ".join([f"{instr}({count:,})" for instr, count in main_instrs])
        
        print(f"  {rank:2d}. {addr}: {total_count:,} 次 - 主要指令: {main_instr_str}")
        
        # 分析是否为2的幂次
        if total_count > 0:
            log2_val = math.log2(total_count)
            if log2_val.is_integer():
                print(f"        👉 执行次数是2的幂: 2^{int(log2_val)} = {total_count:,}")
    
    # 5. Warp地址偏好分析
    print(f"\n🌀 Warp地址偏好分析:")
    
    warps = sorted(stats['warp_address_counter'].keys())
    print(f"  共 {len(warps)} 个warp")
    
    for warp in warps[:min(8, len(warps))]:  # 最多显示8个warp
        warp_stats = stats['warp_address_counter'][warp]
        
        # 计算warp的总指令数
        warp_total = sum(
            sum(instr_dist.values()) 
            for addr_dist in warp_stats.values() 
            for instr_dist in [addr_dist]
        )
        
        print(f"\n  Warp {warp} (总指令: {warp_total:,}):")
        
        # 获取warp的热门地址
        warp_address_counts = Counter()
        for addr, instr_dist in warp_stats.items():
            warp_address_counts[addr] = sum(instr_dist.values())
        
        # 显示warp的前5个热门地址
        for addr, count in warp_address_counts.most_common(5):
            percentage = count / warp_total * 100
            global_count = stats['addresses_counter'][addr]
            global_percentage = global_count / stats['instruction_lines'] * 100
            
            # 获取该地址在warp中的主要指令
            main_instrs = sorted(
                warp_stats[addr].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:2]
            
            main_instr_str = ", ".join([f"{instr}({c:,})" for instr, c in main_instrs])
            
            print(f"    {addr}: {count:,} ({percentage:.1f}% of warp) - {main_instr_str}")
            
            # 计算warp偏好度
            warp_preference = count / global_count * 100 if global_count > 0 else 0
            if warp_preference > 20:  # 如果warp执行了该地址超过20%的指令
                print(f"        ⭐ 该warp执行了此地址 {warp_preference:.1f}% 的指令")

def export_detailed_analysis(stats, output_prefix):
    """导出详细分析结果"""
    output_dir = f"{output_prefix}_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 导出分析结果到 {output_dir}/")
    
    # 1. 导出地址-指令关联矩阵
    with open(f"{output_dir}/address_instruction_matrix.csv", 'w', encoding='utf-8') as f:
        f.write("Address,Instruction,Count,PercentageOfAddress,PercentageOfInstruction\n")
        
        addresses = sorted(stats['addresses_counter'].keys())
        instructions = sorted(stats['instruction_counter'].keys())
        
        for addr in addresses:
            total_at_addr = stats['addresses_counter'][addr]
            for instr in instructions:
                count = stats['address_instr_counter'][addr].get(instr, 0)
                if count > 0:
                    pct_of_addr = count / total_at_addr * 100
                    total_of_instr = stats['instruction_counter'][instr]
                    pct_of_instr = count / total_of_instr * 100 if total_of_instr > 0 else 0
                    
                    f.write(f"{addr},{instr},{count},{pct_of_addr:.2f},{pct_of_instr:.2f}\n")
    
    # 2. 导出每个地址的详细统计
    with open(f"{output_dir}/address_detail.txt", 'w', encoding='utf-8') as f:
        f.write("地址详细统计\n")
        f.write("="*100 + "\n\n")
        
        sorted_addresses = sorted(
            stats['addresses_counter'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for addr, total_count in sorted_addresses:
            f.write(f"\n{addr} (总执行: {total_count:,} 次, {total_count/stats['instruction_lines']*100:.2f}%)\n")
            f.write("-"*80 + "\n")
            
            instr_dist = stats['address_instr_counter'][addr]
            sorted_instrs = sorted(instr_dist.items(), key=lambda x: x[1], reverse=True)
            
            for instr, count in sorted_instrs:
                pct = count / total_count * 100
                f.write(f"  {instr:15s}: {count:12,} ({pct:6.2f}%)\n")
    
    # 3. 导出每个指令的地址分布
    with open(f"{output_dir}/instruction_address_distribution.txt", 'w', encoding='utf-8') as f:
        f.write("指令地址分布\n")
        f.write("="*100 + "\n\n")
        
        sorted_instructions = sorted(
            stats['instruction_counter'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for instr, total_count in sorted_instructions:
            f.write(f"\n{instr:15s} (总执行: {total_count:,} 次, {total_count/stats['instruction_lines']*100:.2f}%)\n")
            f.write("-"*80 + "\n")
            
            addr_dist = stats['instr_address_counter'][instr]
            sorted_addrs = sorted(addr_dist.items(), key=lambda x: x[1], reverse=True)
            
            for addr, count in sorted_addrs:
                pct = count / total_count * 100
                f.write(f"  {addr}: {count:12,} ({pct:6.2f}%)\n")
    
    # 4. 导出Warp地址偏好
    with open(f"{output_dir}/warp_address_preferences.txt", 'w', encoding='utf-8') as f:
        f.write("Warp地址偏好分析\n")
        f.write("="*100 + "\n\n")
        
        for warp in sorted(stats['warp_address_counter'].keys()):
            warp_stats = stats['warp_address_counter'][warp]
            
            # 计算warp总指令数
            warp_total = sum(
                sum(instr_dist.values()) 
                for addr_dist in warp_stats.values() 
                for instr_dist in [addr_dist]
            )
            
            f.write(f"\nWarp {warp} (总指令: {warp_total:,})\n")
            f.write("-"*80 + "\n")
            
            # 按地址汇总warp的指令
            warp_address_summary = {}
            for addr, instr_dist in warp_stats.items():
                warp_address_summary[addr] = sum(instr_dist.values())
            
            # 按执行次数排序
            for addr, count in sorted(warp_address_summary.items(), key=lambda x: x[1], reverse=True)[:20]:
                pct_warp = count / warp_total * 100
                global_count = stats['addresses_counter'][addr]
                pct_global = count / global_count * 100 if global_count > 0 else 0
                
                f.write(f"  {addr}: {count:12,} ({pct_warp:5.1f}% of warp, {pct_global:5.1f}% of global)\n")
                
                # 详细指令分布
                instr_details = sorted(
                    warp_stats[addr].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                for instr, instr_count in instr_details:
                    f.write(f"      {instr:15s}: {instr_count:8,}\n")
    
    print(f"✓ 分析结果已保存到 {output_dir}/ 目录")

def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_instruction_precise.py <日志文件> [选项]")
        print("选项:")
        print("  --max-lines N     最多处理N行（用于测试）")
        print("  --chunk-size N    读取块大小（字节，默认64MB）")
        print("  --export          导出详细分析结果")
        sys.exit(1)
    
    # 解析参数
    log_file = None
    max_lines = None
    chunk_size = 64 * 1024 * 1024
    export_flag = False
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == "--max-lines":
            if i + 1 < len(sys.argv):
                max_lines = int(sys.argv[i + 1])
                i += 1
        elif arg == "--chunk-size":
            if i + 1 < len(sys.argv):
                chunk_size = int(sys.argv[i + 1])
                i += 1
        elif arg == "--export":
            export_flag = True
        else:
            log_file = arg
        
        i += 1
    
    if not log_file:
        print("错误: 未指定日志文件", file=sys.stderr)
        sys.exit(1)
    
    if not Path(log_file).exists():
        print(f"错误: 文件 '{log_file}' 不存在", file=sys.stderr)
        sys.exit(1)
    
    # 分析文件
    stats = analyze_log_file_precise(
        log_file, 
        max_lines=max_lines,
        chunk_size=chunk_size
    )
    
    if stats:
        # 基本统计信息
        print(f"\n{'='*80}")
        print("📊 基本统计信息")
        print(f"{'='*80}")
        print(f"总日志行数: {stats['total_lines']:,}")
        print(f"指令行数: {stats['instruction_lines']:,} ({stats['instruction_lines']/stats['total_lines']*100:.1f}%)")
        print(f"唯一指令地址数: {len(stats['addresses_counter']):,}")
        print(f"唯一指令类型数: {len(stats['instruction_counter']):,}")
        
        if stats['address_range'][0] != float('inf'):
            min_addr = stats['address_range'][0]
            max_addr = stats['address_range'][1]
            addr_span = max_addr - min_addr
            print(f"地址范围: 0x{min_addr:08X} - 0x{max_addr:08X}")
            print(f"地址跨度: {addr_span:,} 字节 ({addr_span/1024:.2f} KB)")
        
        # 指令地址偏好分析
        analyze_instruction_address_preferences(stats)
        
        # 导出结果
        if export_flag:
            output_prefix = Path(log_file).stem
            export_detailed_analysis(stats, output_prefix)
    else:
        print("分析失败", file=sys.stderr)

if __name__ == '__main__':
    main()