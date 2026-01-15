#!/usr/bin/env python3
"""
Ventus GPGPU指令地址分析器（优化版）
针对Windows系统和大文件进行了优化
"""

import re
import sys
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

class ProgressTracker:
    """进度跟踪器，不依赖文件位置"""
    def __init__(self, total_bytes, desc="处理进度"):
        self.total_bytes = total_bytes
        self.desc = desc
        self.start_time = time.time()
        self.processed_bytes = 0
        self.last_update = 0
        self.update_interval = 2  # 每2秒更新一次
        
    def update(self, bytes_read):
        self.processed_bytes = bytes_read
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
        self._display()  # 确保最后显示完整进度
        sys.stderr.write("\n")
        sys.stderr.flush()

def extract_instruction_info(line):
    """
    从单行提取指令信息（高效版本）
    返回：字典或None（如果不是指令行）
    """
    # 快速检查是否为指令行
    if not line.startswith('SM') or 'warp' not in line:
        return None
    
    # 查找地址（更快速的方法）
    parts = line.split()
    if len(parts) < 4:
        return None
    
    # 找到包含0x的字段
    addr = None
    for part in parts[2:6]:  # 只检查前几个字段
        if part.startswith('0x') and len(part) == 10:  # 0x + 8位十六进制
            addr = part
            break
    
    if not addr:
        return None
    
    # 提取SM和warp编号
    try:
        sm_part = parts[0]  # 如 "SM" 或 "SM1"
        if sm_part.startswith('SM'):
            sm_num = sm_part[2:] if len(sm_part) > 2 else parts[1]
            sm_num = int(sm_num) if sm_num.isdigit() else 1
        
        warp_part = parts[2] if 'warp' in parts[1] else parts[3]
        warp_num = int(warp_part) if warp_part.isdigit() else 0
    except (IndexError, ValueError):
        sm_num = 1
        warp_num = 0
    
    # 提取指令名称（简化版）
    instr = "UNKNOWN"
    for part in parts:
        if '_0x' in part and len(part) > 10:
            instr_part = part.split('_')[0]
            if instr_part.isalpha():
                instr = instr_part
                break
    
    # 提取时间信息
    time_ns = 0
    for part in parts:
        if part.startswith('@') and 'ns,' in part:
            time_str = part[1:].split('ns,')[0]
            try:
                time_ns = int(time_str)
            except ValueError:
                pass
            break
    
    return {
        'sm': sm_num,
        'warp': warp_num,
        'address': addr,
        'address_int': int(addr, 16) if addr.startswith('0x') else 0,
        'instruction': instr,
        'time_ns': time_ns
    }

def analyze_log_file_chunks(log_file, max_lines=None, chunk_size=64*1024*1024):
    """
    按块分析日志文件，避免内存问题
    chunk_size: 每次读取的字节数
    """
    print(f"开始分析文件: {log_file}", file=sys.stderr)
    print(f"文件大小: {os.path.getsize(log_file) / (1024**3):.2f} GB", file=sys.stderr)
    
    # 统计信息
    stats = {
        'total_bytes': os.path.getsize(log_file),
        'total_lines': 0,
        'instruction_lines': 0,
        'addresses_counter': Counter(),
        'instruction_counter': Counter(),
        'sm_warp_counter': defaultdict(Counter),
        'address_range': [float('inf'), float('-inf')],
        'start_time': time.time()
    }
    
    try:
        # 创建进度跟踪器
        progress = ProgressTracker(stats['total_bytes'], "分析进度")
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            buffer = ""
            bytes_read = 0
            
            while True:
                # 读取一块数据
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                bytes_read += len(chunk)
                buffer += chunk
                
                # 更新进度
                progress.update(bytes_read)
                
                # 按行处理缓冲区
                lines = buffer.split('\n')
                # 保留最后一行（可能不完整）
                buffer = lines[-1]
                
                for line in lines[:-1]:
                    stats['total_lines'] += 1
                    
                    # 限制最大处理行数
                    if max_lines and stats['total_lines'] > max_lines:
                        progress.close()
                        print(f"\n已达到最大行数限制: {max_lines}", file=sys.stderr)
                        return stats
                    
                    # 提取指令信息
                    instr_info = extract_instruction_info(line)
                    if not instr_info:
                        continue
                    
                    stats['instruction_lines'] += 1
                    
                    # 更新地址统计
                    addr = instr_info['address']
                    addr_int = instr_info['address_int']
                    stats['addresses_counter'][addr] += 1
                    
                    # 更新地址范围
                    if addr_int < stats['address_range'][0]:
                        stats['address_range'][0] = addr_int
                    if addr_int > stats['address_range'][1]:
                        stats['address_range'][1] = addr_int
                    
                    # 更新指令类型统计
                    stats['instruction_counter'][instr_info['instruction']] += 1
                    
                    # 更新SM/Warp统计
                    sm = instr_info['sm']
                    warp = instr_info['warp']
                    stats['sm_warp_counter'][sm][warp] += 1
            
            # 处理缓冲区剩余内容
            if buffer:
                stats['total_lines'] += 1
                instr_info = extract_instruction_info(buffer)
                if instr_info:
                    stats['instruction_lines'] += 1
                    stats['addresses_counter'][instr_info['address']] += 1
        
        progress.close()
        
        # 计算处理时间
        stats['elapsed_time'] = time.time() - stats['start_time']
        
        print(f"\n分析完成!", file=sys.stderr)
        print(f"处理了 {stats['total_lines']:,} 行日志", file=sys.stderr)
        print(f"其中 {stats['instruction_lines']:,} 行是指令行", file=sys.stderr)
        print(f"处理时间: {stats['elapsed_time']:.1f} 秒", file=sys.stderr)
        
        if stats['instruction_lines'] > 0:
            print(f"平均速度: {stats['total_lines']/stats['elapsed_time']:,.0f} 行/秒", file=sys.stderr)
        
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

def print_basic_summary(stats, top_n=300):
    """打印基本摘要信息"""
    if not stats:
        return
    
    print(f"\n{'='*80}")
    print("📊 指令地址分析摘要")
    print(f"{'='*80}")
    
    # 基本信息
    print(f"📁 基本信息:")
    print(f"  总日志行数: {stats['total_lines']:,}")
    print(f"  指令行数: {stats['instruction_lines']:,} "
          f"({(stats['instruction_lines']/stats['total_lines']*100):.1f}%)")
    
    if 'elapsed_time' in stats:
        print(f"  处理时间: {stats['elapsed_time']:.1f} 秒")
    
    # 地址信息
    print(f"\n📍 地址信息:")
    unique_addresses = len(stats['addresses_counter'])
    print(f"  唯一地址数: {unique_addresses:,}")
    
    if stats['address_range'][0] != float('inf'):
        min_addr = stats['address_range'][0]
        max_addr = stats['address_range'][1]
        addr_span = max_addr - min_addr
        print(f"  地址范围: 0x{min_addr:08X} - 0x{max_addr:08X}")
        print(f"  地址跨度: {addr_span:,} 字节 ({addr_span/1024:.2f} KB)")
    
    # 热门地址
    print(f"\n🎯 热门地址 (Top {min(top_n, unique_addresses)}):")
    total_instr = stats['instruction_lines']
    for i, (addr, count) in enumerate(stats['addresses_counter'].most_common(top_n), 1):
        percentage = count / total_instr * 100
        print(f"  {i:2d}. {addr}: {count:8,} ({percentage:5.1f}%)")
    
    # 指令类型统计
    print(f"\n📝 指令类型统计 (Top {top_n}):")
    for i, (instr, count) in enumerate(stats['instruction_counter'].most_common(top_n), 1):
        percentage = count / total_instr * 100
        print(f"  {i:2d}. {instr:15s}: {count:8,} ({percentage:5.1f}%)")
    
    # SM和Warp统计
    print(f"\n🏭 SM和Warp统计:")
    for sm in sorted(stats['sm_warp_counter'].keys()):
        warp_counts = stats['sm_warp_counter'][sm]
        total_in_sm = sum(warp_counts.values())
        active_warps = len(warp_counts)
        
        if active_warps <= 5:
            warp_details = ", ".join([f"warp{w}({c:,})" for w, c in sorted(warp_counts.items())])
        else:
            # 只显示前3个最活跃的warp
            top_warps = sorted(warp_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            warp_details = f"{active_warps}个warp，最活跃: " + ", ".join([f"warp{w}({c:,})" for w, c in top_warps])
        
        print(f"  SM{sm}: {total_in_sm:,} 条指令, {warp_details}")

def export_key_addresses(stats, output_file, top_addresses=100):
    """导出关键地址信息"""
    if not stats:
        return
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Ventus GPGPU指令地址分析报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"总日志行数: {stats['total_lines']:,}\n")
            f.write(f"指令行数: {stats['instruction_lines']:,}\n")
            f.write(f"唯一地址数: {len(stats['addresses_counter']):,}\n\n")
            
            if stats['address_range'][0] != float('inf'):
                min_addr = stats['address_range'][0]
                max_addr = stats['address_range'][1]
                f.write(f"地址范围: 0x{min_addr:08X} - 0x{max_addr:08X}\n")
                f.write(f"地址跨度: {max_addr - min_addr:,} 字节\n\n")
            
            f.write(f"最频繁访问的地址 (Top {top_addresses}):\n")
            f.write("-"*60 + "\n")
            for addr, count in stats['addresses_counter'].most_common(top_addresses):
                percentage = count / stats['instruction_lines'] * 100
                f.write(f"{addr}: {count:,} ({percentage:.1f}%)\n")
            
            f.write(f"\n指令类型统计:\n")
            f.write("-"*60 + "\n")
            for instr, count in stats['instruction_counter'].most_common(50):
                percentage = count / stats['instruction_lines'] * 100
                f.write(f"{instr:15s}: {count:,} ({percentage:.1f}%)\n")
        
        print(f"✓ 统计信息已导出到: {output_file}")
        
    except Exception as e:
        print(f"导出失败: {e}", file=sys.stderr)

def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_address.py <日志文件> [选项]")
        print("选项:")
        print("  --max-lines N     最多处理N行（用于测试）")
        print("  --chunk-size N    读取块大小（字节，默认64MB）")
        print("  --export          导出统计信息到文件")
        print("  --help            显示帮助信息")
        print("\n示例:")
        print("  python analyze_address.py ventus.log")
        print("  python analyze_address.py ventus.log --max-lines 1000000")
        print("  python analyze_address.py ventus.log --chunk-size 100000000 --export")
        sys.exit(1)
    
    # 解析参数
    log_file = None
    max_lines = None
    chunk_size = 64 * 1024 * 1024  # 64MB
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
        elif arg == "--help":
            print("帮助信息:")
            print("此程序用于分析Ventus GPGPU日志文件中的指令地址")
            print("支持流式处理，可处理大文件")
            sys.exit(0)
        elif arg.startswith("-"):
            print(f"未知选项: {arg}", file=sys.stderr)
            sys.exit(1)
        else:
            log_file = arg
        
        i += 1
    
    if not log_file:
        print("错误: 未指定日志文件", file=sys.stderr)
        sys.exit(1)
    
    if not Path(log_file).exists():
        print(f"错误: 文件 '{log_file}' 不存在", file=sys.stderr)
        sys.exit(1)
    
    # 检查文件大小
    file_size = os.path.getsize(log_file)
    print(f"文件大小: {file_size/(1024**3):.1f} GB", file=sys.stderr)
    
    if file_size > 10 * 1024**3:  # 大于10GB
        print(f"警告: 文件较大，处理可能需要较长时间", file=sys.stderr)
        if max_lines is None:
            print("建议使用 --max-lines 参数先测试处理部分数据", file=sys.stderr)
    
    # 分析文件
    stats = analyze_log_file_chunks(
        log_file, 
        max_lines=max_lines,
        chunk_size=chunk_size
    )
    
    if stats:
        print_basic_summary(stats)
        
        if export_flag:
            output_file = f"{Path(log_file).stem}_addresses.txt"
            export_key_addresses(stats, output_file)
    else:
        print("分析失败", file=sys.stderr)

if __name__ == '__main__':
    main()