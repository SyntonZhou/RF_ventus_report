#!/usr/bin/env python3
"""
Ventus GPGPU日志指令统计器
功能：统计log文件中所有指令的出现次数
用法：python count_instructions.py <日志文件1> <日志文件2> ...
"""

import re
import sys
from collections import Counter
from pathlib import Path

def extract_instructions_from_chunk(chunk):
    """
    从日志块中提取所有指令名称
    支持格式：
    1. SM X warp Y 0xADDRESS INSTRUCTION_NAME_OPCODE ...
    2. SM X warp Y JUMP to ...
    """
    instructions = []
    
    for line in chunk.split('\n'):
        if not line.startswith('SM'):
            continue
        
        # 检查是否为JUMP指令
        if 'JUMP to' in line:
            instructions.append('JUMP')
            continue
        
        # 常规指令：匹配地址后的指令名_操作码格式
        # 例如：0x80000000 AUIPC_0x00004197
        pattern = re.compile(r'0x[0-9a-fA-F]{8}\s+([A-Z][A-Z0-9_]+?)_[0-9a-fA-Fx]+')
        match = pattern.search(line)
        
        if match:
            instr = match.group(1)
            # 过滤掉非指令的误匹配
            if instr and not instr.startswith(('MEMADDR', 'ADDR', 'DATA', 'mask')):
                instructions.append(instr)
    
    return instructions

def count_instructions(log_file, chunk_size=1024*1024*64):  # 64MB chunks
    """统计单个文件的指令"""
    stats = Counter()
    total_instructions = 0
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # 使用缓冲区逐块读取文件
            buffer = ""
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # 将缓冲区与新块合并
                chunk = buffer + chunk
                
                # 确保我们不在指令中间截断
                # 查找最后一个换行符，确保完整行
                last_newline = chunk.rfind('\n')
                if last_newline != -1:
                    # 处理完整块，保留最后不完整的行到缓冲区
                    complete_chunk = chunk[:last_newline]
                    buffer = chunk[last_newline:]
                    
                    # 提取指令
                    instructions = extract_instructions_from_chunk(complete_chunk)
                    stats.update(instructions)
                    total_instructions += len(instructions)
                else:
                    # 如果没有找到换行符，整个块可能是一个超长行
                    # 保留到缓冲区继续读取
                    buffer = chunk
                    continue
            
            # 处理缓冲区中剩余的内容
            if buffer:
                instructions = extract_instructions_from_chunk(buffer)
                stats.update(instructions)
                total_instructions += len(instructions)
                
    except Exception as e:
        print(f"读取文件失败 '{log_file}': {e}", file=sys.stderr)
        return None
    
    if not stats:
        print(f"警告: '{log_file}' 中未找到任何指令", file=sys.stderr)
        return None
    
    return {
        'filename': Path(log_file).name,
        'total': total_instructions,
        'unique': len(stats),
        'stats': stats
    }

def print_stats(result, top_n=100):
    """打印统计结果"""
    if not result:
        return
    
    print(f"\n{'='*60}")
    print(f"📊 指令统计: {result['filename']}")
    print(f"{'='*60}")
    print(f"总指令数  : {result['total']:,}")
    print(f"唯一指令数: {result['unique']}")
    print(f"\n指令数量排行 (Top {top_n}):")
    print(f"{'-'*40}")
    
    # 按数量排序
    sorted_stats = result['stats'].most_common(top_n)
    
    for i, (instr, count) in enumerate(sorted_stats, 1):
        percentage = (count / result['total']) * 100
        print(f"{i:3d}. {instr:25s}: {count:6,} ({percentage:5.1f}%)")
    
    # 显示剩余指令总数
    if result['unique'] > top_n:
        shown = sum(c for _, c in sorted_stats)
        remaining = result['total'] - shown
        print(f"{'-'*40}")
        print(f"其他 {result['unique'] - top_n} 种指令: {remaining:,}条")
    
    return sorted_stats

def main():
    if len(sys.argv) < 2:
        print("用法: python count_instructions.py <日志文件1> [日志文件2] ...")
        print("示例: python count_instructions.py 444.log 888.log")
        sys.exit(1)
    
    files = sys.argv[1:]
    all_results = []
    
    for file_path in files:
        if not Path(file_path).exists():
            print(f"错误: 文件 '{file_path}' 不存在", file=sys.stderr)
            continue
        
        print(f"正在处理文件: {file_path}...", file=sys.stderr)
        result = count_instructions(file_path)
        if result:
            print_stats(result)
            all_results.append(result)
    
    # 汇总分析（多文件时）
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("📈 跨文件对比分析")
        print(f"{'='*60}")
        
        print(f"{'文件名':<15} {'总指令':<10} {'唯一指令':<10} {'TOP3指令'}")
        print(f"{'-'*60}")
        
        for r in all_results:
            top3 = ', '.join([f"{i[0]}({i[1]})" for i in r['stats'].most_common(3)])
            print(f"{r['filename']:<15} {r['total']:<10,} {r['unique']:<10} {top3}")

if __name__ == '__main__':
    main()