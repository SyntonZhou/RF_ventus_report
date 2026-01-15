#!/usr/bin/env python3
"""
地址空间分析脚本 - 内存优化版
适用于10-50GB的中等文件
"""
import os
import sys
import re
import csv
import json
import heapq
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import argparse

class MemoryOptimizedAnalyzer:
    def __init__(self, log_file: str):
        self.log_file = log_file
        
        # 使用更高效的数据结构
        self.address_stats = defaultdict(int)  # 地址访问计数
        self.pc_stats = defaultdict(int)       # PC执行计数
        self.opcode_stats = defaultdict(int)   # 操作码计数
        
        # 地址空间统计
        self.address_spaces = {
            '0x70': {'name': '数据缓冲区', 'count': 0, 'min': float('inf'), 'max': float('-inf'), 'addresses': set()},
            '0x80': {'name': '指令空间', 'count': 0, 'min': float('inf'), 'max': float('-inf'), 'addresses': set()},
            '0x90': {'name': '矩阵存储', 'count': 0, 'min': float('inf'), 'max': float('-inf'), 'addresses': set()},
            'other': {'name': '其他', 'count': 0, 'min': float('inf'), 'max': float('-inf'), 'addresses': set()}
        }
        
        # 操作码分类
        self.load_ops = {'VLW12_V', 'VLW_V', 'VLW', 'VLE32_V', 'VLE64_V', 'VLB_V', 'VLH_V', 'VLD_V'}
        self.store_ops = {'VSW12_V', 'VSW_V', 'VSW', 'VSE32_V', 'VSE64_V', 'VSB_V', 'VSH_V', 'VSD_V'}
        
        # 抽样统计
        self.sample_addresses = {}  # 抽样热点地址
        self.sample_size = 100000   # 抽样大小
    
    def parse_line(self, line: str) -> Optional[dict]:
        """快速解析行"""
        line = line.strip()
        if not line:
            return None
        
        # 快速检测常见格式
        if line.startswith('SM') and '0x' in line:
            # 格式1: SM0 warp0 0x800002f4 VLW12_V_0x70002b80
            parts = line.split()
            if len(parts) >= 3:
                pc_match = re.search(r'(0x[0-9a-fA-F]+)', parts[2])
                op_match = re.search(r'([A-Z0-9_]+)_0x([0-9a-fA-F]+)', line)
                if pc_match and op_match:
                    return {
                        'pc': pc_match.group(1),
                        'opcode': op_match.group(1),
                        'address': f"0x{op_match.group(2)}",
                        'type': 'exec'
                    }
        
        elif 'pc=' in line and 'opcode=' in line:
            # 格式2: pc=0x800002f4 opcode=VLW12_V count=524288 min_addr=0x70002b80 max_addr=0x70002bfc
            result = {'type': 'memory_stats'}
            
            # 快速提取
            pc_match = re.search(r'pc=([0-9a-fA-Fx]+)', line)
            op_match = re.search(r'opcode=([A-Z0-9_]+)', line)
            count_match = re.search(r'count=(\d+)', line)
            min_match = re.search(r'min_addr=([0-9a-fA-Fx]+)', line)
            max_match = re.search(r'max_addr=([0-9a-fA-Fx]+)', line)
            
            if pc_match and op_match:
                result['pc'] = pc_match.group(1)
                result['opcode'] = op_match.group(1)
                if count_match:
                    result['count'] = int(count_match.group(1))
                if min_match:
                    result['min_addr'] = min_match.group(1)
                if max_match:
                    result['max_addr'] = max_match.group(1)
                return result
        
        elif line.startswith('0x'):
            # 格式3: 0x800002f4 VLW12_V 524288 0x70002b80 0x70002bfc 128 Bytes
            parts = line.split()
            if len(parts) >= 5:
                result = {
                    'type': 'table',
                    'pc': parts[0],
                    'opcode': parts[1],
                    'min_addr': parts[3],
                    'max_addr': parts[4]
                }
                if parts[2].isdigit():
                    result['count'] = int(parts[2])
                return result
        
        return None
    
    def classify_address_space(self, address: str) -> str:
        """快速地址空间分类"""
        if address.startswith('0x70'):
            return '0x70'
        elif address.startswith('0x80'):
            return '0x80'
        elif address.startswith('0x90'):
            return '0x90'
        else:
            return 'other'
    
    def process_parsed_data(self, data: dict):
        """处理解析数据，内存优化"""
        if data['type'] in ['memory_stats', 'table']:
            pc = data['pc']
            opcode = data['opcode']
            count = data.get('count', 1)
            
            # 更新统计
            self.pc_stats[pc] += count
            self.opcode_stats[opcode] += count
            
            # 处理地址
            for addr_key in ['min_addr', 'max_addr', 'address']:
                if addr_key in data and data[addr_key]:
                    addr = data[addr_key]
                    self.address_stats[addr] += count
                    
                    # 地址空间统计
                    space = self.classify_address_space(addr)
                    space_info = self.address_spaces[space]
                    space_info['count'] += count
                    space_info['addresses'].add(addr)
                    
                    # 更新最小最大地址
                    addr_int = int(addr, 16)
                    if addr_int < space_info['min']:
                        space_info['min'] = addr_int
                    if addr_int > space_info['max']:
                        space_info['max'] = addr_int
        
        elif data['type'] == 'exec':
            pc = data['pc']
            opcode = data['opcode']
            address = data.get('address')
            
            self.pc_stats[pc] += 1
            self.opcode_stats[opcode] += 1
            
            if address:
                self.address_stats[address] += 1
                
                space = self.classify_address_space(address)
                space_info = self.address_spaces[space]
                space_info['count'] += 1
                space_info['addresses'].add(address)
                
                addr_int = int(address, 16)
                if addr_int < space_info['min']:
                    space_info['min'] = addr_int
                if addr_int > space_info['max']:
                    space_info['max'] = addr_int
    
    def analyze(self, max_lines: int = None):
        """分析日志文件，支持限制行数"""
        print(f"分析文件: {self.log_file}")
        
        total_lines = 0
        parsed_lines = 0
        
        # 使用迭代器读取
        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                total_lines += 1
                
                # 解析行
                parsed = self.parse_line(line)
                if parsed:
                    parsed_lines += 1
                    self.process_parsed_data(parsed)
                
                # 进度报告
                if total_lines % 1000000 == 0:
                    print(f"已处理: {total_lines:,} 行, 解析: {parsed_lines:,} 行")
                    # 定期清理内存
                    if total_lines % 5000000 == 0:
                        self._optimize_memory()
                
                # 限制行数（用于测试）
                if max_lines and total_lines >= max_lines:
                    break
        
        print(f"总计行数: {total_lines:,}")
        print(f"解析成功: {parsed_lines:,}")
        
        # 最终内存优化
        self._optimize_memory()
    
    def _optimize_memory(self):
        """内存优化：清理和压缩数据"""
        # 抽样热点地址（只保留高频地址）
        if len(self.address_stats) > self.sample_size:
            # 使用堆获取top N地址
            top_addresses = heapq.nlargest(self.sample_size, 
                                          self.address_stats.items(),
                                          key=lambda x: x[1])
            self.address_stats = dict(top_addresses)
            
            # 重建地址空间集合
            for space_info in self.address_spaces.values():
                # 只保留抽样中的地址
                space_info['addresses'] = set(
                    addr for addr in space_info['addresses']
                    if addr in self.address_stats
                )
        
        # 触发垃圾回收
        import gc
        gc.collect()
    
    def generate_report(self, output_dir: str, top_n: int = 1000):
        """生成精简报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 地址空间统计
        self._save_address_space_report(output_dir)
        
        # 2. PC统计（top N）
        self._save_pc_report(output_dir, top_n)
        
        # 3. 操作码统计
        self._save_opcode_report(output_dir)
        
        # 4. 热点地址统计（top N）
        self._save_hot_addresses_report(output_dir, top_n)
        
        # 5. 综合摘要
        self._save_summary(output_dir)
        
        print(f"\n报告已生成到: {output_dir}")
    
    def _save_address_space_report(self, output_dir: str):
        """保存地址空间统计"""
        filename = os.path.join(output_dir, 'address_spaces.csv')
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['地址空间', '名称', '访问次数', '唯一地址', '大小(字节)'])
            
            for prefix, info in self.address_spaces.items():
                size = info['max'] - info['min'] if info['min'] != float('inf') else 0
                writer.writerow([
                    prefix,
                    info['name'],
                    info['count'],
                    len(info['addresses']),
                    size
                ])
    
    def _save_pc_report(self, output_dir: str, top_n: int):
        """保存PC统计（top N）"""
        filename = os.path.join(output_dir, 'pc_stats.csv')
        
        # 获取top N的PC
        top_pcs = heapq.nlargest(top_n, self.pc_stats.items(), key=lambda x: x[1])
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['PC', '执行次数', '占比(%)'])
            
            total = sum(self.pc_stats.values())
            for pc, count in top_pcs:
                percentage = (count / total * 100) if total > 0 else 0
                writer.writerow([pc, count, f"{percentage:.2f}%"])
    
    def _save_opcode_report(self, output_dir: str):
        """保存操作码统计"""
        filename = os.path.join(output_dir, 'opcode_stats.csv')
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['操作码', '次数', '类型', '占比(%)'])
            
            total = sum(self.opcode_stats.values())
            for opcode, count in sorted(self.opcode_stats.items(), 
                                       key=lambda x: x[1], reverse=True):
                percentage = (count / total * 100) if total > 0 else 0
                op_type = '加载' if opcode in self.load_ops else '存储' if opcode in self.store_ops else '其他'
                writer.writerow([opcode, count, op_type, f"{percentage:.2f}%"])
    
    def _save_hot_addresses_report(self, output_dir: str, top_n: int):
        """保存热点地址"""
        filename = os.path.join(output_dir, 'hot_addresses.csv')
        
        # 获取top N地址
        top_addresses = heapq.nlargest(top_n, self.address_stats.items(), 
                                      key=lambda x: x[1])
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['地址', '访问次数', '空间', '占比(%)'])
            
            total = sum(self.address_stats.values())
            for addr, count in top_addresses:
                percentage = (count / total * 100) if total > 0 else 0
                space = self.classify_address_space(addr)
                writer.writerow([addr, count, space, f"{percentage:.2f}%"])
    
    def _save_summary(self, output_dir: str):
        """保存摘要"""
        filename = os.path.join(output_dir, 'summary.txt')
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("地址空间分析摘要\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"总地址访问次数: {sum(self.address_stats.values()):,}\n")
            f.write(f"唯一地址数量: {len(self.address_stats):,}\n")
            f.write(f"总指令执行次数: {sum(self.pc_stats.values()):,}\n")
            f.write(f"唯一PC数量: {len(self.pc_stats):,}\n")
            f.write(f"唯一操作码数量: {len(self.opcode_stats):,}\n\n")
            
            f.write("地址空间分布:\n")
            for prefix, info in self.address_spaces.items():
                f.write(f"  {prefix} ({info['name']}):\n")
                f.write(f"    访问次数: {info['count']:,}\n")
                f.write(f"    唯一地址: {len(info['addresses']):,}\n")
                if info['min'] != float('inf'):
                    f.write(f"    地址范围: 0x{info['min']:x} - 0x{info['max']:x}\n\n")

def main():
    parser = argparse.ArgumentParser(description='地址空间分析工具 - 内存优化版')
    parser.add_argument('--log', required=True, help='日志文件路径')
    parser.add_argument('--out', default='address_analysis', help='输出目录')
    parser.add_argument('--max-lines', type=int, default=None, help='最大处理行数（用于测试）')
    parser.add_argument('--top-n', type=int, default=1000, help='保存的top N记录数')
    
    args = parser.parse_args()
    
    analyzer = MemoryOptimizedAnalyzer(args.log)
    analyzer.analyze(args.max_lines)
    analyzer.generate_report(args.out, args.top_n)
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()