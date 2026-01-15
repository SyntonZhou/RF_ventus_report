#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
import argparse
import json
import csv
import sys
import time
import os
import shutil
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean, median, variance, stdev
from typing import Dict, List, Optional, Tuple

# 匹配"真正的指令行"（带 PC + 指令token + @xxns）
INSTR_RE = re.compile(
    r"SM\s+(?P<sm>\d+)\s+warp\s+(?P<warp>\d+)\s+0x(?P<pc>[0-9a-fA-F]+)\s+"
    r"(?P<tok>[A-Z0-9]+(?:_[A-Z0-9]+)?)_0x[0-9a-fA-F]+.*?@(?P<ns>\d+)ns"
)

class InstructionAnalyzer:
    def __init__(self, log_file: str, ns_per_cycle: Optional[int] = None, 
                 sm: Optional[int] = None, warp: Optional[int] = None):
        self.log_file = log_file
        self.ns_per_cycle = ns_per_cycle
        self.sm_filter = sm
        self.warp_filter = warp
        self.completed_instructions = []
        print(f"初始化分析器: 日志文件={log_file}, SM={sm}, WARP={warp}, 周期时间={ns_per_cycle}")
        
    def parse_log(self):
        """解析日志文件，提取指令信息"""
        print(f"开始解析日志文件: {self.log_file}")
        print("正在解析指令...")
        
        pending = defaultdict(deque)
        completed = []
        line_count = 0
        instr_count = 0
        
        try:
            with open(self.log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    line_count += 1
                    
                    # 每处理10万行输出一次进度
                    if line_count % 100000 == 0:
                        print(f"  已处理 {line_count} 行，找到 {instr_count} 条指令...")
                    
                    m = INSTR_RE.search(line)
                    if not m:
                        continue
                    
                    sm = int(m.group("sm"))
                    warp = int(m.group("warp"))
                    
                    # 应用过滤器
                    if self.sm_filter is not None and sm != self.sm_filter:
                        continue
                    if self.warp_filter is not None and warp != self.warp_filter:
                        continue
                    
                    pc = int(m.group("pc"), 16)
                    opcode = m.group("tok")
                    ns = int(m.group("ns"))
                    
                    # 调试输出（仅显示前几条指令）
                    if instr_count < 5:
                        print(f"  找到指令: line={line_num}, SM={sm}, WARP={warp}, OP={opcode}, PC=0x{pc:x}, ns={ns}")
                    
                    has_wb = (" WB " in line) or line.strip().endswith("WB") or (" WB" in line)
                    is_store_or_branch_like = (" ADDR " in line) or ("JUMP=" in line) or ("SIMTSTK" in line)
                    
                    key = (sm, warp, pc, opcode)
                    
                    if has_wb:
                        if pending[key]:
                            rec = pending[key].popleft()
                            rec["done_ns"] = ns
                            completed.append(rec)
                        else:
                            completed.append({
                                "sm": sm, "warp": warp, "pc": pc, "opcode": opcode, 
                                "issue_ns": ns, "done_ns": ns, "line_num": line_num
                            })
                    else:
                        if is_store_or_branch_like:
                            completed.append({
                                "sm": sm, "warp": warp, "pc": pc, "opcode": opcode, 
                                "issue_ns": ns, "done_ns": ns, "line_num": line_num
                            })
                        else:
                            pending[key].append({
                                "sm": sm, "warp": warp, "pc": pc, "opcode": opcode, 
                                "issue_ns": ns, "done_ns": None, "line_num": line_num
                            })
                    
                    instr_count += 1
        except FileNotFoundError:
            print(f"错误: 找不到文件 {self.log_file}")
            sys.exit(1)
        except Exception as e:
            print(f"解析日志时出错: {e}")
            sys.exit(1)
        
        # 丢弃未完成的指令
        self.completed_instructions = [r for r in completed if r["done_ns"] is not None]
        self.completed_instructions.sort(key=lambda r: r["done_ns"])
        
        print(f"解析完成: 总行数={line_count}, 有效指令数={len(self.completed_instructions)}")
        
        # 自动探测 ns_per_cycle 如果未指定
        if self.ns_per_cycle is None:
            print("自动探测周期时间...")
            self.ns_per_cycle = self._auto_detect_cycle_time()
            print(f"探测到的周期时间: {self.ns_per_cycle} ns/cycle")
    
    def _auto_detect_cycle_time(self) -> int:
        """自动探测周期时间（ns/cycle）"""
        if len(self.completed_instructions) < 2:
            print("警告: 指令太少，使用默认周期时间 10 ns/cycle")
            return 10
        
        diffs = []
        for i in range(1, min(1000, len(self.completed_instructions))):  # 只检查前1000个样本
            d = self.completed_instructions[i]["done_ns"] - self.completed_instructions[i-1]["done_ns"]
            if d > 0:
                diffs.append(d)
        
        if not diffs:
            print("警告: 未找到有效的完成间隔，使用默认周期时间 10 ns/cycle")
            return 10
        
        # 计算最大公约数
        g = 0
        for x in diffs:
            g = math.gcd(g, x)
        
        result = g if g > 0 else 10
        print(f"自动探测: 找到 {len(diffs)} 个间隔，最大公约数={result}")
        return result
    
    def get_instruction_data(self, opcode: str) -> Dict:
        """获取指定指令的数据"""
        print(f"\n获取指令数据: {opcode}")
        
        if not self.completed_instructions:
            print("开始解析日志...")
            self.parse_log()
        
        # 筛选指定opcode的指令
        filtered = [r for r in self.completed_instructions if r["opcode"] == opcode]
        
        print(f"找到 {len(filtered)} 条 {opcode} 指令")
        
        if not filtered:
            return {"error": f"指令 {opcode} 未找到"}
        
        # 计算Throughput CPI
        print("计算Throughput CPI...")
        throughput_cpis = []
        for i in range(1, len(self.completed_instructions)):
            if self.completed_instructions[i]["opcode"] == opcode:
                d_ns = self.completed_instructions[i]["done_ns"] - self.completed_instructions[i-1]["done_ns"]
                cpi = d_ns / self.ns_per_cycle
                throughput_cpis.append({
                    "cpi": cpi,
                    "done_ns": self.completed_instructions[i]["done_ns"],
                    "issue_ns": self.completed_instructions[i]["issue_ns"],
                    "pc": self.completed_instructions[i]["pc"],
                    "line_num": self.completed_instructions[i]["line_num"]
                })
        
        print(f"计算得到 {len(throughput_cpis)} 个Throughput CPI样本")
        
        # 计算Latency（issue到done的延迟）
        print("计算Latency...")
        latency_values = []
        for r in filtered:
            lat_ns = r["done_ns"] - r["issue_ns"]
            if lat_ns > 0:
                latency_values.append({
                    "latency": lat_ns / self.ns_per_cycle,
                    "issue_ns": r["issue_ns"],
                    "done_ns": r["done_ns"],
                    "pc": r["pc"],
                    "line_num": r["line_num"]
                })
        
        print(f"计算得到 {len(latency_values)} 个Latency样本")
        
        # 计算统计信息
        throughput_stats = self._calculate_stats([d["cpi"] for d in throughput_cpis])
        latency_stats = self._calculate_stats([d["latency"] for d in latency_values])
        
        return {
            "opcode": opcode,
            "total_count": len(filtered),
            "throughput_count": len(throughput_cpis),
            "latency_count": len(latency_values),
            "ns_per_cycle": self.ns_per_cycle,
            "throughput_cpis": throughput_cpis,
            "latency_values": latency_values,
            "throughput_stats": throughput_stats,
            "latency_stats": latency_stats,
            "all_instructions": [
                {
                    "issue_ns": r["issue_ns"],
                    "done_ns": r["done_ns"],
                    "pc": f"0x{r['pc']:08x}",
                    "sm": r["sm"],
                    "warp": r["warp"],
                    "line_num": r.get("line_num", 0)
                }
                for r in filtered
            ]
        }
    
    def _calculate_stats(self, values: List[float]) -> Dict:
        """计算统计信息"""
        if not values:
            return {}
        
        try:
            return {
                "mean": mean(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
                "variance": variance(values) if len(values) > 1 else 0,
                "stdev": stdev(values) if len(values) > 1 else 0,
                "p25": self._percentile(values, 25),
                "p50": self._percentile(values, 50),
                "p75": self._percentile(values, 75),
                "p90": self._percentile(values, 90),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99),
            }
        except Exception as e:
            print(f"计算统计信息时出错: {e}")
            return {}
    
    def _percentile(self, values: List[float], p: int) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        
        try:
            sorted_vals = sorted(values)
            k = (len(sorted_vals) - 1) * (p / 100)
            f = math.floor(k)
            c = math.ceil(k)
            
            if f == c:
                return sorted_vals[int(k)]
            
            return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)
        except Exception as e:
            print(f"计算百分位数时出错: {e}")
            return 0.0
    
    def create_unique_output_dir(self, base_dir: str, opcode: str) -> Path:
        """创建唯一的输出目录，包含时间戳和随机标识"""
        import uuid
        
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成短随机标识符
        random_id = str(uuid.uuid4())[:8]
        
        # 构建目录名
        sm_str = f"_SM{self.sm_filter}" if self.sm_filter is not None else ""
        warp_str = f"_WARP{self.warp_filter}" if self.warp_filter is not None else ""
        cycle_str = f"_CYC{self.ns_per_cycle}" if self.ns_per_cycle else ""
        
        dir_name = f"{opcode}_{timestamp}_{random_id}{sm_str}{warp_str}{cycle_str}"
        full_path = Path(base_dir) / dir_name
        
        # 确保目录不存在，如果存在则添加序号
        counter = 1
        original_path = full_path
        while full_path.exists():
            full_path = Path(base_dir) / f"{opcode}_{timestamp}_{random_id}_{counter}{sm_str}{warp_str}{cycle_str}"
            counter += 1
        
        # 创建目录
        full_path.mkdir(parents=True, exist_ok=True)
        
        # 创建README文件说明
        readme_path = full_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# 指令分析结果: {opcode}\n\n")
            f.write(f"## 分析信息\n")
            f.write(f"- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 日志文件: {self.log_file}\n")
            f.write(f"- 指令码: {opcode}\n")
            f.write(f"- SM过滤器: {self.sm_filter if self.sm_filter is not None else '无'}\n")
            f.write(f"- WARP过滤器: {self.warp_filter if self.warp_filter is not None else '无'}\n")
            f.write(f"- 周期时间: {self.ns_per_cycle} ns/cycle\n")
            f.write(f"- 目录标识: {dir_name}\n\n")
            f.write("## 包含的文件\n")
            f.write("- `*_full_data.json`: 完整统计信息(JSON格式)\n")
            f.write("- `*_throughput_cpi.csv`: Throughput CPI详细数据\n")
            f.write("- `*_throughput_cpi_values.txt`: Throughput CPI纯数值列表\n")
            f.write("- `*_latency.csv`: Latency详细数据\n")
            f.write("- `*_latency_values.txt`: Latency纯数值列表\n")
        
        return full_path
    
    def analyze_opcode(self, opcode: str, save_raw_data: bool = True, 
                      output_dir: str = "analysis_output"):
        """分析特定指令码"""
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"指令分析: {opcode}")
        print(f"{'='*60}")
        
        data = self.get_instruction_data(opcode)
        
        if "error" in data:
            print(f"错误: {data['error']}")
            return
        
        # 打印基本信息
        print(f"总指令数: {data['total_count']}")
        print(f"Throughput CPI样本数: {data['throughput_count']}")
        print(f"Latency样本数: {data['latency_count']}")
        print(f"周期时间: {data['ns_per_cycle']} ns/cycle")
        
        # 打印统计信息
        if data['throughput_stats']:
            print(f"\nThroughput CPI统计:")
            stats = data['throughput_stats']
            print(f"  平均值: {stats['mean']:.2f}")
            print(f"  中位数: {stats['median']:.2f}")
            print(f"  最小值: {stats['min']:.2f}")
            print(f"  最大值: {stats['max']:.2f}")
            print(f"  方差: {stats['variance']:.2f}")
            print(f"  标准差: {stats['stdev']:.2f}")
            print(f"  百分位数:")
            print(f"    P25: {stats['p25']:.2f}, P50: {stats['p50']:.2f}, P75: {stats['p75']:.2f}")
            print(f"    P90: {stats['p90']:.2f}, P95: {stats['p95']:.2f}, P99: {stats['p99']:.2f}")
        
        if data['latency_stats']:
            print(f"\nLatency统计:")
            stats = data['latency_stats']
            print(f"  平均值: {stats['mean']:.2f}")
            print(f"  中位数: {stats['median']:.2f}")
            print(f"  最小值: {stats['min']:.2f}")
            print(f"  最大值: {stats['max']:.2f}")
            print(f"  方差: {stats['variance']:.2f}")
            print(f"  标准差: {stats['stdev']:.2f}")
            print(f"  百分位数:")
            print(f"    P25: {stats['p25']:.2f}, P50: {stats['p50']:.2f}, P75: {stats['p75']:.2f}")
            print(f"    P90: {stats['p90']:.2f}, P95: {stats['p95']:.2f}, P99: {stats['p99']:.2f}")
        
        # 分析分布
        self._analyze_distribution(data, opcode)
        
        # 保存原始数据
        if save_raw_data:
            unique_output_dir = self.create_unique_output_dir(output_dir, opcode)
            print(f"\n创建输出目录: {unique_output_dir}")
            self._save_raw_data(data, opcode, unique_output_dir)
        
        elapsed_time = time.time() - start_time
        print(f"\n分析完成，耗时: {elapsed_time:.2f} 秒")
    
    def _analyze_distribution(self, data: Dict, opcode: str):
        """分析数据分布"""
        if data['throughput_stats'] and data['throughput_count'] > 0:
            print(f"\nThroughput CPI分布分析:")
            cpis = [d["cpi"] for d in data["throughput_cpis"]]
            self._print_distribution_histogram(cpis, opcode + "_throughput")
        
        if data['latency_stats'] and data['latency_count'] > 0:
            print(f"\nLatency分布分析:")
            latencies = [d["latency"] for d in data["latency_values"]]
            self._print_distribution_histogram(latencies, opcode + "_latency")
    
    def _print_distribution_histogram(self, values: List[float], label: str, num_bins: int = 20):
        """打印分布直方图"""
        if not values:
            print("  无数据")
            return
        
        min_val = min(values)
        max_val = max(values)
        
        # 如果所有值相同，特殊处理
        if min_val == max_val:
            print(f"  所有值都相同: {min_val}")
            return
        
        # 创建分桶
        bins = [0] * num_bins
        bin_width = (max_val - min_val) / num_bins
        
        for val in values:
            bin_idx = min(num_bins - 1, int((val - min_val) / bin_width) if bin_width > 0 else 0)
            bins[bin_idx] += 1
        
        # 打印直方图
        max_bin_count = max(bins)
        
        print(f"  范围: [{min_val:.1f}, {max_val:.1f}], 样本数: {len(values)}")
        print(f"  {'区间':<15} {'数量':>8} {'百分比':>8} {'直方图'}")
        
        for i in range(num_bins):
            lower = min_val + i * bin_width
            upper = lower + bin_width
            count = bins[i]
            percentage = count / len(values) * 100 if len(values) > 0 else 0
            
            # 创建ASCII柱状图
            bar_length = min(50, int(percentage * 0.5))  # 缩放因子，最多50个字符
            bar = "█" * bar_length
            
            if count > 0:  # 只显示有数据的桶
                print(f"  [{lower:6.1f}-{upper:6.1f}): {count:8d} {percentage:7.1f}% {bar}")
    
    def _save_raw_data(self, data: Dict, opcode: str, output_path: Path):
        """保存原始数据到文件"""
        print(f"\n保存数据到目录: {output_path}")
        
        # 保存JSON格式的完整数据
        json_file = output_path / f"{opcode}_full_data.json"
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                # 简化数据以减小文件大小
                simplified_data = {
                    "opcode": data["opcode"],
                    "counts": {
                        "total": data["total_count"],
                        "throughput": data["throughput_count"],
                        "latency": data["latency_count"]
                    },
                    "stats": {
                        "throughput": data["throughput_stats"],
                        "latency": data["latency_stats"]
                    },
                    "ns_per_cycle": data["ns_per_cycle"],
                    "analysis_time": datetime.now().isoformat(),
                    "log_file": self.log_file,
                    "filters": {
                        "sm": self.sm_filter,
                        "warp": self.warp_filter
                    }
                }
                json.dump(simplified_data, f, indent=2)
            print(f"✓ 完整统计信息已保存到: {json_file}")
        except Exception as e:
            print(f"✗ 保存JSON文件时出错: {e}")
        
        # 保存Throughput CPI原始值（CSV格式）
        if data["throughput_cpis"]:
            csv_file = output_path / f"{opcode}_throughput_cpi.csv"
            try:
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["CPI", "Done_Time_ns", "Issue_Time_ns", "PC", "Line_Number"])
                    for item in data["throughput_cpis"]:
                        writer.writerow([
                            f"{item['cpi']:.6f}",
                            item['done_ns'],
                            item['issue_ns'],
                            f"0x{item['pc']:08x}",
                            item['line_num']
                        ])
                print(f"✓ Throughput CPI原始数据已保存到: {csv_file}")
                
                # 保存简单的CPI值列表（每行一个，便于绘图）
                simple_file = output_path / f"{opcode}_throughput_cpi_values.txt"
                with open(simple_file, 'w', encoding='utf-8') as f:
                    for item in data["throughput_cpis"]:
                        f.write(f"{item['cpi']:.6f}\n")
                print(f"✓ Throughput CPI值列表已保存到: {simple_file}")
                
                # 保存数据摘要
                summary_file = output_path / f"{opcode}_throughput_summary.txt"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"Throughput CPI 数据摘要\n")
                    f.write(f"=====================\n")
                    f.write(f"指令码: {opcode}\n")
                    f.write(f"样本数: {len(data['throughput_cpis'])}\n")
                    f.write(f"平均值: {data['throughput_stats'].get('mean', 0):.2f}\n")
                    f.write(f"中位数: {data['throughput_stats'].get('median', 0):.2f}\n")
                    f.write(f"最小值: {data['throughput_stats'].get('min', 0):.2f}\n")
                    f.write(f"最大值: {data['throughput_stats'].get('max', 0):.2f}\n")
                    f.write(f"方差: {data['throughput_stats'].get('variance', 0):.2f}\n")
                    f.write(f"标准差: {data['throughput_stats'].get('stdev', 0):.2f}\n")
            except Exception as e:
                print(f"✗ 保存Throughput CPI文件时出错: {e}")
        else:
            print(f"⚠ 无Throughput CPI数据可保存")
        
        # 保存Latency原始值（CSV格式）
        if data["latency_values"]:
            csv_file = output_path / f"{opcode}_latency.csv"
            try:
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Latency_Cycles", "Issue_Time_ns", "Done_Time_ns", "PC", "Line_Number"])
                    for item in data["latency_values"]:
                        writer.writerow([
                            f"{item['latency']:.6f}",
                            item['issue_ns'],
                            item['done_ns'],
                            f"0x{item['pc']:08x}",
                            item['line_num']
                        ])
                print(f"✓ Latency原始数据已保存到: {csv_file}")
                
                # 保存简单的Latency值列表
                simple_file = output_path / f"{opcode}_latency_values.txt"
                with open(simple_file, 'w', encoding='utf-8') as f:
                    for item in data["latency_values"]:
                        f.write(f"{item['latency']:.6f}\n")
                print(f"✓ Latency值列表已保存到: {simple_file}")
                
                # 保存数据摘要
                summary_file = output_path / f"{opcode}_latency_summary.txt"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"Latency 数据摘要\n")
                    f.write(f"================\n")
                    f.write(f"指令码: {opcode}\n")
                    f.write(f"样本数: {len(data['latency_values'])}\n")
                    f.write(f"平均值: {data['latency_stats'].get('mean', 0):.2f}\n")
                    f.write(f"中位数: {data['latency_stats'].get('median', 0):.2f}\n")
                    f.write(f"最小值: {data['latency_stats'].get('min', 0):.2f}\n")
                    f.write(f"最大值: {data['latency_stats'].get('max', 0):.2f}\n")
                    f.write(f"方差: {data['latency_stats'].get('variance', 0):.2f}\n")
                    f.write(f"标准差: {data['latency_stats'].get('stdev', 0):.2f}\n")
            except Exception as e:
                print(f"✗ 保存Latency文件时出错: {e}")
        else:
            print(f"⚠ 无Latency数据可保存")
        
#         # 创建绘图脚本
#         self._create_plot_script(output_path, opcode)
    
#     def _create_plot_script(self, output_path: Path, opcode: str):
#         """创建用于绘图的Python脚本"""
#         plot_script = output_path / f"plot_{opcode}.py"
        
#         script_content = f'''#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 自动生成的绘图脚本
# 用于可视化 {opcode} 指令的性能数据
# """

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# from pathlib import Path

# # 设置中文字体
# matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# matplotlib.rcParams['axes.unicode_minus'] = False

# # 设置图形风格
# plt.style.use('seaborn-v0_8-darkgrid')

# def load_cpi_data(cpi_file):
#     """加载CPI数据"""
#     with open(cpi_file, 'r') as f:
#         data = [float(line.strip()) for line in f if line.strip()]
#     return np.array(data)

# def load_latency_data(latency_file):
#     """加载Latency数据"""
#     with open(latency_file, 'r') as f:
#         data = [float(line.strip()) for line in f if line.strip()]
#     return np.array(data)

# def plot_histogram(data, title, xlabel, output_file, bins=50):
#     """绘制直方图"""
#     if len(data) == 0:
    
#         print("警告: ")
#         print(title)
#         print("没有数据")
#         return
    
#     plt.figure(figsize=(12, 8))
    
#     # 计算统计信息
#     mean_val = np.mean(data)
#     median_val = np.median(data)
#     std_val = np.std(data)
#     min_val = np.min(data)
#     max_val = np.max(data)
    
#     # 绘制直方图
#     n, bins, patches = plt.hist(data, bins=bins, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
#     # 添加统计线
#     plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'平均值: {mean_val:.2f}')
#     plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_val:.2f}')
#     plt.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1标准差')
#     plt.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
#     # 设置图形属性
#     plt.xlabel(xlabel, fontsize=12)
#     plt.ylabel('频率', fontsize=12)
#     plt.title(f'{title}\\n样本数: {{len(data)}}, 范围: [{min_val:.2f}, {max_val:.2f}]', fontsize=14)
#     plt.legend(fontsize=10)
#     plt.grid(True, alpha=0.3)
    
#     # 保存图形
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     print(f"已保存: {{output_file}}")
#     plt.close()

# def plot_boxplot(data, title, ylabel, output_file):
#     """绘制箱线图"""
#     if len(data) == 0:
#         print(f"警告: {title} 没有数据")
#         return
    
#     plt.figure(figsize=(10, 8))
    
#     # 绘制箱线图
#     box = plt.boxplot(data, vert=True, patch_artist=True, 
#                       boxprops=dict(facecolor='lightblue', color='darkblue'),
#                       medianprops=dict(color='red', linewidth=2),
#                       whiskerprops=dict(color='darkblue'),
#                       capprops=dict(color='darkblue'),
#                       flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
#     # 设置图形属性
#     plt.title(title, fontsize=14)
#     plt.ylabel(ylabel, fontsize=12)
#     plt.grid(True, alpha=0.3, axis='y')
    
#     # 添加统计信息
#     stats_text = f'样本数: {{len(data)}}\\n'
#     stats_text += f'平均值: {{np.mean(data):.2f}}\\n'
#     stats_text += f'中位数: {{np.median(data):.2f}}\\n'
#     stats_text += f'标准差: {{np.std(data):.2f}}'
    
#     plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
#              fontsize=10, verticalalignment='top',
#              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     # 保存图形
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     print(f"已保存: {{output_file}}")
#     plt.close()

# def plot_cdf(data, title, xlabel, output_file):
#     """绘制累积分布函数(CDF)图"""
#     if len(data) == 0:
#         print(f"警告: {title} 没有数据")
#         return
    
#     plt.figure(figsize=(12, 8))
    
#     # 排序数据
#     sorted_data = np.sort(data)
    
#     # 计算CDF
#     cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
#     # 绘制CDF
#     plt.plot(sorted_data, cdf, linewidth=2.5, color='darkgreen')
    
#     # 添加百分位线
#     percentiles = [25, 50, 75, 90, 95, 99]
#     colors = ['red', 'orange', 'blue', 'purple', 'brown', 'black']
    
#     for p, color in zip(percentiles, colors):
#         percentile_val = np.percentile(data, p)
#         plt.axvline(percentile_val, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
#         plt.text(percentile_val, 0.02, f'P{{p}}={{percentile_val:.1f}}', 
#                 rotation=90, verticalalignment='bottom', fontsize=9, color=color)
    
#     # 设置图形属性
#     plt.xlabel(xlabel, fontsize=12)
#     plt.ylabel('累积概率', fontsize=12)
#     plt.title(f'{title} 累积分布函数(CDF)', fontsize=14)
#     plt.grid(True, alpha=0.3)
#     plt.ylim(0, 1)
    
#     # 保存图形
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     print(f"已保存: {{output_file}}")
#     plt.close()

# def main():
#     # 获取当前目录
#     current_dir = Path(__file__).parent
    
#     # 数据文件路径
#     cpi_file = current_dir / '{opcode}_throughput_cpi_values.txt'
#     latency_file = current_dir / '{opcode}_latency_values.txt'
    
#     print("开始绘图...")
    
#     # 加载数据
#     if cpi_file.exists():
#         cpi_data = load_cpi_data(cpi_file)
#         print(f"加载 {len(cpi_data)} 个CPI数据点")
        
#         # 绘制CPI图形
#         plot_histogram(cpi_data, '{opcode} Throughput CPI分布', 'CPI (Cycles per Instruction)', 
#                       current_dir / '{opcode}_cpi_histogram.png')
#         plot_boxplot([cpi_data], '{opcode} Throughput CPI箱线图', 'CPI', 
#                     current_dir / '{opcode}_cpi_boxplot.png')
#         plot_cdf(cpi_data, '{opcode} Throughput CPI', 'CPI', 
#                 current_dir / '{opcode}_cpi_cdf.png')
    
#     if latency_file.exists():
#         latency_data = load_latency_data(latency_file)
#         print(f"加载 {len(latency_data)} 个Latency数据点")
        
#         # 绘制Latency图形
#         plot_histogram(latency_data, '{opcode} Latency分布', 'Latency (Cycles)', 
#                       current_dir / '{opcode}_latency_histogram.png')
#         plot_boxplot([latency_data], '{opcode} Latency箱线图', 'Latency (Cycles)', 
#                     current_dir / '{opcode}_latency_boxplot.png')
#         plot_cdf(latency_data, '{opcode} Latency', 'Latency (Cycles)', 
#                 current_dir / '{opcode}_latency_cdf.png')
    
#     print("\\n绘图完成！")
#     print("\\n生成的文件:")
#     print("1. *_histogram.png - 直方图")
#     print("2. *_boxplot.png - 箱线图")
#     print("3. *_cdf.png - 累积分布函数图")
#     print("\\n使用以下命令运行绘图脚本:")
#     print(f"  python {{plot_script.name}}")

# if __name__ == '__main__':
#     main()
# '''
        
#         try:
#             with open(plot_script, 'w', encoding='utf-8') as f:
#                 f.write(script_content)
            
#             # 使脚本可执行（在Unix-like系统上）
#             try:
#                 plot_script.chmod(0o755)
#             except:
#                 pass
            
#             print(f"✓ 绘图脚本已保存到: {plot_script}")
#             print(f"  运行命令: python {plot_script.name}")
#         except Exception as e:
#             print(f"✗ 创建绘图脚本时出错: {e}")
    
    def list_all_opcodes(self):
        """列出日志中所有出现的指令码"""
        print(f"开始解析日志以列出所有指令码...")
        
        if not self.completed_instructions:
            self.parse_log()
        
        opcode_counts = defaultdict(int)
        for instr in self.completed_instructions:
            opcode_counts[instr["opcode"]] += 1
        
        print(f"\n{'='*60}")
        print(f"日志中所有指令码统计:")
        print(f"{'='*60}")
        print(f"{'指令码':<20} {'数量':>10} {'占比':>8}")
        print(f"{'-'*40}")
        
        total = sum(opcode_counts.values())
        for opcode, count in sorted(opcode_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total * 100
            print(f"{opcode:<20} {count:>10} {percentage:>7.1f}%")
        
        print(f"{'-'*40}")
        print(f"{'总计':<20} {total:>10}")
        
        return list(opcode_counts.keys())


def main():
    parser = argparse.ArgumentParser(description="指令级性能分析工具")
    parser.add_argument("log", help="trace日志文件")
    parser.add_argument("--opcode", help="要分析的指令码（如'LW', 'CSRRS'等）")
    parser.add_argument("--list", action="store_true", help="列出所有指令码而不进行分析")
    parser.add_argument("--warp", type=int, default=None, help="只统计指定warp id")
    parser.add_argument("--sm", type=int, default=None, help="只统计指定SM id")
    parser.add_argument("--ns-per-cycle", type=int, default=None, help="手动指定1cycle=多少ns")
    parser.add_argument("--output-dir", default="analysis_output", help="输出目录（默认: analysis_output）")
    parser.add_argument("--no-save", action="store_true", help="不保存原始数据文件")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的输出目录")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("指令级性能分析工具")
    print("=" * 60)
    print(f"日志文件: {args.log}")
    print(f"SM过滤器: {args.sm}")
    print(f"WARP过滤器: {args.warp}")
    print(f"周期时间: {args.ns_per_cycle}")
    print(f"输出目录: {args.output_dir}")
    print(f"强制覆盖: {args.force}")
    
    # 检查日志文件是否存在
    if not Path(args.log).exists():
        print(f"错误: 日志文件 '{args.log}' 不存在")
        sys.exit(1)
    
    # 创建分析器
    analyzer = InstructionAnalyzer(
        log_file=args.log,
        ns_per_cycle=args.ns_per_cycle,
        sm=args.sm,
        warp=args.warp
    )
    
    # 如果指定--list，列出所有指令码
    if args.list:
        analyzer.list_all_opcodes()
        return
    
    # 如果指定--opcode，分析特定指令
    if args.opcode:
        analyzer.analyze_opcode(
            opcode=args.opcode,
            save_raw_data=not args.no_save,
            output_dir=args.output_dir
        )
    else:
        print("\n错误: 请指定要分析的指令码（--opcode）或使用--list列出所有指令码")
        print("\n示例:")
        print("  列出所有指令码:")
        print("    python cpi_data_analyze.py trace.log --list")
        print("  分析特定指令:")
        print("    python cpi_data_analyze.py trace.log --opcode LW")
        print("    python cpi_data_analyze.py trace.log --opcode CSRRS --ns-per-cycle 10 --sm 1 --warp 0")
        print("  更多选项:")
        print("    --output-dir DIR    指定输出目录")
        print("    --no-save           不保存原始数据文件")
        print("    --force             强制覆盖已存在的输出目录")
        print("    --warp N            只分析特定warp")
        print("    --sm N              只分析特定SM")
        print("    --ns-per-cycle N    手动指定周期时间")


if __name__ == "__main__":
    main()