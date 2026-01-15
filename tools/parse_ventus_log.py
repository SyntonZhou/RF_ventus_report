import re
from collections import defaultdict

# 定义要处理的文件列表
log_files = ['512.log', '512512512.log']

# 定义正则表达式
re_inst = re.compile(r'^SM\s+(\d+)\s+warp\s+(\d+)\s+0x[0-9a-fA-F]+\s+([A-Z0-9_]+)')
re_jump_evt = re.compile(r'^SM\s+(\d+)\s+warp\s+(\d+)\s+JUMP to 0x[0-9a-fA-F]+ @(\d+)ns')
re_time = re.compile(r'@(\d+)ns')

def parse_log_file(log_path, output_path):
    """解析日志文件并将结果保存到输出文件"""
    
    # 统计数据结构
    inst_cnt_sm = defaultdict(lambda: defaultdict(int))  # inst_cnt_sm[sm][mnemonic]
    jump_cnt_sm = defaultdict(int)
    first_ts_sm = {}
    last_ts_sm = {}
    
    def upd_ts(sm, ts):
        if sm not in first_ts_sm:
            first_ts_sm[sm] = ts
        last_ts_sm[sm] = ts
    
    # 读取并解析日志文件
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            if not line.startswith('SM '):
                continue

            # 提取时间戳
            mt = re_time.search(line)
            if not mt:
                continue
            ts = int(mt.group(1))

            mj = re_jump_evt.match(line)
            if mj:
                sm = int(mj.group(1))
                jump_cnt_sm[sm] += 1
                upd_ts(sm, ts)
                continue

            mi = re_inst.match(line)
            if mi:
                sm = int(mi.group(1))
                raw = mi.group(3)
                # 例如 AUIPC_0x00004197 -> AUIPC
                mnemonic = raw.split('_')[0]
                inst_cnt_sm[sm][mnemonic] += 1
                upd_ts(sm, ts)
    
    # 写入输出文件
    with open(output_path, 'w') as out_f:
        # 获取所有SM并排序
        sms = sorted(set(list(inst_cnt_sm.keys()) + list(jump_cnt_sm.keys())))
        
        # 写入汇总信息
        out_f.write(f"== Log File: {log_path} ==\n")
        out_f.write("== SM Summary ==\n")
        for sm in sms:
            total = sum(inst_cnt_sm[sm].values())
            t0 = first_ts_sm.get(sm, None)
            t1 = last_ts_sm.get(sm, None)
            dur = (t1 - t0) if (t0 is not None and t1 is not None) else 0
            ipc_proxy = (total / dur) if dur > 0 else 0.0
            summary_line = f"SM {sm}: instr={total}, jump_evt={jump_cnt_sm[sm]}, dur_ns={dur}, instr_per_ns={ipc_proxy:.6f}\n"
            out_f.write(summary_line)
            print(summary_line.strip())  # 同时在控制台显示
        
        # 写入每个SM的指令统计
        out_f.write("\n== Top instructions per SM ==\n")
        for sm in sms:
            items = sorted(inst_cnt_sm[sm].items(), key=lambda x: x[1], reverse=True)[:100]
            out_f.write(f"\nSM {sm}:\n")
            print(f"\nSM {sm}:")  # 同时在控制台显示
            for k, v in items:
                line = f"  {k:10s} {v}\n"
                out_f.write(line)
                print(f"  {k:10s} {v}")  # 同时在控制台显示
    
    print(f"\n结果已保存到: {output_path}")
    print("=" * 50)

# 处理每个日志文件
for log_file in log_files:
    try:
        # 生成输出文件名（将.log替换为_parsed.txt）
        output_file = log_file.replace('.log', '_parsed.txt')
        
        print(f"正在处理文件: {log_file}")
        parse_log_file(log_file, output_file)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {log_file}")
    except Exception as e:
        print(f"处理文件 {log_file} 时出错: {e}")