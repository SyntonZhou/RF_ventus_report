# 配置参数
SOURCE_FILE = "256256256.log"  # 源log文件路径
TARGET_FILE = "extracted_lines.log"  # 输出新文件路径
START_LINE = 1  # 起始行号（含）
END_LINE = 200    # 结束行号（含）

# 逐行读取并提取目标行
try:
    # with语句自动管理文件打开/关闭，避免资源泄露
    with open(SOURCE_FILE, 'r', encoding='utf-8') as infile, \
         open(TARGET_FILE, 'w', encoding='utf-8') as outfile:
        
        line_number = 0  # 行号计数器（从1开始匹配自然行号）
        for line in infile:
            line_number += 1
            
            # 跳过起始行之前的内容
            if line_number < START_LINE:
                continue
            
            # 写入目标行
            if START_LINE <= line_number <= END_LINE:
                outfile.write(line)
            
            # 超过结束行后提前退出，提升效率
            if line_number > END_LINE:
                break
    
    print(f"成功提取{START_LINE}-{END_LINE}行，保存至：{TARGET_FILE}")

# 异常处理（文件不存在/权限不足等）
except FileNotFoundError:
    print(f"错误：找不到文件 {SOURCE_FILE}")
except PermissionError:
    print(f"错误：没有权限读取/写入文件")
except Exception as e:
    print(f"未知错误：{e}")