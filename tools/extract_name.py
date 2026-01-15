import re

def extract_lines_without_unknown(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    # 提取符合条件的行
    filtered_lines = []
    for line in lines:
        # 找到所有尖括号内容
        matches = re.findall(r'<([^>]+)>', line)
        
        if matches:
            # 检查是否至少有一个尖括号内容不是'unknown'
            if any(match.strip().lower() != 'unknown' for match in matches):
                filtered_lines.append(line.rstrip('\n'))
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(filtered_lines))
    
    print(f"提取了 {len(filtered_lines)} 行")
    return filtered_lines

# 使用示例
extract_lines_without_unknown('disasm.txt', 'output.txt')