# 文件：find_wb.py
def find_first_wb_line(filename):
    """
    查找文件中第一次出现'WB'的那一行
    
    参数:
        filename: 要搜索的文件名
    
    返回:
        str: 包含'WB'的第一行内容，如果没有找到则返回None
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                if 'ENDPRG' in line:
                    return line.strip(), line_number
        return None, None  # 没有找到
    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 不存在")
        return None, None
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return None, None

if __name__ == "__main__":
    # 方式1：直接在代码中指定文件名
    filename = "256256256.log"
    
    
    line_content, line_num = find_first_wb_line(filename)
    
    if line_content is not None:
        print(f"在第 {line_num} 行找到 'WB'：")
        print(line_content)
    else:
        print("文件中没有找到包含 'WB' 的行")