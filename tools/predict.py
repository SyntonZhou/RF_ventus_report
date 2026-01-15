# 原代码中导入的prod未使用，此处注释（若需保留可取消注释）
# from math import prod

# 定义基础参数B
n = 4096
tile = 16
B = n / tile

# 初始化存储指令计算值的字典
pred = {}

# 核心：各类指令的数值计算公式（基于B的多项式）
pred['VLW12_V'] = 544 * (B**3)
pred['VFMADD_VV'] = 128 * (B**3)
pred['VADD_VI'] = 128 * (B**3)
pred['VMADD_VX'] = 8 * (B**3)
pred['SETPRC'] = 304 * (B**3) + 32 * (B**2) + 16 * B + 96
pred['JAL'] = 288 * (B**3) + 48 * (B**2) + 48 * B + 192
pred['VBEQ'] = 288 * (B**3) + 64
pred['JUMP'] = 144 * (B**3) + 24 * (B**2) + 24 * B + 152
pred['VMV_V_X'] = 176 * (B**3) + 40 * (B**2) + 24 * B + 128
pred['AUIPC'] = 152 * (B**3) + 16 * (B**2) + 8 * B + 96
pred['LUI'] = 16 * (B**3) + 16 * (B**2) + 8
pred['VADD_VV'] = 152 * (B**3) + 16 * (B**2) + 8 * B + 24
pred['VADD_VX'] = 152 * (B**3) + 8 * (B**2) + 48
pred['VADD12_VI'] = 136 * (B**3) + 8 * (B**2) + 8 * B
pred['VSLL_VI'] = 16 * (B**3) + 8 * (B**2) + 32
pred['VAND_VV'] = 16 * (B**3) + 8 * (B**2)
pred['VSW12_V'] = 16 * (B**3) + 8 * (B**2)
pred['LW'] = 32 * (B**3) + 32 * (B**2) + 416
pred['VMSLT_VX'] = 32 * (B**3) + 16 * (B**2)
pred['JOIN'] = 24 * (B**3) + 16 * (B**2) + 8 * B + 32
pred['VBNE'] = 16 * (B**2) + 16
pred['BGE'] = 8 * (B**2) + 8 * B + 8
pred['REGEXT'] = 12 * B + 68
pred['VBLT'] = 16 * (B**3) + 16 * (B**2) + 16 * B + 16
pred['VMUL_VX'] = 16 * B
pred['ADDI'] = 128 * (B**3) + 272

# 定义常量类指令的固定值
consts = {
    'CSRRS': 144, 'SW': 104, 'JALR': 96, 'VLW_V': 40, 'MUL': 24,
    'VSW_V': 24, 'ADD': 16, 'CSRRW': 16, 'VID_V': 16, 'VREMU_VX': 16,
    'VSETVLI': 8, 'BEQ': 8, 'VDIVU_VX': 8, 'ENDPRG': 8
}

# 将常量指令值合并到pred字典中
pred.update(consts)

# 可选：打印最终计算结果（便于验证）
if __name__ == "__main__":
    print("各指令的计算结果：")
    for instr, value in sorted(pred.items()):
        print(f"{instr:12s} = {value}")