import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import pandas as pd

# 原始数据
data = """pc	count
0x80000000	8
0x80000004	8
0x80000008	8
0x8000000c	8
0x80000010	8
0x80000014	16
0x80000018	8
0x8000001c	16
0x80000020	16
0x80000024	8
0x80000028	8
0x8000002c	8
0x80000030	8
0x80000034	16
0x80000038	8
0x8000003c	8
0x80000040	8
0x80000044	8
0x80000048	8
0x8000004c	8
0x80000050	8
0x80000054	8
0x80000064	16
0x80000068	16
0x8000006c	16
0x80000070	16
0x80000074	16
0x80000078	8
0x8000007c	8
0x80000080	8
0x80000084	8
0x80000088	8
0x8000008c	8
0x80000090	8
0x80000094	8
0x80000098	16
0x8000009c	16
0x800000a0	8
0x800000b8	8
0x800000bc	8
0x800000c0	8
0x800000c4	8
0x800000c8	16
0x800000cc	16
0x800000d0	8
0x800000d4	16
0x800000d8	8
0x800000dc	16
0x800000e0	8
0x800000e4	16
0x800000e8	8
0x800000ec	16
0x800000f0	8
0x800000f4	16
0x800000f8	8
0x800000fc	16
0x80000100	8
0x80000104	8
0x80000108	16
0x8000010c	16
0x80000110	24
0x80000114	8
0x80000118	8
0x8000011c	16
0x80000120	24
0x80000124	40
0x80000128	16
0x8000012c	16
0x80000130	16
0x80000134	16
0x80000138	16
0x8000013c	16
0x80000140	8
0x80000144	16
0x80000148	8
0x8000014c	8
0x80000150	8
0x80000154	8
0x80000158	8
0x8000015c	8
0x80000160	8
0x80000164	8
0x80000168	8
0x8000016c	8
0x80000170	8
0x80000174	8
0x80000178	8
0x8000017c	8
0x80000180	8
0x80000184	256
0x80000188	512
0x8000018c	256
0x80000190	256
0x80000194	256
0x80000198	256
0x8000019c	512
0x800001a0	8
0x800001a4	16
0x800001a8	8
0x800001ac	8
0x800001b0	8
0x800001b4	396
0x800001b8	272
0x800001bc	16
0x800001c0	256
0x800001c4	256
0x800001c8	256
0x800001cc	256
0x800001d0	512
0x800001d4	512
0x800001d8	16
0x800001dc	8192
0x800001e0	16384
0x800001e4	8192
0x800001e8	8192
0x800001ec	8192
0x800001f0	16384
0x800001f4	8192
0x800001f8	16384
0x800001fc	8192
0x80000200	8192
0x80000204	16384
0x80000208	8192
0x8000020c	8192
0x80000210	8192
0x80000214	8192
0x80000218	8192
0x8000021c	8192
0x80000220	16384
0x80000224	16384
0x8000022c	262144
0x80000230	262144
0x80000234	262144
0x80000238	262144
0x8000023c	262144
0x80000240	524288
0x80000244	262144
0x80000248	262144
0x8000024c	262144
0x80000250	524288
0x80000254	524288
0x80000258	524288
0x8000025c	262144
0x80000260	262144
0x80000264	262144
0x80000268	524288
0x8000026c	262144
0x80000270	262144
0x80000274	262144
0x80000278	262144
0x8000027c	262144
0x80000280	262144
0x80000284	262144
0x80000288	524288
0x8000028c	262144
0x80000290	262144
0x80000294	262144
0x80000298	524288
0x8000029c	524288
0x800002a0	524288
0x800002a4	262144
0x800002a8	262144
0x800002ac	262144
0x800002b0	524288
0x800002b4	262144
0x800002b8	262144
0x800002c0	262144
0x800002c4	262144
0x800002c8	524288
0x800002cc	262144
0x800002d0	262144
0x800002d4	262144
0x800002dc	262144
0x800002e0	524288
0x800002e4	524288
0x800002e8	16384
0x800002ec	4194304
0x800002f0	4194304
0x800002f4	8388608
0x800002f8	8388608
0x800002fc	4194304
0x80000300	4194304
0x80000304	4194304
0x80000308	4194304
0x8000030c	4194304
0x80000310	4194304
0x80000314	8388608
0x80000318	8388608
0x8000031c	7864320
0x80000320	8192
0x80000324	8192
0x80000328	8192
0x8000032c	8192
0x80000330	8192
0x80000334	8192
0x80000338	8192
0x8000033c	8192
0x80000340	16384
0x80000344	16384
0x80000348	512
0x8000034c	16
0x80000350	16
0x80000354	16
0x80000358	16
0x8000035c	16
0x80000360	16
0x80000364	32
0x80000368	32
0x8000036c	32
0x80000370	16
0x80000374	16
0x80000378	16
0x8000037c	32
0x80000380	32
0x80000384	16
0x80000388	8
0x8000038c	8
0x80000390	8
0x80000394	16
0x80000398	16
0x8000039c	16
0x800003a0	16
0x800003a4	16
0x800003a8	16
0x800003ac	16
0x800003b4	16
0x800003b8	32
0x800003bc	16
0x800003c0	32
0x800003c4	8
0x800003c8	8
0x800003cc	16
0x800003d0	16
0x800003d4	16
0x800003d8	8
0x800003dc	8
0x800003e0	16
0x800003e4	8
0x800003e8	16
0x800003ec	8
0x800003f0	16
0x800003f4	8
0x800003f8	8
0x800003fc	16
0x80000400	16
0x80000404	16
0x80000408	8
0x8000040c	8
0x80000410	16
0x80000414	16
0x80000418	8
0x8000041c	8
0x80000420	8
0x80000424	8
0x80000428	8
0x8000042c	16
0x80000430	16
0x8000043c	8
0x80000440	16
0x80000444	8
0x80000448	16"""

# 解析数据
lines = data.strip().split('\n')
addresses = []
counts = []
offsets = []  # 地址偏移量

base_address = 0x80000000

for line in lines[1:]:  # 跳过标题行
    parts = line.split('\t')
    if len(parts) >= 2:
        addr = int(parts[0], 16)
        count = int(parts[1])
        addresses.append(addr)
        counts.append(count)
        offsets.append(addr - base_address)

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 整体视图（对数坐标）
axes[0, 0].scatter(offsets, counts, s=10, alpha=0.6, color='blue')
axes[0, 0].set_yscale('log')
axes[0, 0].set_xlabel('Address Offset from 0x80000000')
axes[0, 0].set_ylabel('Access Count (log scale)')
axes[0, 0].set_title('Memory Access Pattern (Overall View)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].ticklabel_format(style='plain', axis='x')

# 2. 前200个地址的详细视图
sample_size = min(200, len(offsets))
axes[0, 1].bar(offsets[:sample_size], counts[:sample_size], width=4, alpha=0.7, color='green')
axes[0, 1].set_xlabel('Address Offset from 0x80000000')
axes[0, 1].set_ylabel('Access Count')
axes[0, 1].set_title(f'Detailed View (First {sample_size} Addresses)')
axes[0, 1].grid(True, alpha=0.3)

# 3. 按访问次数分组的统计
# 统计不同访问次数范围的数据点数量
count_ranges = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
range_labels = ['1-10', '10-100', '100-1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M']
range_counts = []

for i in range(len(count_ranges)-1):
    low = count_ranges[i]
    high = count_ranges[i+1]
    count_in_range = sum(1 for c in counts if low <= c < high)
    range_counts.append(count_in_range)

axes[1, 0].bar(range_labels, range_counts, color='orange', alpha=0.7)
axes[1, 0].set_xlabel('Access Count Range')
axes[1, 0].set_ylabel('Number of Addresses')
axes[1, 0].set_title('Distribution of Access Frequencies')
for i, v in enumerate(range_counts):
    axes[1, 0].text(i, v, str(v), ha='center', va='bottom')

# 4. 热力图风格的可视化（按地址段着色）
# 根据访问次数分配颜色
norm_counts = np.log10(np.array(counts) + 1)
scatter = axes[1, 1].scatter(offsets, counts, c=norm_counts, 
                             cmap='viridis', s=20, alpha=0.7)
axes[1, 1].set_yscale('log')
axes[1, 1].set_xlabel('Address Offset from 0x80000000')
axes[1, 1].set_ylabel('Access Count (log scale)')
axes[1, 1].set_title('Access Heatmap (Color by Frequency)')
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1], label='Log10(Access Count)')

plt.tight_layout()
plt.show()

# 显示统计信息
print("统计信息:")
print(f"总地址数量: {len(addresses)}")
print(f"最小访问次数: {min(counts)}")
print(f"最大访问次数: {max(counts)}")
print(f"平均访问次数: {sum(counts)/len(counts):.2f}")
print(f"中位数访问次数: {sorted(counts)[len(counts)//2]}")

# 识别热点地址
hot_spots = sorted(zip(addresses, counts), key=lambda x: x[1], reverse=True)[:10]
print("\n热点地址（访问次数最高的10个）:")
for addr, count in hot_spots:
    print(f"地址: 0x{addr:08x}, 访问次数: {count:,}")