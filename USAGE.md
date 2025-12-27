# rCSFs 使用说明

rCSFs 是一个高性能的 Rust/Python 混合库，用于处理原子物理中的 CSF (Configuration State Function) 数据。

## 安装

```bash
# 开发环境安装
cd /path/to/rCSFs
pixi install
pixi shell
maturin develop

# 或者使用 pip 安装（需要先构建）
pip install -e .
```

## 核心功能

### 1. CSF 文件转换（CSF → Parquet）

将 CSF 文本文件转换为高效的 Parquet 格式。

#### 功能式 API

```python
from rcsfs import convert_csfs, convert_csfs_parallel

# 串行转换（适合小文件）
convert_csfs(
    input_path="input.csf",
    output_path="output.parquet",
    max_line_len=256,      # 最大行长度
    chunk_size=100000      # 批处理大小
)

# 并行转换（适合大文件，推荐）
convert_csfs_parallel(
    input_path="input.csf",
    output_path="output.parquet",
    max_line_len=256,
    chunk_size=50000,      # 批处理大小
    num_workers=None       # 工作线程数（None=自动检测CPU核心数）
)
```

#### 面向对象 API

```python
from rcsfs import CSFProcessor

# 创建处理器实例
processor = CSFProcessor(
    max_line_len=256,
    chunk_size=30000
)

# 查看当前配置
print(processor.get_config())
# {'max_line_len': 256, 'chunk_size': 30000}

# 动态修改配置
processor.max_line_len = 512
processor.chunk_size = 50000

# 串行转换
processor.convert("input.csf", "output.parquet")

# 并行转换（推荐）
processor.convert_parallel("input.csf", "output.parquet")
```

#### 返回值说明

```python
result = convert_csfs_parallel("input.csf", "output.parquet")

# result 是一个 ParallelConversionStats 字典，包含：
# - success: bool          # 是否成功
# - input_file: str        # 输入文件路径
# - output_file: str       # 输出文件路径
# - header_file: str       # 生成的头文件路径
# - total_lines: int       # 总行数
# - csf_count: int         # CSF 条目数量
# - truncated_count: int   # 被截断的行数
# - num_workers: int       # 使用的工作线程数
# - max_line_len: int      # 最大行长度配置
# - chunk_size: int        # 批处理大小配置
```

#### 读取 Parquet 文件

```python
import polars as pl

# 读取转换后的 Parquet 文件
df = pl.read_parquet("output.parquet")

# Parquet 文件结构：
# - idx: uint64       # CSF 索引
# - line1: str        # CSF 第一行（轨道配置）
# - line2: str        # CSF 第二行（中间 J 耦合值）
# - line3: str        # CSF 第三行（最终耦合和总 J 值）

print(df.head())
```

#### 提取 CSF 文件头信息

```python
from rcsfs import csfs_header

header = csfs_header("input.csf")

# header 是一个 HeaderInfo 字典，包含：
# - header_lines: int    # 头文件行数
# - file_path: str       # 文件路径
# - line1: str           # 第1行内容
# - line2: str           # 第2行内容
# - line3: str           # 第3行内容
# - line4: str           # 第4行内容
# - line5: str           # 第5行内容

print(f"Header lines: {header['line1']}")
```

---

### 2. CSF 描述符生成（用于机器学习）

将 CSF 数据转换为固定长度的数值描述符数组。

#### 基本用法

```python
from rcsfs import CSFDescriptorGenerator

# 1. 定义 peel subshells（轨道列表）
peel_subshells = ['5s', '4d-', '4d', '5p-', '5p', '6s']

# 2. 创建描述符生成器
gen = CSFDescriptorGenerator(peel_subshells)

# 3. 查看生成器配置
print(f"轨道数量: {gen.orbital_count()}")  # 6
print(f"轨道列表: {gen.peel_subshells()}")
```

#### 解析单个 CSF

```python
# CSF 格式示例：
# line1: 轨道配置和电子数
# line2: 中间 J 耦合值
# line3: 最终耦合和总 J 值

line1 = "  5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)"
line2 = "                   3/2               2        "
line3 = "                                           4-  "

# 解析得到描述符数组
descriptor = gen.parse_csf(line1, line2, line3)

# 描述符数组格式：
# 对于每个轨道，有 3 个值：[e_count, middle_J, coupling_J]
# 长度 = 3 * orbital_count

print(descriptor)
# [2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 6.0, 3.0, 3.0, 2.0, 0.0, 0.0, 4.0, 2.0, 2.0, 2.0, 0.0, 8.0]
#  5s:  [2.0, 0.0, 0.0]  <- 2个电子，无中间J，无耦合J
#  4d-: [4.0, 0.0, 0.0]  <- 4个电子，无中间J，无耦合J
#  4d:  [6.0, 3.0, 3.0]  <- 6个电子，中间J=3/2，耦合J=3/2
#  5p-: [2.0, 0.0, 0.0]  <- 2个电子，无中间J，无耦合J
#  5p:  [4.0, 2.0, 2.0]  <- 4个电子，中间J=1，耦合J=1
#  6s:  [2.0, 0.0, 8.0]  <- 2个电子，无中间J，耦合J=4 (最后一个轨道使用最终J)
```

#### 从列表解析

```python
csf_lines = [
    "  5s ( 2)  4d-( 4)  4d ( 6)",
    "                   3/2      ",
    "                        4-  "
]

descriptor = gen.parse_csf_from_list(csf_lines)
```

#### 批量解析

```python
# 多个 CSF 的列表
csf_list = [
    [
        "  5s ( 2)  4d-( 4)  4d ( 6)",
        "                   3/2      ",
        "                        4-  "
    ],
    [
        "  5s ( 1)  4d-( 3)  4d ( 5)",
        "                   1/2      ",
        "                        2-  "
    ],
    # ... 更多 CSF
]

descriptors = gen.batch_parse_csfs(csf_list)

# descriptors 是一个列表，每个元素是一个描述符数组
print(f"生成了 {len(descriptors)} 个描述符")
```

---

### 3. J 值转换工具

将 J 值字符串转换为双倍整数表示（2J）。

```python
from rcsfs import j_to_double_j

# 分数 J 值
j_to_double_j("3/2")  # 返回: 3
j_to_double_j("5/2")  # 返回: 5
j_to_double_j("1/2")  # 返回: 1

# 整数 J 值
j_to_double_j("2")    # 返回: 4   (2 * 2 = 4)
j_to_double_j("3")    # 返回: 6   (2 * 3 = 6)
j_to_double_j("4")    # 返回: 8   (2 * 4 = 8)

# 带宇称符号
j_to_double_j("4-")   # 返回: 8   (忽略 '-' 符号)
j_to_double_j("3+")   # 返回: 6   (忽略 '+' 符号)

# 空值
j_to_double_j("")     # 返回: 0
```

---

## 完整工作流示例

### 示例 1：从 GRASP 程序处理 CSF 数据

```python
import graspkit as gk
from pathlib import Path
from rcsfs import CSFProcessor, CSFDescriptorGenerator
import polars as pl

# 1. 加载 GRASP CSF 文件
test_csf_path = Path("/home/qqqyy/test/3d8_4s2_3.c")
test_csf = gk.GraspFileLoad.from_filepath(test_csf_path, "csfs").get_csfs_data()

print(f"加载了 {len(test_csf)} 个 CSF")

# 2. 转换为 Parquet 格式
processor = CSFProcessor()
processor.convert_parallel(
    input_path=str(test_csf_path),
    output_path="output.parquet"
)

# 3. 读取 Parquet 文件
df = pl.read_parquet("output.parquet")
print(df.head())

# 4. 批量生成描述符
peel_subshells = ['5s', '4d-', '4d', '5p-', '5p', '6s']
gen = CSFDescriptorGenerator(peel_subshells)

# 从 DataFrame 中提取数据并生成描述符
csf_list = [
    [row['line1'], row['line2'], row['line3']]
    for row in df.iter_rows(named=True)
]

descriptors = gen.batch_parse_csfs(csf_list)

# 5. 转换为 NumPy 数组用于 ML
import numpy as np
X = np.array(descriptors)
print(f"描述符数组形状: {X.shape}")  # (n_csfs, 3 * n_orbitals)

# 6. 可视化或进一步处理
# ...
```

### 示例 2：结合机器学习

```python
from rcsfs import CSFDescriptorGenerator
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 1. 准备数据
peel_subshells = ['5s', '4d-', '4d', '5p-', '5p', '6s']
gen = CSFDescriptorGenerator(peel_subshells)

# 2. 从 Parquet 读取 CSF 数据并生成描述符
import polars as pl
df = pl.read_parquet("csf_data.parquet")

X = []
y = []  # 假设你有对应的能量值或其他目标变量

for row in df.iter_rows(named=True):
    descriptor = gen.parse_csf(row['line1'], row['line2'], row['line3'])
    X.append(descriptor)
    # y.append(...) # 添加你的目标值

X = np.array(X)

# 3. 训练模型
model = RandomForestRegressor()
# model.fit(X, y)

# 4. 预测
# predictions = model.predict(X_new)
```

---

## API 参考

### 类型定义

```python
from rcsfs import (
    ConversionStats,        # CSF 转换统计信息
    ParallelConversionStats, # 并行转换统计信息
    HeaderInfo              # CSF 文件头信息
)
```

### 主要函数和类

| 函数/类 | 说明 |
|---------|------|
| `convert_csfs()` | 串行转换 CSF 到 Parquet |
| `convert_csfs_parallel()` | 并行转换 CSF 到 Parquet |
| `csfs_header()` | 提取 CSF 文件头信息 |
| `get_parquet_info()` | 获取 Parquet 文件信息 |
| `CSFProcessor` | CSF 处理器类 |
| `CSFDescriptorGenerator` | CSF 描述符生成器 |
| `j_to_double_j()` | J 值转换函数 |

---

## 性能建议

1. **大文件使用并行转换**：`convert_csfs_parallel()` 比串行版本快 2-4 倍
2. **调整 chunk_size**：
   - 小文件 (< 1MB): `chunk_size=30000`
   - 中等文件 (1-100MB): `chunk_size=50000`
   - 大文件 (> 100MB): `chunk_size=100000`
3. **num_workers**：默认自动检测 CPU 核心数，通常不需要手动设置

---

## 常见问题

### Q: CSF 文件格式是什么？
A: CSF 文件格式：
   - 前 5 行：头信息（元数据）
   - 第 6 行开始：CSF 数据，每 3 行一组
     - Line 1: 轨道配置和电子数
     - Line 2: 中间 J 耦合值
     - Line 3: 最终耦合和总 J 值

### Q: 为什么使用 Parquet 格式？
A: Parquet 是列式存储格式，具有：
   - 高压缩率
   - 快速查询
   - 类型安全
   - 广泛的生态支持

### Q: 如何选择 peel_subshells？
A: peel_subshells 取决于你的原子体系。可以从 GRASP 输出文件或理论计算中获取。

---

## 更多信息

- GitHub: https://github.com/YenochQin/rCSFs
- Rust 版本要求: >=1.92.0, <1.93
- Python 版本要求: 3.13
