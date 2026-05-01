# rCSFs

[English README](README.md)

`rcsfs` 是 GraspKit 工作流中的 Rust/PyO3 加速层，用于在 Python 中处理
GRASP 风格的 CSF（Configuration State Function）数据。

它负责把大规模 CSF 文本文件转换为 Parquet 表，并基于这些表生成定长、
适合机器学习使用的描述符矩阵。Python 侧公开接口保持简洁，核心解析、
流式处理和并行 descriptor 生成逻辑由 Rust 实现。

## 核心能力

- 将 GRASP CSF 文本转换为 Parquet，并保持原始 CSF 顺序。
- 转换时生成配套的 `*_header.toml`，保存原始头信息和转换统计。
- 从 header TOML 中提取 peel subshell 列表。
- 生成列名稳定的 descriptor Parquet：`col_0..col_N`。
- 支持为机器学习流程生成归一化 descriptor。
- 使用 Rust、Arrow/Parquet 和 Rayon 提供高吞吐批处理能力。

## 仓库结构

```text
rCSFs/
├── src/                  # Rust 核心逻辑和 PyO3 绑定
├── rcsfs/                # Python 包装层、类型提示、py.typed 标记
├── tests/                # Rust 集成测试和 Python API 测试
├── docs/                 # 设计说明、变更记录、代码审查记录
├── scripts/              # 验证和对比工具
├── Cargo.toml            # Rust crate 元数据
└── pyproject.toml        # Python/maturin 构建配置
```

## 环境要求

- Python `3.14+`
- Rust toolchain 和 Cargo
- `uv`，用于管理 Python 环境
- `maturin`，用于构建 Python 扩展

当前仓库中的 `rcsfs` 主要按源码构建方式使用。编译后的 Python 扩展模块名为
`rcsfs._rcsfs`；日常使用时建议从 `rcsfs` 直接导入包装函数。

## 安装与构建

本地开发环境：

```bash
uv sync --group dev --group lint
uv run maturin develop
```

本地 release 构建：

```bash
uv sync --group dev --group lint
uv run maturin develop --release
```

构建可分发 wheel：

```bash
uv run maturin build --release
```

默认情况下，Maturin 会把 wheel 写入 `target/wheels/`；如果指定了其他输出目录，
则以 Maturin 参数为准。

## 快速开始

```python
from pathlib import Path

from rcsfs import (
    convert_csfs,
    generate_descriptors_from_parquet,
    get_parquet_info,
    read_peel_subshells,
)

input_csf = Path("tests/fixtures/sample.csf")
csf_parquet = Path("sample.parquet")
descriptor_parquet = Path("sample_descriptors.parquet")

# 1. CSF 文本转 Parquet。
conversion = convert_csfs(
    input_csf,
    csf_parquet,
    chunk_size=3_000_000,
    num_workers=None,
)
print(conversion)

# 2. 查看转换后的 Parquet 元数据。
print(get_parquet_info(csf_parquet))

# 3. 从自动生成的 header TOML 中读取 peel subshells。
peel_subshells = read_peel_subshells(conversion["header_file"])

# 4. 生成 descriptor 列。
descriptors = generate_descriptors_from_parquet(
    csf_parquet,
    descriptor_parquet,
    peel_subshells=peel_subshells,
    normalize=True,
)
print(descriptors)
```

如果需要在 Python 中查看输出表，可以在环境中安装 Polars 或 PyArrow：

```python
import polars as pl

df = pl.read_parquet("sample_descriptors.parquet")
print(df.head())
```

## 处理流程

### 1. CSF 文本转 Parquet

`convert_csfs(...)` 读取 CSF 文本文件，跳过前 5 行头信息，然后将剩余内容按
三行一组写入 Parquet：

| 列名 | 含义 |
| --- | --- |
| `idx` | 从 0 开始的 CSF 行号 |
| `line1` | 轨道占据行 |
| `line2` | 中间耦合值行 |
| `line3` | 最终耦合值和总 `J` 行 |

示例：

```python
from rcsfs import convert_csfs

stats = convert_csfs(
    "input.csf",
    "csfs.parquet",
    max_line_len=256,
    chunk_size=3_000_000,
    num_workers=8,
)
```

转换时会在输出 Parquet 旁边生成 header 文件：

```text
input_header.toml
```

返回的转换统计通常包括：

- `success`
- `input_file`
- `output_file`
- `header_file`
- `max_line_len`
- `chunk_size`
- `total_lines`
- `csf_count`
- `truncated_count`
- 失败时的 `error`

### 2. 读取 Peel Subshells

`read_peel_subshells(...)` 会从 `convert_csfs(...)` 生成的 header TOML 中读取
有序 peel subshell 列表。

```python
from rcsfs import read_peel_subshells

peel_subshells = read_peel_subshells("input_header.toml")
```

典型输出：

```python
["5s", "4d-", "4d", "5p-", "5p", "6s"]
```

### 3. 生成 Descriptor Parquet

`generate_descriptors_from_parquet(...)` 读取转换后的 CSF Parquet，并输出
descriptor 矩阵。输出表每个 descriptor 位置对应一列：

```text
col_0, col_1, ..., col_N
```

每个 peel subshell 的 descriptor 按以下三元组展开：

```text
[n_i, 2Q_i, 2J_cum,i]
```

示例：

```python
from rcsfs import generate_descriptors_from_parquet

stats = generate_descriptors_from_parquet(
    "csfs.parquet",
    "descriptors.parquet",
    peel_subshells=["5s", "4d-", "4d", "5p-", "5p", "6s"],
    num_workers=8,
    normalize=False,
)
```

descriptor 输出说明：

- 原始 descriptor 写为 `Int32` 列。
- 归一化 descriptor 写为 `Float32` 列。
- descriptor Parquet 使用 ZSTD 压缩。
- descriptor 宽度为 `3 * len(peel_subshells)`。

### 4. 查看 Parquet 元数据

```python
from rcsfs import get_parquet_info

info = get_parquet_info("descriptors.parquet")
```

返回信息通常包括：

- `file_path`
- `file_size`
- `num_rows`
- `num_columns`
- `compression`
- 可用时还会包含 `created_by` 等 writer 元数据

## Python 公共 API

| 函数 | 说明 |
| --- | --- |
| `convert_csfs(input_path, output_path, max_line_len=256, chunk_size=3000000, num_workers=None)` | 将 GRASP CSF 文本转换为 Parquet。 |
| `read_peel_subshells(header_path)` | 从生成的 `*_header.toml` 中读取 peel subshell 名称。 |
| `generate_descriptors_from_parquet(input_parquet, output_parquet, peel_subshells, num_workers=None, normalize=False)` | 从转换后的 CSF Parquet 生成 descriptor 列。 |
| `get_parquet_info(input_path)` | 返回 Parquet 文件的基础元数据。 |

当前导出的 Python 包装层定义在 [`rcsfs/__init__.py`](rcsfs/__init__.py)。更底层
的 Rust 函数属于实现细节，后续版本可能调整。

## 输入格式约定

`rcsfs` 期望 CSF 文件符合以下结构：

- 前 5 行：头信息和元数据
- 剩余内容：每 3 行为一个 CSF
- 每组第 1 行：轨道占据信息
- 每组第 2 行：中间耦合值
- 每组第 3 行：最终耦合值和总 `J`

如果输入文件不符合这个结构，转换可能失败，或者生成的 descriptor 不可靠。

## 性能建议

- 独占机器上通常保留 `num_workers=None`，交给 Rayon 使用默认线程配置。
- 共享服务器上建议显式设置 `num_workers`，避免占用过多 CPU。
- 除非已经通过 profiling 证明其他值更合适，否则保持
  `chunk_size=3_000_000`。这个默认值约等于每个流式批次处理一百万个三行 CSF。
- 如果合法 CSF 行超过默认的 256 个字符，可以增大 `max_line_len`。超过限制的行
  会计入 `truncated_count`。
- descriptor 生成通常是最耗 CPU 的步骤；启用归一化时，每个 CSF 的归一化也会
  并行执行。

## 开发命令

常用验证命令：

```bash
cargo test
uv run pytest tests/rcsfs_test.py
uv run ruff check .
uv run mypy rcsfs
```

性能优化或归一化逻辑变更后，可以使用 descriptor 对比脚本做回归检查：

```bash
uv run python scripts/compare_descriptor_outputs.py --help
```

## 相关文档

- [`docs/CHANGELOG.md`](docs/CHANGELOG.md)：面向发布的变更记录
- [`docs/CSF_DESCRIPTOR_GUIDE.md`](docs/CSF_DESCRIPTOR_GUIDE.md)：descriptor 说明
- [`docs/normalization_analysis.md`](docs/normalization_analysis.md)：归一化逻辑分析
- [`docs/performance_optimization_log.md`](docs/performance_optimization_log.md)：性能优化记录

## 兼容性说明

一些较早文档中可能出现 `convert_csfs_parallel`、`CSFProcessor`、
`CSFDescriptorGenerator`、`csfs_header`、`j_to_double_j` 等名称。这些不是当前
Python 包装层公开 API。新代码请优先使用上方 API 表中的四个函数。

## 许可证

MIT，详见 [`LICENSE`](LICENSE)。
