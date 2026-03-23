# rCSFs

[English README](README.md)

rCSFs 是一个基于 Rust 的高性能工具库，用于在 Python 中处理原子物理里的 CSF（Configuration State Function）数据。

这个库主要解决两类问题：

1. 将大规模 CSF 文本文件转换为 Parquet，方便后续分析和存储。
2. 生成适合机器学习工作流使用的定长描述符表。

当前对外暴露的 Python API 以函数式接口为主，重点是流式处理、并行执行和 Parquet 优先的数据流程。

## 项目简介

rCSFs 是一个基于 Rust + PyO3 的库，面向原子结构计算、光谱分析等场景中的 CSF 数据处理。

它可以帮助你：

- 将 CSF 文本文件转换为列式 Parquet 文件。
- 在转换过程中保持原始 CSF 顺序。
- 从自动生成的头信息 TOML 中提取 `peel subshells`。
- 基于转换后的 CSF 数据生成描述符 Parquet 文件。
- 为机器学习下游流程生成可选归一化的描述符数据。

## 为什么用 rCSFs

- Rust 核心实现，吞吐高且内存行为更稳定。
- 流式批处理，适合大文件。
- 通过 `rayon` 做并行执行。
- Python 侧接口直接支持 `Path`。
- 输出是标准 Parquet，便于接入 Polars、PyArrow 等生态。

## 安装

`rcsfs` 当前要求 Python `3.14+`。

从源码构建：

```bash
git clone https://github.com/YenochQin/rCSFs.git
cd rCSFs
uv sync
maturin develop --release
```

如果按仓库的开发流程使用，也可以执行：

```bash
uv sync --group dev --group lint
maturin develop
```

## 快速开始

```python
from pathlib import Path

import polars as pl
from rcsfs import (
    convert_csfs,
    generate_descriptors_from_parquet,
    get_parquet_info,
    read_peel_subshells,
)

input_csf = Path("tests/fixtures/sample.csf")
csf_parquet = Path("sample.parquet")
desc_parquet = Path("sample_descriptors.parquet")

# 1. CSF 文本转 Parquet
stats = convert_csfs(input_csf, csf_parquet)
print(stats)

# 2. 查看 parquet 元数据
info = get_parquet_info(csf_parquet)
print(info)

# 3. 从自动生成的 header TOML 中读取 peel subshells
peel_subshells = read_peel_subshells(stats["header_file"])
print(peel_subshells[:6])

# 4. 生成描述符 parquet
desc_stats = generate_descriptors_from_parquet(
    csf_parquet,
    desc_parquet,
    peel_subshells=peel_subshells,
    normalize=True,
)
print(desc_stats)

# 5. 读取描述符表
df = pl.read_parquet(desc_parquet)
print(df.head())
```

## 典型工作流

### 1. CSF 文本转 Parquet

`convert_csfs(...)` 会读取 CSF 文件，跳过前 5 行头信息，并将后续内容按三行一组写成以下列：

- `idx`
- `line1`
- `line2`
- `line3`

同时会在输出目录生成一个配套的 TOML 文件：

```text
<input_stem>_header.toml
```

这个文件包含：

- 原始 5 行头信息
- 转换统计信息

示例：

```python
from rcsfs import convert_csfs

stats = convert_csfs(
    "input.csf",
    "output.parquet",
    max_line_len=256,
    chunk_size=3_000_000,
    num_workers=None,
)
```

返回结果中通常包含：

- `success`
- `input_file`
- `output_file`
- `header_file`
- `max_line_len`
- `chunk_size`
- `csf_count`
- `total_lines`
- `truncated_count`

### 2. 读取 peel subshells

`read_peel_subshells(...)` 用于从 header TOML 中解析 `peel subshells` 列表。

```python
from rcsfs import read_peel_subshells

peel_subshells = read_peel_subshells("output_header.toml")
```

典型输出：

```python
["5s", "4d-", "4d", "5p-", "5p", "6s"]
```

### 3. 生成描述符 Parquet

`generate_descriptors_from_parquet(...)` 会读取转换后的 CSF Parquet，并输出描述符表，列名格式为：

```text
col_0, col_1, ..., col_N
```

描述符按轨道展开，结构为：

```text
[n_i, 2Q_i, 2J_cum,i]
```

补充说明：

- 原始描述符写为 `Int32` 列。
- 归一化描述符写为 `Float32` 列。
- 输出 Parquet 使用 ZSTD 压缩。

示例：

```python
from rcsfs import generate_descriptors_from_parquet

stats = generate_descriptors_from_parquet(
    "output.parquet",
    "descriptors.parquet",
    peel_subshells=["5s", "4d-", "4d", "5p-", "5p", "6s"],
    num_workers=8,
    normalize=False,
)
```

### 4. 查看 Parquet 元数据

```python
from rcsfs import get_parquet_info

info = get_parquet_info("output.parquet")
```

返回内容通常包括：

- `file_path`
- `file_size`
- `num_rows`
- `num_columns`
- `compression`

## Python 公共 API

| 函数 | 说明 |
| --- | --- |
| `convert_csfs(input_path, output_path, max_line_len=256, chunk_size=3000000, num_workers=None)` | 将 CSF 文本转换为 Parquet |
| `get_parquet_info(input_path)` | 读取 Parquet 元数据 |
| `read_peel_subshells(header_path)` | 从头文件 TOML 中提取 peel subshells |
| `generate_descriptors_from_parquet(input_parquet, output_parquet, peel_subshells, num_workers=None, normalize=False)` | 从转换后的 CSF 数据生成描述符 Parquet |

## 输入数据格式

rCSFs 期望的 CSF 文本结构如下：

- 前 5 行：头信息 / 元数据
- 后续内容：每 3 行为一个 CSF
- 第 1 行：轨道占据信息
- 第 2 行：中间耦合值
- 第 3 行：最终耦合值和总 `J`

## 适用场景

- 为 CSF 数据建立分析用 Parquet 数据集。
- 将大规模文本格式 CSF 数据迁移到列式存储。
- 生成适合机器学习训练的描述符矩阵。
- 构建带归一化的描述符数据集。

## 性能建议

- 如果是在独占机器上运行，通常保留 `num_workers=None` 即可，交给 rayon 使用默认线程配置。
- 如果是在共享服务器上运行，建议显式设置 `num_workers`，避免和其他任务抢占 CPU。
- `chunk_size=3_000_000` 是当前默认值，除非你已经测量出更合适的参数，否则建议先保持默认。
- 如果输入文件中存在特别长的 CSF 行，并且你不希望被截断，可以适当增大 `max_line_len`。

## 关于旧文档

仓库里一些较早的文档提到过 `convert_csfs_parallel`、`CSFProcessor`、`CSFDescriptorGenerator`、`csfs_header`、`j_to_double_j` 这类接口。

这些内容并不属于当前 [rcsfs/__init__.py](rcsfs/__init__.py) 导出的公共 Python 包装层，因此本 README 只保留当前实际支持、且与代码一致的接口说明。

## 开发命令

本地常用命令：

```bash
cargo test
pytest tests/rcsfs_test.py
ruff check .
mypy rcsfs
```

## 许可证

MIT，详见 [LICENSE](LICENSE)。
