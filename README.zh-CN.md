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
uv run maturin develop --release
```

如果按仓库的开发流程使用，也可以执行：

```bash
uv sync --group dev --group lint
uv run maturin develop
```

`maturin` 安装在 uv 管理的 Python 环境中。请使用 `uv run maturin ...`，或者先激活 `.venv` 后再运行裸 `maturin`。

## 快速开始

```python
from pathlib import Path

import polars as pl
from rcsfs import (
    convert_csfs,
    generate_descriptors_from_parquet,
    get_parquet_info,
    read_csfs,
    read_peel_subshells,
)

input_csf = Path("tests/fixtures/sample.csf")
csf_parquet = Path("sample.parquet")
desc_parquet = Path("sample_descriptors.parquet")

# 直接读取 header 元数据和 CSF 数据（不生成中间 Parquet 文件）
header, csf_df = read_csfs(input_csf, num_workers=8)
print(header["block_info"])
print(csf_df.head())

# 需要保留 J^P block 归属时，设置 include_block_id=True
blocked_header, blocked_csf_df = read_csfs(
    input_csf, num_workers=8, include_block_id=True
)

# 1. 将 CSF 文本转为 Parquet，用于需要持久化的数据流程
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

`convert_csfs(...)` 会读取 CSF 文件，跳过前 5 行头信息，跳过只包含 `*` 的 GRASP
block 分隔行，并将 CSF 数据按三行一组写成以下列：

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
- block 元数据，包括 `block_info.block_lengths`
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

### 5. 使用完整整数表示解析和还原 CSF

开发中的 `complete_csf` API 可以将完整的 GRASP CSF 文件读入紧凑的内存整数结构。与机器学习描述符不同，该结构会保留子壳层占据、显式写出的零状态、seniority、实际打印出的中间耦合、总 `2J`、宇称、block 边界和 CSF 顺序。

可以用往返工具验证一个 CSF 文件：

```bash
uv run cargo run --release --example roundtrip_csf -- \
  /path/to/input.c \
  /path/to/output.c
```

输出示例：

```text
records=225157 blocks=7 occupied_entries=1974490 coupling_entries=670869 allocated_bytes=27263731 byte_identical=true
```

该命令会解析输入文件，从整数结构重新写出 CSF，然后以流式方式比较两个文本文件。若输出不是逐字节一致，命令会返回错误。输出路径必须尚不存在；已有文件（包括指向输入的符号链接或硬链接）会在写入前被拒绝，以保护原始基准文件。

Rust 代码也可以直接使用该表示：

```rust
use _rcsfs::complete_csf::CompleteCsfFile;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let csfs = CompleteCsfFile::parse_path(Path::new("input.c"))?;
    println!("CSF 数量：{}", csfs.records.len());
    println!("J/P block 数量：{}", csfs.blocks.len());
    println!("已分配字节数：{}", csfs.allocated_bytes());

    let first = &csfs.records[0];
    let occupied = csfs.occupied(first)?;
    let intermediate_couplings = csfs.couplings(first)?;
    println!(
        "占据子壳层={} 中间耦合={}",
        occupied.len(),
        intermediate_couplings.len()
    );

    csfs.write_path(Path::new("output.c"))?;
    Ok(())
}
```

`allocated_bytes()` 统计整数结构自身拥有的堆容量，不包含分配器元数据和解析、格式化时的临时缓冲，因此不等于进程峰值 RSS。

`couplings()` 只返回 GRASP 实际打印出的中间耦合，不是完整的耦合链。`kopp2.f90` 中的 `first` 标志会抑制前导的耦合，因此返回结果是稀疏的，必须通过 `boundary` 字段索引 —— 在 `e1_cc1as1.c` 上，1,524,176 个内部耦合位中只有 670,869 个被打印。需要每个边界都有取值的消费方（例如机器学习描述符的累积 `2J` 列）必须自行重建被抑制的前缀。

解析是严格的，只接受原版 GRASP 写出端实际产生的写法：block 分隔符必须恰好是 `" *"`，空 block 会被拒绝，三个表头标签会被校验，seniority 必须占据字段偏移 3 和 4，J 字段必须是约简形式的纯十进制数 —— `"+4"` 和 `"8/2"` 会报错，而不是在写出时被静默改写。每条记录都会与规范格式逐字节对照，包括填充空白和未使用的列。所有行必须以 LF 结尾；CRLF 和末行缺少 LF 都会被拒绝。这保证解析与格式化互为逆运算，因此往返成功即意味着逐字节一致。

目前这是 Rust 开发接口。串行生成器已可从一个明确给定的相对论占据组态枚举子壳层态与耦合，并直接返回该整数表示；该表示的 Python 绑定尚未实现。

### 6. 从固定相对论占据组态生成 CSF

```bash
uv run cargo run --release --example generate_csfs -- \
  examples/fixed_configuration.toml /path/to/new-output.c
```

示例生成闭核 `1s` 加 `2p_{3/2}^2` 的两个合法 CSF。TOML 请求明确指定每个相对论子壳层的占据、总 `2J` 范围和记录数上限；输出路径必须尚不存在。

Rust 接口为 `csf_generation::generate_csfs(&GenerationRequest)`，直接返回整数记录，保留原版态表顺序、seniority 标签与耦合输出规则。其上层的 `csf_generation::enumerate_occupations(&ExcitationRequest)` 解析 `rcsfgenerate` 交互输入记录，完成激发枚举、参考宇称筛选、非相对论占据拆分与多参考合并；已有列表扩展模式与 Python 生成入口仍待实现。用法、限制与 Fortran 对照测试见 [生成指南](docs/CSF_GENERATION.md)。

## Python 公共 API

| 函数 | 说明 |
| --- | --- |
| `read_csfs(input_path, max_line_len=256, num_workers=None, *, include_block_id=False, include_coupling_signature=False, strict=True)` | 无需中间 Parquet，直接返回 `(header, dataframe)`；可选列保留 block 归属和定宽 coupling signature，严格模式会拒绝末尾不完整的 CSF |
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
uv run cargo test
uv run pytest tests/rcsfs_test.py
uv run ruff check .
uv run mypy rcsfs
```

Cargo 测试也应通过 `uv run` 运行，这样 PyO3 会链接到项目 uv 环境中的 Python 3.14。裸 `cargo test` 可能会发现系统 Python，例如 macOS 上 Xcode 的 Python 3.9，并在链接阶段报 `library 'python3.9' not found`。

## 许可证

MIT，详见 [LICENSE](LICENSE)。
