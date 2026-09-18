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
    select_interacting_csfs,
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

# 4. 生成描述符 parquet（默认 descriptor_version=2）
desc_stats = generate_descriptors_from_parquet(
    csf_parquet,
    desc_parquet,
    peel_subshells=peel_subshells,
    header_path=stats["header_file"],
)
print(desc_stats)

# 5. 读取描述符表
df = pl.read_parquet(desc_parquet)
print(df.head())

# 6. 生成保守、非精确的相互作用候选上界
interaction_stats = select_interacting_csfs(
    "reference.csf", "candidates.csf", "selected.csf", num_workers=8
)
assert interaction_stats["exact"] is False
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

`generate_descriptors_from_parquet(...)` 会读取转换后的 CSF Parquet，并输出描述符表。
通过 `descriptor_version` 可选择两种格式：

#### V2（默认）

每个 peel subshell 对应四个整数列（具名列）：

```text
sub{i}_n, sub{i}_2j, sub{i}_v, sub{i}_2k   （占据数、打印的 2J、seniority、打印的耦合 2K）
```

外加两个全局列：

```text
total_two_j, parity   （parity 取值 +1/-1）
```

GRASP 未打印的字段值为 `-1`（`MISSING`），与显式打印的 `0` 区分。V2 始终写为 `Int32` 列，
**不支持** `normalize=True`。

当提供 `header_path`（或从 `input_parquet` 同目录自动检测）时，其 SHA-256 会记录在输出
Parquet 的 key-value metadata 的 `source_header_sha256` 字段中，将描述符文件与生成它的确切
header 文件绑定。`get_parquet_info(...)` 会在 `key_value_metadata` 中返回该哈希以及完整的
格式约定（`descriptor_version`、`channels_per_subshell`、`peel_subshells`、
`feature_columns`、`global_columns`、`missing_sentinel`、`normalized`）。

#### V1（旧版）

列名为位置式 `col_0, col_1, ..., col_N`，按轨道展开为稠密三元组：

```text
[n_i, 2Q_i, 2J_cum,i]
```

显式传入 `descriptor_version=1` 即可使用该格式。原始 V1 描述符写为 `Int32` 列；
`normalize=True`（仅 V1 支持）会写为 `Float32` 列。

两种格式的输出 Parquet 均使用 ZSTD 压缩。

示例：

```python
from rcsfs import generate_descriptors_from_parquet

# V2（默认）
stats = generate_descriptors_from_parquet(
    "output.parquet",
    "descriptors.parquet",
    peel_subshells=["5s", "4d-", "4d", "5p-", "5p", "6s"],
    num_workers=8,
    header_path="output_header.toml",
)

# V1，附加归一化
stats_v1 = generate_descriptors_from_parquet(
    "output.parquet",
    "descriptors_v1.parquet",
    peel_subshells=["5s", "4d-", "4d", "5p-", "5p", "6s"],
    num_workers=8,
    normalize=True,
    descriptor_version=1,
)
```

### 3a. 从 V2 描述符还原 CSF

`restore_csfs_from_descriptors(...)` 会根据 V2 描述符 Parquet 文件及其来源
`{stem}_header.toml` 重建 CSF 文本文件。如果描述符文件记录了 `source_header_sha256`，
在写出任何内容之前会先校验该 header 是否与之匹配——不匹配意味着 header 在生成描述符之后
被重新生成或修改过。

```python
from rcsfs import restore_csfs_from_descriptors

stats = restore_csfs_from_descriptors(
    "descriptors.parquet",
    "output_header.toml",
    "restored.c",
)
# 只还原指定索引的子集，按给定顺序：
stats = restore_csfs_from_descriptors(
    "descriptors.parquet", "output_header.toml", "subset.c", indices=[0, 5, 12],
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
- `created_by`
- `key_value_metadata` —— `dict[str, str | None]`，Parquet 文件的 key-value metadata。
  对 V2 描述符文件而言，包含完整的格式约定：`descriptor_version`、`channels_per_subshell`、
  `subshell_count`、`peel_subshells`、`missing_sentinel`、`normalized`、`feature_columns`、
  `global_columns`、`source_header_sha256`、`source_header_filename`。对不带 key-value
  metadata 的文件（如普通 CSF Parquet 或 V1 描述符文件）该字典为空。

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

### 6. 从单个固定相对论占据组态生成 CSF（Rust 开发接口）

```bash
uv run cargo run --release --example generate_csfs -- \
  examples/fixed_configuration.toml /path/to/new-output.c
```

示例生成闭核 `1s` 加 `2p_{3/2}^2` 的两个合法 CSF。TOML 请求明确指定每个相对论子壳层的占据、总 `2J` 范围；输出路径必须尚不存在。这是一个 Rust 开发期入口（`csf_generation::generate_csfs(&GenerationRequest)`），不是面向用户的产品 CLI——从 Python 生成完整 CSF 列表见下面的[命令行工具](#命令行工具)一节。

## 命令行工具

安装 `rcsfs` 会同时安装一个 `rcsfs` 命令行脚本（`uv run rcsfs ...`），提供五个子命令。

### `rcsfs csfsgenerate` —— 生成新的 CSF 列表

除了交互式问答外，也可以使用 TOML 配置进行可复现的批处理。默认只生成 CSF 文本；将 `generate_descriptors` 设为 `true` 后，还会生成 CSF Parquet、header TOML、描述符 Parquet 和描述符 TOML sidecar。CSV 描述符输出不再支持。**`csfsgenerate` 目前始终写出 V1 描述符**，与库级别默认的 V2 无关（暂缓实现；参见设计文档的迁移计划）；需要 V2 输出时请改用已生成好的 CSF Parquet 配合 `gen-descriptors`。

```toml
[generate]
order = "*"
core = 3
references = ["3d(10,i)4s(2,*)4p(6,*)4d(6,*)", "3d(10,*)4s(2,i)4p(6,i)4d(6,*)"]
active_orbitals = "5s,5p,5d,4f"
j_min = 0
j_max = 12
excitations = 2
continue_lists = false

[output]
generate_descriptors = true
csf = "out.c"
parquet = "out.parquet"
descriptor_parquet = "out_descriptors.parquet"
normalize = false
```

运行：`uv run rcsfs csfsgenerate --config generation.toml`。

### `rcsfs gen-descriptors` —— 从 CSF Parquet 生成描述符 Parquet

```bash
# V2（默认）
uv run rcsfs gen-descriptors csf.parquet descriptors.parquet --header csf_header.toml

# V1，附加归一化
uv run rcsfs gen-descriptors csf.parquet descriptors.parquet \
  --header csf_header.toml --descriptor-version 1 --normalize
```

`--descriptor-version {1,2}` 选择格式（默认 `2`）；`--normalize` 仅 V1 支持，与
`--descriptor-version 2` 同时使用会报错。该命令还会写出 `{output_stem}.toml` sidecar，
镜像描述符版本与轨道列表，供不打开 Parquet 文件的工具读取。

### `rcsfs restore-csfs` —— 从 V2 描述符还原 CSF 文本文件

```bash
uv run rcsfs restore-csfs --descriptors descriptors.parquet --header csf_header.toml \
  --output restored.c

# 只还原指定索引的子集，按给定顺序
uv run rcsfs restore-csfs --descriptors descriptors.parquet --header csf_header.toml \
  --output subset.c --indices 0 5 12
```

`--header` 必须是生成该描述符文件时使用的确切 `{stem}_header.toml`。如果描述符文件记录了
`source_header_sha256`，写出任何内容之前会先与该文件校验一致性。

### `rcsfs zero-first` —— 将 CSF 列表重排为零级 + 一级空间

对应 GRASP2018 的 `rcsfzerofirst`：在每个对称性分块内，零级参考 CSF 固定在块首，随后追加一级补集。

```bash
uv run rcsfs zero-first zero.csf full.csf out.csf
```

### `rcsfs interacting` —— 生成相互作用候选上界

第一阶段实现把每个 MR 块置于输出块首，然后追加满足二体占据差预筛的候选
CSF。它保持输入顺序且并行结果确定，但尚未实现 GRASP 的重耦合、Coulomb
角因子和 Breit/SNRC 判定。因此结果只是可能含假阳性的结构上界，返回统计固定
为 `exact=False`；Dirac–Coulomb 与 Dirac–Coulomb–Breit 当前使用同一上界规则。

CLI 默认写入 `rcsf.out`，并使用 8 个 worker 线程。可通过 `--output PATH`
和 `--threads N` 覆盖这两个默认值。

```bash
uv run rcsfs interacting rcsfsmr.inp rcsf.inp --hamiltonian dc

# 可选覆盖项
uv run rcsfs interacting rcsfsmr.inp rcsf.inp --hamiltonian dc \
  --threads 4 --output selected.csf
```

## Python 公共 API

| 函数 | 说明 |
| --- | --- |
| `read_csfs(input_path, max_line_len=256, num_workers=None, *, include_block_id=False, include_coupling_signature=False, strict=True)` | 无需中间 Parquet，直接返回 `(header, dataframe)`；可选列保留 block 归属和定宽 coupling signature，严格模式会拒绝末尾不完整的 CSF |
| `convert_csfs(input_path, output_path, max_line_len=256, chunk_size=3000000, num_workers=None)` | 将 CSF 文本转换为 Parquet |
| `get_parquet_info(input_path)` | 读取 Parquet 元数据 |
| `read_peel_subshells(header_path)` | 从头文件 TOML 中提取 peel subshells |
| `generate_descriptors_from_parquet(input_parquet, output_parquet, peel_subshells, num_workers=None, normalize=False, compression=None, *, descriptor_version=2, header_path=None)` | 从转换后的 CSF 数据生成描述符 Parquet；`descriptor_version=2`（默认）写具名列，`1` 写旧版 `col_{i}` 列且是 `normalize=True` 的前提 |
| `restore_csfs_from_descriptors(descriptor_parquet, header_path, output, indices=None)` | 根据 V2 描述符 Parquet 文件及其来源 header TOML 重建 CSF 文本文件 |
| `partition_csfs(zero_parquet, zero_header, full_parquet, full_header, output_csf)` | 按对称性分块将 CSF 列表重排为零级 + 一级空间 |
| `select_interacting_csfs(reference_csf, candidate_csf, output_csf, *, hamiltonian="dirac_coulomb", method="structural_upper_bound", num_workers=None, overwrite=False)` | 写出保守且非精确的相互作用候选上界；统计固定包含 `exact=False` |
| `generate_csfs_from_transcript(transcript, output_path, normalize=False, threads=None)` | 从内存中的 `rcsfgenerate.log` 格式 transcript 生成 CSF；`rcsfs csfsgenerate` 的底层实现（始终写出 V1 描述符） |

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

## 开发命令

本地常用命令：

```bash
uv run cargo test
uv run pytest
uv run ruff check .
uv run basedpyright rcsfs/
```

Cargo 测试也应通过 `uv run` 运行，这样 PyO3 会链接到项目 uv 环境中的 Python 3.14。裸 `cargo test` 可能会发现系统 Python，例如 macOS 上 Xcode 的 Python 3.9，并在链接阶段报 `library 'python3.9' not found`。

## 许可证

MIT，详见 [LICENSE](LICENSE)。

`uv run rcsfs csfsgenerate` 默认输出到 `rcsf.out`；输出路径可省略。
