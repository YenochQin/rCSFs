# rCSFs

[English README](README.md)

rCSFs 是一个基于 Rust 的高性能工具库，用于在 Python 中处理原子物理里的 CSF（Configuration State Function）数据。

这个库主要解决两类问题：

1. 将大规模 CSF 文本文件转换为 Parquet，方便后续分析和存储。
2. 生成适合机器学习工作流使用的定长描述符表。

当前对外暴露的 Python API 以函数式接口为主，重点是流式处理、并行执行和 Parquet 优先的数据流程。

## 描述符 CSF 数据集

此版本生成和处理 V3 描述符 Parquet 及配套 TOML header。Int8 列保存占据数、
物理 seniority、子壳层 2J+1、累计 2K+1、总 2J+1 和宇称。header 保存轨道表、电子数和显式块长度，
Parquet 元数据用 SHA-256 校验配套 header 的内容。

Python wheel 同时包含 `rcsfs` 包和 Rust 原生 CLI。源码构建见
[开发说明](docs/development.md)，打包流程见[发布说明](docs/releasing.md)。

## 生成和处理

在计算目录运行 `rcsfs csfsgenerate init-config`，然后编辑 `rcsfs.toml`。例如：

```toml
conf = "helium"

[csfsgenerate]
inactive_core = 0
reference_configuration = ["1s(2,*)"]
active_space = "2s"
j_min = 0
j_max = 0
excitations = 2
```

```fish
rcsfs -c rcsfs.toml
rcsfs info helium_desc.parquet --json
rcsfs convert reference.c reference_desc.parquet
rcsfs zero-first reference_desc.parquet helium_desc.parquet ordered_desc.parquet --json
rcsfs interacting reference_desc.parquet helium_desc.parquet -o filtered_desc.parquet --json
rcsfs jsplit ordered_desc.parquet --prefix block
rcsfs restore-csfs --descriptors ordered_desc.parquet --header ordered_header.toml --output ordered.c
```

生成默认采用磁盘模式。用 TOML 的 `output` 或命令行的 `--output` 指定文件名，
可以用 `--estimate-only` 只查看规模和空间估计。设置 `conf` 和 `as` 后，
生成文件名为 `{conf}_as{level}raw_desc.parquet`。
`name_desc.parquet` 和旧的 `name_descriptors.parquet` 都配对 `name_header.toml`。
用 `csfs-split` 拆分较小 active spaces，各输出均带自己的 header，文件名需与输入对不同。

`zero-first` 和 `interacting` 直接读写 V3 描述符及配套 header。
`zero-first` 在每个显式对称性块中先写零阶参考，再按原顺序写参考中没有的全空间 CSF。
输出 header 的 `[block_info].zero_order_counts` 按文件分块顺序记录各块开头的零阶 CSF 数。
`interacting` 先写参考，再写参考中没有且通过占据数结构上界筛选的候选 CSF。
这个上界可能保留完整角代数筛选会排除的候选。

Python 中用 `convert_csfs` 显式导入文本，`read_csfs` 读取 header 和数值行，
`select_csfs` 按递增且唯一的零基索引选择，`restore_csfs_from_descriptors` 显式恢复
文本。直接生成用 `generate_csfs_from_transcript(transcript, output_path)`。
完整示例见[英文说明](README.md)，变更见[更新日志](docs/CHANGELOG.md)。

[CSF 生成源码布局](docs/GENERATION_SOURCE_LAYOUT.md)
