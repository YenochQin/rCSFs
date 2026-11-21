# CSF 计算描述符 V3 生成计划

2026-11-21 · 待实施 · 源码基线 `b885cec`。rCSFs 负责定义磁盘格式、生成和转换数据、提供 CLI/Python 数据集操作。本文用 `descriptor_version=3` 区分新字段语义，版本号指文件格式。

## 磁盘契约

一行表示一条有序 jj-CSF。令 `M` 为配套 TOML 中有序 peel 轨道表的长度。Parquet 每行有 `4M+2` 个非空、未归一化的逻辑 `Int8` 值。建议列名如下，`i` 是从 0 开始的轨道索引。

| 顺序 | 建议列名 | 定义 |
| --- | --- | --- |
| 每轨道第 1 位 | `sub{i}_occupation` | 占据数 q，范围为 0 至该轨道容量 |
| 每轨道第 2 位 | `sub{i}_seniority` | 物理 seniority v，空壳和满壳取 0 |
| 每轨道第 3 位 | `sub{i}_two_j_plus_one` | 子壳层自身的 2Jᵢ+1，空壳和满壳取 1 |
| 每轨道第 4 位 | `sub{i}_cumulative_two_k_plus_one` | 沿共享轨道表耦合到该位置后的 2Kᵢ+1，空壳和满壳沿用前一步 |
| 倒数第 2 位 | `total_two_j_plus_one` | 总 2J+1，始终为正 |
| 最后一位 | `parity` | 独立宇称，偶宇称 +1，奇宇称 −1 |

初始 K=0，初始累计编码为 1。角动量以宽整数进行运算和校验，在 2J、2K 为 0 至 126 时才编码成 1 至 127。所有轨道都有确定值；−1 只可能出现在负宇称列。M>0 时，最后一个轨道的累计编码等于总角动量编码。M=0 时，纯闭核用一条 `[1, +1]` 表示。

文件保留有序 core/peel 表、电子数、原始 header 字节的 SHA-256、显式块次序和长度；按适用情况保存每块的 `zero_order_counts`。使用版本、通道顺序、完整列名、逻辑类型和元数据联合识别布局。相邻的同对称性块各自有块身份。V2 读取使用自己的版本分支；两版行宽相同。[现行格式与 API](CSF_DESCRIPTOR_GUIDE.md) · [布局和元数据](../src/descriptor_transport.rs)

拟写入 `channels_per_subshell=4`、`descriptor_dtype=int8`、`seniority_encoding=physical`、`max_two_j=126`、`normalized=false`、有序 `feature_columns`、`global_columns` 及 `source_header_sha256`。V3 的元数据省去 `missing_sentinel`。生成配置的 `j_min`、`j_max` 继续以物理 2J 表示，写出行末时才加一。

## 生成与数据集操作

1. **直接生成。** 在角动量递归的 `emit()` 处，把所选子壳层态和已占据轨道的 `cumulative` 交给 V3 编码器。把局部轨道映射到 header 顺序，输出完整 J、K 码，遇到空轨道继承累计码；省略的打印标签通过已有状态表确定物理 v。内存、磁盘分段及并行缓冲路径都要在临时借用结束前写入或保留所需信息。共用同一行编码规则和行序。[角动量递归与输出接口](../src/configuration_generation/shared_generation/angular_coupling_generator.rs) · [内存生成](../src/configuration_generation/csf_generation/complete_csf_generation_pipeline.rs) · [磁盘生成](../src/configuration_generation/csf_generation/disk_csf_generation_pipeline.rs) · [seniority 状态表](../src/seniority.rs)
2. **文本转换与恢复。** 从三行 CSF 读入的显式 0 是一次真实选择。结合占据、态表、总 J 和耦合链补齐唯一可定的值；非平凡中间分支缺失且无法唯一确定时报告块、行和轨道。恢复文本时用规范打印规则选择输出位置。往返验收以同一物理 CSF 和同一 V3 整数行为准；原文本在哪一格留白无法由 V3 一般性地还原。[文本转换](../src/csfs_descriptor.rs) · [现有编码与恢复入口](../src/descriptor_v2.rs)
3. **数据集操作。** 让读取、写出、选择、分块、匹配、合并、去重及 zero-first 显式分派版本。合并轨道表时保留耦合顺序；插入的空轨道为 `(0,0,1,前一步累计码)`。不同占据表上的记录须先映射到同一有序基再匹配。选取或重分块后重建块长度和配套 header 哈希，按操作的分区语义保留或重算零阶计数。文件级 CSF 身份包含完整的占据、v、J、K 和总对称性。[数据集读写](../src/descriptor_dataset.rs) · [匹配、合并和分割](../src/descriptor_operations.rs)
4. **对外接口。** CLI、Python 与类型声明同步暴露 V3 版本和实际列名。原始数据列提供计算码；`read_csfs(..., include_coupling_signature=True)` 的派生分组列继续提供 `List(Int32)`，尾项为物理总 2J。按规范打印边界生成既有分组语义：每个已占据 peel 位置依次取打印的 2K、打印的子壳层 2J、或 0，末项使用总 2J；纯闭核签名为 `[0]`。精确选择用完整记录或原始行号。更新格式指南及 API 示例。[Python 读取端](../src/descriptor_python.rs) · [公共 API](../rcsfs/__init__.py)

## 验收样例

以 `1s, 2s, 3s, 4s` 全部列入 peel 表为例。按四元组逐列断言，最后两列分别断言。

| CSF | 1s | 2s | 3s | 4s | total_two_j_plus_one | parity |
| --- | --- | --- | --- | --- | ---: | ---: |
| A₀，在 3s 取 K=0 | (2,0,1,1) | (1,1,2,2) | (1,1,2,1) | (1,1,2,2) | 2 | +1 |
| A₁，在 3s 取 K=1 | (2,0,1,1) | (1,1,2,2) | (1,1,2,3) | (1,1,2,2) | 2 | +1 |
| B，只有 3s 开壳 | (2,0,1,1) | (2,0,1,1) | (1,1,2,2) | (0,0,1,2) | 2 | +1 |

- 校验电子数含共同闭核，宇称由占据与轨道 l 得出。对每一步的解码值检查角动量三角范围、步长和末步总 J。子壳层检查使用物理 v 和允许态表；针对普通 Suite 当前支持域校验额外态标记 w=0。
- 加入单个 `2p-` 电子的奇宇称样例（末两列 `[2, -1]`）、纯闭核、部分占据且自身 Jᵢ=0、相同 Jᵢ 不同 v、相邻同对称性块。覆盖角动量编码 127 的范围检查，以及编码 0、越界值、错误宇称、错误继承值和错配 header 的拒绝路径。
- 对上述小数据经直接生成、文本转换、规范文本往返以及内存/磁盘生成断言适用的逐行预期值；覆盖 CLI 和 Python 的筛选与派生签名。用轨道表头部、中部、尾部插入空轨道的样例验收匹配和合并。跨仓库消费者可独立复用本表，无需依赖私有计算数据。

## 跨仓库交付

rCSFs 先交付字段规范及可重建的小型 Parquet/header 样本。Suite 以原始列验收读块和成对角积分，GraspKit 以派生 DataFrame 验收分组与 mixing 对齐。Tools 在相应读写能力可用后接入作业和 ML 入口，文档示例同时注明所需文件格式与软件版本。
