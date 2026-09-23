# P4：单遍最终编码的计量（2026-09-23，`fbcf884` 七相位复测）

本文登记 [CSF_V2_GENERATION_PERFORMANCE_PLAN.md](../CSF_V2_GENERATION_PERFORMANCE_PLAN.md)
P4 的计量：最终产物改为单遍构建（取消 descriptor 回读），并与上一轮的两遍尾部对照。

> **勘误（相对本文 2026-09-22 早先版本）**
>
> 1. 早先版本报告"尾部提速 2.47×/2.36×"，其相位计时**不完整**：两个 writer 的
>    `close()`（footer）、文本 flush、发布，以及按位图筛选/收集所选列与释放格式化字符串
>    这两段随数据增长的工作，都在相位之外；B1 的相位之和（4.19 秒）比调用墙钟
>    （5.82 秒）少 1.6 秒。该结论**撤回**，本版按完整计时重报（下表的相位之和与调用墙钟
>    相差 1–6%）。`fbcf884` 又补齐 segment IPC 读取/解压，并把 writer row group 限为
>    8,192 行；本页现已用七相位口径重测。
> 2. 早先版本曾称 `parquet` crate"没有受支持的并行编码入口、需自建编码器与 footer 重建"。
>    该说法**错误**，已在计划中更正：crate 提供 `ArrowRowGroupWriterFactory` +
>    `ArrowColumnChunk::append_to_row_group` 这条受支持的多线程编码路径（验证见
>    `temp/parquet_probe`）。该实现仍未做，理由见 §结论 4。

测量来自**干净工作树**的提交 `fbcf884`（tree `1a6d542…`，扩展 SHA-256 `782f9bdd…`，均记录在 JSON 的
`environment.git.tree` 与 `environment.extension.module_sha256` 中）；Apple M4 10 核
16 GiB，APFS；8 线程；每个组合预热 1 次后测量 3 次，每次测量在独立进程中进行，表中为
中位数。对照数据来自同一机器、同一输入、同一线程数与同一去重策略的上一轮
[dedup 报告](v2_disk_generation_dedup_20260922.md)（提交 `47e04ea`，两遍尾部，其阶段
计时覆盖整个 merge 与 restore 函数，因此可与本表逐项对照）。

原始报告：
[final encoding b1](v2_disk_generation_final_encoding_b1_20260922.json)、
[final encoding b2](v2_disk_generation_final_encoding_b2_20260922.json)。

## 计时的完整性与可核对性

单遍尾部分七个相位上报，命名对应实际操作：

| 相位 | 覆盖 |
| --- | --- |
| `final_encoding_read` | segment IPC 读取、解压与批次结构校验；source batch 在解码前预留 |
| `final_encoding_select` | 按幸存位图筛选每批的行、收集所选列（串行，唯一逐行接触数据的相位） |
| `final_encoding_prepare` | V2 解码、合法性校验与文本格式化（`threads` 线程池内并行） |
| `final_encoding_descriptor_encode` | descriptor Parquet 的转换与压缩（不含其文件写入） |
| `final_encoding_descriptor_write` | descriptor 的文件写入、footer（由写入器内部计数） |
| `final_encoding_csf_outputs_encode` | CSF 文本渲染与 CSF Parquet 转换 |
| `final_encoding_csf_outputs_write` | CSF 文本与 Parquet 的文件写入、flush、footer、发布 |

`setup`（transcript 解析、scratch 创建）与 `header_write` 也各自成项，因为它们同样在
调用内。七个相位加其余阶段与调用墙钟的差额即本文所称"未计开销"；默认组合的残差
中位数为 B1 0.013 秒、B2 0.009 秒。基准 harness **逐次记录残差并在其绝对值超过调用
时长 10%（或 50 ms）时拒绝写出报告**：正残差拦漏计，负残差拦相位重叠/重复计时。

## 结果

B1（2,695,762 条，`verified_unique`）：

| codec | 端到端中位数（秒） | 最小 / 最大 | 相位之和 | 未计开销 | 尾部合计 | scratch 峰值 | RSS 中位数 | 两遍尾部（对照） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | **4.075** | 4.005 / 4.127 | 4.048 | 0.013 | 3.422 | 1073.3 MiB | 246.2 MiB | 8.536 |
| zstd | 4.198 | 4.188 / 4.242 | 4.184 | 0.015 | 3.622 | 12.0 MiB | 205.3 MiB | 9.118 |

B1 尾部相位（none）：read 0.093、select 0.193、prepare 0.810（CPU 6.278，7.8× 并行）、
descriptor encode 1.526（CPU≈墙钟）、descriptor write 0.005、CSF outputs encode 0.480、
CSF outputs write 0.315。

B2（560,351 条，`verified_unique`）：

| codec | 端到端中位数（秒） | 最小 / 最大 | 相位之和 | 未计开销 | 尾部合计 | scratch 峰值 | RSS 中位数 | 两遍尾部（对照） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | **1.512** | 1.509 / 1.513 | 1.503 | 0.009 | 1.209 | 506.1 MiB | 390.9 MiB | 2.575 |
| zstd | 1.592 | 1.592 / 1.607 | 1.581 | 0.010 | 1.309 | 4.5 MiB | 273.1 MiB | 2.981 |

B2 尾部相位（none）：read 0.047、select 0.092、prepare 0.173（CPU 1.352）、
descriptor encode 0.727、descriptor write 0.002、CSF outputs encode 0.100、
CSF outputs write 0.068。

`exact` 策略下尾部与 `verified_unique` 相同（none：B1 3.400 秒，B2 1.210 秒）：
策略只改变去重阶段，尾部只认最终的记录序列。

## 内容差分

同一输入的 none/zstd 与 verified/exact 四种组合发布完全一致的四个摘要。CSF 文本与
header 延续历史摘要；受 8,192 行 row-group 上限影响，descriptor 与 CSF Parquet 的物理
摘要按预期改变。B1 为 CSF `76fef6d844…`、descriptor `e6a08e0716…`、CSF Parquet
`bc3bfc5b0c…`、header `60d7650587…`；B2 为 `31eff7a1c5…`、`a776bbaa7b…`、
`4b604e34d6…`、`c24a9e6a80…`。逻辑行由实时 Rust 差分覆盖，不冻结 Parquet 布局。

## 结论

1. **尾部（转换+写出）提速 2.49×（B1 8.536 → 3.422 秒）与 2.13×（B2 2.575 → 1.209 秒）**，
   七相位完整计时下仍达到 P4 的 ≥2× 调研目标。
2. **累计相对基线 `7ad18b1` 的 8 线程：B1 14.403 → 4.075 秒（3.53×）、B2 4.825 →
   1.512 秒（3.19×）**。开 zstd 时为 3.43× / 3.03×。这些是本机数字，目标机器仍需测量。
3. 解码前 source-batch 预留与 8,192 行 row-group 上限均已进入测量；受管峰值中位数为
   B1 62.0 MiB、B2 130.4 MiB。o1 的维护预算阶梯为 8 MiB 在生成批次拒绝、12–24 MiB
   在 writer/source/final batch 拒绝、28 MiB 通过。
4. **descriptor encode 仍是尾部最大单项**：B1 1.526/3.422 秒（45%），B2
   0.727/1.209 秒（60%），且 CPU≈墙钟。这为下一步实现受支持的并行列块编码提供了明确
   的端到端收益上限；落地后仍需重新登记布局摘要与性能。
5. zstd 的端到端代价为 B1 +3.0%、B2 +5.3%；新增 read 相位显示解压主要增加
   0.197/0.075 秒，归因不再混入 select/prepare。

## 未测量与边界

- 目标机器与目标文件系统；本机 NVMe/APFS、页缓存未清空，macOS 无 `process_io`，
  因此没有物理 I/O 与写放大数字。
- 1/2/4/8 线程的字节对照由 Rust 测试
  `the_combined_final_pass_is_thread_invariant` 覆盖（同一输入跨线程发布相同的 CSF
  字节、相同的 V2 行与相同的 CSF Parquet 行），本页只测 8 线程。
- 发布路径：流水线内部（暂存→发布）已是 rename 语义并含竞争写入测试；CLI 从 staging
  到目的地的复制、部分发布失败策略仍是 P4 未实施的子步骤，本页不含其数字。
- 两遍尾部自 `de97127` 起只作为测试内参考实现存在，无法再通过公开 API 复测；表中的
  对照取自 `47e04ea` 已登记的报告。
