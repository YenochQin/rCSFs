# P4：单遍最终编码的计量（2026-09-22，`45618fc` 修正版）

本文登记 [CSF_V2_GENERATION_PERFORMANCE_PLAN.md](../CSF_V2_GENERATION_PERFORMANCE_PLAN.md)
P4 的计量：最终产物改为单遍构建（取消 descriptor 回读），并与上一轮的两遍尾部对照。

> **状态（2026-09-23）：历史报告，已被实现修正取代。** 后续评审发现本报告没有把
> segment IPC 的读取/解压纳入独立相位，且两个 `ArrowWriter` 未设置内存模型假定的
> 8,192 行 row-group 上限。当前代码已经修正这两点，并在解码前预留 source batch；这会
> 改变七相位计时和 descriptor 的物理 Parquet 布局。因此下列 `45618fc` 数字只说明当时
> 实现，不再完成当前 P4 性能验收。新结果必须在修正提交后的干净树上重新登记。

> **勘误（相对本文 2026-09-22 早先版本）**
>
> 1. 早先版本报告"尾部提速 2.47×/2.36×"，其相位计时**不完整**：两个 writer 的
>    `close()`（footer）、文本 flush、发布，以及按位图筛选/收集所选列与释放格式化字符串
>    这两段随数据增长的工作，都在相位之外；B1 的相位之和（4.19 秒）比调用墙钟
>    （5.82 秒）少 1.6 秒。该结论**撤回**，本版按完整计时重报（下表的相位之和与调用墙钟
>    相差 1–6%）。
> 2. 早先版本曾称 `parquet` crate"没有受支持的并行编码入口、需自建编码器与 footer 重建"。
>    该说法**错误**，已在计划中更正：crate 提供 `ArrowRowGroupWriterFactory` +
>    `ArrowColumnChunk::append_to_row_group` 这条受支持的多线程编码路径（验证见
>    `temp/parquet_probe`）。该实现仍未做，理由见 §结论 4。

测量来自**干净工作树**的提交 `45618fc`（tree 与扩展 SHA-256 `34383b62…` 记录在 JSON 的
`environment.git.tree` 与 `environment.extension.module_sha256` 中）；Apple M4 10 核
16 GiB，APFS；8 线程；每个组合预热 1 次后测量 3 次，每次测量在独立进程中进行，表中为
中位数。对照数据来自同一机器、同一输入、同一线程数与同一去重策略的上一轮
[dedup 报告](v2_disk_generation_dedup_20260922.md)（提交 `47e04ea`，两遍尾部，其阶段
计时覆盖整个 merge 与 restore 函数，因此可与本表逐项对照）。

原始报告：
[final encoding b1](v2_disk_generation_final_encoding_b1_20260922.json)、
[final encoding b2](v2_disk_generation_final_encoding_b2_20260922.json)。

## 计时的完整性与可核对性

单遍尾部分六个相位上报，命名对应实际操作：

| 相位 | 覆盖 |
| --- | --- |
| `final_encoding_select` | 按幸存位图筛选每批的行、收集所选列（串行，唯一逐行接触数据的相位） |
| `final_encoding_prepare` | 解码、校验、格式化（`threads` 线程池内并行） |
| `final_encoding_descriptor_encode` | descriptor Parquet 的转换与压缩（不含其文件写入） |
| `final_encoding_descriptor_write` | descriptor 的文件写入、footer（由写入器内部计数） |
| `final_encoding_csf_outputs_encode` | CSF 文本渲染与 CSF Parquet 转换 |
| `final_encoding_csf_outputs_write` | CSF 文本与 Parquet 的文件写入、flush、footer、发布 |

`setup`（transcript 解析、scratch 创建）与 `header_write` 也各自成项，因为它们同样在
调用内。六个相位加其余阶段与调用墙钟的差额即本文所称"未计开销"，实测 1–6%
（B1 0.02–0.30 秒、B2 0.06–0.14 秒），来自 transcript 解析、空间预检、`fs::metadata`、
绑定层编组与分配器尾账。基准 harness 现在**逐次记录该残差并在超过调用时长 10%
（或 50 ms）时拒绝写出报告**，因此"某个相位没有计时器"会在测量时被拦住，而不是事后
由端到端时间反推出来。

## 结果

B1（2,695,762 条，`verified_unique`）：

| codec | 端到端中位数（秒） | 最小 / 最大 | 相位之和 | 未计开销 | 尾部合计 | scratch 峰值 | RSS 中位数 | 两遍尾部（对照） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | **4.472** | 4.414 / 4.591 | 4.388 | 0.084 | 3.682 | 1073.3 MiB | 285.3 MiB | 8.536 |
| zstd | 4.846 | 4.802 / 4.865 | 4.542 | 0.304 | 3.826 | 12.0 MiB | 277.5 MiB | 9.118 |

B1 尾部相位（none）：select 0.195、prepare 1.331（CPU 10.46，7.8× 并行）、
descriptor encode 1.369（CPU≈墙钟）、descriptor write 0.000、CSF outputs encode 0.497、
CSF outputs write 0.290。

B2（560,351 条，`verified_unique`）：

| codec | 端到端中位数（秒） | 最小 / 最大 | 相位之和 | 未计开销 | 尾部合计 | scratch 峰值 | RSS 中位数 | 两遍尾部（对照） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | **1.594** | 1.580 / 1.598 | 1.537 | 0.057 | 1.203 | 506.1 MiB | 460.9 MiB | 2.575 |
| zstd | 1.692 | 1.671 / 1.838 | 1.557 | 0.135 | 1.254 | 4.5 MiB | 375.8 MiB | 2.981 |

B2 尾部相位（none）：select 0.092、prepare 0.275（CPU 2.17）、descriptor encode 0.655、
descriptor write 0.000、CSF outputs encode 0.106、CSF outputs write 0.075。

`exact` 策略下尾部与 `verified_unique` 相同（B1 3.667–3.764 秒，B2 1.200–1.224 秒）：
策略只改变去重阶段，尾部只认最终的记录序列。

## 内容差分

单遍尾部发布的 descriptor、CSF 文本与 header 的 SHA-256 与修正前、以及与 `exact` 策略
**逐一相同**（B1：`76fef6d844…` / `a9163c371f…` / `60d7650587…`；B2：`31eff7a1c5…` /
`28de0896cb…` / `c24a9e6a80…`）；只有 CSF Parquet 的摘要随批次划分变化，其 row group
边界不属于契约，逻辑行由 Rust 差分测试逐行比对。

## 结论

1. **尾部（转换+写出）提速 2.32×（B1 8.536 → 3.682 秒）与 2.14×（B2 2.575 → 1.203 秒）**，
   按完整计时达到 P4 的"并行转换阶段至少 2×"目标。端到端 B1 9.263 → 4.472 秒
   （2.07×）、B2 2.914 → 1.594 秒（1.83×）。
2. **累计相对基线 `7ad18b1` 的 8 线程：B1 14.403 → 4.472 秒（3.22×）、B2 4.825 →
   1.594 秒（3.03×）**，远超 ≥2× 目标（本机、本输入、默认去重策略）。开 zstd 时为
   2.97× / 2.85×。这些是本机数字，目标机器仍需自行测量。
3. **当时的受管内存结论已被更严格的维护测试替代**：当前实现还在 IPC 解码前预留 source
   batch，并把两个 writer 的 row group 限为 8,192 行。预算阶梯现为 8 MiB 在生成批次
   拒绝、12–24 MiB 在 writer/source/final batch 拒绝、28 MiB 通过；本报告的 24 MiB
   通过结论不再适用。B1/B2 的 RSS 数字仍是 `45618fc` 的历史观测。
4. **编码阶段是尾部最大单项**：B1 的 descriptor encode 1.369 秒是 3.682 秒尾部的 37%，
   且 CPU≈墙钟（单线程）。这是并行 Parquet 列编码的量化理由；该实现仍未做，因为它的
   产出布局会变（`temp/parquet_probe`：逻辑行相同、字节不同），需要连同重测与重新登记
   一起做。
5. zstd 的历史端到端代价为 B1 +8.4%、B2 +6.1%；当前实现把 IPC 解压明确归入
   `final_encoding_read`，需重测后才能更新相位归因。

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
