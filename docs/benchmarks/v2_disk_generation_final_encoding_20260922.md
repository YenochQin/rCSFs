# P4：单遍最终编码的计量（2026-09-22，`e6f3b3e` 修正版）

本文登记 [CSF_V2_GENERATION_PERFORMANCE_PLAN.md](../CSF_V2_GENERATION_PERFORMANCE_PLAN.md)
P4 的计量：最终产物改为单遍构建（取消 descriptor 回读），并与上一轮的两遍尾部对照。

> **勘误（相对本文 2026-09-22 早先版本）**
>
> 1. 早先版本报告"尾部提速 2.47×/2.36×"，其相位计时**不完整**：两个 writer 的
>    `close()`（footer）、文本 flush、发布，以及按位图筛选/收集所选列与释放格式化字符串
>    这两段随数据增长的工作，都在相位之外；B1 的相位之和（4.19 秒）比调用墙钟
>    （5.82 秒）少 1.6 秒。该结论**撤回**，本版按完整计时重报（下表的相位之和与调用墙钟
>    相差 2–4%）。
> 2. 早先版本曾称 `parquet` crate"没有受支持的并行编码入口、需自建编码器与 footer 重建"。
>    该说法**错误**，已在计划中更正：crate 提供 `ArrowRowGroupWriterFactory` +
>    `ArrowColumnChunk::append_to_row_group` 这条受支持的多线程编码路径（验证见
>    `temp/parquet_probe`）。该实现仍未做，理由见 §结论 4。

测量来自**干净工作树**的提交 `e6f3b3e`（tree 与扩展 SHA-256 `34383b62…` 记录在 JSON 的
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
调用内。六个相位加其余阶段与调用墙钟的差额即本文所称"未计开销"，实测 2–4%
（B1 0.10–0.33 秒、B2 0.05–0.14 秒），来自 transcript 解析、空间预检、`fs::metadata`、
绑定层编组与分配器尾账；不再有随数据增长却未归位的段落。

## 结果

B1（2,695,762 条，`verified_unique`）：

| codec | 端到端中位数（秒） | 最小 / 最大 | 相位之和 | 未计开销 | 尾部合计 | scratch 峰值 | RSS 中位数 | 两遍尾部（对照） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | **4.503** | 4.480 / 4.945 | 4.406 | 0.097 | 3.704 | 1073.3 MiB | 276.5 MiB | 8.536 |
| zstd | 4.695 | 4.531 / 4.832 | 4.398 | 0.297 | 3.714 | 12.0 MiB | 277.5 MiB | 9.118 |

B1 尾部相位（none）：select 0.194、prepare 1.334（CPU 10.39，7.8× 并行）、
descriptor encode 1.373（CPU≈墙钟）、descriptor write 0.000、CSF outputs encode 0.499、
CSF outputs write 0.304。

B2（560,351 条，`verified_unique`）：

| codec | 端到端中位数（秒） | 最小 / 最大 | 相位之和 | 未计开销 | 尾部合计 | scratch 峰值 | RSS 中位数 | 两遍尾部（对照） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | **1.581** | 1.560 / 1.628 | 1.532 | 0.049 | 1.213 | 506.1 MiB | 447.9 MiB | 2.575 |
| zstd | 1.669 | 1.649 / 1.703 | 1.551 | 0.118 | 1.238 | 4.5 MiB | 384.6 MiB | 2.981 |

B2 尾部相位（none）：select 0.092、prepare 0.282（CPU 2.23）、descriptor encode 0.656、
descriptor write 0.000、CSF outputs encode 0.108、CSF outputs write 0.075。

`exact` 策略下尾部与 `verified_unique` 相同（B1 3.621–3.640 秒，B2 1.180–1.234 秒）：
策略只改变去重阶段，尾部只认最终的记录序列。

## 内容差分

单遍尾部发布的 descriptor、CSF 文本与 header 的 SHA-256 与修正前、以及与 `exact` 策略
**逐一相同**（B1：`76fef6d844…` / `a9163c371f…` / `60d7650587…`；B2：`31eff7a1c5…` /
`28de0896cb…` / `c24a9e6a80…`）；只有 CSF Parquet 的摘要随批次划分变化，其 row group
边界不属于契约，逻辑行由 Rust 差分测试逐行比对。

## 结论

1. **尾部（转换+写出）提速 2.30×（B1 8.536 → 3.704 秒）与 2.12×（B2 2.575 → 1.213 秒）**，
   按完整计时达到 P4 的"并行转换阶段至少 2×"目标。端到端 B1 9.263 → 4.503 秒
   （2.06×）、B2 2.914 → 1.581 秒（1.84×）。
2. **累计相对基线 `7ad18b1` 的 8 线程：B1 14.403 → 4.503 秒（3.20×）、B2 4.825 →
   1.581 秒（3.05×）**，远超 ≥2× 目标（本机、本输入、默认去重策略）。开 zstd 时为
   3.07× / 2.89×。这些是本机数字，目标机器仍需自行测量。
3. **受管内存记账已修正**：批次的预留发生在分配之前、按真实在途结构计费、两个 Parquet
   writer 分别计费。可观察的证据是 o1 单线程下的预算阶梯：8 MiB 在生成批次被拒、
   12 MiB 在 CSF Parquet writer 被拒、16–20 MiB 在 final-encoding batch 被拒、24 MiB
   通过（峰值 21.2 MiB）；该阶梯是维护测试。B1/B2 的诚实峰值分别为 276.5/447.9 MiB（RSS
   中位数），受管峰值 130.4 MiB（B2）。
4. **编码阶段是尾部最大单项**：B1 的 descriptor encode 1.373 秒是 3.704 秒尾部的 37%，
   且 CPU≈墙钟（单线程）。这是并行 Parquet 列编码的量化理由；该实现仍未做，因为它的
   产出布局会变（`temp/parquet_probe`：逻辑行相同、字节不同），需要连同重测与重新登记
   一起做。
5. zstd 的代价在完整计时下仍只是合并阶段的一次解压（B1 +4.3%、B2 +5.6% 端到端）。

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
