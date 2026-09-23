# P4：并行 descriptor 列编码（2026-09-23）

本次在 [七相位单遍报告](v2_disk_generation_final_encoding_20260922.md)之后，
使用 Parquet 58 的 `ArrowRowGroupWriterFactory` 在 Rayon 线程池中分别编码列，
再由单一 `SerializedFileWriter` 按 schema 顺序追加完成的列块并写 footer。
每次只有一个行组在途；列编码缓冲与 chunk 在分配前计入受管预算。

测量来自干净提交 `c91e51e`（完整 commit、tree、扩展 SHA-256 见
[B1 JSON](v2_disk_generation_parallel_descriptor_b1_20260923.json) 与
[B2 JSON](v2_disk_generation_parallel_descriptor_b2_20260923.json)），Apple M4、
APFS、8 线程；每个 none/zstd × verified_unique/exact 组合预热 1 次、
独立进程测量 3 次。下表只列默认的 verified_unique；同一轮也测了 exact，
并由 harness 核对四种组合的发布摘要一致。

| 输入 / segment codec | 端到端中位数，最小–最大 | descriptor encode 墙钟 / CPU | 对照单列 encode | 受管峰值 | scratch 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 / none | 3.317 秒，3.098–3.456 | 0.395 / 2.559 秒 | 1.526 秒 | 62.0 MiB | 1073.3 MiB |
| B1 / zstd | 3.161 秒，3.089–3.169 | 0.396 / 2.563 秒 | — | 62.0 MiB | 12.0 MiB |
| B2 / none | 0.979 秒，0.965–0.980 | 0.176 / 1.159 秒 | 0.727 秒 | 130.4 MiB | 506.1 MiB |
| B2 / zstd | 1.049 秒，1.044–1.053 | 0.178 / 1.173 秒 | — | 130.4 MiB | 4.5 MiB |

默认组合相对前一干净报告 `fbcf884`，B1 端到端 4.075→3.317 秒
（−18.6%），B2 1.512→0.979 秒（−35.3%）；descriptor encode 分别
1.526→0.395 秒（3.86×）与 0.727→0.176 秒（4.13×）。这次的 CPU/墙钟
约 6.5×，而前一轮约 1×，证明加速发生在真实列编码，而不只是并行准备。
累计相对 `7ad18b1` 的 8 线程端到端为 B1 14.403→3.317 秒（4.34×）、
B2 4.825→0.979 秒（4.93×）。B1 的 zstd 运行快于 none，但 none 的
0.36 秒极差和写入相位波动足以覆盖这种差异，不能据此更改默认 codec。

默认组合 RSS 中位数为 B1 211.3 MiB、B2 283.6 MiB；受管峰值分别仍由
其他阶段主导，为 62.0/130.4 MiB。七相位未计残差中位数分别为
0.015/0.009 秒，均低于 harness 拒绝阈值。macOS 未提供可用的物理进程 I/O，
表中的 scratch 是采样峰值，不能当作物理写放大。

CSF 文本、header 与 CSF Parquet 的 SHA-256 与 `fbcf884` 完全一致；
descriptor 因行组/列块布局改变，B1 为 `91f24db02e4d…`，B2 为
`51a876669b8f…`。逻辑 V2 行、schema 和文件元数据由实时两遍差分测试检查，
线程 1/2/4/8 的逻辑行不变性也在维护测试中覆盖。Parquet 的物理摘要
不是输出契约，本报告只用于识别实际测量的产物。

下一子步骤是 CLI 发布原子化。本文只证明单文件写出与编码的性能，
不声称五个 CLI 产物能作为一个原子事务发布；目标机器容量与 RSS 尚未验收。
