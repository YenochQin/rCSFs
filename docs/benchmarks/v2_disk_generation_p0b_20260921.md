# V2 磁盘生成基线（2026-09-21，P0b）

本文件登记 [CSF_V2_GENERATION_PERFORMANCE_PLAN.md](../CSF_V2_GENERATION_PERFORMANCE_PLAN.md)
P0b 要求的可重复基线。输入是仓库内登记的 transcript，不依赖 GRASP 可执行文件或私有数据。

## 输入

| 名称 | 文件 | SHA-256 | 组态 | 去重前 CSF |
|---|---|---:|---:|---:|
| B1 | `tests/fixtures/b1_cc1_5spdfg_3exc.rcsfgenerate` | `dc5f73baeb3bd5ee6d460393213e14b42bd06c5bf5186ffdb346fc310d0c1d99` | 19,480 | 2,695,762 |
| B2 | `tests/fixtures/b2_cc1_fullas_2exc.rcsfgenerate` | `280822983d7f008d8a5a62b0a0fe5b507c2c33e6aa24316834c6ca9f3a8abbf8` | 19,243 | 560,351 |

两者都使用 §2.1 的两个参考组态与 2J = 8：B1 改用 `5s,5p,5d,5f,5g`、3 次激发，B2 使用完整活性空间 `9s,9p,9d,9f,7g`、2 次激发。登记值由 `tests/fixtures/transcripts.toml` 绑定；哈希不符时基准脚本拒绝运行。计数与 §2.3 记录的 19,480 / 2,695,762 和 19,243 / 560,351 一致，因此这就是该段所用的输入。

## 测量条件

Apple M4，10 核，16 GiB，macOS 26.6.2（Darwin 25.6.0），APFS；release 扩展，共享 Tools 环境；Rust 源码为 `a993ca4`，测量时工作树含本次登记的基准脚本与 fixture（`describe` 记录为 `a993ca4-dirty`，原始 JSON 中可见）。未清空页缓存，未测物理设备 I/O（macOS 不上报 `/proc/self/io`），因此表内数值是本机对照，不是目标机器性能结论。

每种组合预热 1 次后测量 3 次，按 1 → 8 线程顺序执行；表中为中位数。计时区间只覆盖生成调用（含发布 staging 产物），临时目录的创建与删除不计入，删除时间单独记录为 `cleanup_seconds`。scratch 峰值由 0.2 秒采样的临时目录监控给出，是有损观测；受管内存来自 `resource_stats`，与该进程 RSS 是不同指标。

| 输入 | 线程 | 端到端中位数（秒） | 最小 / 最大 | 清理（秒） | scratch 峰值 / 结束时 | 结束文件数 | 受管内存峰值 | 进程 RSS 峰值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 1 | 15.942 | 15.865 / 16.460 | 0.03 | 2142 / 1073 MiB | 24 | 29 MiB | 371 MiB |
| B1 | 8 | 14.403 | 13.909 / 14.851 | 0.07 | 2142 / 1073 MiB | 24 | 43 MiB | 408 MiB |
| B2 | 1 | 5.658 | 5.653 / 5.896 | 0.03 | 1002 / 506 MiB | 8 | 61 MiB | 660 MiB |
| B2 | 8 | 4.825 | 4.786 / 5.065 | 0.03 | 998 / 506 MiB | 8 | 88 MiB | 660 MiB |

`rss_peak_bytes` 是自进程启动以来的峰值（Linux/macOS 的 `ru_maxrss` 语义），是单调量：靠后的行可能包含更早运行已经达到的峰值，因此同一文件内后续行的 RSS 只能与前面的行比较，不能当作单次运行的独立测量。受管内存只覆盖 P1 记账的受管结构，与 RSS 相差一个数量级以上，差距来自 Arrow/Parquet 运行库、allocator 元数据、线程栈与页缓存。

阶段墙钟中位数（秒）：

| 输入 | 线程 | 枚举 | CSF 生成 | 去重 | 合并描述符 | 文本与 CSF Parquet 还原 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 1 | 0.015 | 3.259 | 4.234 | 2.868 | 5.595 |
| B1 | 8 | 0.015 | 1.216 | 4.340 | 2.938 | 5.695 |
| B2 | 1 | 0.043 | 1.183 | 1.737 | 1.428 | 1.259 |
| B2 | 8 | 0.043 | 0.433 | 1.693 | 1.405 | 1.262 |

结论与 §2.3 一致：只有生成阶段随线程数缩放，去重、合并与还原三段的 CPU 时间各自接近其墙钟时间，合计约 10.9 秒（B1，8 线程），占端到端约 76%。这两组输入都未删除重复行（`duplicate_count` 为 0）。

本机数值高于计划 §2.3 记录的 11.5 秒，是因为机器不同（10 核 M4 与当时的 8 核环境）且此处包含发布 staging 阶段；两者不可直接相减比较。

## 复现

```bash
source ../graspkit-tools/.venv/bin/activate
python scripts/benchmark_v2_generation.py tests/fixtures/b1_cc1_5spdfg_3exc.rcsfgenerate \
  --threads 1 8 --repeats 3 --warmup 1 \
  --output docs/benchmarks/v2_disk_generation_p0b_b1_20260921.json
python scripts/benchmark_v2_generation.py tests/fixtures/b2_cc1_fullas_2exc.rcsfgenerate \
  --threads 1 8 --repeats 3 --warmup 1 \
  --output docs/benchmarks/v2_disk_generation_p0b_b2_20260921.json
```

原始测量见 `v2_disk_generation_p0b_b1_20260921.json` 与 `v2_disk_generation_p0b_b2_20260921.json`。

## 来源可审计性

这两份报告测量于 2026-09-21，早于基准脚本记录源码身份（`git.tree` 与已加载扩展的
SHA-256）。原始 JSON 保留了 `commit` 与 `dirty`，但被测扩展当时未重建，今天无法重
建该二进制来复核，因此这两份报告不能作为可复核基线使用；后续对照请以
[2026-09-22 矩阵](v2_disk_generation_matrix_20260922.md) 为准，它记录了 tree 哈希与
扩展哈希，并来自干净工作树。

## 尚未覆盖

- 目标机器上的测量：本表只代表一台本地机器，scratch 位于系统临时目录所在卷（APFS），未施加并发 I/O 或空间压力。
- 物理设备 I/O 与页缓存状态未记录。
- 本表未测 `memory_budget_mib`：这些运行都使用未指定预算。低/中/高三档预算见
  [2026-09-22 矩阵](v2_disk_generation_matrix_20260922.md)。
- B1/B2 的重复率为 0，不能据此推断一般输入无需去重。
