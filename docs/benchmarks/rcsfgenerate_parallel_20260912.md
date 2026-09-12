# 本机单 J 并行回归（2026-09-12）

输入由登记的 e1_cc1as1 transcript 将 `0,12` 改为 `6,6`，其他回答不变。
原版使用 workspace 中固定版本的 `grasp/build-debug/bin/rcsfgenerate` 重新生成；Rust 每份输出均与它逐字节 SHA-256 相同。此处原版只作正确性对照，不比较 Debug Fortran 与 Release Rust 的速度。

硬件：Apple M4，10 核，16 GiB；Rust `cargo build --release --examples`，默认 release 优化。每线程数预热 1 次，随后测量 3 次，表中为中位数。顺序测量 1/2/4 线程，缓存未清空；未测物理 I/O、服务器 RSS 或高线程数。这是本机单 J 验证，不能推广为服务器性能结论。

|线程|生成阶段（秒）|端到端（秒）|生成加速|端到端加速|
|---|---|---|---|---|
|1|0.007456|0.067862|1.00×|1.00×|
|2|0.004594|0.064798|1.62×|1.05×|
|4|0.002831|0.063439|2.63×|1.07×|

记录数：86587；唯一占据任务：1374。

输出 SHA-256：`28f4e1a3d908e6689f6b7d08ff70e735f07239b9a958a42b0dc4677b28bf2cbd`。

[原始测量](rcsfgenerate_parallel_20260912.json)。生成耗时包括结果组织所需的分支归并；端到端包括输入、占据枚举、生成和最终文本写出。前缀细分只覆盖至少 64 个态组合的任务，不代表所有单占据任务都能充分并行。

复现：先运行 `uv run cargo build --release --examples`，再运行
`uv run python scripts/benchmark_single_j_generation.py`。需要同级
`grasp/build-debug/bin/rcsfgenerate`。脚本会更新本报告及原始 JSON。

本轮验证：Rust 常规测试 141 项通过，另显式启用完整基准、原版 GEN 和输入改写三项外部对照，全部通过。Python 全套包含真实 CLI 以及两种描述符输入的原始/归一化对照。`ruff check .` 和 `basedpyright rcsfs/` 通过。开发 CLI 脚本的 1/2 线程归一化输出一致，o1 CSF 哈希仍为登记值，修复后的描述符 CSV SHA-256 为 `cafc0934f7693f80e09872b22b2aab8040158dfdd879073dc6cd2d7e0b8dfc5e`。
