# rcsfgenerate 串行性能基线（2026-09-10）

本次在 Apple M4、10 核、16 GiB、macOS 26.6.2 上测量两个登记的完整算例。所有程序串行运行，每个程序/算例预热 1 次、正式运行 3 次；下表采用中位数。正式 24 次及预热 8 次输出均通过登记基准的完整文件 SHA-256 检查，包含文件头、记录三行文本、分块及顺序。

测量结果适用于这台开发机及这两个算例，不代表 128 GB 服务器、多核或更大单 J 算例。原始逐轮数值、哈希和存活文件快照见 [JSON 记录](rcsfgenerate_serial_20260910.json)。该记录不包含真实 CSF 文件或本机绝对路径。

## 端到端与内存

外部墙钟从启动 `/usr/bin/time` 到子进程退出，包括进程启动、生成、输出及退出清理；哈希校验和清理不计入耗时。操作系统计时工具提供峰值 RSS，仅统计被测程序，不含 Python 驱动。所有运行顺序执行，未同时运行编译/测试；不清除 OS 缓存，不调用 fsync，因此不是冷缓存或持久化到介质的测量。

| 算例 | 版本 | 端到端中位数 (s) | 三次范围 (s) | 峰值 RSS 中位数 (MiB) |
|---|---|---:|---:|---:|
| e1_cc1as1 | 登记 Fortran Debug | 3.947836 | 3.924529–3.987578 | 11.125 |
| e1_cc1as1 | Fortran Release | 3.133330 | 3.125965–3.141545 | 2.250 |
| e1_cc1as1 | Rust Release | 0.342083 | 0.337927–0.351278 | 69.203 |
| o1_cc1as1 | 登记 Fortran Debug | 1.026271 | 1.022395–1.027887 | 11.047 |
| o1_cc1as1 | Fortran Release | 0.773545 | 0.769520–0.791153 | 2.234 |
| o1_cc1as1 | Rust Release | 0.082322 | 0.081550–0.082888 | 24.891 |

Fortran 固定源码提交为 `9006157730a82ac839f2b4ff4e938bcba63a539e`。Debug 使用原先登记且哈希未变的可执行文件。Release 在仓库外复制 CMake 列出的应用源码后，用 GNU Fortran 16.2.0、`-O3 -fno-automatic -fallow-argument-mismatch` 独立构建；该应用可独立链接，未链接原版 Debug 构建中未使用的 GRASP/BLAS 库。Rust 使用 rustc 1.92.0、`opt-level=3, lto=true, codegen-units=1`。这些是各自优化构建，并非逐项相同的编译设置。

Rust 的测量入口是开发示例 `benchmark_generation`：沿用已有占据枚举/归并与串行生成器，保留每个任务的整数 chunk，以块引用组织顺序，最后一次输出 CSF。不采用文本中间文件，不引入并行、哈希提前去重或新的耦合算法。它不是已稳定的 Python/CLI 产品接口，只有全局记录上限，尚未实现进程内存预算。

- e1_cc1as1 的端到端比值为 9.16×。Rust 峰值 RSS 为 69.2 MiB，高于 Fortran 的 2.25 MiB；全内存保留结果并没有降低该算例的进程峰值内存。
- o1_cc1as1 的端到端比值为 9.40×。Rust 峰值 RSS 为 24.9 MiB，高于 Fortran 的 2.23 MiB；全内存保留结果并没有降低该算例的进程峰值内存。

## 分阶段诊断

Fortran 阶段数值来自额外计时副本，不来自上表的未修改程序。`SYSTEM_CLOCK` 的计时器按调用累计；GEN 内每条记录有计时点，因此包含可见的测量开销。BLANDA/GEN 是包含子阶段的时间，以下用差值给出排他时间。差值不是完全剔除了计时开销的纯算法耗时。

| 阶段 (s，中位数) | e1 Fortran 计时副本 | e1 Rust | o1 Fortran 计时副本 | o1 Rust |
|---|---:|---:|---:|---:|
| 输入改写 | 0.000194 | 0.000029 | 0.000180 | 0.000030 |
| 占据枚举/归并 | 0.003235 | 0.000879 | 0.003803 | 0.001083 |
| GEN 除三行格式化/写入；Rust 整数构造 | 0.010831 | 0.023027 | 0.002427 | 0.008557 |
| 三行格式化及写入 | 0.273050 | 0.308817 | 0.060451 | 0.064442 |
| 文本合并 | 0.981909 | 不需要 | 0.224844 | 不需要 |
| 重读/分块及最终写入 | 2.004443 | 不需要 | 0.494624 | 不需要 |
| Rust 块引用组织 | 不适用 | 0.000518 | 不适用 | 0.000626 |

Fortran 的“三行格式化及写入”是每个 GEN 输出分支内 KOPP1/KOPP2 和三次 WRITE；RAD1 的每占据初始化仍归入 GEN 的剩余时间。Rust 生成时间包含整数记录构造，Fortran 不保留同类数据结构，因此不能把该行当作纯耦合算法速度比较。Fortran 的“重读/分块”同时包括读取合并结果、写入临时 scratch、重读 scratch 和写出最终文件，不能再次加到端到端时间上。

- e1_cc1as1：计时副本端到端 3.267470 s，未修改 Release 3.133330 s，差异 +4.3%。
- o1_cc1as1：计时副本端到端 0.789635 s，未修改 Release 0.773545 s，差异 +2.1%。

阶段结果支持下一步优先评估结果存储与输出：Fortran 的文本合并/分块占多数时间；Rust 的最终文本格式化及写出占多数时间。上述端到端收益混合了免去文本中间文件、已有占据预归并、数据表示和编译差异，不能全部归因于耦合算法或未来的并行化。

## 规模与逻辑 I/O

| 指标 | e1_cc1as1 | o1_cc1as1 |
|---|---:|---:|
| 唯一占据任务（含零 CSF 任务） | 1374 | 1614 |
| Fortran GEN 调用次数 | 1474 | 1878 |
| Fortran 合并前生成记录 | 458212 | 103194 |
| 最终 CSF 数 | 452373 | 89786 |
| J/P 块数 | 7 | 2 |
| 最终文件字节数 | 108555592 | 24002854 |
| Rust 整数 chunk 已分配容量（字节） | 70102420 | 15638467 |
| Fortran 记录体逻辑读取（字节） | 375053103 | 88565885 |
| Fortran 记录体逻辑写入（字节） | 544268471 | 123588256 |
| Rust CSF 中间文件读取/写入（字节） | 0 | 0 |

逻辑 I/O 计数覆盖 GEN 的三行写入、MERGE 主输出及 clist.new 的记录体写入、LASA1/LASA2 的记录体读取、RCSFBLOCK 的记录体读取与 scratch/最终写入；定宽记录长度遵守原版 9N/9N/9N+2，换行按 1 字节计算。不包括文件头、日志、输入解析、运行库内部读取，因此是明确范围内的记录体数据量，不是系统总 I/O。只对当前登记的新建/双参考路径验证，其他输入模式需重新审计计数点。

临时文件涉及 GEN 的 7/8 号输出、合并用的 rcsf.out/fil1.dat/clist.new，以及按 J/P 分组的 Fortran SCRATCH 文件；后者关闭后删除，无法从结束时目录大小反推出累计读写量。JSON 中 surviving_files 只记录结束时仍存在的文件。Rust 只写一次最终 CSF 文件，最终写入字节数见表。

本机 `/usr/bin/time -l` 的 block input/output operations 全部为 0。该值不能解释为没有磁盘 I/O，故不换算物理读写字节。真实存储设备读写量、冷缓存条件及 128 GB 服务器测量仍待补充。Rust 的 chunk 容量包括 chunk 内 Vec/String 容量及外层 chunk Vec，不含占据任务、排序树、分配器元数据和临时缓冲；它不等于峰值 RSS。

## 复跑

在 rCSFs 目录运行；所有输出目录必须尚不存在，且准备脚本拒绝在原 GRASP checkout 内生成副本。基准文件放在仓库外。

```bash
uv run cargo build --release --example benchmark_generation
uv run python scripts/rcsfgenerate_benchmark/prepare_fortran.py \
  --grasp-source ../grasp --output /tmp/rcsf-release \
  --optimization release --uninstrumented
uv run python scripts/rcsfgenerate_benchmark/prepare_fortran.py \
  --grasp-source ../grasp --output /tmp/rcsf-profile --optimization release
uv run python scripts/rcsfgenerate_benchmark/run.py \
  --variant fortran-debug fortran ../grasp/build-debug/bin/rcsfgenerate \
  --variant fortran-release fortran /tmp/rcsf-release/build/rcsfgenerate \
  --variant fortran-profile fortran /tmp/rcsf-profile/build/rcsfgenerate \
  --variant rust-release rust target/release/examples/benchmark_generation \
  --baseline-dir /path/to/baselines --output /tmp/rcsf-results --repeats 3
```

脚本保存逐轮资源报告、程序日志、校验哈希和 JSON；校验通过后删除本轮生成的 CSF/大中间文件以节省空间，不会删除外部基准。校验失败立即停止并保留现场。构建和哈希比较不计入生成耗时。
