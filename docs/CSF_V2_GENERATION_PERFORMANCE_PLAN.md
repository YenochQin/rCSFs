# V2 描述符与 CSF 生成性能实施计划

状态：P0、P1 已实施，P2–P6 待实施。基线：`7ad18b1`。制定日期：2026-09-21。

本文规划当前磁盘生成路径的性能改造。表中的测量是已有事实；阶段任务、接口、性能门槛是待实现目标，不代表已经具备。本文补充 [CSF_GENERATION.md](CSF_GENERATION.md) 和 [原生成开发计划](rcsfgenerate_rust_development_plan.md)，不沿用旧计划中“全部生成结果默认驻留内存”的资源假设。

## 已实施内容（P0/P1）

以下内容已经进入当前代码和维护测试；它们不是后续阶段的预期目标。

- **P0 观测和基准入口**：`scripts/benchmark_v2_generation.py` 可接收任意 transcript，输出可保存的 JSON。磁盘路径现在分别记录 `enumeration`、`csf_generation`、`deduplication`、`descriptor_merge` 和 `csf_restore` 阶段的墙钟时间、Unix 进程 CPU 时间（平台可用时）、输入/输出记录数以及阶段边界的逻辑文件字节数。逻辑字节数不代表物理设备 I/O 或页缓存流量。
- **P0 进度和输出契约**：range 进度通过 stderr 按约 2 秒限频报告 `started`、`completed` 和已生成 CSF 行数，不再用临时目录数量推断完成度。`--json` 的 stdout 仍是机器可解析的 JSON；新增 `stage_stats` 不改变既有统计字段。B0 维护测试覆盖统计、进度、V2 schema、行顺序、元数据和 round-trip，B1/B2 由显式基准脚本运行，不进入默认 pytest。
- **P1 统一执行选项**：内部 `GenerationOptions` 统一线程数、scratch 路径、批次设置和资源预算。现有 `--threads` 保持有效；Python 磁盘生成 API 追加 keyword-only 的 `memory_budget_mib`，PyO3、stub、TypedDict 和 CLI/TOML 配置同步支持。
- **P1 受管内存计量**：`memory_budget_mib` 约束占据组态、在途 Arrow batch、去重桶、幸存位图和 Parquet writer 的受管数据结构预留。预算不足会返回明确的资源错误；不会静默提高预算。每次运行通过 `resource_stats` 返回 `memory_budget_mib`、`budget_bytes`、`peak_managed_bytes`、`current_managed_bytes` 和 `occupation_bytes`。
- **预算边界**：该预算是内部字节记账，不是操作系统 RSS 硬限制；allocator 元数据、线程栈、Arrow/Parquet 运行库和其它进程内存不在其中。未指定预算时保持现有保守参数并报告未指定，不自动按物理内存百分比设置。
- **P1 有界生成和稳定发布**：range worker 直接写有界 segment，任务错误会在 join 后传播；segment 元数据携带 J/parity、range/segment 序号、记录数和字节数，merge 按稳定序号发布，因此 worker 完成顺序不改变结果，也不会因慢消费者保留完整 CSF batch。
- **V2 路径收口**：启用描述符的 TOML/config `csfsgenerate` 运行默认选择磁盘路径，直接生成可逆 V2 描述符，且磁盘路径禁止自动回退 V1；V2 的 `normalize=true` 明确拒绝。独立 `gen-descriptors` 仍可显式选择 V1。交互式 `csfsgenerate --generate-descriptors` 的旧 in-memory 兼容路径仍使用 V1，不属于本次 P0/P1 迁移。

本次实施已验证 79 项 Rust 测试、76 项 Python 测试、release 扩展构建，以及受影响 `rcsfs/cli.py` 的 Ruff 检查。`basedpyright rcsfs/` 仍会报告既有的 `polars` 类型缺失；`rcsfs/cli.py` 单独检查通过。慢消费者完整故障注入、目标机器上的长任务基准以及 P2 之后的压力矩阵仍未完成。

## 1. 目标与约束

目标是在 V2 结果及顺序不变的前提下，减少中间数据量、消除长时间串行后处理，并在明确的内存预算内提高多核吞吐量。

必须保持：

- 最终 V2 schema：每个 Peel 子壳层四列 `n/2j/v/2k`，另有 `total_two_j/parity`，均为 Int32。
- seniority、未打印值 `MISSING=-1` 与显式零的区别，以及已有 header 绑定和元数据契约。
- GRASP 文本格式、J/parity 块顺序、块内首条记录顺序、去重时保留首次出现者。
- 线程数、任务切分、批次大小、预算和临时压缩方式不改变逻辑结果。
- 外部数据的严格合法性检查；内部优化不能把未经验证的数据当成可信记录。
- 输出不得覆盖既有文件；失败不报告成功，不删除其他进程的文件。多文件发布不是天然的原子事务，不能通过文案宣称具备尚未实现的全组原子性。
- TOML/config 的默认磁盘描述符路径使用 V2；不能通过自动切回 V1 或旧归一化逻辑获得性能。V2 `normalize=true` 继续明确拒绝。交互式 in-memory 兼容路径的 V1 迁移尚未包含在 P0/P1。

本计划不改变原子物理选择规则，不删除用户请求的输出，不把完整数十亿条结果放入内存，不要求维护测试依赖 GRASP 可执行文件或私有数据集。全面移除旧 V1 接口属于独立迁移任务。

## 2. 已确认的事实及未确认事项

### 2.1 固定复现输入

下面是规范化后的合法 TOML，`j_min/j_max` 在当前实现中表示 **2J**，此处为 J=4。

```toml
[generate]
order = "*"
core = 1
references = [
  "2s(2,i)2p(6,i)3s(2,1)3p(6,i)3d(8,*)4s(2,*)",
  "2s(2,i)2p(6,i)3s(2,i)3p(6,5)3d(8,*)4s(2,*)",
]
active_orbitals = "9s,9p,9d,9f,7g"
j_min = 8
j_max = 8
excitations = 4
continue_lists = false

[output]
generate_descriptors = true
csf = "calculation.c"
parquet = "calculation.parquet"
descriptor_parquet = "calculation_descriptors.parquet"
normalize = false
```

### 2.2 规模与内存

使用当前库的真实占据枚举，并复用当前合法态表进行独立动态规划计数：

| 指标 | 基线结果 |
| --- | ---: |
| 唯一占据组态 | 7,031,941 |
| 每 range 4096 组态时的任务数 | 1717 |
| Peel 子壳层 / V2 列数 | 56 / 226 |
| 有序 `(kappa, occupation)` 模式数 | 270,592 |
| 去重前 CSF 计数 | 5,814,175,207 |
| 每 range CSF 数：最小 / 中位 / 最大 | 18,885 / 1,930,518 / 30,382,053 |
| 枚举及计数峰值 RSS | 3425 MiB，约 3.34 GiB |
| 枚举及计数墙钟时间 | 21.89 秒 |

完整输入未生成最终文件，58.14 亿是计数值，尚未逐条核验，也不是已经证明的去重后数量。计数在下述 B1、B2 上与真实生成数量一致。

当前临时 Arrow 每行是 `226×4+4+8=916` 字节数值载荷；完整输入约 4.844 TiB。根去重桶每行是 `226×4+8+16=928` 字节，约 4.907 TiB。源 Arrow 保留期间会写完整套根桶，两者合计约 9.75 TiB，尚不包括格式开销、递归分桶、文本和最终 Parquet。

1611 个 range 目录只能说明这些任务已经开始，不能视为已完成百分比。4.1 GB 用量也不能证明缓冲区充分利用了内存：枚举本身已有约 3.34 GiB 的峰值，且进程 RSS 和系统页缓存是不同指标。

### 2.3 实际耗时

本地可见 8 个 CPU，工作目录位于 NVMe 文件系统；release 扩展，使用共享 Tools Python 环境。未清空页缓存，数据不代表用户服务器性能。

B1 保留参考组态与 2J，改用 `5s,5p,5d,5f,5g`、3 次激发：19,480 个组态、5 个 ranges、2,695,762 条 CSF。

| 阶段 | 1 线程 | 8 线程 | 8 线程复测 |
| --- | ---: | ---: | ---: |
| 枚举与 Arrow 生成 | 1.935 s | 0.681 s | 0.663 s |
| 去重 | 3.515 s | 3.548 s | 3.491 s |
| 合并描述符 | 2.083 s | 2.069 s | 2.037 s |
| 文本与 CSF Parquet 还原 | 5.034 s | 5.082 s | 5.037 s |
| 含启动的总耗时 | 12.681 s | 11.494 s | 11.346 s |

后三阶段的 CPU 时间各自接近墙钟时间，符合约一个 CPU 的占用。测量直接调用磁盘 API，不含 CLI 发布时的最终文件复制。B1 只有五个 ranges，不能据此推断大任务生成阶段的并行扩展性。

B2 使用完整活性空间、2 次激发：19,243 个组态、560,351 条 CSF。8 线程生成、去重、合并、还原分别约 0.244、1.500、1.034、1.184 秒。B1、B2 均未删除重复行，这不足以证明所有输入无需去重。

### 2.4 代码定位

| 位置（相对 rCSFs 根目录） | 已确认的成本 |
| --- | --- |
| `src/csf_generation/streaming.rs`：`generate_range_segments` | 固定组态数划分，每任务同步写 Arrow |
| 同文件：`deduplicate_v2_descriptor_segments`、`process_bucket` | 顺序分桶和顺序消费，固定 65,536 行叶桶上限 |
| 同文件：`merge_v2_deduplicated_segments` | 顺序筛选、复制并交给单个 Parquet writer |
| `src/csfs_descriptor.rs`：`restore_v2_descriptor_parquet_to_outputs` | 顺序解码、验证、格式化、写出两种产物 |
| `src/descriptor_schema.rs`：`validate_record` | 逐记录解析标签并获取会分配 Vec 的合法态表 |
| `src/csf_generation/mod.rs`：`generate_prepared_records` | 嵌套并行先缓存完整分支记录，再顺序送入 sink |
| 同文件：`Generator::couple` | 目标 J 的最终判断在末级，缺少后缀可达性剪枝 |
| `rcsfs/cli.py`：`_generate_outputs` | 最终产物从 staging 完整复制到目标 |

待确认：用户运行的具体阶段、CPU/内存/存储、Arrow 是否仍增长、磁盘等待情况；紧凑暂存的收益、并行 Parquet 编码方案及生成唯一性均尚未实现验证。

## 3. 分阶段交付

依赖顺序：`P0 → P1 → P2 → P3 → P4 → P5 → P6`。每阶段单独提交、独立给出基线对照；P5 的状态缓存和计数算法可在 P1 后提前开发，但不以其局部收益代替 P2—P4 的端到端改造。

### P0：建立可重复基线和阶段观测

改动范围：`streaming.rs`、`rcsfs/cli.py`、`scripts/`、`tests/`、`docs/benchmarks/`。

- [x] 将临时探针整理成不依赖外部 GRASP 的维护基准入口；`scripts/benchmark_v2_generation.py` 接受任意 transcript，输出可保存的 JSON 测量。
- [x] 分别报告枚举、CSF 生成、分桶/去重、最终编码和 CSF 还原阶段的墙钟、Unix 进程 CPU 时间、行数和阶段边界/临时写入字节。发布复制仍由 Python CLI 管理，未伪装成 Rust 阶段。
- [x] range 报告 started/completed 和已完成 CSF 行数，按 2 秒限频；不能用目录数计算进度。
- [x] 日志走 stderr，`--json` 的 stdout 保持机器可解析；新增 `stage_stats` 字段保持旧统计字段不变。
- [x] B0 小型语义样例覆盖新增统计和进度；B1/B2 可通过性能脚本显式运行。原始聚合测量仍需在目标机器上登记到 `docs/benchmarks/`，不记录机器专属绝对路径或生成的完整 CSF。

完成条件（P0）：同一 API/脚本能运行 B1/B2 并生成阶段报告；B0 的文本、V2 schema/行顺序和元数据回归通过；1/2/4/8 线程比较入口存在且不进入默认 pytest。目标机器的完整基准报告和物理设备 I/O 仍是 P0 的运行记录工作，不在代码中虚构。

### P1：统一资源预算和内部批次协议

改动范围：`src/csf_generation/` 中的资源管理模块、`streaming.rs`、PyO3 绑定、Python wrapper/stub、CLI。

- [x] 引入内部 `GenerationOptions`，集中传递线程数、预算和现有 scratch 位置；不向用户暴露所有 bucket/range/batch 常量。
- [x] 首批新增可选 `memory_budget_mib`，CLI 对应 `--memory-budget-mib`。它表示受管理数据结构的总预算：占据组态、缓存、在途 batches、重排序、去重、编码均需计入。
- [x] 不宣称该值等于操作系统 RSS 硬限制：另行报告 allocator、线程栈和运行库余量及实测峰值。组态容量也必须纳入预留/增长检查；不能等枚举完才发现超限。预算不足以容纳当前不可 spill 的组态结构时，明确返回资源错误，不静默提高预算。
- [x] 省略预算时暂时保持现有保守参数，明确报告预算未指定；自动占用物理内存百分比不在本阶段引入。
- [x] 所有主要在途数据通过按字节计量的 permit 管理，含占据组态、Arrow batch、去重桶、幸存位图和 Parquet writer 的保守预留；阶段切换时释放或转移所有权，不能重复拥有整份预算。
- [x] 定义批次元信息：J/parity、range/segment 序号、记录数及占用字节。稳定序号不依赖 worker 完成顺序。
- [x] Rayon range worker 在错误时完成 join 并传播首个错误；每个 worker 直接写入自己的有界 segment，后续 merge 按稳定序号发布，慢消费者不会积累完整 CSF batch。

完成条件：小预算、乱序完成、worker/写入故障下无死锁；已计量的受管内存不越界；不足预算不造成无提示 OOM；预算变化不改变逻辑结果。现有位置参数兼容，新 Python 参数仅追加为 keyword-only，wrapper、stub、PyO3 一起更新。维护测试覆盖了预算拒绝、预算统计、线程顺序一致性和 V2 round-trip；慢消费者的完整故障注入仍留在后续压力矩阵。

### P2：减少暂存数据与分桶 I/O

改动范围：内部 segment codec、`streaming.rs` 的 writer/reader/桶记录格式。

分为两个可独立验收的子步骤：

- [ ] **P2a：临时 Arrow 压缩实验。** 比较当前无压缩、依赖实际支持的 LZ4/ZSTD；确认 Arrow IPC feature 配置，测 CPU、编码速度、写入量和峰值内存。协议显式记录 codec，不能把尚未启用的压缩当成有效设置。
- [ ] **P2b：紧凑暂存记录。** 重用整数记录语义，按 batch 存储占据子壳层、合法局域态、耦合 arena 及记录 offsets；省略未占据槽位，保留 MISSING 和 seniority。内存中不为每条记录创建多个独立 Vec。
- [ ] range ID 等批次常量放到 segment 元数据，连续 ordinal 尽量存 base/count；读取时仍验证连续性和范围。
- [ ] 去重 key 从规范化紧凑记录产生，哈希碰撞后仍精确比较。用于比较的全局 Peel 映射必须相同，不能把不同配置的局部下标直接比较。
- [ ] 分桶文件也使用紧凑记录/有界块；不能只压缩源 Arrow，却继续写完整 226 列根桶。
- [ ] 物理 segment 按目标字节聚合，不因后续调度变细而生成大量小文件。内部格式带版本号，本阶段不承诺旧 scratch 断点恢复。

完成条件：新旧暂存路径在 B0/B1/B2 上输出相同文本和 V2 逻辑行；损坏 offsets/截断文件明确失败；记录临时空间、逻辑 I/O 与 CPU 代价。B2 暂存峰值空间降低至少 50% 作为默认启用目标；未达目标则记录原因并继续实验，不直接宣称完成优化。

### P3：去重全链路并行化

改动范围：分桶、递归拆桶、幸存索引、并行执行管理。

- [ ] 对 segment/batch 并行计算规范化 key 和 hash，将结果送往有界 bucket 写入任务，避免每行竞争全局锁。
- [ ] 显式携带原始稳定 ordinal，桶写入顺序可以变化；同 key 的 survivor 取最小原始 ordinal，不能取最快 worker 的记录。
- [ ] 根据预计字节量和预算规划桶数、叶桶容量及并发数，替代全场景固定 65,536 行；分桶决定仅影响执行方式，不影响输出顺序。
- [ ] 并行消费互相独立的桶；使用局部幸存索引或分区位图归并，避免每个 worker 都复制一份完整 bitset。
- [ ] 超大桶按确定性规则进一步拆分。对大量完全相同 key 单独处理，不能无限依赖再次 hash 将它们分散。
- [ ] 用较大的有界 I/O buffer 合并小写入；哈希先保持确定性，是否替换 SHA-256 作为单独微基准决策，精确比较始终保留。

完成条件：单 J/parity 也能并行；跨 segment/线程/递归层的重复、强制碰撞、全部相同行和乱序任务均保持首次出现记录。B1/B2 去重阶段在 8 个可用 CPU 上相对新实现 1 线程达到至少 2× 加速作为目标，并报告相对旧版本的绝对耗时及 I/O；未达到时先定位瓶颈再进入发布。

### P4：并行构建最终产物，取消不必要的回读

改动范围：segment merge、V2 编码、`csfs_descriptor.rs` 输出路径、`rcsfs/cli.py` 发布路径。

- [ ] 去重完成后，从有序幸存记录生成带稳定序号的 batch；共享该批数据，分别生成 descriptor 和 CSF 三行文本/Parquet。
- [ ] 文本格式化、合法性校验和批次转换在线程池内完成，发布端按序写出；正确处理跨 batch 的 J/parity 块分隔符和全局 idx。
- [ ] 描述符不再先写最终 Parquet，再回读解压以生成文本。独立的外部 descriptor 还原 API 继续保留完整输入校验。
- [ ] 先验证当前 Parquet 依赖能否并行编码独立 row group/列块，再由单一文件提交端写入 footer。若采用临时分片，必须通过受支持的读取/重编码或元数据重建方式合并，禁止字节拼接多个 Parquet 文件。
- [ ] 不以“并行准备 batch + 全部压缩仍在一个 ArrowWriter”冒充编码并行；分别计时准备、压缩和写入。
- [ ] 同文件系统探索无覆盖的原子单文件发布，目标旁 staging 与重命名/链接策略先验证平台能力；跨文件系统保留复制。注入竞争写入，确认不覆盖其他进程的产物。
- [ ] 明确部分发布失败后的报告和清理策略；所有产物完成发布后才输出任务成功。

完成条件：生成流程中没有读取刚写出的最终 descriptor 以还原文本；线程数变化不改变 CSF 字节及 V2 行；发布竞争测试作用于真实新路径。B1/B2 上并行转换阶段目标至少 2× 加速，累计端到端相对 `7ad18b1` 的 8 线程基线目标至少 2×；这些是验收目标，不是预计必达收益。

### P5：目标 J 剪枝、角动量缓存和工作量均衡

改动范围：`states.rs`、`mod.rs`、`descriptor_schema.rs`、`occupations.rs`、range planner。

- [ ] 合法态表改为可复用只读数据；Peel 标签只解析一次。外部验证继续使用完整规则，但通过缓存查表而非逐记录分配 Vec。
- [ ] 建立目标-J 的后缀可达位集/DP，对无法完成的耦合前缀提前剪枝，保留其余分支的原始遍历顺序。
- [ ] 生产级计数器使用 checked arithmetic；计数溢出返回明确错误，调度估计不得因截断而变成零。高于 32 位的记录/ordinal 贯穿协议、文件和统计。
- [ ] 缓存键包含有序角动量/占据模式、目标 2J 范围及影响输出的状态语义；主量子数、Peel 映射、parity 在构建最终记录时正确补回。先缓存小型状态/DP，不缓存所有 CSF 模板。
- [ ] 按预计 CSF 数切分连续配置任务；一个配置过大时按稳定状态前缀拆分。调度任务大小与 segment 大小独立。
- [ ] 替换嵌套并行中无限增长的完整分支 Vec，使用 P1 的有界 batch；外层任务充足时避免对所有配置再做大量细粒度并行。
- [ ] 占据组态与排序 key 改为紧凑 arena/共享存储，合并时尽量移动而非克隆；不改变 reference 合并顺序。若完整组态仍不能放进指定预算，明确失败；外部排序/spill 作为独立后续任务，不伪称已经支持。

完成条件：计数与真实生成在小规模覆盖集合上逐项一致；剪枝开关、缓存开关、线程数和切分方式不改变顺序；缓存不超预算。完整输入只运行计数与 planner，登记任务预计 CSF 数的 p50/p95/max；在可拆分任务上，最大预计工作量不超过规划目标的 2 倍。单独列出无法满足的不可拆分任务，不掩盖长尾。

### P6：证明唯一性后削减全局去重

本阶段有正确性门槛，未通过时继续使用 P3 的精确去重。

- [ ] 证明跨配置：规范化 occupation key 的唯一性与 V2 occupation 列一一对应，包括多 reference 合并和零占据处理。
- [ ] 证明同配置：状态选择与耦合路径到可逆 V2 记录的映射单射，覆盖重复 J 不同 seniority、隐藏耦合、闭壳层和 MISSING/0。
- [ ] 对支持的态表及有界小系统做穷举差分，比较原始生成、局部去重、全局精确去重的内容和顺序；证明文档必须解释穷举范围以外为何仍成立。
- [ ] 只有满足构造不变量的内部生成记录可走已验证路径；不能让外部输入靠用户标志绕过验证。
- [ ] 若证明只支持将去重限制到单配置，就先落地局部去重；如果单射证明失败，保留通用去重并记录反例。
- [ ] 在满足条件的路径消除根桶/幸存位图往返；统计中的 generated/unique/duplicate 仍有明确、真实的含义。

完成条件：证明和回归测试一起评审；未证明的路径仍精确去重。计量实际减少的 scratch 峰值和全量读写轮次，不把省去去重时间当成其它阶段的加速。

## 4. 接口收口

以下为 **P1 已实现接口**：

```toml
[generate]
# 与物理输入字段同表；示意值，不是针对未知机器的推荐配置。
threads = 8
memory_budget_mib = 8192
```

- 现有 CLI `--threads` 继续有效，并新增配置读取；定义优先级为 CLI 显式值 > TOML > 默认。线程预算适用于整个执行流程，避免生成、去重和编码各自启动完整线程池造成过度并行。
- Python 磁盘生成函数保留现有位置参数，追加 keyword-only `memory_budget_mib=None`；同步 TypedDict、stub 和 PyO3 参数验证。
- 非整数、布尔值、零、负数、超范围预算/线程数在边界明确拒绝。
- `storage` 与现有 `scratch_dir` 语义保留。压缩方式、range 大小、桶数先作为内部策略和基准开关，不增加大量面向用户的调优字段。
- CLI JSON 保留现有统计字段，只增添结构化 `stage_stats/resource_stats`；成功路径中的产物地址必须指向最终文件，不能留 staging 路径。
- 随接口提交更新 README、`CSF_GENERATION.md`、CHANGELOG；修正与默认 V2 路径冲突的旧描述，不能等全部阶段完成才同步文档。

## 5. 验证与基准矩阵

| 层级 | 内容 | 运行方式 |
| --- | --- | --- |
| B0 | 小型真实格式样例；单/多 J、奇偶与半整数 J、闭壳层、seniority、显式零与 MISSING、多 reference | 普通 Rust/Python 维护测试 |
| B1 | §2.3 的 270 万条输入 | 显式性能基准，不进默认 CI |
| B2 | 完整活性空间、2 次激发的 56 万条输入 | 显式性能基准 |
| B3 | 完整输入：703 万组态与计数/planner | 手动规模验收，不生成 58 亿条文件 |
| B4 | 目标机器上的代表性生成及最终完整输入 | P0—P5 通过后，先预测空间，再按实际资源逐级放大 |

正确性比较包括 CSF 文本逐字节、V2 schema/Int32/行顺序/值、header/block lengths、全部统计和首次重复保留顺序。Parquet 压缩布局允许改变，不能要求压缩文件字节相同。除旧实现差分外，还需手工可核验的 B0 样例和编码/解码不变量，避免只继承旧错误。

压力场景包括：最早任务极慢、后续任务快速完成、零产出任务、单一超大配置、重复倾斜桶、强制哈希碰撞、超过 2^32 的 ordinal 小型合成边界、截断临时文件、写入异常和发布竞争。

性能基准每个组合预热 1 次、测量至少 3 次，报告中位数及波动。比较线程 1/2/4/8（以及目标机器适用的更高值），预算至少取低/中/高三档；预算不足组态基线时应报告资源拒绝，不能用其伪造慢速基线。记录缓存条件、CPU 配额、RSS、逻辑 I/O、可获得时的物理 I/O、scratch 峰值和文件数。

同机比较的默认启用门槛：正确性与资源约束全部通过，目标阶段有实测收益，B0/B1/B2 不出现无法解释的端到端性能回退。超过 10% 的稳定回退需明确处理或保持旧策略可选，不能仅以 CPU 占用更高解释。

所有 Python/Rust/PyO3 检查使用唯一共享环境。从 `rCSFs/` 执行：

```bash
source ../graspkit-tools/.venv/bin/activate
cargo test
cargo build --release --features pyo3/extension-module
# Linux CPython 3.14 示例；其它平台使用对应扩展后缀。
cp target/release/lib_rcsfs.so rcsfs/_rcsfs.cpython-314-x86_64-linux-gnu.so
pytest
ruff check .
basedpyright rcsfs/
```

涉及 Python binding/CLI/文件行为的阶段同时补 Python 回归；Rust 纯算法改动补相应 Rust 测试。未改变源码的文档阶段只检查事实、链接和 diff，不重复运行整套构建。

## 6. 交付与决策门槛

建议提交序列：P0 基准 → P1 预算及批次 → P2a 压缩 → P2b 紧凑暂存 → P3 分桶及去重并行 → P4 批量转换及编码 → P4 发布优化 → P5 缓存/剪枝 → P5 调度/占据存储 → P6 证明及快速路径。每项实现提交包含适用测试与阶段测量，不用一个巨型提交掩盖回归来源。

P2 压缩默认值、P4 Parquet 并行实现、P6 去重快速路径以测量或证明结果决策；默认保留可正确完成任务的已验证路径。排查优化失败时使用内部基准开关，不让终端用户承担复杂实现选择。

最终交付必须说明：实际完成了哪些阶段、哪类机器/输入受益、最终文件与物理语义是否一致、预算的覆盖范围、总耗时及峰值空间的变化。完整用户输入在目标机验证之前，不宣称六分钟低 CPU 的具体原因已被完全解决，也不承诺固定加速倍数。
