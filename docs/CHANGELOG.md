# 修改历史 (Changelog)

本文档记录 rCSFs 项目的重要修改和改进。

---

## [Unreleased]

### P2a：临时 segment 的 Arrow IPC codec 开关（2026-09-22）

- 新增 `RCSFS_SEGMENT_CODEC`（`none`/`lz4`/`zstd`）：临时 Arrow IPC segment 可选用
  Arrow 级压缩。默认仍为不压缩——压缩是待测量的决策，不是既定默认；`arrow-ipc` 的
  `lz4`/`zstd` feature 已启用，无法兑现的取值直接报错，而不是悄悄写未压缩文件却
  记录为已压缩。
- `segment_codec` 出现在生成统计、估算报告与 CLI JSON 中，基准脚本逐次运行核对
  扩展实际报告的 codec，不一致即失败——报告里写一个没真正启用的 codec 会让 P2a 的
  压缩比变成虚构。
- 容量模型继续按未压缩 segment 计费（新增一条 assumption 说明），因此开启压缩的运行
  仍落在估算上界之内，模型不会被单次测量的比例“调优”。
- 测试：Rust 侧验证每个 codec 都能到达写入器、能被读回、且确实压缩（`none` 与
  `lz4`/`zstd` 的字节数对比），以及 codec 不改变任何已发布字节（同一输入的 CSF 文本、
  descriptor 行、计数一致）；Python 侧覆盖默认值、编解码往返与非法取值的明确拒绝。

### 测试环境：pytest 必须加载工作树而不是已安装的 wheel（2026-09-22）

- `[tool.maturin] python-source = "."` 让仓库根目录本身就是包根，但裸 `pytest`
  脚本不会把当前目录放进 `sys.path`，于是 `import rcsfs` 解析到共享环境
  site-packages 里的 wheel——本仓库的测试因此一直在测"上次 `uv sync` 时"的代码，
  而不是工作树。实测：裸 `pytest` 加载 `graspkit-tools/.venv/.../rcsfs/_rcsfs*.so`，
  `python -m pytest` 才加载 `rCSFs/rcsfs/_rcsfs*.so`。
- 在 `[tool.pytest.ini_options]` 中固定 `pythonpath = ["."]`，使 CLAUDE.md 中
  "pytest 加载 in-tree 扩展"的说法真正成立；未刷新 in-tree 扩展时测试会立刻暴露，
  而不是悄悄通过。全套 Python 测试在修正后仍全部通过。

### P6a：内部生成路径的唯一性证明与穷举程序（2026-09-22）

- 新增 [V2_GENERATION_UNIQUENESS.md](V2_GENERATION_UNIQUENESS.md)：证明内部生成路径
  （transcript → 占据枚举 → 固定组态生成 → V2 行编码）产出的行两两不同——跨配置由
  "枚举 key ⇔ 占据向量"与"行的 n 列就是占据向量"给出，配置内由"态表在 (2J, seniority)
  上单射"与"隐藏耦合由行唯一决定"给出。据此**精确去重在该路径上是空操作**。
- 决策：**P3 不适用**（不实施根桶并行化）；P2b 不为已证明无用的根桶格式投入；
  P6b 具备启动门槛，实施时保留精确去重作为未证明路径的安全网。生产路径在 P6b 落地前
  行为不变。
- 新增 `tests/p6a_uniqueness_test.rs`：用生产生成器与编码器对有界系统做穷举差分
  （单子壳层全部态表、1–4 子壳层耦合链、`FIRST`/隐藏耦合、仅占据不同、多参考），
  逐条比较原始生成 / 配置内去重 / 全 run 去重的内容与顺序，并把冲突记录作为反例报出；
  反面控制证明检查非空跑（同一配置排两次时报出 100 对重复，局部层一条都发现不了）。
- 新增 `tests/p6a_disk_uniqueness_test.py`：注册 fixture `o1_cc1as1`（89,786 条，奇总 2J）
  与 `e1_cc1as1`（452,373 条，偶总 2J）完整走磁盘路径，断言 `duplicate_count == 0`、
  产物行数与生成数一致、P5a 计数等于实际产出。
- 新增 `states.rs` 维护测试：穷举全部可达 `2j_max` 与电子数，断言每张态表的
  `(2J, seniority)` 两两不同，并钉住支持的 `(2j, 电子/空穴占据)` 集合——新增态表会让
  它失败，逼使唯一性论证被重新审视。


- 修正未跟踪目录内改动不被指纹察觉的缺陷：`git status` 默认把一个未跟踪目录折叠成
  一行 `?? dir/`，其内容变化既不改变条目也不改变指纹，`--allow-dirty-source` 下的
  运行中漂移因此会被错误接受。现在使用 `--untracked-files=all` 逐文件列出；即使
  仍收到目录条目也**递归哈希**而不是写入固定占位符，无法读取的路径直接拒绝登记
  （固定占位符会让两棵不同的树得到相同指纹）。回归测试使用真实临时 git 仓库，并
  验证过它在旧实现下失败（条目为 `newdir/` 而非 `newdir/a.rs`）。
- `source identity` 改为 `GitIdentity`/`ExtensionIdentity`/`SourceIdentity` 三个
  TypedDict，完整性关键字段（commit/tree/dirty/fingerprint/扩展哈希）有类型可查，
  不再以嵌套 `dict[str, Any]` 传递。
- `_run_git` 与 `_run_git_raw` 合并为一个带 `raw` 开关的函数。
- 修正 `_parse_status_z` 中 rename 的注释顺序（实现在先、注释写反）。

### 第四轮评审修正：源码身份的前后快照（2026-09-22）

- 源码身份改为**运行前采集并立即门禁**：脏树在开始任何 warmup 或测量之前就被拒绝，
  长任务不再为无法登记的结果白跑；报告记录的是运行前快照（进程实际加载的代码与扩展
  不会在运行中改变）。
- 运行结束后再次采集身份并比较 commit、tree、dirty、fingerprint 与扩展哈希；任何一项
  在测量期间变化就**拒绝登记**，因为此时报告会同时描述两个状态。`--allow-dirty-source`
  接受的是"稳定的脏树"，不是"会变的树"。已用一个真实的运行中改动做过端到端验证：
  进程完成后拒绝写出报告并给出前后指纹。
- 两个脚本共用 `add_source_identity_arguments()` 与 `finalize_report()`，参数定义、
  门禁与报告收尾不再各写一份。
- `git status` 改为 `--porcelain=v1 -z` 结构化解析：以 NUL 分隔、不做引号转义，因此
  带空格、引号、换行的文件名与 rename 的源路径都能正确解析，fingerprint 与
  `dirty_paths` 共用同一份条目。

### 第三轮评审修正：产物唯一性、脏源码与死代码（2026-09-22）

- 每种产物只允许一个目标：`check_space` 拒绝同一 artifact kind 出现两次（否则会把一个
  产物的字节数计两遍），`estimate_v2_generation.py` 的 `--destination` 遇到重复 kind
  直接报错；此前文档声称可以重复，实际是后一个路径静默覆盖前一个。
- 脏源码的指纹补全：此前 `git diff HEAD` 不含未跟踪文件、也未套用 `docs/benchmarks`
  排除规则，可能出现"未跟踪源码 + 空 diff 哈希"或被报告内容污染。现在指纹由
  `git status --porcelain`（同一排除规则）构造，含已跟踪改动与每个未跟踪文件的内容
  哈希，并记录 `dirty_paths`；`describe` 不再带 `-dirty` 后缀，改由 `dirty` 字段表达。
  两个基准脚本默认**拒绝在脏工作树上写报告**，`--allow-dirty-source` 才记录为"已标识
  但未提交"的测量。
- 删除尚未接线的压缩抽象（`SegmentCompression`、`RCSFS_SEGMENT_COMPRESSION` 与
  `GenerationOptions::segment_compression`）：仓库内外无任何引用，P2a 会在真正测量时
  连同结论一起引入；`Cargo.toml` 的 arrow-ipc lz4/zstd feature 保留，它是依赖配置而
  非死代码。
- `space.rs` 的三个策略测试合并为一个策略矩阵（三种容量结论 × 三种策略），覆盖
  已知不足必然拒绝、opt-out 仅覆盖无法测量、报告始终给出结论，并断言两种拒绝的
  错误信息可区分。
- 修正 `estimate_disk_generation` 文档中残留的 `metadata` kind 与"opt-out 同时放宽
  不足与未知"的旧描述。

### 第二轮评审修正：空间开关语义与报告可审计性（2026-09-22）

- `allow_unchecked_space` 此前同时放宽"空间不足"与"无法测量"两种情况，导致直接用
  公开 API 的调用者可以在已知空间不足的卷上开始生成。现在该开关只覆盖"无法测量"，
  已知空间不足对任何准备写入的调用都会拒绝；固化了旧行为的测试已替换。
- `ArtifactKind` 拆分为 `Header` 与 `DescriptorMetadata`：此前 header 与描述符
  sidecar 共用一个 `metadata` 目标，CLI 只传入 sidecar，因此发布到另一卷的 header
  从未参与预检。两者现在各自计入自己的目标卷；同一类产物可对应多个目标路径。
- 报告记录源码身份而不只是提交号：`git.tree`、脏树时的 `dirty_diff_sha256`、以及
  进程实际加载的扩展的 SHA-256。脏树判定忽略 `docs/benchmarks`（写报告不应让下一份
  报告声称源码被修改）。新增测试要求这些字段；2026-09-21 的两份基线报告在测试与
  文档中标记为"早于该要求"，其二进制今天无法重建，因此只作历史记录。
- `normalize_path` 现在同时处理 Windows 形式（`C:\...`、`\\server\share`），
  Windows 根按大小写不敏感匹配，POSIX 根保持精确匹配。
- `capacity.rs` 拆出 `space.rs`：前者只回答"需要多少字节"，后者回答"放得下吗、
  放不下怎么办"，标定比例与文件系统逻辑不再相互耦合。
- `cargo fmt --check` 通过。此前也不通过——计划基线 `7ad18b1` 已有 23 处差异，其中 3 处
  在本系列从未改动的 `pipeline.rs`——因此这次对整个 crate 执行了格式化，而不仅是新代码；
  评审指出的未使用 import 与文件尾空行一并消失，P1 遗留的两处警告改为有说明的
  `allow(dead_code)`。

### 评审修正：漏生成记录与预检绕过（2026-09-22）

- 修正状态前缀合并会漏生成记录的缺陷：合并多个小前缀时 `pending` 在每个非空前缀
  上被前移，任务区间只覆盖最后一个前缀却计入整组记录数。新增"任意任务切分下每个
  非空前缀都被覆盖且仅覆盖一次"的回归测试（已验证该测试能复现旧缺陷），并把状态
  前缀测试输入换成会真正触发前缀拆分的组态——原输入每个 (组态, 2J) 至多一条记录，
  从未进入该路径。
- 生成结束时校验实际记录数等于计数总量，每个超标组态校验其任务记录数等于计数结果；
  错误的切分会直接失败而不是产出更短的文件。
- 估算路径不再拒绝"计数回退"的组态：无法用 DP 精确计数（中间耦合可能超出输出字段）
  时改为按 2J 拆分并把仍超标的单个目标报告为不可拆分长尾，而不是让规划整体失败。
- 空间预检按卷而不是按目录分组：同一卷上的 scratch、staging 与发布副本按阶段取合计
  最大值，避免各自通过而合计不足。destinations 以产物种类（csf_text/csf_parquet/
  descriptor/metadata）传入，由模型决定各卷应承担多少字节。
- 空间检查分为两种语义：估算只**报告**结果（空间不足时 `sufficient=false`、无法测量时
  `null`，都仍然给出答案），而**真正要写入的调用会拒绝**无法完成的运行
  （`allow_unchecked_space` / CLI `--allow-unchecked-space` 才继续）。CLI 在创建任何
  暂存目录之前根据报告做出该决定，未知值不再被当作通过。
- `estimate_disk_generation` 现在把调用者的 `memory_budget_mib` 施加到占据枚举上，
  低预算估算与实际生成在同一阶段以同样原因失败。
- CLI 在失败时也删除自己创建的 scratch 目录，失败后的重试不再与残留目录冲突。
- 基准脚本共用 `scripts/benchmark_support.py`（注册校验、环境与文件系统元数据、路径
  归一化），报告写入前把机器绝对路径替换为 `<system-temp>`/`<repo-root>`/`<path>`；
  预算或空间拒绝记录为 `outcome: "rejected"` 并从计时汇总中排除。
- 计数代码从 `csf_generation/mod.rs` 移入 `counting.rs`，mod.rs 保留生成基础结构。

### V2 生成容量预估与磁盘预检（2026-09-21）

- 新增 `csf_generation::capacity`：用 P5a 的 checked 计数估算 segment、根桶、
  递归桶、幸存位图、描述符、CSF 文本与 CSF Parquet 的字节数，应用 25% 安全
  余量，并在写入任何 segment 之前检查 scratch 与 staging 卷的可用空间；CLI
  另外检查各最终目标目录，因为发布是复制，且目标路径只有 CLI 知道。
- 新增 `estimate_disk_generation`（Python/PyO3）与 CLI `--estimate-only`：
  执行与真实运行相同的枚举、计数与调度，但不创建 scratch、不写文件。
- 估算模型的所有比例取自登记的 B1/B2 实测，报告以 `assumptions` 列出每个比例
  与未实测项（递归分桶按再整体重写一遍计入，不默认为零）。B2 上估算为实测的
  1.01–1.62 倍。
- 登记 B3 完整输入（`b3_cc1_9spdfg_4exc.rcsfgenerate`）：计数得到 7,031,941
  个组态与 5,814,175,207 条去重前 CSF，与计划 §2.2 的独立数值完全一致；容量
  报告见 `docs/benchmarks/v2_disk_generation_b3_capacity_20260921.md`。
- 明确失败恢复策略为"重新开始"：scratch 未绑定输入哈希与格式版本，不从其中
  恢复；`plan_stats` 之外新增报告字段 `failure_recovery`。
- 明确受管预算并发超出时的处理是立即返回资源错误（不是等待/背压），并在
  文档中登记 B1/B2 的受管峰值与 RSS 差距。

### V2 生成计数与工作量规划（2026-09-21）

- 新增 `csf_generation::planning`：生成前用动态规划精确计数每个占据组态的
  去重前 CSF 数，再按记录数（而非固定组态数）切分调度任务。任务规模目标
  由计数总量与线程数推导，并限制在内部上下界内；`RCSFS_RECORDS_PER_TASK`
  仅供基准实验固定调度。
- 超过目标的单个组态先按 2J 目标、再按稳定状态前缀拆分；无法继续拆分的
  长尾仍会被调度并单独报告，不会被丢弃或隐藏在统计里。
- 计数使用 checked 算术，溢出返回错误而不是截断为零；计数与真实生成在小
  规模覆盖集合上逐项对照，包括被生成器拒绝的组态。
- 既有 4096 组态/range 的固定切分（`configurations_per_range`）被移除；
  新的切分只改变执行方式，不改变发布顺序，B1 上 250k/50k/4M 三种任务粒度
  输出的 CSF 文本、header 与描述符行完全一致。
- 结果新增 `plan_stats`：任务数、任务规模目标、每任务估计记录数的
  min/p50/p95/max、零产组态数与不可拆分长尾，以及阶段
  `workload_planning`。

### V2 磁盘生成观测与资源预算（2026-09-21）

- TOML/config `csfsgenerate` 的描述符生成默认选择磁盘路径并直接写可逆
  V2；该路径不自动回退到 V1，V2 与 `normalize=true` 的组合会明确报错。
  交互式 in-memory 兼容路径仍保留旧 V1 行为。
- 增加 `stage_stats` 和 `resource_stats`：报告枚举、CSF 生成、去重、描述符
  合并和 CSF 还原的阶段耗时、CPU 时间、记录数、逻辑字节数，以及受管内存
  预算和峰值。
- 增加统一的 `GenerationOptions` 和 `memory_budget_mib`（CLI
  `--memory-budget-mib`、TOML `[generate]`、Python keyword-only API、PyO3
  binding/stub），预算不足时返回明确错误。该预算是受管内存记账，不是 RSS
  硬限制。
- 增加 range started/completed/CSF 行数进度报告、V2 disk round-trip 与资源
  预算回归测试，并保留 CLI JSON 的既有字段。

### Structural interaction upper bound (2026-09-15)

- Add a Rust/Python `select_interacting_csfs` file API and a
  `rcsfs interacting` CLI, defaulting to `rcsf.out` and 8 workers. The
  first-stage method applies the
  two-electron occupation bound in parallel while preserving block and record
  order, exact-reference skipping, and atomic output publication.
- Mark every result `exact=False`: recoupling and Coulomb/Breit angular tests
  are not implemented yet, so selected CSFs are a conservative upper bound.
- Add deterministic worker-count, header/block validation, alias protection,
  adjacent equal-symmetry block, Python API, and CLI regression coverage.

### Repository test scope (2026-09-14)

- Remove the three external GRASP comparison tests and their unused Fortran driver.
- Keep maintained tests self-contained and focused on rCSFs code; put temporary test code and external probes under `temp/`.

### Generation Parquet review fixes (2026-09-14)

- Align the generation CLI, Python wrapper, extension, and Rust callers after CSV removal.
- Reject existing or aliased generation destinations and publish staged Parquet outputs with exclusive creation.
- Restore versioned descriptor TOML metadata and validate TOML configuration value types.
- Cover real interactive/config generation, metadata, destination collisions, and concurrent file creation.


### Python 交互生成与验收收尾（2026-09-12）

- 产品入口改为 `rcsfs csfsgenerate [output]`，默认 `rcsf.out`；问答经内存 transcript 进入共享 Rust 流水线，cargo examples 保留作开发回归。
- 移除 Rust/Python/CLI 的记录数上限；保留整数溢出和格式边界检查。
- 增加单占据任务的有序态前缀并行，保持串行记录顺序。
- 修复直接描述符导出的全局轨道补零、sidecar 列序和稀疏耦合字段映射，使其与既有文本描述符路径一致。旧 CSV 哈希不再适用，CSF 文本哈希保持不变。
- 新增真实 Python CLI 的 e1/o1 1/2/4 线程哈希回归，以及原始/归一化描述符逐项回归。


### CLI 生成与并行阶段进展（2026-09-10）

- 新增 `generate_transcript_csfs` CLI：直接读取 `rcsfgenerate` transcript，执行
  占据枚举、并行 CSF 生成和确定性 J/宇称块合并。
- CLI 支持可选 CSV 描述符输出及 `--normalize` 归一化；描述符从完整整数记录派生。
- 新增 `generate_csfs_parallel`，支持 Rayon 线程数配置并保持任务顺序。
- `e1_cc1as1` 已通过 1/2 线程逐字节回归，`o1_cc1as1` 已通过 1/2/4 线程逐字节回归。
- 修复 roundtrip 测试并行运行时临时目录名称碰撞。
- transcript 描述符 CSV 现在伴随版本化 TOML sidecar，记录编码、归一化状态、记录数和轨道顺序。

### CSF 串行生成开发接口

- 新增串行全内存测量示例 `benchmark_generation` 与可重跑的隔离 Fortran
  计时脚本；登记完整输出逐字节校验、阶段耗时、RSS 和记录体逻辑 I/O。
- 新增 `CompleteCsfFile::write_record_to`，可从整数 chunk 逐条导出三行文本，
  不写文件头/块分隔符，也不逐条 flush。

- 新增 `csf_generation::generate_csfs`，从固定相对论占据组态直接枚举整数 CSF，
  包含原版态表、seniority、角动量耦合和按 J 分块的顺序。
- 新增 `csf_generation::enumerate_occupations`，对应原版 `BLANDA`：解析
  `rcsfgenerate` 交互输入记录、按 `slug.f90` 的上下界枚举非相对论占据、施加
  `blanda.f90` 的参考宇称判据、拆分相对论分量，并按 `TEST/LIKA` 降序归并多个
  参考组态。已有列表扩展模式及 Python 生成接口尚未实现。
- 新增 `generate_csfs` TOML 开发示例、记录上限和 Fortran `GEN` 差分测试。
- 登记 `e1_cc1as1`、`o1_cc1as1` 两份交互输入记录，其逐 J 块记录数与原版
  `rcsfgenerate` 输出完全一致（452,373 / 89,786 条），并写入回归测试。
- 严格 CSF writer 拒绝写出空列表，避免生成无法由严格 parser 读取的文件。

### 修复

- 完整整数 CSF 解析器逐条验证规范格式，拒绝会被静默改写的占据数、J 值、
  字段空白和未使用列；同时拒绝 CRLF 和末行缺少 LF，保证解析/写出无损。
- `roundtrip_csf` 示例仅创建新输出文件，拒绝覆盖已有文件或输入文件的链接。
- 完整 CSF 解析回归测试迁至 `tests/complete_csf_test.rs`，共享稳定 fixture。

### ✨ 新功能

- `read_csfs()` 新增 `strict=True`；默认拒绝末尾不足三行的 CSF，显式设置
  `strict=False` 时才沿用丢弃不完整末尾记录的转换行为。
- `read_csfs()` 新增可选的 `include_coupling_signature=True`，通过 descriptor
  共用的定宽解析器追加非空 `List(Int32)` 列；列表按已占据 peel subshell 顺序
  保存 coupling `2J`，末项为总 `2J`，并可直接在 Polars 中切片和分组。
- 默认关闭 coupling signature，原有 schema 与快速路径不变；可与
  `include_block_id=True` 组合使用，在 block 内比较不含 parity 的 coupling key。
- Windows x64 development build 微基准（`tests/fixtures/sample.csf`，28 CSFs，
  预热 20 次后调用 2,000 次）：disabled/default 为 0.462 ms/call、进程峰值
  working set 81.9 MiB；enabled 的 1/8/default workers 分别为
  1.108/0.823/0.654 ms/call，峰值 87.6/89.5/88.5 MiB。该小文件结果主要反映
  调用与线程池开销，不代表大文件吞吐量。
- 新增 `read_csfs(...) -> tuple[CsfHeaderData, polars.DataFrame]`，通过 Arrow C
  Stream 将 Rust 构造的 Arrow 数据直接交给 Polars，无需写入和重新读取
  Parquet；返回的 header 字典与 `convert_csfs` 写出的 TOML 结构一致。
- 支持包含多个 J 值块的 CSF 文件：`*` 分隔符不会进入 DataFrame；设置
  `include_block_id=True` 时增加零起始的 `UInt32` 块编号列。
- 新增 Polars 运行时依赖以及多 block、参数校验、异常输入和不完整末尾数据测试。
- 为 PyO3 扩展新增 `_rcsfs.pyi`，集中定义公开 `TypedDict`，并将 Python API
  更新为 Python 3.14 的 `X | None` 注解风格；严格 `basedpyright` 检查不再产生
  `Unknown`、裸 `dict` 或缺失类型参数错误。

## [1.3.1-beta.1] - 2026-07-17

### ✨ 新功能

#### ✅ CSF 零阶/一阶划分（zero-first partition）

**功能描述**:
- 新增 CSF 对称块（J^P）内的零阶/一阶空间划分能力，对标 GRASP2018 的 `rcsfzerofirst`
  Fortran 工具，但走 Parquet 中介路径。
- 每个块内：零阶参考 CSF 锁定到块首，完整列表中不在零阶的 CSF（一阶补集）追加其后。
- 匹配键为三行记录 `(line1, line2, line3)` 的精确字符串相等，与 `lodcsl_Part.f90` 一致。

**新增内容**:
- `src/csf_partition.rs`：Rust 核心，流式读取两个 Parquet + 按 `block_lengths` 切块 +
  `HashSet` 反匹配 + 写回 CSF 文本。
- `src/lib.rs`：注册 `partition_csfs` PyO3 函数。
- `src/csfs_conversion.rs`：`HeaderData`/`BlockInfo`/`HeaderInfo` 设为 `pub`，供 partition 复用。
- `rcsfs/__init__.py`：`partition_csfs` Python 包装 + `PartitionStats` TypedDict。
- `rcsfs/cli.py`：`zero-first` 子命令（CSF 进 → Parquet 中介 → CSF 出，含临时文件管理）。
- `tests/csf_partition_test.rs`：6 个 Rust 集成测试。
- `tests/cli_test.py`：5 个 CLI 测试。

**CLI 用法**:
```
uv run rcsfs zero-first zero.csf full.csf [reordered.csf]
```

**验证结果**:
- ✅ `uv run cargo test` 全绿（6 partition + 23 integration + 22 normalization + 1 doctest）。
- ✅ `uv run pytest` 全绿（10 个测试，含 5 个新 CLI 测试）。
- ✅ `uv run ruff check .` 通过。
- ✅ 端到端实测：`sample.csf` 拆分为 zero(4)+full(28)，输出 28 CSF（4 锁定到首 + 24 补集），
  header 一致、无丢失/重复。

---

## [1.2.2-beta.1] - 2026-04-28

### ⚡ 性能优化

#### ✅ Descriptor 并行生成结果侧列式缓冲

**问题描述**:
- 旧的并行 descriptor pipeline 中，worker 返回 `Vec<(idx, Vec<i32>)>`。
- writer 线程需要逐行执行归一化和列构建，`normalize=True` 时容易成为瓶颈。
- descriptor 输出本身是多列 `col_0..col_N`，逐行结果在写入前还需要再转换成列式结构。

**优化方案**:
- 将并行 worker 的输出改为 batch 级列式 buffer。
- raw 路径输出 `Vec<Vec<i32>>`，writer 直接转换为 `Int32Array`。
- normalized 路径在 worker 侧并行完成 per-CSF 归一化，输出 `Vec<Vec<f32>>`，writer 直接转换为 `Float32Array`。
- 保持 reader 到 worker 的输入结构不变，未启用 RecordBatch work item，避免引入历史上已观察到的性能回退风险。

**影响文件**:
- `src/csfs_descriptor.rs`
- `tests/integration_test.rs`
- `scripts/compare_descriptor_outputs.py`

**验证结果**:
- ✅ `cargo test` 全部通过。
- ✅ `pytest tests/rcsfs_test.py` 全部通过。
- ✅ 使用 385,600 个 CSFs 的真实数据验证，normalized descriptor 与 `1.2.1-beta3` 正确基线完全一致：
  - shape: `(385600, 87)`
  - schema equal: `True`
  - columns equal: `True`
  - exact equal: `True`
  - overall max abs diff: `0.0`
  - changed columns: `0`
- ✅ 同一份真实数据 descriptor 生成耗时从约 `2.4s` 降至约 `0.8s`。
- ✅ 服务器压力测试通过：14,585,607 个 CSFs、168 列 normalized descriptor，耗时 `1m38.4s`；运行期间调用全部 CPU，btop 观察多数核心保持 `75%+` 使用率；输出与旧基线完全一致：
  - shape: `(14585607, 168)`
  - schema equal: `True`
  - columns equal: `True`
  - exact equal: `True`
  - overall max abs diff: `0.0`
  - changed columns: `0`

### 🧪 测试工具

#### ✅ 新增 descriptor parquet 对比脚本

新增 `scripts/compare_descriptor_outputs.py`，用于比较新旧 descriptor parquet：

- 检查 shape、schema、列名是否一致。
- 检查逐元素完全相等。
- 输出最大绝对差异和容差内是否一致。
- 当存在差异时，按 `col_N` 映射到 subshell 和字段名。
- 可选输出对应 raw CSF parquet 行，便于定位解析差异。

---

## [1.1dev2] - 进行中

### 🐛 Bug 修复

#### ✅ 问题 1: Rayon 线程池重复配置导致多次调用失败

**问题描述**:
- `build_global()` 只能在程序生命周期内调用一次
- 用户多次调用 `convert_csfs()` 并指定 `num_workers` 时，后续调用会失败

**修复方案**:
```rust
// 修复前
rayon::ThreadPoolBuilder::new()
    .num_threads(n)
    .build_global()?;

// 修复后
match rayon::ThreadPoolBuilder::new()
    .num_threads(n)
    .build_global()
{
    Ok(_) => println!("配置 Rayon 线程池，使用 {} 个 worker", n),
    Err(_) => eprintln!("警告: Rayon 线程池已配置，忽略 num_workers={} 参数", n),
}
```

**影响文件**:
- `src/csfs_conversion.rs:128-145`

**测试验证**:
```python
# 第一次调用
stats1 = convert_csfs("file.csf", "output1.parquet", num_workers=4)
# 输出: 配置 Rayon 线程池，使用 4 个 worker

# 第二次调用（之前会失败，现在正常）
stats2 = convert_csfs("file.csf", "output2.parquet", num_workers=8)
# 输出: 警告: Rayon 线程池已配置，忽略 num_workers=8 参数
```

**验证结果**:
- ✅ 处理 428 万 CSF 成功
- ✅ 多次调用不会崩溃
- ✅ 警告信息清晰

---

#### ✅ 问题 3: 资源泄漏风险

**问题描述**:
- 如果后续操作失败或 panic，文件句柄可能泄漏
- `writer.close()` 在后面调用，但如果中途出错，文件可能不会正确关闭
- 不完整的输出文件可能残留在磁盘上

**修复方案**:
添加了 RAII 包装器 `ParquetFileGuard`，确保：
1. 文件句柄正确关闭
2. 出错时自动清理不完整的输出文件
3. panic 时也能正确清理资源

**影响文件**:
- `src/csfs_conversion.rs:16-61` (添加 `ParquetFileGuard` 结构体)
- `src/csfs_conversion.rs:212-217` (并行版本使用 guard)
- `src/csfs_conversion.rs:451-452` (顺序版本使用 guard)
- `src/csfs_conversion.rs:334, 340` (并行版本使用 guard 方法)
- `src/csfs_conversion.rs:563, 572` (顺序版本使用 guard 方法)

**修改内容**:
```rust
/// RAII wrapper for ArrowWriter that ensures proper cleanup on errors.
struct ParquetFileGuard<'a> {
    writer: Option<ArrowWriter<File>>,
    path: &'a Path,
    cleanup_on_drop: bool,
}

impl<'a> ParquetFileGuard<'a> {
    fn new(writer: ArrowWriter<File>, path: &'a Path) -> Self {
        Self {
            writer: Some(writer),
            path,
            cleanup_on_drop: true,
        }
    }

    fn finish(mut self) -> Result<(), ParquetError> {
        self.cleanup_on_drop = false;
        if let Some(writer) = self.writer.take() {
            writer.close()?;
        }
        Ok(())
    }
}

impl<'a> Drop for ParquetFileGuard<'a> {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            let _ = std::fs::remove_file(self.path);
        }
        let _ = self.writer.take().map(|w| w.close());
    }
}
```

**验证结果**:
- ✅ 文件句柄在任何情况下都能正确关闭
- ✅ 错误时不完整的输出文件会被自动清理
- ✅ 即使 panic 也能正确释放资源

---

#### ✅ 问题 5: 边界条件处理 - 无限循环风险

**问题描述**:
- 当 `lines_read > 0` 但 `num_full_csfs == 0` 时，存在无限循环风险
- 没有最大迭代次数保护
- 不完整的数据没有明确的处理策略

**修复方案**:
在 `convert_csfs_to_parquet` 和 `convert_csfs_to_parquet_parallel` 函数中添加了边界条件检查：

```rust
// 修复前
let num_full_csfs = batch_lines.len() / 3;
if num_full_csfs == 0 {
    if lines_read == 0 {
        break;
    }
    continue;  // 可能无限循环
}

// 修复后
let num_full_csfs = batch_lines.len() / 3;
if num_full_csfs == 0 {
    if lines_read == 0 {
        break;
    }
    // 防止无限循环：如果读取了行但无法组成完整 CSF
    if batch_lines.len() < 3 {
        eprintln!(
            "警告: 文件末尾有 {} 行不完整的数据，将被忽略",
            batch_lines.len()
        );
        break;
    }
    continue;
}
```

**影响文件**:
- `src/csfs_conversion.rs:258-273` (并行版本)
- `src/csfs_conversion.rs:511-528` (顺序版本)

**验证结果**:
- ✅ 防止无限循环
- ✅ 对文件末尾不完整数据发出明确警告
- ✅ 程序能够正常退出

---

#### ✅ 问题 4: 统一错误处理

**问题描述**:
- 某些函数返回 `Result<T, String>`
- 某些函数返回 `Result<T, Box<dyn Error>>`
- 错误消息混合使用中文和英文
- 错误上下文丢失（文件路径、行号等）

**修复方案**:
统一使用 `anyhow::Result` (即 `Result<T, anyhow::Error>`)：

1. **添加 anyhow 导入**:
```rust
use anyhow::{Context, Result};
```

2. **统一错误类型转换**:
```rust
// 修复前
fn foo() -> Result<T, String> {
    bar().map_err(|e| format!("Failed: {}", e))?;
}

// 修复后
fn foo() -> Result<T> {
    bar().with_context(|| "Failed")?;
}
```

**影响文件**:
- `src/csfs_conversion.rs`: 添加 `use anyhow::{Context, Result};`
- `src/csfs_descriptor.rs`:
  - 添加 `use anyhow::{Context, Result};`
  - `read_peel_subshells_from_header()`: `Result<Vec<String>>`
  - `j_to_double_j()`: `Result<i32>`
  - `parse_csf()`: `Result<Vec<i32>>`
  - `generate_descriptors_from_parquet()`: `Result<BatchDescriptorStats>`
  - `generate_descriptors_from_parquet_parallel()`: `Result<BatchDescriptorStats>`
  - 所有线程返回类型改为 `Result<T, anyhow::Error>`
  - PyO3 绑定函数使用 `e.to_string()` 转换错误
- `src/descriptor_normalization.rs`:
  - 添加 `use anyhow::{Context, Result};`
  - `normalize_electron_count()`: `Result<f32>`
  - `get_subshell_properties()`: `Result<[i32; 3]>`
  - `get_subshells_properties()`: `Result<Vec<i32>>`
  - `compute_properties_reciprocals()`: `Result<Vec<f32>>`
  - `normalize_descriptor()`: `Result<Vec<f32>>`
  - `batch_normalize_descriptors()`: `Result<Vec<Vec<f32>>>`

**修改模式**:
- `.map_err(|e| format!(...))` → `.with_context(|| ...)`
- `.ok_or("...")` → `.ok_or_else(|| anyhow::anyhow!(...))`
- `return Err(format!(...))` → `return Err(anyhow::anyhow!(...))`
- PyO3: `.map_err(|e| PyIOError::new_err(e))` → `.map_err(|e| PyIOError::new_err(e.to_string()))`

**验证结果**:
- ✅ 代码成功编译，无错误
- ✅ 统一的错误类型处理
- ✅ 保留完整的错误上下文
- ✅ 更好的错误信息追踪

---

### ✅ 无需修复

#### 问题 2: Python GIL 释放方式

**结论**: 原代码使用 `py.detach()` 是正确的

**原因**: 项目使用 PyO3 0.27.2，在此版本中：
- `py.detach()` = ✅ 推荐方式（PyO3 0.20+）
- `py.allow_threads()` = ⚠️ 已废弃

CODE_REVIEW.md 的建议基于旧版 PyO3，不适用于当前版本。

---

## 相关文档

- [代码审查报告](./CODE_REVIEW.md)
- [性能优化日志](./performance_optimization_log.md)
- [CSF 描述符指南](./CSF_DESCRIPTOR_GUIDE.md)
