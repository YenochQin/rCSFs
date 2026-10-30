# rCSFs 直接生成 GRASPG CSFG：实现方案与兼容契约

> 状态：2026-10-30 首版已实现；本文保留设计依据，实际接口和限制以 [使用说明](CSFG_GENERATION.md) 为准。
> 已完成共同连续空间的内存生成、Python/CLI 和 Gd 文件兼容验证；有限内存模式及复杂列表仍未实现。
> 上游依据：GRASPG `e157d9fc61e8d2882e0b29d644b2bb011f5f745d`，本工作区 suite 集成版本 `50ca581`。
> 本文源码链接按工作区的 `rCSFs/`、`grasp-atomic-suite/` 并列布局解析。代码事实以链接处实现为准；推导和建议另作标识。

## GRASPG 实际接受的文件契约

`rcsfg.out` 不是一种带显式 `type=...` 字段的新文本格式。它沿用五行 header 和每条三行 CSF 的布局；
普通标记空间 CSF 与 CSFG 代表项混合存储，以 ` *` 行分隔 `Jπ` 块。CSFG 类型由轨道顺序、
`rlabel.out` 所声明的标记空间和占据数推断。[展开读取](../../grasp-atomic-suite/src/lib/lib9290_csfg/rcsfsymexpand.f90)（123–173、269–298 行）

### 轨道顺序和两份文件的配套关系

- 第一组 GRASPG 生成输入建立标记轨道集合；其顺序按主量子数递增、同一 n 内按 l 递增，拆成相对论子壳层。
  后续相关轨道排列为连续的 κ 通道，每个通道中 n 递增。通道顺序为 `s, p-, p, d-, d, ...`，不是把全部轨道继续按 n 排列。
  [标记集合的建立](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/rcsfexcitation.f90)（512–609 行）、
  [相关通道 header 的建立](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/rcsfsym.f90)（150–183、307–315 行）、
  [通道表](../../grasp-atomic-suite/src/lib/lib9290_csfg/indexsym.f90)（26–54 行）。
- `rlabel.out` 第一行的计数 **包含闭核子壳层**；第二行按固定五字符槽存放全部标记子壳层（闭核在前，标记 peel 随后）。
  下游先扣除 header 闭核数，得到 peel 中的标记前缀长度，然后把余下部分视作相关通道。
  [写入](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/rcsfsym.f90)（118–132 行）、
  [读取](../../grasp-atomic-suite/src/lib/lib9290_csfg/setlaborb.f90)（28–32 行）、
  [扣除/恢复闭核计数](../../grasp-atomic-suite/src/lib/lib9290_csfg/rcsfsymexpand.f90)（166–203、474–479 行）。
- Gd I 的已核实例子是 `rlabel.out` 计数 22：14 个闭核子壳层加 8 个标记 peel 子壳层。`rcsfg.out` 第四行则有 44 个 peel 子壳层，前 8 个标记、后 36 个相关。
  这组数字用于解释格式，不应硬编码进实现。
- header 必须列出最终展开空间的全部相关轨道，而不只是出现在代表项中的最高两层。展开器通过通道起始 n 和连续 header 索引做替换；
  任意缺层、交错 κ 通道或独立重排 header 都可能改变/破坏展开。
  [展开循环](../../grasp-atomic-suite/src/lib/lib9290_csfg/rcsfsymexpand.f90)（318–452 行）。

### 五类代表项

下面的“同一对称性”指同一 κ；`p-` 和 `p` 是不同通道。设相关通道有连续的 q 个径向轨道，最高 n 为 N。
在单一、统一相关范围的模型中，代表项规则如下；标记轨道部分、子壳层态和耦合链仍必须完整保留。

| 类型 | 相关空间占据 | 代表项 | 展开个数 |
| --- | --- | --- | --- |
| 1 | 无相关电子 | 原来的标记空间 CSF | 1 |
| 2 | 一个单占相关子壳层 | 该 κ 的 N 层，保留全部合法耦合态 | q |
| 3 | 两个单占相关子壳层，κ 不同 | 两个 κ 各自的最高层 | q₁q₂ |
| 4 | 两个单占相关子壳层，κ 相同、n 不同 | 同一 κ 的 N−1、N 两层 | q(q−1)/2 |
| 5 | 一个双占相关子壳层 | 该 κ 的 N 层，保留允许的双电子子壳层态 | q |

依据：[类型定义、运行时分类与跨度计算](../../grasp-atomic-suite/src/lib/lib9290_csfg/findtype.f90)（10–22、129–219 行）、
[生成器对最高层及相邻层的筛选](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/rcsfsym.f90)（203–275 行）、
[最终筛选](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/rcsfsym_cyc.f90)（305–408 行）。

类型 4 和类型 5 必须分开生成：两个不同径向轨道的单占组合不等于同一子壳层双占，后者有自己的等价电子态约束。
q=1 时没有类型 4；不能构造一个落入标记空间的 `N−1` 伪代表。

### 顺序是兼容契约的一部分

每个 `Jπ` 块的标记空间 CSF 应放在相关代表项之前。下游不是只数全体类型 1，还用首次遇到相关项之前的前缀长度
计算 `ncsfDF1/scfDF1`，用于初始本征对及转换流程。因此“相同集合、任意排序”不够。
[findtype](../../grasp-atomic-suite/src/lib/lib9290_csfg/findtype.f90)（127–155 行）、
[rcsfsymexpand](../../grasp-atomic-suite/src/lib/lib9290_csfg/rcsfsymexpand.f90)（213–219、297–316 行）。

块内及展开顺序还决定 `map1/map2` 的连续编号和 `.m/.cm` 中系数的对应关系。
新生成的 CSFG 不能因总 CSF 数相同就与已有混合系数文件直接拼用。
[展开编号](../../grasp-atomic-suite/src/lib/lib9290_csfg/rcsfsymexpand.f90)（253–266、334–352、389–449、456–460 行）。

## 不能直接照抄的算法细节

上游流水线先生成/合并 CSF，再执行 `rcsfsym → rcsfblock → rcsfsym_cyc`。
[调用顺序](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/jjgen15b.f90)（125–154 行）。

第二及后续输入列表只对部分范围限制为最高两层：对 `n > max(reference n)` 的轨道使用最高、次高层规则；
参考最高 n 以内的未列轨道仍可能成为活动轨道。这解释了 Gd I 直接翻译输入为何产生许多随后被删除的低 n 中间项。
Rust 直接生成时应在代表项规划阶段确定实际参与生成的轨道，不复刻这个输入转译偶然产生的工作量。
[输入转译](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/rcsfexcitation.f90)（881–925 行）。

`rcsfsym_cyc` 对不能立即判断的记录执行同一块内向后比较。比较包含标记占据、相关占据/κ，以及去空格后的两行角动量信息；
不是按“非相对论组态＋Jπ”去重。其 `freeorb` 分支不能使用局部连续分组提前停止，可能扫描块内大量后续记录。
Rust 实现应使用完整的离散 CSF/耦合键及明确的覆盖关系，但不能在尚未证明径向跨度包含关系时只抹去 n 做全局去重。
[逐条比较](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/rcsfsym_cyc.f90)（469–533 行）。

## 建议首版支持范围

以下是根据读取方契约提出的实现范围，不是上游所有输入都已验证的声明。

1. 标记集合显式给定；全部参考组态的已占据子壳层都位于标记集合内。
2. 一个共同的标记集合、一个共同的相关上界集合；允许同一模型中的多参考组态并集。
3. 所有相关通道从标记上界之后连续展开；无径向洞和任意逐 n 占据例外。
4. 至多两个电子进入相关空间。首版可采用非负 S/SD 激发约束，并在规划后再次检查相关占据总数，不能只数不同相关子壳层数。
5. 保留现有 `c/i/*/minimum occupation` 的物理约束；只有能证明与代表项展开保持一致的约束才接受。不能把“代表项生成时抑制低 n”的内部措施当成用户物理冻结条件。
6. 首版拒绝不同 VV/CV/CC 活性范围列表混合、任意 n 依赖约束、三重/四重进入相关空间。以后逐项增加覆盖证明和外部基准。

拒绝超范围有明确依据：生成器在发现超过两个相关子壳层时停止；读取方只实现上述五类，并在遇到一个双占相关子壳层或两个单占相关子壳层后停止扫描。
因此“原交互层接受一个更大的激发整数”不等于现有 GRASPG CSFG 读取方支持任意高阶激发。
[生成侧检查](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/rcsfsym_cyc.f90)（336–408 行）、
[读取侧扫描](../../grasp-atomic-suite/src/lib/lib9290_csfg/findtype.f90)（129–145 行）。

多列表还涉及径向覆盖并集。上游对正好两次列表输入有专门的最高层快速筛选，而其他情况保留部分低层候选供进一步比较；
它明确把不同 VV/CV/CC 范围列为复杂情况。不能简单对每个 κ 取全局最大 n，否则可能把原本没有的通道组合加入空间；
也不能只按去 n 后的角动量键留一个代表，因为较小跨度可能属于不同物理分支。
[两列表特例](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/rcsfsym.f90)（239–276 行）、
[柔性范围比较](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/rcsfsym_cyc.f90)（411–533 行）。

输出兼容检查还应覆盖读取方的固定宽度和编译容量。当前上游生成器使用 n≤25 的枚举、21 个 κ 通道至 n 轨道，
多个展开路径的单条 CSF 行缓冲为 300 字符、header 缓冲为 1500 字符；suite 公共轨道容量默认 127。
这些不应被误写成原子物理上的普遍限制，也不能因为 Rust 支持更大数组就忽略目标程序的读取限制。
[生成器数组/通道约束](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/rcsfexcitation.f90)（22–46、575–596 行）、
[展开缓冲](../../grasp-atomic-suite/src/lib/lib9290_csfg/rcsfsymexpand.f90)（59–71 行）、
[共享容量](../../grasp-atomic-suite/src/lib/libmod/suite_parameters_M.f90)。

## 验收应分层进行

1. **文本协议**：header/核心计数/标记前缀/κ 通道顺序/固定宽度/块分隔全部正确；`rcsfg.out` 与 `rlabel.out` 作为一对校验，只有全部发布成功才报告成功；部分发布必须明确报告且不得继续调用下游（现有 helper 不是两文件原子事务）。
2. **五类小模型**：覆盖 q=0/1/2、多 κ、`p-` 与 `p`、同 κ 单占对与双占、闭核与无闭核、多个 `Jπ` 块。精确比较占据、子壳层态、seniority/额外态标签和完整耦合链。
3. **独立外部基准**：由原 GRASPG 生成参考文件；Rust 输出经原版 `rcsfgexpand_csfg` 展开，并与同一轨道顺序下独立完整生成结果做精确 CSF 键比较、重复检查和缺失检查。若要求原版顺序兼容，再比较逐字节或逐条序列。
4. **真实 Gd I 回归**：已有目标为 11,299,739 完整 CSF、1,125,587 条 CSF(G)，六个奇宇称块；现有实验确认 41,990 个占据/Jπ 分组维数相同，但这只是一层检查，不能代替精确耦合态检验。
5. **消费者链路**：小模型跑原版 `findtype/rcsfsymexpand`、角向程序和同一参数下的 RCI 对照；检查映射、矩阵维数、矩阵元/能量与系数对齐。物理精度验收与文件兼容验收分开报告。
6. **性能**：分别报告代表项规划、耦合枚举、合并、写文件、可选展开；记录线程数、峰值内存和磁盘量。10.62 秒的 Rust 完整生成结果表明有开发价值，但不能据压缩比承诺直接 CSFG 生成必定再快 10 倍。

外部 GRASPG 对照和本地 Gd 数据属于探索性验证，放在忽略的 `temp/`；rCSFs 常规 Cargo/pytest 测试应封存小型固定 fixture，不能依赖外部二进制或私人数据目录。
这遵循 [rCSFs 验证规则](../AGENTS.md)。Gd 实测数字来自本次本地实验汇总，属于单机单次结果，不是性能保证。

## rCSFs 已具备的基础与实际缺口

当前审查的 rCSFs 基线为 `4e57b463f6ad1599fdc5a6a27bf7e0c6c933eff6`。
以下复用判断来自现有代码，不表示已经提供 CSFG 功能。

| 现有模块 | 可以复用的行为 | CSFG 所需变化 |
| --- | --- | --- |
| [`electron_occupation_enumeration.rs`](../src/configuration_generation/shared_generation/electron_occupation_enumeration.rs) | c/i/*/最小占据规则、激发计数、nl→κ 分配、多参考占据配置并集 | 在“占据配置”层规划代表项；保留物理约束和径向覆盖信息 |
| [`angular_coupling_generator.rs`](../src/configuration_generation/shared_generation/angular_coupling_generator.rs) 的 `GenerationRequest` / `prepare_generation` | 已按传入的 configuration 顺序构造耦合链，不强制 n 顺序；子壳层状态选择只依赖 j 和占据数 | 在调用前提供 GRASPG 耦合顺序，不必重写角动量递归 |
| 同文件的 `GeneratedRecordSink`、`generate_configuration_records` | 整数记录流、可分割状态前缀、与存储分离的生成接口 | 增加 CSFG 输出 adapter，保留所有离散态标签和耦合信息 |
| [`generation_task_planning.rs`](../src/configuration_generation/shared_generation/generation_task_planning.rs) / [`csf_record_counting.rs`](../src/configuration_generation/shared_generation/csf_record_counting.rs) | 每占据配置精确计数、J/状态前缀任务拆分、确定的 task ordinal | 对代表占据配置规划，另报告展开后的逻辑计数 |
| [`complete_csf.rs`](../src/complete_csf.rs) 的 `format_record_parts` / `write_record_to` | 三行格式、seniority、显式零与未打印值的区分 | 复用记录编码；由 CSFG writer 提供自己的完整 header |
| [`complete_csf_generation_pipeline.rs`](../src/configuration_generation/csf_generation/complete_csf_generation_pipeline.rs) | Jπ 分块、缓冲写出、统计 | 当前 header 只收集实际用到的轨道并按 `(n,l,κ分支)` 排序，**不能原样用于 CSFG** |
| [`disk_csf_generation_pipeline.rs`](../src/configuration_generation/csf_generation/disk_csf_generation_pipeline.rs) | 分段顺序、资源预算、准确去重基础设施 | 当前围绕完整 CSF/V2 descriptors，且 `prepare_multiple_lists` / `precompute_peel_subshells` 重排 header；不能直接套用 |
| [`_publication.py`](../rcsfs/_publication.py) | 完成后发布、路径处理、部分发布错误报告 | 两个文件同批准备和校验；现有实现是逐文件原子发布，**不是两文件原子事务** |

还应区分两种“顺序”：

- **占据配置之间的枚举/合并顺序**：上游 `TEST` 仍按 n、l 遍历，比较 `(总占据, 高 j 占据, 低 j 占据)`。
  rCSFs 的 `EnumeratedConfiguration::key` 与此对应，不应把所有 n 排序统统改成 κ 排序。
  [上游比较器](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/test.f90)、
  [`enumerate_reference` / `merge_two`](../src/configuration_generation/shared_generation/electron_occupation_enumeration.rs)。
- **一条 CSF 内部的轨道及耦合顺序**：上游 `GEN` 使用 `POSN/POSL`；CSFG 模式需要标记前缀和相关 κ 通道顺序。
  改变它必须在生成耦合链之前完成，不能对已经耦合的 CSF 只改 header 或交换字段。
  [上游 GEN](../../grasp-atomic-suite/src/appl/rcsfggenerate90_csfg/genb.f90)（409–466 行）、
  [`prepare_generation`](../src/configuration_generation/shared_generation/angular_coupling_generator.rs)。

当前 transcript parser 明确只接受 `orbital_order="*"`。新模式应走结构化请求和显式 CSFG layout，
不能偷偷放宽原入口并使旧调用改变顺序。核心选择 5/6 的规范化也应复用现有逻辑，
不能把用户 core 编号直接当作 `ExcitationRequest` 中已经规范化的低层编号。

## 推荐的直接生成算法

### 先压缩占据配置，再展开角动量

推荐流程为：

```text
物理模型 + 显式标记空间
    ↓ 复用现有占据枚举
允许的相对论占据配置集合
    ↓ 检查径向覆盖，按五类规则规范化
唯一的代表占据配置 + 完整轨道 layout + 展开重数
    ↓ 以 GRASPG 顺序调用现有角动量递归
完整的代表 CSF 耦合态
    ↓ 确定性分块/前缀/顺序与文本编码
rcsfg.out + rlabel.out
```

“直接”指不生成完整 CSF 状态列表和中间 `.c` 文件；不必排斥在较小的占据配置层进行完整枚举。
Gd 已有 rCSFs 日志报告 7,154 个相对论占据配置，却有 11,299,739 条完整 CSF。
先复用这一步，再压缩配置，能避免最复杂的激发规则重复实现。

具体步骤：

1. **构造物理 layout。** 分离 closed core、labeling peel 与 correlation channels。每个 κ 通道保存真实的连续 n 范围，
   完整 header 和用于耦合的顺序都来自此 layout；代表项选择不修改用户模型。
2. **枚举允许占据配置。** 按原来的 `c/i/*/minimum`、参考并集、电子数、宇称和 SD 规则枚举。
   检查全部非零参考占据在标记空间，检查相关电子数≤2。
3. **规范化为代表配置。** 类型 1 不变；类型 2/5 改为本通道最高 n；类型 3 改为两通道各自最高 n；
   类型 4 改为同 κ 的最高两个不同 n。保留标记部分的每一个相对论占据数，不合并等价电子双占与不同径向轨道双单占。
4. **证明覆盖没有改变。** 对首版模型，在占据层展开这些模板，要求：

   ```text
   expand_occupation_templates(representatives) == allowed_relativistic_occupations
   ```

   此检查比较的是占据配置集合，不是千万条耦合后的 CSF。它可发现将空轨道的物理 `i` 约束误当成代表项优化、
   不连续径向范围、错误跨列表并集等导致的增项/漏项。计数相等不够，必须集合相等；模板去重后，还要求每个允许占据配置恰好被展开覆盖一次，以提前发现重叠代表项。
   需要同时验证核心/电子数、各 J 范围以及模板范围一致；超范围时在写文件前失败。
5. **占据模板去重。** 仅在上述共同径向范围和覆盖检查成立后，按完整代表相对论占据结构去重。
   这里尚未生成角动量态，因而同一模板只生成一次是正确的。不能把这个去重规则用于已经生成的 CSF 记录；
   后者的键还必须包含子壳层态、seniority/附加标签、完整耦合链、J 和宇称。
6. **生成耦合态。** 将每个模板按 layout 重排后交给现有生成器，枚举其全部合法角动量态。
   不移动已生成的耦合链。这样可直接利用 `4f⁷` 已有状态表和并行任务拆分。
7. **稳定输出。** 每个 Jπ 块先写标记空间 CSF，再写相关代表项；内部使用上游配置比较器和 GEN 状态遍历次序。
   工作者完成次序和 HashMap 迭代次序不能影响结果。固定顺序后，以相同输入的 GRASPG 文件作为逐条/逐字节基准。

该算法已用于首版实现，并通过当前 Gd 模型的 Fortran 文件对照；对于一般模型，
“角动量状态对 n 不敏感”不能替代覆盖、顺序和等价电子约束的验证。

### 不推荐把全文转换作为最终生产路径

先按 GRASPG 顺序生成全部完整 CSF、再以 Rust 筛选 CSFG，可以用作小规模独立参考实现或开发中间阶段。
它复用现有成熟递归，便于发现直接规划的遗漏，但仍承担完整 CSF 的生成与存储成本。
**已有普通 n 顺序 `.c` 文件不能作为这种参考实现的直接输入而只重排文本**；需重新按 CSFG layout 生成耦合态，
或者另做完整重耦合算法，后者不是本需求的一部分。

也不建议逐句移植 Fortran 的字符串处理、临时 CSF 文件及逐条向后比较；Rust 应生成代表配置并使用整数记录，
而不是仅用另一种语言重复冗余工作。

## 模块划分与拟议接口

设计建议将代表规则和兼容格式集中到 `src/configuration_generation/csfg_generation/`。目录整理后使用 `direct_csfg_generation_pipeline.rs`、`representative_occupation_planning.rs`、`graspg_csfg_output_writer.rs`，`mod.rs` 仅声明与导出；下表保留原来的职责划分：

| 内部模块 | 职责 |
| --- | --- |
| `representative_occupation_planning.rs` | 标记/core/κ 布局、五类模板、覆盖检查与确定性代表顺序 |
| `direct_csfg_generation_pipeline.rs` | 代表生成任务、计数与共享耦合内核衔接 |
| `graspg_csfg_output_writer.rs` | 标签前缀、块顺序、三行文本与标签文件编码 |
| `python_csfg_bindings.rs` | 薄 PyO3 适配，公开名称由 src/lib.rs 注册 |

外部只暴露一个生成入口；`CsfgRequest`、输出路径及选项组成小接口，调用方无需组装 Fortran 的两段问答，
也无需手动追加 `5f(0,i)` 等内部代表项限制。物理模型的解释由 Rust 负责。
复用现有 `GenerationRequest`/record sink/resource accounting；不为新模式复制一套 `GEN`、状态表或线程池。

Python/PyO3 保持薄层：配置读取、类型/路径检查和结果报告在 Python；科学枚举与 CSFG 规则在 Rust。
若新增公开 `generate_csfgs(...)`，需同步更新 `src/configuration_generation/csfg_generation/python_csfg_bindings.rs`（由 `src/lib.rs` 注册）、`rcsfs/__init__.py`、`rcsfs/_rcsfs.pyi` 和统计类型。
`ExcitationRequest` 的 core 规范化与 reference parser 宜提取为共享的内部输入处理，避免旧 transcript 与新结构化入口漂移。

建议新增命令 **`rcsfs csfggenerate`** 和配置表 **`[csfggenerate]`**，保留当前 `csfsgenerate` 完整 CSF 行为。
下面的首版配置现已支持；批量和资源预算等后续选项仍以现行使用说明为准：

```toml
[csfggenerate]
inactive_core = 4
reference_configuration = ["4d(10,c)5s(2,i)5p(6,i)6s(2,i)4f(7,*)5d(1,*)"]
labeling_space = "6s,5p,5d,4f"
active_space = "10s,9p,9d,8f,8g"
j_min = 2
j_max = 12
excitations = 2
rcsfg_out = "rcsfg.out"
rlabel_out = "rlabel.out"
```

- `j_min/j_max` 延续 2J 语义。`labeling_space` 首版要求显式提供，避免自动选择改变表示和输出顺序。
- 新命令固定使用 GRASPG layout，不复用 `orbital_order="*"` 的旧含义。批量条目和 `as/conf` 文件命名可后续按现有配置约定加入；不能把 `as=4` 当作物理 active-space 定义。
- 首版仅生成两个文本文件和统计，不自动生成完整 `rcsf.out`；展开是独立验证/可选步骤。
- 不能将当前完整 CSF 的 V2 descriptors、训练标签或混合系数直接用于 CSFG。若以后增加 CSFG descriptors，
  必须记录 representation、标记集合、径向范围/类型等语义并单独设计，不在本次首版复用旧含义。
- 两文件先在私有 staging 目录写完、校验配套关系，再使用现有发布机制；只有全部发布成功才返回成功。
  现有 helper 逐文件发布；发生部分发布时明确报告已发布/失败路径，不宣称两文件全有或全无。
  若未来要求整对原子发布，应设计新的输出目录契约，而不是在两个独立路径上假装已有事务。

建议统计至少包含 `label_csf_count`、`generator_count`、`record_count`、`expanded_csf_count`、
按类型/Jπ 的数量、峰值内存、各阶段计时与输出字节数。展开计数可由每个代表态乘其径向重数累加得到，
使用 checked 64 位整数；必须与独立展开校验，不能将公式自身再算一遍作为唯一测试。

## 分阶段实施与停止条件

### P0：固定契约与最小对照

- 封存五类小模型的上游输出、明确每类 order/前缀/长度；加入只有一个径向轨道、标记空间内空轨道、全标记无相关项等边界。
- 将 Gd 实验的输入、计数与哈希作为外部基准索引，不能把大文件或本机路径放进常规测试。
- 产出可独立检查的 layout/代表类型规范；这一阶段不改变当前 CSF 生成入口。

### P1：单一统一空间的直接 Rust 生成

- 实现占据层规划、覆盖检查与自定义耦合顺序；复用内存生成器/记录格式先取得正确的小模型结果。
- 写出 `rcsfg.out/rlabel.out`，先打通相同模型多参考、同一个活性范围和 J 区间；其它情况显式拒绝。
- 验收：小模型精确耦合态集合与数量无遗漏/重复，单线程和多线程输出稳定，兼容原 GRASPG 读入和展开。

### P2：可交付命令、确定性与资源控制

- 接入拟议 Python/CLI/TOML 入口及完整的类型/错误契约，补足文档、输出发布和路径错误测试。
- 保留确定的任务序号，按计数分批生成/合并。先支持内存模式；再将必要的有限内存文本/整数段接到同一 record sink，
  不为了借用当前 disk pipeline 而强制写不需要的完整 V2 descriptors。
- 没有实现的 memory/disk/预算选项应拒绝或不暴露，不能接受后忽略。明确区分受控分配预算和进程峰值 RSS。
- 验收：该 Gd 基准最终应为 1,125,587 条 CSF(G)，展开为 11,299,739 条；在相同 GRASPG 表示和已固定排序下，
  目标是复现参考 `.g/.l` 的逐字节结果。维数一致只是必要条件。

### P3：消费者验证与扩大模型范围

- 在小型输入上检查原版 angular/findtype/展开映射和 RCI 数值一致性，再进入真实原子计算。
- 混合 VV/CV/CC 不同 n 上界、多列表不同 J、任意径向例外、负激发/双激发限定及精确 jj-MR 子集等，
  每一种都要定义范围并集和去重语义后再开放。不得直接套用已有 `[[csfsgenerate.lists]]` 的完整 CSF 并集逻辑。
  现有多列表实现与上游 occupation-first merge 的差异已在 [CSF_GENERATION.md](CSF_GENERATION.md#multiple-lists-in-one-toml-file) 说明。
- 三个及以上电子进入相关空间不属于现有 GRASPG 五类读取契约，不能只扩展 Rust 生成器就宣称兼容。

当前 Rust 子壳层构造限制为 n≤15，而上游输入层可枚举到 25；本方案首版保留现有 Rust 范围，
不顺便扩大 n、状态表或 occupied-subshell 上限。Gd 的 n≤10 在此范围内。

## 性能预期与需要实测的事项

Gd 的已完成单机计时：完整 rCSFs 2.0.1（8 线程）10.62 秒；现有 Fortran CSFG 普通输入 494.75 秒，
减少冗余代表输入 286.48 秒；CSFG 展开约 12 秒。两份 Fortran 输入的 `.g/.l` 及展开文件哈希相同。
直接 Rust 方案避免了完整 CSF 文本中转和低 n 冗余状态，且能复用已验证的快速递归，因此有明确的实现价值。

方案阶段尚无 Rust CSFG 原型，因此当时没有承诺具体秒数或加速倍数；首版实测见 [Gd 基准](benchmarks/csfg_gd_20261030.md)。代表配置数量、耦合树、稳定合并和写出成本不按文件压缩比简单缩放。
实施后分别测 P1 内存版本和后续有限内存版本；最终报告耗时、RSS、输出大小和兼容性级别。
CSFG 下游角向/SCF/RCI 的收益必须独立测量。

本文最初只记录方案；后续首版实现与验证见 [使用说明](CSFG_GENERATION.md) 和 [Gd 基准](benchmarks/csfg_gd_20261030.md)。设计段落中的拟议功能不表示全部已交付。
