# V2 生成路径的唯一性：证明与穷举验证（P6a）

状态：结论已给出（2026-09-22），P6b 已据此让内部生产路径默认使用
`verified_unique`，并保留 `exact` 作为核验路径。结论只适用于 rCSFs 自身的内部生成路径，
不适用于外部 CSF 文本输入；P6b 的内容差分与自动回退验收状态以性能计划为准。

结论摘要：

1. 内部生成路径（transcript → 占据枚举 → 固定组态生成 → V2 行编码）**不会产出两条相同的 V2 行**。
   这既是"跨配置"的（不同组态的行绝不相同），也是"配置内"的（同一次遍历的不同产出绝不相同）。
2. 因此对这条路径，`deduplicate_v2_descriptor_segments` 的精确去重是**空操作**：它删除 0 行，
   也不改变发布顺序。
3. 决策：**P3 不适用**（不实施根桶并行化）；P2 不必为根桶格式投入；P6b 具备启动门槛所需的
   证明与回归证据，但实施时必须保留精确路径作为未证明输入的安全网。

本文件是论证，不是测量报告。测量（B1/B2 的 `duplicate_count = 0`）在
[benchmarks](benchmarks/README.md) 中登记，只是证据的一部分，不能替代证明。

## 1. 这个结论决定什么

计划中的 P2/P3/P6b 都围绕"精确去重"的成本展开：根桶与递归分桶要写完整 226 列的行，
在 B3 规模上约 4.9 TiB，是 scratch 的主要成分之一。是否值得为它做紧凑记录（P2b）、
并行化（P3）或整体跳过（P6b），取决于这条链路**是否真的会删掉东西**。

如果内部路径全局唯一，那么：

- 对内部路径优化根桶格式是把成本花在死代码上；
- 反过来，把根桶整条链路从内部路径移除（P6b）不会改变任何一条输出；
- 但如果只是"本机测到重复数为零"，这个结论不成立——B1/B2 只覆盖两个输入，
  并且它们的重复数为零也可能是巧合，而不是结构性质。所以 P6a 要求证明或反例。

## 2. 术语

**配置（configuration）**：一组相对论子壳层占据 `(subshell, electrons)`，即 `GenerationRequest.configuration`。
零占据的子壳层不出现。

**遍历（traversal）**：生成器的一次状态选择 + 耦合取值的完整组合，即
`Generator`（`src/csf_generation/mod.rs`）从 `selected` 到 `cumulative` 的一条完整路径。

**记录（record，可观测量）**：写出到输出、能被解码回来的全部内容——有占据子壳层的
`(subshell_index, occupation, state)`、GRASP 打印的中间耦合 `(boundary, two_j)`、
总 `2J` 与 parity。**隐藏耦合**（生成器算过但 GRASP 不打印的累积 `2J`）不属于可观测量。

**行（row）**：V2 的一行，每子壳层四通道 `n/2j/v/2k` 加全局 `total_two_j/parity`
（`descriptor_schema::DescriptorLayout`）。行就是去重链路的 key
（`descriptor_v2::write_feature_row` 写出，`streaming::stable_row_hash` 哈希，桶内再按行逐列精确比较）。

## 3. 前提

### 3.1 L1：占据键与占据向量互为确定

`occupations.rs::split_configuration`/`split_branches` 为每个非闭合字段枚举
`split_occupation` 给出的 `(lower, upper)` 对，并写 key 位 `(occupation, upper, lower)`；
`lower ≤ 2l`、`upper ≤ 2l+2` 且二者之和等于该轨道的非相对论占据，所以 `(lower, upper)`
与该轨道两个相对论伙伴的电子数**一一对应**（`l = 0` 时只有一个伙伴，`upper` 恒为 0）。
每个 reference 的字段表都由 `fields_for_reference` 在同一全局 slot 表上按同一顺序生成，
因此不同 reference 的 key 逐位可比。

于是 key 与占据向量互为函数：key 由占据向量唯一决定，占据向量也能由 key 唯一重建。
特别地 **key 相等 ⇔ 占据向量相等**。两条推论：

- `merge_two` 对相等 key 只保留一份（`split_configuration` 的 key、`merge_lists` 的归并），
  因此合并结果中任意两个配置的占据向量**两两不同**。这条合并不丢配置：key 相等意味着
  两个配置完全相同。
- 不同 key ⇒ 不同占据向量。

穷举验证：`tests/p6a_uniqueness_test.rs::the_enumeration_key_and_the_occupation_vector_agree`
对注册 fixture 的 1,374 个配置双向查表（key→占据、占据→key 都不冲突），并断言枚举出的
配置不含零占据；`the_multi_reference_merge_is_the_union_of_its_references` 断言合并结果
恰好等于各 reference 单独枚举的并集，且并集**小于**各 reference 之和——即该输入确实触发了
合并的"相等 key"分支，测试不是空跑。

### 3.2 L2：行的占据列唯一决定配置

`descriptor_v2::write_feature_row` 把每个子壳层的电子数写进 `n` 通道，未占据写 0；
`validate_record` 要求占据在 `1..=capacity`。因此行的 `n` 列编码的正是占据向量。
结合 L1：**不同配置的行不可能相等**，即不同配置产出的行集合互不相交。

这一条把"跨配置重复"从"需要去重兜底"变成了结构不可能：跨配置重复的前提是两个配置
占据向量相同，而枚举器已经排除了它。

### 3.3 L3：态表在 `(2J, seniority)` 上单射

`states.rs::subshell_states` 是生成器唯一的态来源（`prepare_configuration` 用它填
`choices`）。`states.rs` 的维护测试 `state_tables_are_injective_and_cover_the_documented_set`
穷举**所有可达的** `2j_max`（`capacity = 2|kappa|`、`1 ≤ |kappa| ≤ 11` ⇒ 全部奇数 `2j_max ≤ 21`）
与**所有**电子数，断言：

- 每张表的 `(2J, seniority)` 两两不同（相同两态在 V2 行里就是一列，无法区分）；
- 支持的 `(2j, electron/hole occupation)` 组合恰好是文档化的集合（`population ≤ 2` 全部，
  加上 `(5,3)(7,3)(7,4)(9,3)(9,4)(9,5)`）；新增一张表会让该测试失败，逼使论证重新审视。

同时 `emit` 的 `None` 约定（`two_j == 0 && choices[index].len() == 1` 才不写状态）与
表长 1 ⇒ 表项唯一一起，使"行里 `2j = MISSING`"恰好对应一个确定的态，解码无歧义。

### 3.4 L4：一次遍历最多产出一条记录

`Generator::select_states`/`couple`（`src/csf_generation/mod.rs`）在每个
`(状态选择, 中间耦合取值串)` 上只调用一次 `emit`：单子壳层时直接比较目标，
多子壳层时在末级判断可达性后调用一次。因此产出次数等于遍历数，不会有一次遍历写两行。

### 3.5 L5：行唯一决定遍历（隐藏耦合是可恢复的）

行里可用的信息是：每个子壳层的 `n`、`2j/seniority`（或 `MISSING`）、
boundary 落在 `2..occupied-1` 的 `2K`（其余为 `MISSING`），以及全局 `total_two_j`/`parity`。
设 `cumulative[0] = selected[0].two_j`、`cumulative[last] = target`，
中间 `cumulative[i]` 是 boundary `i+1` 处的累积 `2J`。对内部下标 `i`（`1 ≤ i ≤ last-1`）：

- 若该子壳层选中的态 `2J = 0`，则耦合区间退化为 `[cumulative[i-1], cumulative[i-1]]`，
  故 `cumulative[i] = cumulative[i-1]`，由前一个值决定；
- 否则若该 boundary 被打印，`cumulative[i]` 就是行里的 `2K`；
- 否则（未打印且 `2J ≠ 0`）：`emit` 只有在 `first` 为真时才不打印——`first` 为真意味着
  `selected[0].two_j == 0` 且此前每个内部态都是 `2J = 0` 且表长 1，于是
  `cumulative[i-1] = 0`，从而 `cumulative[i] ∈ [|0 - s|, 0 + s] = {s}`，等于该态的 `2J`。

归纳可得：**整条 `cumulative` 序列、连同状态选择，都由行唯一决定**。`first` 标志本身
是状态的函数（不是独立自由度），所以它不能成为两行相同而遍历不同的来源。

于是"行 → 遍历"是一个良定义的函数；`descriptor_v2::decode_v2_into` 是它在可观测量上的实现。
若两次产出 `r₁ ≠ r₂` 却有相同的行，则两者都由该行的逆映射得到，矛盾。**编码是单射。**

穷举验证不靠这段文字而是逐条构造逆映射：`tests/p6a_uniqueness_test.rs` 对每个产出记录
执行 `decode(encode(r)) == r`，只要有一条不成立就报出该行与两份内容。在单射成立时，
"逆映射逐条正确"与"没有两条记录同码"是等价的。

### 3.6 L6：写入器只接受佩尔序记录

`streaming::DescriptorBatchSink::push` 在编码前调用
`validate_record(peel_subshells, remapped_occupied, couplings)`，要求占据的全局下标
**严格递增**、耦合 boundary 落在 `2..occupied`、状态在合法态表内。生成器的耦合顺序与
运行级 Peel 表顺序一致（`enumerate_occupations` 先推 `kappa > 0` 伙伴再推 `kappa < 0`，
与 `precompute_peel_subshells` 的 `(n, l, kappa < 0)` 排序一致），所以内部路径满足该前提；
不满足的记录会被明确拒绝，而不是被悄悄编码成另一种坐标下的行。

## 4. 定理

对内部生成路径的任意一次运行，**所有产出的 V2 行两两不同**。

- 跨配置：由 L1 + L2，不同配置的占据列不同。
- 配置内：由 L3 + L4 + L5，同一配置的两条产出若行相同则是同一条记录，而 L4 保证一次遍历
  只产出一条。
- 顺序前提由 L6 保证；`RecordSelection` 的切分是同一遍历的后缀，不改变可观测量。

推论：`deduplicate_v2_descriptor_segments` 在这种运行上删除 0 行，`merge_v2_deduplicated_segments`
按稳定序发布的顺序就是原始生成顺序（去重保留首次出现，而无重复可删）。

## 5. 穷举程序与证据

`tests/p6a_uniqueness_test.rs`（debug 构建，约 3.5 秒）用**生产**生成器与编码器跑有界系统，
把"原始生成 / 配置内去重 / 全run去重"三层按记录序号逐条比较，并保留**反例**而不是布尔失败：
每条检查在失败时打印两条冲突记录的来源（哪个配置的第几条）。

| 家族 | 组态数 | 记录数 | 覆盖的现象 |
| --- | ---: | ---: | --- |
| 单子壳层态表 | 75 | 317 | 所有可达 `2j_max` 与所有电子数；表内 `(2J, seniority)` 唯一性 |
| 耦合链 | 324 | 242,602 | 1–4 个子壳层的所有子集组合、`4f/5g` 的重复 `2J` 不同 seniority |
| 隐藏耦合 | 99 | 9,305 | 首壳层填满时的 `FIRST` 抑制、内部 `2J = 0` 态不打印 |
| 仅占据不同 | 84 | 17,231 | 同一组子壳层、不同电子数（含"一个配置省略、另一个占据"） |
| 多参考运行 | 13 | 48 | 真实枚举器输出（含合并） |

每个家族的结论都是：0 个行重复、0 个 CSF 文本重复、0 个"解码回来不是原来那条记录"，
且三层给出一致的顺序。

反面控制（保证检查不是空跑）：

- `the_audit_reports_duplicates_and_the_layers_disagree`：把同一个配置排进任务两次，
  审计必须报出 100 对重复，全 run 层必须只保留 100 条，而配置内层必须**一条都发现不了**。
  这正是精确去重仍然存在的理由，也是 P6b 不能把"跳过根桶"扩大到可能重复输入的路径的理由。
- `a_repeated_reference_is_merged_not_generated_twice`：同一个 reference 写两遍，
  配置列表必须与写一遍完全相同（merger 消费相等 key），随后审计仍为 0 重复。

端到端证据（release 扩展，`tests/p6a_disk_uniqueness_test.py`，约 2 秒）：注册 fixture
`o1_cc1as1`（89,786 条，奇数总 `2J`）与 `e1_cc1as1`（452,373 条，偶数总 `2J`）完整走
生产磁盘路径，`duplicate_count == 0`、产物行数与生成数一致、P5a 的计数等于实际产出。

规模证据（测量，非证明）：`docs/benchmarks/` 中 B1（2,695,762 条）与 B2（560,351 条）的
全部登记运行 `duplicate_count = 0`；两者差异在机器日间波动之内，不能单独支撑结论。

## 6. 边界：什么没有被证明

- **不支持态的占据**：`subshell_states` 对未定义的 `(2j, occupation)` 直接报错，
  生成器拒绝该配置。这不是"未证明"，是"不存在"。
- **外部 CSF 文本路径**（`generate_descriptors_from_parquet` 读取用户提供的 CSF 文件）
  不在范围内：那里的重复来自外部数据，与内部构造无关，本证明对它没有任何承诺。
- **不经过枚举器的 `GenerationRequest`**：公开的 `generate_csfs` 接受任意组态。顺序不合
  佩尔序、状态不在态表内、占据超容量的输入会被 `validate_record` 拒绝；本证明只覆盖
  枚举器产出的、因而满足这些不变量的组态。
- **手工构造的零占据项**：`prepare_configuration` 会**静默丢弃**电子数为 0 的子壳层，
  所以 `{4f(2), 5g(0)}` 与 `{4f(2)}` 会产出相同的行。枚举器不产出零占据项
  （`split_branches` 只在 `> 0` 时压入，`the_enumeration_key_and_the_occupation_vector_agree`
  对注册 fixture 断言了这一点），因此内部路径不会遇到；但直接调用公开 `generate_csfs`
  的调用方如果自己造出零占据项，本结论不覆盖它。
- **V1 与旧内存路径**不在此范围。
- **依赖的代码位置**：L1（`occupations.rs` 的 key 构造）、L3（`states.rs` 的表）、
  L4/L5（`mod.rs` 的 `emit`/`couple`/`FIRST` 逻辑）、L2（`descriptor_v2.rs` 的通道布局）、
  L6（`descriptor_schema.rs` 的 `validate_record`）。这些位置的改动会让本文件失效，
  而回归测试只会在**覆盖到的**家族上失败——这正是 P6b 必须保留精确路径作为安全网的原因。
- 本结论不涉及"两次独立运行"或"人工拼接多个输出"的场景。

## 7. 决策

- **P3：不适用。** 启动条件（"P6a 发现一般反例，或只能把精确去重缩小到仍然构成显著瓶颈的边界"）
  都不成立：内部路径全局唯一，根桶并行化不会带来任何输出变化，只是把死代码并行化。
  决策记录：P3 标记为不适用，不实施。
- **P2：不改验收口径。** scratch 的 I/O 主体是源 segment（每行 916 字节）与最终产物，
  P2a/P2b 仍按原目标（B2 scratch 峰值降低 ≥ 50%）推进；但"为根桶做紧凑记录/优化"只在
  仍需保留根桶的路径上有意义，对内部路径应随 P6b 一起取消，而不是先优化再取消。
- **P6b：已实施，验收未完成。** 证明与回归都在本文件与 `tests/p6a_*` 中；实现保留了
  精确去重核验路径，并只让当前已证明的内部构造默认走快速路径。“发现不变量不满足时自动
  回退”在当前构造下不可达且尚未实现，等待评审明确豁免或在出现新的未证明构造时落地。
- **任何把"跳过去重"扩大到外部输入的改动都不在本结论范围内**，需要单独的证据。
