# 描述符归一化方案分析报告

基于 GRASP `rcsfgenerate90` 源码审查（`genb.f90`, `kopp1.f90`, `kopp2.f90`）及 rCSFs 当前实现（`src/descriptor_normalization.rs`, `src/csfs_descriptor.rs`）。

---

## 符号说明

本文统一使用以下符号，与代码变量名对应：

| 符号 | 代码变量 / 来源 | 含义 |
|------|----------------|------|
| `N` | `peel_subshells.len()` | 参与描述符的子壳层总数 |
| `i` | 子壳层在 `peel_subshells` 中的下标，`0 ≤ i < N` | 子壳层位置索引 |
| `n_i` | `descriptor[3*i]` | 子壳层 `i` 的占据电子数（来自 line1）|
| `2Q_i` | `descriptor[3*i+1]` | 子壳层 `i` 自身内部角动量（doubled，来自 line2）|
| `2J_cum,i` | `descriptor[3*i+2]` | 前 `i+1` 个子壳层逐步耦合后的累积角动量（doubled，来自 line3）|
| `g_i` | `get_max_subshell_electrons(subshell_i)` | 子壳层 `i` 的最大电子容量：`g_i = 2j_i + 1 = 2|kappa_i|` |
| `kappa_i` | 由 subshell 类型决定 | 相对论量子数（如 `s=-1`, `p-=1`, `p=-2`, `d-=2`, `d=-3`, ...） |
| `2J_target` | 当前 CSF 的 line3 末态，或单一 `Jpi` block 下由调用方统一传入 | 当前目标总角动量（doubled） |

---

## CSF 格式确认

每个 CSF 占三行：

| 行 | 描述符字段 | 内容 | GRASP 子程序 |
|----|-----------|------|------------|
| line1 | `n_i` | 各子壳层占据电子数 | — |
| line2 | `2Q_i` | 各子壳层自身内部角动量（doubled） | `kopp1.f90` |
| line3 | `2J_cum,i` | 逐步耦合的累积角动量（doubled），末态字段包含最终 `J` 和 parity，其中角动量部分对应 `2J_target` | `kopp2.f90` |

关键调用链（`genb.f90`）：

```fortran
J(i)  = JKVANT(ORBIT(i), PLUS(i), ANTEL(i), Ii)  ! 子壳层自身 2Q_i
CALL KOPP1(POS, RAD2, J,  S, PAR)                ! J  -> line2（各 2Q_i）
CALL KOPP2(POS, RAD3, JK, J, PAR, ANTKO)         ! JK -> line3（各 2J_cum,i）
```

其中：

- `J(i)` 对应 `2Q_i`：单个子壳层的内部允许角动量
- `JK(i)` 对应 `2J_cum,i`：沿耦合链累积生成的中间角动量

因此，rCSFs 将 line2 视为 subshell-local 特征、将 line3 视为 cumulative coupling 特征，这一理解是正确的。

---

## 各归一化项分析

### 1. 电子数 `n_i`

**当前分母：** `g_i = 2|kappa_i|`

归一化后：

```text
n_i / g_i   ∈ [0, 1]
```

结论：

- ✅ 完全正确
- ✅ 严格落在 `[0, 1]`
- ✅ 物理意义明确：壳层填充分数

**不需要修改。**

---

### 2. 子壳层内部角动量 `2Q_i`

#### 2.1 当前分母：`kappa_i^2`

当前实现对 line2 使用：

```text
2Q_i / kappa_i^2
```

`kappa_i^2` 的来源：对于 jj-coupled 子壳层，固定占据数 `n_i` 时，子壳层内部 `2Q_i` 的允许上界为：

```text
u_i(n_i) = n_i * (g_i - n_i)
```

对所有可能的占据数 `n_i ∈ {0, 1, ..., g_i}` 取最大值：

```text
max_{n_i} u_i(n_i) = floor(g_i^2 / 4) = kappa_i^2
```

最大值在半满时取到（`n_i = g_i / 2`）。因此 `kappa_i^2` 是对所有占据数都成立的**静态最大上界**，但对于非半满的 CSF 是宽松的。

#### 2.2 推荐分母：`n_i * (g_i - n_i)`

对固定 `n_i` 的当前 CSF，更紧的上界是：

```text
u_i(n_i) = n_i * (g_i - n_i)
```

归一化后：

```text
2Q_i / [n_i * (g_i - n_i)]   ∈ [0, 1]（当 n_i ∉ {0, g_i} 时）
```

边界情况：

- `n_i = 0`（空壳层）或 `n_i = g_i`（满壳层）时，分母为 0，但此时 `2Q_i` 物理上也必须为 0（空壳无粒子，满壳角动量对消），直接输出 `0.0`

优点：

- 与当前 CSF 的实际占据数直接对应
- 上界更紧：非半满时避免过度压缩
- 保留 particle-hole 对称性：`u_i(n_i) = u_i(g_i - n_i)`

#### 2.3 两种分母对比（`d ` 轨道，`g_i = 6`，`kappa_i^2 = 9`）

| `n_i` | `n_i*(g_i-n_i)` | `kappa_i^2` | 用 `kappa_i^2` 时 `2Q_i_max` 归一化结果 |
|--------|-----------------|-------------|----------------------------------------|
| 0 | 0 | 9 | 0.0（直接赋 0）|
| 1 | 5 | 9 | 5/9 ≈ 0.56（实际最大值仅 0.56，未达 1.0）|
| 2 | 8 | 9 | 8/9 ≈ 0.89 |
| 3 | **9** | 9 | **1.0**（半满时两者相等）|
| 4 | 8 | 9 | 8/9 ≈ 0.89 |
| 5 | 5 | 9 | 5/9 ≈ 0.56 |
| 6 | 0 | 9 | 0.0（直接赋 0）|

用 `n_i*(g_i-n_i)` 时，物理最大值始终归一化为 `1.0`。

#### 2.4 `n_i*(g_i-n_i)` 与 `kappa_i^2` 的对应关系表

##### s 壳层

| subshell | `kappa_i` | `g_i` | `kappa_i^2` | `n_i` | `n_i*(g_i-n_i)` |
|---------|----------:|------:|------------:|------:|----------------:|
| `s ` | `-1` | 2 | 1 | 0 | 0 |
| `s ` | `-1` | 2 | 1 | 1 | 1 |
| `s ` | `-1` | 2 | 1 | 2 | 0 |

##### p 壳层

| subshell | `kappa_i` | `g_i` | `kappa_i^2` | `n_i` | `n_i*(g_i-n_i)` |
|---------|----------:|------:|------------:|------:|----------------:|
| `p-` | `1` | 2 | 1 | 0 | 0 |
| `p-` | `1` | 2 | 1 | 1 | 1 |
| `p-` | `1` | 2 | 1 | 2 | 0 |
| `p ` | `-2` | 4 | 4 | 0 | 0 |
| `p ` | `-2` | 4 | 4 | 1 | 3 |
| `p ` | `-2` | 4 | 4 | 2 | 4 |
| `p ` | `-2` | 4 | 4 | 3 | 3 |
| `p ` | `-2` | 4 | 4 | 4 | 0 |

##### d 壳层

| subshell | `kappa_i` | `g_i` | `kappa_i^2` | `n_i` | `n_i*(g_i-n_i)` |
|---------|----------:|------:|------------:|------:|----------------:|
| `d-` | `2` | 4 | 4 | 0 | 0 |
| `d-` | `2` | 4 | 4 | 1 | 3 |
| `d-` | `2` | 4 | 4 | 2 | 4 |
| `d-` | `2` | 4 | 4 | 3 | 3 |
| `d-` | `2` | 4 | 4 | 4 | 0 |
| `d ` | `-3` | 6 | 9 | 0 | 0 |
| `d ` | `-3` | 6 | 9 | 1 | 5 |
| `d ` | `-3` | 6 | 9 | 2 | 8 |
| `d ` | `-3` | 6 | 9 | 3 | 9 |
| `d ` | `-3` | 6 | 9 | 4 | 8 |
| `d ` | `-3` | 6 | 9 | 5 | 5 |
| `d ` | `-3` | 6 | 9 | 6 | 0 |

##### f 壳层

| subshell | `kappa_i` | `g_i` | `kappa_i^2` | `n_i` | `n_i*(g_i-n_i)` |
|---------|----------:|------:|------------:|------:|----------------:|
| `f-` | `3` | 6 | 9 | 0 | 0 |
| `f-` | `3` | 6 | 9 | 1 | 5 |
| `f-` | `3` | 6 | 9 | 2 | 8 |
| `f-` | `3` | 6 | 9 | 3 | 9 |
| `f-` | `3` | 6 | 9 | 4 | 8 |
| `f-` | `3` | 6 | 9 | 5 | 5 |
| `f-` | `3` | 6 | 9 | 6 | 0 |
| `f ` | `-4` | 8 | 16 | 0 | 0 |
| `f ` | `-4` | 8 | 16 | 1 | 7 |
| `f ` | `-4` | 8 | 16 | 2 | 12 |
| `f ` | `-4` | 8 | 16 | 3 | 15 |
| `f ` | `-4` | 8 | 16 | 4 | 16 |
| `f ` | `-4` | 8 | 16 | 5 | 15 |
| `f ` | `-4` | 8 | 16 | 6 | 12 |
| `f ` | `-4` | 8 | 16 | 7 | 7 |
| `f ` | `-4` | 8 | 16 | 8 | 0 |

由上表可见：

- `n_i*(g_i-n_i)` 关于半满占据严格对称（particle-hole 对称）
- 在半满时，`n_i*(g_i-n_i)` 达到最大值，该最大值恰好等于 `kappa_i^2`
- `kappa_i^2` 是 `n_i*(g_i-n_i)` 在所有可能占据数上的最大包络

---

### 3. 累积耦合角动量 `2J_cum,i`

#### 3.1 当前分母：单一常数 `max_cumulative_doubled_j`

当前实现对 line3 的所有位置使用相同分母：

```text
2J_cum,i / max_cumulative_doubled_j   （对所有 i = 0, 1, ..., N-1）
```

在当前 API 的单一 `Jpi` block 用法下，`max_cumulative_doubled_j` 可视为 `2J_target`，由调用方传入。

这不够严格，原因如下：

- `2J_cum,i` 是沿耦合链**逐步积累**的中间值，不同位置 `i` 的允许范围本质上不同
- 前期位置（`i` 较小）的 `2J_cum,i` 受限于"前 `i+1` 个壳层最多能产生多大角动量"
- 后期位置（`i` 接近 `N-1`）的 `2J_cum,i` 还必须满足"后续壳层能把它拉回到 `2J_target`"的约束
- 统一使用 `2J_target` 作为分母，对前期位置可能过度压缩，对某些中间位置则可能出现归一化值超过 1

#### 3.2 推荐分母：占据相关的位置依赖上界 `U_i_occ`

记每个子壳层在当前 CSF 占据数 `n_i` 下的内部角动量上界（与第 2 节相同）：

```text
u_i = n_i * (g_i - n_i)
```

定义前缀累积和（前 `i+1` 个壳层的上界之和）：

```text
prefix_i = sum_{k=0}^{i} u_k   （k 从 0 到 i，共 i+1 项）
```

定义后缀累积和（第 `i+1` 到末尾的上界之和）：

```text
suffix_i = sum_{k=i+1}^{N-1} u_k   （k 从 i+1 到 N-1，共 N-i-1 项；i=N-1 时 suffix=0）
```

则第 `i` 个累积角动量 `2J_cum,i` 的占据相关上界为：

```text
U_i_occ = min(prefix_i,  2J_target + suffix_i)
```

两项约束来源：

- `prefix_i`：前 `i+1` 个壳层最多能形成多大的累积角动量（前方上限）
- `2J_target + suffix_i`：最终必须耦合到 `2J_target`，而后续壳层最多只能将累积值改变 `suffix_i`，因此中间值不能离终点太远（后方上限）

归一化后：

```text
2J_cum,i / U_i_occ   ∈ [0, 1]
```

对合法 CSF 且上述上界模型成立时，该归一化值应落在 `[0, 1]`。

边界情况：

- 若 `U_i_occ = 0`（前缀和为 0，即前 `i+1` 个壳层全空或全满），则 `2J_cum,i` 物理上也必须为 0，直接输出 `0.0`

#### 3.3 末位（`i = N-1`）的特殊情况

在 `parse_csf` 中（`src/csfs_descriptor.rs`），最后一个位置的耦合值被强制覆盖为 `final_double_j`（即 `2J_target`）：

```rust
// Special handling: last subshell uses final J value
if is_last {
    temp_coupling_item = final_double_j;
}
```

同时，末位的后缀和 `suffix_{N-1} = 0`，因此：

```text
U_{N-1}_occ = min(prefix_{N-1}, 2J_target + 0) = min(prefix_{N-1}, 2J_target)
```

若 `prefix_{N-1} ≥ 2J_target`（对合法可实现的 CSF 应成立），则 `U_{N-1}_occ = 2J_target`，归一化后：

```text
2J_cum,{N-1} / U_{N-1}_occ = 2J_target / 2J_target = 1.0（恒为常数）
```

这意味着：在单一 `Jpi` block 内，末位归一化值对所有 CSF 都是 `1.0`，该列不携带任何区分信息。对 ML 任务，可考虑在特征构建阶段移除该列（不在本次修改范围内）。

#### 3.4 关于 `2J_target` 的依赖

`U_i_occ` 的计算需要 `2J_target`。在单一 `Jpi` block 场景中，它可由 `max_cumulative_doubled_j` 统一提供；在多 `Jpi` block 场景中，则应从每条 CSF 的 line3 末态值中读取：

- 若输入文件是单一 `Jpi` block：`2J_target` 固定，调用方传入一次即可，`U_i_occ` 可逐 CSF 实时计算
- 若数据集混有多个 `Jpi` block：不同 CSF 的 `2J_target` 不同，需从每条 CSF 的 line3 末态值中读取，不能使用统一的 `max_cumulative_doubled_j`

当前 API 设计（`max_cumulative_doubled_j: Option<i32>`）适用于单一 `Jpi` block 场景。

---

## 推荐修改方案（更物理版本）

| 归一化项 | 当前分母 | 推荐分母 | 是否需要逐 CSF 计算 |
|---------|---------|---------|-------------------|
| `n_i`（descriptor[3*i]） | `g_i` | `g_i`（不变）| 否（静态预计算）|
| `2Q_i`（descriptor[3*i+1]） | `kappa_i^2` | `n_i * (g_i - n_i)` | **是**（依赖当前 CSF 的 `n_i`）|
| `2J_cum,i`（descriptor[3*i+2]） | 统一常数分母（当前 API 中通常由 `max_cumulative_doubled_j` 提供） | `U_i_occ = min(prefix_i, 2J_target + suffix_i)` | **是**（依赖当前 CSF 的各 `n_i`）|

完整的逐 CSF 归一化逻辑（对每个子壳层位置 `i = 0, ..., N-1`）：

```text
归一化后 descriptor[3*i]   = n_i / g_i

若 n_i == 0 或 n_i == g_i:
    归一化后 descriptor[3*i+1] = 0.0
    （u_i = 0，2Q_i 物理上必须为 0）
否则:
    归一化后 descriptor[3*i+1] = 2Q_i / [n_i * (g_i - n_i)]

u_i = n_i * (g_i - n_i)
prefix_i = sum_{k=0}^{i} u_k
suffix_i = sum_{k=i+1}^{N-1} u_k
U_i_occ  = min(prefix_i, 2J_target + suffix_i)

若 U_i_occ == 0:
    归一化后 descriptor[3*i+2] = 0.0
否则:
    归一化后 descriptor[3*i+2] = 2J_cum,i / U_i_occ
```

### `U_i_occ = min(prefix_i, 2J_target + suffix_i)` 的物理含义

这个分母的目标不是为某个单独子壳层构造一个“局域上界”，而是为位置 `i` 上的累计耦合量 `2J_cum,i` 构造一个与整条 CSF 一致的、逐位置变化的可实现上界。

先定义三个量：

- `J_left = J_cum,i`：从第一个子壳层耦合到第 `i` 个子壳层后得到的累计角动量
- `J_right`：第 `i` 个子壳层之后所有剩余子壳层最终合起来所形成的总角动量
- `J_target`：整条 CSF 的目标总角动量

最终态必须满足角动量耦合三角不等式：

```text
|J_left - J_right| <= J_target <= J_left + J_right
```

其中与当前位置上界直接相关的是左侧约束：

```text
|J_left - J_right| <= J_target
```

它给出一个必要条件：

```text
J_left <= J_target + J_right
```

若采用 doubled-J 记号，则变为：

```text
2J_left <= 2J_target + 2J_right
```

这说明：即使前半部分已经耦合出了很大的累计角动量，后半部分也必须仍有足够的耦合能力把它“接回”最终的 `J_target`。否则，该位置上的 `J_cum,i` 就不可能属于一条合法的、最终总角动量为 `J_target` 的 CSF。

在归一化方案里，我们并不试图精确求解后半部分的真实最大 `2J_right`，而是用一个 occupancy-dependent 的耦合能力尺度去近似刻画每个子壳层还能提供多少角动量自由度：

```text
u_k = n_k * (g_k - n_k)
```

它有几个直观性质：

- 空壳层和满壳层时 `u_k = 0`，因为它们不再提供内部耦合自由度
- 半满附近 `u_k` 较大，表示该子壳层的耦合自由度更丰富
- `u_k` 依赖当前 CSF 的占据数 `n_k`，因此是逐 CSF 变化的

于是我们定义：

```text
prefix_i = sum_{k=0}^{i} u_k
suffix_i = sum_{k=i+1}^{N-1} u_k
```

其中：

- `prefix_i` 表示前半部分自身最多能积累出的耦合能力预算
- `suffix_i` 表示后半部分还能提供的耦合调整预算

因此，对 `2J_cum,i` 的上界需要同时满足两类限制：

```text
2J_cum,i <= prefix_i
2J_cum,i <= 2J_target + suffix_i
```

最终取二者较小值：

```text
U_i_occ = min(prefix_i, 2J_target + suffix_i)
```

这里的第二项 `2J_target + suffix_i` 正是由三角不等式中的

```text
2J_left <= 2J_target + 2J_right
```

启发而来，其中 `2J_right` 用后半部分的耦合能力预算 `suffix_i` 近似替代。换句话说，`suffix_i` 不是说“后面的角动量直接线性加到前面”，而是说“后面的壳层总共还能提供多少耦合调整空间”，从而约束当前位置允许出现的累计角动量大小。

### 一个具体例子：为什么 `4f (4)` 的分母是 `13`

考虑以下占据：

```text
5s (2)  4d-(4)  4d (6)  5p-(2)  5p (4)  6s (2)  4f-(3)  4f (4)  5d (1)
```

对应的 line2 / line3 中，与 `4f (4)` 和总态相关的角动量信息为：

- `4f (4)` 对应的 line3 元素是 `"7/2"`，因此 `2J_cum,i = 7`
- 最终总角动量是 `"4-"`，因此 `2J_target = 8`

对各占据子壳层计算 `u_k = n_k * (g_k - n_k)`：

- `5s (2)`：`g=2`，`u=2*(2-2)=0`
- `4d-(4)`：`g=4`，`u=0`
- `4d (6)`：`g=6`，`u=0`
- `5p-(2)`：`g=2`，`u=0`
- `5p (4)`：`g=4`，`u=0`
- `6s (2)`：`g=2`，`u=0`
- `4f-(3)`：`g=6`，`u=3*(6-3)=9`
- `4f (4)`：`g=8`，`u=4*(8-4)=16`
- `5d (1)`：`g=6`，`u=1*(6-1)=5`

因此在 `4f (4)` 这个位置：

```text
prefix_i = 0+0+0+0+0+0+9+16 = 25
suffix_i = 5
U_i_occ = min(25, 8 + 5) = 13
```

于是该位置第三个描述符的归一化值为：

```text
2J_cum,i / U_i_occ = 7 / 13 ~= 0.53846
```

这里 `suffix_i = 5` 的原因也很直接：`4f (4)` 之后只有 `5d (1)` 仍然占据，而其耦合预算正是 `u_{5d} = 1*(6-1)=5`；再后面的 peel subshell 都是空的，因此贡献为 `0`。

### 实现注意事项

- `g_i` 仍可静态预计算（仅依赖 `peel_subshells`）
- `u_i`、`prefix_i`、`suffix_i`、`U_i_occ` 均依赖当前 CSF 的 `n_i`，需逐 CSF 计算
- `n_i` 已在 `parse_csf` 中解析并存入 `descriptor[3*i]`，可直接读取，无需重新解析
- 性能影响：每条 CSF 多做 `O(N)` 次整数运算，相比 IO 和解析开销可忽略不计
- 当前的预计算倒数方案（`normalization_reciprocals`）不再适用，需改为逐 CSF 调用归一化函数
