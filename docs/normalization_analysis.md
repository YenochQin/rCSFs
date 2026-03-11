# 描述符归一化方案分析报告
请根据 docs/normalization_analysis.md 实现归一化优化

基于 GRASP `rcsfgenerate90` 源码审查（`genb.f90`, `kopp1.f90`, `kopp2.f90`）及 rCSFs 当前实现（`src/descriptor_normalization.rs`, `src/csfs_descriptor.rs`）。

---

## CSF 格式确认

每个 CSF 占三行：

| 行 | 内容 | GRASP 子程序 |
|----|------|------------|
| line1 | 各子壳层占据电子数 `n_i` | - |
| line2 | 各子壳层自身角动量 `2Q_i` | `kopp1.f90` |
| line3 | 逐步耦合的累积角动量 `2J_cum,i`，以及末态 `J+parity` | `kopp2.f90` |

关键调用链（`genb.f90`）：

```fortran
J(i)  = JKVANT(ORBIT(i), PLUS(i), ANTEL(i), Ii)  ! 子壳层自身 2Q_i
CALL KOPP1(POS, RAD2, J,  S, PAR)                ! J  -> line2
CALL KOPP2(POS, RAD3, JK, J, PAR, ANTKO)         ! JK -> line3
```

其中：

- `J(i)` 是单个 subshell 的允许 `2Q`
- `JK(i)` 是沿耦合链生成的中间累计 `2J`

因此，rCSFs 将 `line2` 视为 subshell-local 特征、将 `line3` 视为 cumulative coupling 特征，这一理解是正确的。

---

## 各归一化项分析

### 1. 电子数：`n_i / g_i`

- 记 `g_i = 2j_i + 1 = 2|kappa_i|`，即子壳层最大电子容量
- 当前实现等价于 `n_i / g_i`

结论：

- ✅ 完全正确
- ✅ 严格落在 `[0, 1]`
- ✅ 物理意义明确：壳层填充分数

---

### 2. 子壳层角动量：`2Q_i`

#### 2.1 当前分母 `kappa_i^2` 的物理意义

当前实现对 `line2` 使用：

```text
2Q_i / kappa_i^2
```

这不是随意经验缩放，而是一个**静态上界**：

- 对 jj-coupled subshell，`g_i = 2|kappa_i|`
- 给定占据数 `n_i` 时，该 subshell 自身最大允许 doubled-J 为

```text
U_i_occ = n_i * (g_i - n_i)
```

- 若再对所有可能占据数取最大值，则有

```text
max_n U_i_occ = floor(g_i^2 / 4) = kappa_i^2
```

因此：

- `kappa_i^2` 是 subshell-local `2Q_i` 的**静态最大上界**
- 它与 GRASP `JKVANT` 表一致

#### 2.2 结论与建议

结论：

- ✅ `2Q_i / kappa_i^2` 在物理上是成立的
- ⚠️ 但它不是最紧的上界，只是一个与占据数无关的静态上界

更推荐的方案：

```text
2Q_i / [n_i * (g_i - n_i)]
```

优点：

- 与当前 CSF 的实际占据匹配
- 比 `kappa_i^2` 更紧
- 保留 particle-hole symmetry

边界情况：

- 当 `n_i = 0` 或 `n_i = g_i` 时，理论上 `U_i_occ = 0`
- 此时该壳层的 `2Q_i` 也应为 0，可直接输出 0 而不是做除法

建议表述：

- `kappa_i^2`：可接受的静态归一化分母
- `n_i * (g_i - n_i)`：更物理、更推荐的占据相关分母

---

### 3. 累积耦合角动量：`2J_cum,i`

当前实现对 `line3` 使用一个统一常数：

```text
2J_cum,i / max_cumulative_doubled_j
```

这在物理上不够严格，因为 `line3` 的量是**沿耦合链的累计中间角动量**，不同位置的允许范围不同。

#### 3.1 为什么统一常数不够好

- 中间位置的 `2J_cum,i` 不一定受最终 `2J_target` 直接约束
- 合法的中间累计值可能大于最终总 `2J`
- 因此统一常数分母可能导致：
  - 有些位置过度压缩
  - 有些位置仍然大于 1

#### 3.2 位置依赖的静态上界

若记每个 subshell 的静态上界为

```text
u_i_static = kappa_i^2
```

则可构造第 `i` 个 cumulative coupling 的静态位置依赖上界：

```text
U_i_static = min( sum_{k<=i} u_k_static,
                  2J_target + sum_{k>i} u_k_static )
```

解释：

- 第一项：前 `i` 个 subshell 最多能形成多大的累计 doubled-J
- 第二项：若最终必须耦合到 `J_target`，则中间值不可能无限偏离最终值；后续 subshell 最多只能“拉回” `sum_{k>i} u_k_static`

这是一个合理的、位置依赖的**静态**上界。

#### 3.3 更推荐的占据相关上界

若进一步使用占据相关 subshell 上界

```text
u_i_occ = n_i * (g_i - n_i)
```

则更紧的 cumulative 上界为：

```text
U_i_occ = min( sum_{k<=i} u_k_occ,
               2J_target + sum_{k>i} u_k_occ )
```

这比纯 `kappa_i^2` 更能反映当前 CSF 的真实可达范围。

#### 3.4 关于 `J_target` 的依赖

这里必须明确：

- 位置依赖上界需要 `J_target`
- `J_target` 不能仅由 `peel_subshells` 推出

它必须来自下列之一：

- 当前 block 的固定 `Jpi`
- 当前 CSF 的 `line3` 末态值
- 调用方显式提供

因此：

- 若输入文件是单一 `Jpi` block，可按 block 预计算
- 若数据集中混有多个 `Jpi` block，则必须分 block 或逐 CSF 处理

#### 3.5 结论

- ❌ 不推荐继续使用单一常数作为所有 `2J_cum,i` 的分母
- ✅ 推荐改为位置依赖上界
- 优先级：
  1. `U_i_occ`（占据相关，最推荐）
  2. `U_i_static`（静态版，易实现）

---

### 4. 末位 cumulative J 的特殊情况

rCSFs 在解析时会将最后一个位置的 coupling 值覆盖为最终 `2J_target`。

这意味着：

- 在单一 `Jpi` block 内，最后一个 `J_coup` 是常数特征
- 它对区分同一 block 内不同 CSF 没有信息量

因此：

- 对单一 `Jpi` block 的 ML 任务，建议移除该列
- 若训练数据跨多个 `Jpi` block 混合，则该列可保留，但应明确它更像 block 标签上下文，而不是普通结构特征

---

## 推荐修改方向

### 最小改动版本

- 保持 `n_i / g_i`
- 保持 `2Q_i / kappa_i^2`
- 将 `2J_cum,i / 常数` 改为 `2J_cum,i / U_i_static`

优点：

- 改动小
- 保留当前静态预计算结构
- 显著优于统一常数方案

### 更物理版本

- `n_i / g_i`
- `2Q_i / [n_i * (g_i - n_i)]`
- `2J_cum,i / U_i_occ`

优点：

- 与当前 CSF 的占据更一致
- 上界更紧
- 更接近 GRASP 生成该 CSF 时的实际角动量可达域

注意：

- 该方案需要在归一化阶段读取 `n_i`
- 不能仅靠 subshell 列表静态预计算全部分母

---

## 总结

| 归一化项 | 当前分母 | 是否可用 | 更严谨建议 |
|---------|---------|---------|-----------|
| `n_i` | `g_i = 2|kappa_i|` | ✅ | 保持 |
| `2Q_i` | `kappa_i^2` | ✅ 静态可用 | 推荐改为 `n_i(g_i-n_i)` |
| `2J_cum,i` | 单一常数 | ❌ | 改为位置依赖上界 `U_i_static` 或 `U_i_occ` |
| 末位 `2J_coup` | 常数 `2J_target` | ⚠️ 仅作上下文 | 单一 `Jpi` block 内建议移除 |

最终结论：

- 当前 `line2` 的归一化方向是对的，但还能更物理
- 当前 `line3` 的归一化方案不够严格，建议优先修改
- 若只做一处关键修正，应先修正 cumulative coupling 的归一化
