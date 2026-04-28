# Descriptor 流式 fast path 可行性与实现计划

## 0. 可行性结论

整体方向可行，但应分阶段推进：

1. **低分配 parser fast path 可行且应优先做。** 当前 `parse_csf()` 的主要额外开销来自补齐字符串、`chunk_string()`、临时 `Vec<&str>` 和 subshell 临时字符串。改为 ASCII 字节切片、按原始 9 字节 block 解析，可以在保持核心语义的前提下降低分配。
2. **用 fast parser 替换 worker 内部解析可作为第一阶段落地。** 这一阶段不改 reader/worker/writer 的 channel 结构，风险最低，便于用现有测试和 benchmark 对比。
3. **RecordBatch 直接传 worker 技术上可行，但性能收益不确定。** 仓库已有性能日志记录过一次 “RecordBatch Instead of Arc<str>” 尝试并回滚，原因是 CPU 利用率下降。因此这一步必须作为独立实验，用 benchmark 证明收益后再合入。
4. **严格错误策略会改变当前公开行为。** 当前实现对未知 subshell 和 per-block J 解析失败采取 warning/写零/跳过等容错行为。改成直接报错退出不是纯性能优化，应通过 strict mode 或单独 breaking change 处理。
5. **若要在并行流水线中遇错即停，需要补充取消机制。** 不能只在 worker 中 `return Err(...)`，还要避免 reader 阻塞在 bounded channel，并确保 writer 能正确结束或清理输出文件。

## 1. 背景

当前 descriptor 生成流程读取的是 `convert_csfs()` 生成的中间 CSF Parquet 文件，例如：

```text
o1j1qas5.c
  -> convert_csfs()
  -> o1j1qas5.parquet
  -> generate_descriptors_from_parquet()
  -> o1j1qas5_desc.parquet
```

其中 `o1j1qas5.parquet` 保存原始 CSF 文本的三行结构，典型列为：

- `idx`
- `line1`
- `line2`
- `line3`

现有实现已经是流式读取 Parquet，不会一次性读完整个文件。并行路径每批读取 65,536 行，再把每行转换为 `(idx, line1, line2, line3)` 发送给 worker，worker 调用 `parse_csf()` 将三行文本转成 descriptor。

目前可以进一步优化的点是：避免在 reader 阶段逐行构造 `Arc<str>`，并减少 `parse_csf()` 内部反复分配、切块和临时字符串处理。目标是在保持流式读取的前提下，对每个 batch 直接从原始固定宽度文本中提取需要的字段并填充 descriptor。

---

## 2. 目标

实现一个 descriptor 生成 fast path，分为保守阶段和实验阶段：

1. 仍然流式读取 `name.parquet`，不把完整 Parquet 一次性读入内存。
2. 第一阶段不改 pipeline 数据结构，只在 worker 中按原始 CSF 固定宽度 block 直接提取字段并生成 descriptor。
3. 严格保持现有 `parse_csf()` 对 `line2`、`line3`、`;`、fallback 和 final J 的语义。
4. 第一阶段默认保持现有容错行为；未知 subshell 直接报错应作为 strict mode 或后续行为变更处理。
5. 第二阶段再实验以 `RecordBatch` 为单位处理，目标是减少逐行字符串复制和 `Arc<str>` 分配；必须用 benchmark 验证。
6. 保持输出格式不变：descriptor Parquet 仍是多列 `col_0..col_N`，raw 为 `Int32`，normalized 为 `Float32`。

非目标：

- 不生成新的“裁切后文本 Parquet”作为中间文件。
- 不改变 descriptor 的列布局。
- 不改变 `convert_csfs()` 的输入输出契约。
- 不默认一次性读完整个 Parquet 到内存。

---

## 3. 关键解析规则

fast path 不能简单删除无关 block 后再调用旧 parser。原因是 CSF 三行之间依赖原始固定宽度位置，且最后一个 subshell 有特殊 final J 语义。

每个 CSF 应按以下步骤处理：

```text
subshells_line = line1.trim_end()
line_length = subshells_line.len()

middle_line = line2.trim_end() 左对齐补空格到 line_length

coupling_raw = line3.trim_end()
final_j = 从 coupling_raw 末尾提取
coupling_trimmed = coupling_raw 去掉前 4 字符和末尾 5 字符
coupling_line = coupling_trimmed 左对齐补空格到 line_length

按原始 line1 的 9 字节 block 顺序扫描:
  line1_block = subshells_line[block_idx * 9 .. block_idx * 9 + 9]
  line2_block = middle_line[block_idx * 9 .. block_idx * 9 + 9]
  line3_block = coupling_line[block_idx * 9 .. block_idx * 9 + 9]
```

注意：

- `line2` 和 `line3` 必须先补齐到 `line1.trim_end()` 的长度，再按同一个 `block_idx` 取 block。
- `block_idx` 必须来自原始 `line1` 的 block 位置，不能来自裁切后的 block 序号。
- “最后一个 subshell”必须按原始 `line1` 的最后一个 block 判断。
- 如果 `subshell` 不在 `peel_subshells` 中：
  - 兼容模式：保持当前行为，warning 后跳过该 subshell。
  - strict 模式：返回错误并终止任务。

---

## 4. 字段提取语义

对每个 9 字节 block：

1. 从 `line1_block[0..5]` 提取 subshell name 并 `trim()`。
2. 从 `line1_block[6..8]` 提取电子数。
3. 用同一个 `block_idx` 读取 `line2_block` 和 `line3_block`。
4. `line2_block` 的 J 值处理：
   - 如果为空，`middle_j = 0`。
   - 如果包含 `;`，取 `;` 后面的值。
   - 否则解析整个 block。
5. `line3_block` 的 coupling 值处理：
   - 如果 `line3_block` 非空，解析它。
   - 如果 `line3_block` 为空但 `line2_block` 非空，使用 `middle_j` 作为 fallback。
6. 如果当前 block 是原始最后一个 subshell，`coupling_j` 强制使用 `final_j`。
7. 如果电子数为 0，则该 subshell 对应 descriptor 三元组写入 `[0, 0, 0]`。
8. 否则写入 `[electron_count, middle_j, coupling_j]`。

这些规则要与当前 `parse_csf()` 保持一致。未知 subshell 从 warning 改为 error 应作为 strict 行为单独控制，不能混入默认 fast path，否则会改变现有 API 语义。

---

## 5. 建议实现步骤

### 5.1 抽出可复用 parser 核心

新增一个低分配 parser 函数，例如：

```rust
fn parse_csf_fast(
    line1: &str,
    line2: &str,
    line3: &str,
    generator: &CSFDescriptorGenerator,
    strict: bool,
) -> Result<Vec<i32>>
```

短期可以仍返回 `Vec<i32>`，先保证行为一致。后续再优化为直接写入预分配输出 buffer。

实现要点：

- 入口校验 ASCII。CSF 固定宽度格式应按字节解析，非 ASCII 输入应返回错误。
- 避免 `chunk_string()` 创建 `Vec<&str>`。
- 按 `block_idx` 用字节范围切片。
- 用原始 block 数判断 `is_last`。
- 默认兼容当前 parser：未知 subshell warning 后跳过，per-block J 解析失败写 0。
- strict 模式下：未知 subshell、非 ASCII、固定宽度不完整、J 解析失败均返回 `Err`。
- parser 核心应避免构造 subshell `String`；可用 `&str` 或 `[u8]` lookup，必要时给 `CSFDescriptorGenerator` 增加只读查询方法，避免暴露可变内部结构。

### 5.2 用 fast parser 替换 worker 内部解析

当前并行 worker 内部逻辑是：

```rust
generator_clone.parse_csf(line1, line2, line3)
```

可以先替换为：

```rust
parse_csf_fast(line1, line2, line3, &generator_clone, false)
```

第一阶段不改 channel 结构，先降低 parser 内部分配并保留兼容错误策略。

注意：第一阶段不应默认修正未知 subshell 为硬错误；否则现有 Python 调用和测试期望会改变。建议先新增 strict 行为，但 Python API 默认保持兼容。

### 5.3 优先优化结果侧列式 buffer

在 parser 行为稳定后，优先把 `Vec<(u64, Vec<i32>)>` 优化为 batch 级列式 buffer：

```rust
struct RawDescriptorBatch {
    idx: Vec<u64>,
    columns: Vec<Vec<i32>>,
}
```

normalized 路径可输出：

```rust
struct NormalizedDescriptorBatch {
    idx: Vec<u64>,
    columns: Vec<Vec<f32>>,
}
```

这样 writer 只需要把列 buffer 转成 Arrow arrays，避免 `Vec<i32>` per row 的分配。该优化不改变 reader 到 worker 的所有权模型，通常比直接传 `RecordBatch` 风险更低。

### 5.4 实验：将 reader 到 worker 的数据从 rows 改为 RecordBatch

后续可单独实验改 pipeline：

```rust
struct WorkItem {
    batch_idx: usize,
    batch: RecordBatch,
}
```

worker 从 `RecordBatch` 内部 downcast 出：

- `UInt64Array`
- `StringArray` for `line1`
- `StringArray` for `line2`
- `StringArray` for `line3`

然后按 row index 直接调用 fast parser。这样可以去掉 reader 线程中的：

```rust
let rows: Vec<(u64, Arc<str>, Arc<str>, Arc<str>)> = ...
```

减少每批大量小分配。

风险：

- 仓库历史性能日志中已有一次 RecordBatch 传 worker 的失败尝试，记录为 CPU 利用率下降。
- 可能的原因包括 Arrow column value 访问成本、batch 生命周期导致的缓存局部性变化、线程竞争方式变化。
- 因此该阶段必须保留在独立分支或 feature flag 下，用同一份中间 Parquet 与阶段一做基准对比。

---

## 6. 错误处理策略

建议把 descriptor 生成的错误处理拆成兼容模式和 strict 模式：

兼容模式，即默认行为：

- 非 ASCII 输入：返回错误，但现有 batch 生成路径仍可按当前策略写零 descriptor。
- 未知 subshell：warning 后跳过，保持当前 `parse_csf()` 行为。
- 固定宽度 block 尾部不完整：尽量保持当前 `chunk_string()` 的容忍行为，除非确认数据格式必须严格。
- `line2` / `line3` J 值解析失败：保持当前 per-block `unwrap_or(0)` 行为。
- normalized 路径中的归一化失败：保持当前 warning 并写零 descriptor 行为。

strict 模式，即后续可选行为变更：

- 未知 subshell：返回错误并终止任务。
- 非 ASCII 输入：返回错误并终止任务。
- 固定宽度 block 不完整：返回错误并终止任务，除非确认历史数据允许尾部空白缺失。
- `line2` / `line3` J 值解析失败：返回错误，而不是默默用 0。
- normalized 路径中的归一化失败：与 raw 路径保持一致，返回错误并终止任务。

并行流水线 strict 模式还需要新增取消策略：

- worker 遇到首个错误后，通过 error channel 或 shared cancellation flag 通知 reader/writer。
- reader 检查 cancellation，停止继续读取和发送 work item，避免阻塞在 bounded channel。
- writer 检查 cancellation，停止等待后续 batch，并通过 `ParquetFileGuard` 清理未完成输出。
- 主线程优先汇总首个根因错误，而不是只报告 channel send/recv 失败。

---

## 7. 测试计划

新增测试应覆盖：

1. fast parser 与现有 `parse_csf()` 在正常 CSF 上输出一致。
2. `line2_block` 包含 `;` 时，取 `;` 后的值。
3. `line3_block` 为空且 `line2_block` 非空时，coupling 使用 `line2` fallback。
4. 原始最后一个 subshell 使用 `line3` 末尾 final J。
5. 即使中间某些 subshell 不参与输出，也不能改变 last block 判断。
6. 未知 subshell 在兼容模式下 warning/跳过，在 strict 模式下返回错误。
7. 非 ASCII 输入返回错误。
8. `line2` / `line3` 比 `line1` 短时，先补齐后取同一 block index。
9. per-block J 解析失败在兼容模式下写 0，在 strict 模式下返回错误。
10. 第一阶段只替换 parser 后，输出行数、顺序和 descriptor 内容保持一致。
11. normalized 输出在 fast path 下 schema 和数值保持一致。
12. strict 模式下 worker 出错时，reader/writer 能正确停止，输出文件不会留下成功态坏文件。
13. 如果实验 `RecordBatch` work item，输出行数、顺序和 descriptor 内容必须与阶段一一致。

建议先构造小型内联 CSF 三行样例做 Rust 单元测试，再用 `tests/fixtures/sample.csf` 做端到端集成测试。

---

## 8. 性能验证计划

建议用同一份中间 Parquet 做三组 benchmark：

1. 当前实现：reader 转 `Arc<str>`，worker 调用 `parse_csf()`。
2. 阶段一：reader 仍转 rows，但 worker 使用 `parse_csf_fast()`。
3. 阶段二：reader 仍转 rows，但 worker/writer 使用 batch 级列式 buffer，避免 `Vec<i32>` per row。
4. 实验阶段：reader 发送 `RecordBatch`，worker 直接从 Arrow arrays 读取并调用 fast parser。

记录指标：

- 总耗时。
- 峰值内存。
- CPU 利用率。
- 每秒处理 CSF 数。
- 输出 Parquet 行数和校验 hash。

合入门槛：

- 阶段一必须在正常样例上与旧 parser 输出完全一致。
- 阶段一若没有明显性能提升，也可以保留为 parser 简化，但不应继续推进更高风险 pipeline 改动。
- RecordBatch 实验只有在同机、同数据、同 worker 数下稳定优于阶段一时才合入；否则保留现有 rows channel。

对 `o1j1qas5.parquet` 这类 1.6G 中间文件，仍建议保持流式处理。操作系统文件缓存会让重复读取接近内存速度，默认全量加载的收益未必能抵消内存峰值和实现复杂度。

---

## 9. 推荐落地顺序

1. 把未知 subshell 策略从 warning 改为 error，并加测试。
2. 新增 `parse_csf_fast()`，严格复刻现有 block、`;`、fallback 和 final J 语义。
3. 用单元测试证明 `parse_csf_fast()` 与现有 parser 在正常输入上一致。
4. 在 worker 内切换到 `parse_csf_fast()`。
5. 再把 `WorkItem.rows` 改成 `WorkItem.batch: RecordBatch`，减少 reader 端分配。
6. 性能测试后决定是否继续做列式 buffer 输出。
