# 合并代码审查 — 2026-04-27

**审查日期:** 2026-04-27  
**审查范围:** 合并 `docs/CODE_REVIEW_1.2.1-beta2.md` 和本文件原 Rust 专项审查中发现的问题；覆盖 `src/` 下的 Rust 核心代码、PyO3 绑定入口、Python 包装层、类型 stub、测试和相关文档。  
**审查方式:** 静态代码审查，重点关注正确性、边界条件、并发行为、资源清理、API 一致性、测试可运行性和大文件处理风险。  
**涉及文件:**

- `src/csfs_conversion.rs`
- `src/csfs_descriptor.rs`
- `src/descriptor_normalization.rs`
- `src/lib.rs`
- `rcsfs/__init__.py`
- `rcsfs/_rcsfs.pyi`
- `tests/integration_test.rs`
- `tests/rcsfs_test.py`
- `docs/CODE_REVIEW_1.2.1-beta2.md`
- `CLAUDE.md`

---

## 1. 总体评价

这个库的核心代码整体结构比较清晰，主要职责拆分合理：

- `src/csfs_conversion.rs` 负责把 CSF 文本文件转换成 Parquet，并保存 header TOML。
- `src/csfs_descriptor.rs` 负责从 Parquet 中读取三行 CSF 数据，生成固定长度 descriptor，并支持并行 pipeline。
- `src/descriptor_normalization.rs` 负责 descriptor 的物理归一化。
- `src/lib.rs` 负责 PyO3 Python 模块注册和 Python API 入口参数校验。

代码已经考虑了很多大文件处理场景，例如流式读取、分批写入 Parquet、释放 Python GIL、使用 bounded channel 控制 pipeline 内存，以及用 multi-column Parquet 格式避免 ListArray 开销。

合并两份 review 后，发现的问题主要集中在以下几类：

1. 部分统计字段语义不一致。
2. 固定宽度 CSF 文本解析混用了字节长度和字符长度。
3. 并行 descriptor 入口缺少 `num_workers=0` 校验，存在死锁风险。
4. descriptor 写入失败时缺少不完整输出文件清理。
5. 归一化逻辑对非法 descriptor 缺少物理合法性校验。
6. Python 暴露 API 与 Rust 内部能力存在不完全一致。
7. Python 包装层、类型 stub 和测试文件存在发布阻塞问题。
8. 部分文档和注释对实际行为描述不准确。

---

## 2. 高优先级问题

### 2.1 `num_workers=0` 会导致 descriptor 并行生成死锁

**位置:** `src/csfs_descriptor.rs:431`、`src/csfs_descriptor.rs:453`、`src/csfs_descriptor.rs:598`、`src/csfs_descriptor.rs:785`

**问题描述:**

`generate_descriptors_from_parquet_parallel` 中允许调用方传入 `num_workers=Some(0)`：

```rust
let num_workers = num_workers.unwrap_or_else(|| {
    let default = num_cpus::get();
    default
});
```

随后：

```rust
let channel_capacity = num_workers * 2;
let (work_tx, work_rx) = bounded(channel_capacity);
```

当 `num_workers == 0` 时：

- `channel_capacity == 0`，work channel 是无缓冲 channel。
- `for _worker_id in 0..num_workers` 不会创建任何 worker。
- reader thread 在向 `work_tx.send(work_item)` 发送第一批数据时会阻塞。
- 主线程随后先 `join` reader thread，等待 reader 完成。
- reader 等 worker 接收，worker 不存在，主线程等 reader，形成死锁。

**影响:**

Python 用户如果调用：

```python
generate_descriptors_from_parquet(..., num_workers=0)
```

程序可能永久挂起，而不是返回错误。

**建议修复:**

在 Rust 层和 Python 入口层都校验 worker 数量：

```rust
if matches!(num_workers, Some(0)) {
    return Err(anyhow::anyhow!("num_workers must be greater than 0"));
}
```

或者归一化为至少 1：

```rust
let num_workers = num_workers.unwrap_or_else(num_cpus::get).max(1);
```

更推荐显式报错，因为 `num_workers=0` 通常是用户配置错误。

---

### 2.2 并行转换和顺序转换的 `truncated_count` 语义不一致

**位置:** `src/csfs_conversion.rs:319`、`src/csfs_conversion.rs:324`、`src/csfs_conversion.rs:490`、`src/csfs_conversion.rs:491`

**问题描述:**

顺序转换函数 `convert_csfs_to_parquet` 中，`truncated_count` 是“被截断的行数”：

```rust
if line.len() > max_line_len {
    truncated_count += 1;
}
```

并行转换函数 `convert_csfs_to_parquet_parallel` 中，`process_line` 返回每一行是否被截断，但最后按 CSF 汇总成一个 bool：

```rust
let (line1, t1) = process_line(&chunk[0], max_line_len);
let (line2, t2) = process_line(&chunk[1], max_line_len);
let (line3, t3) = process_line(&chunk[2], max_line_len);

((csf_count + i) as u64, line1, line2, line3, t1 || t2 || t3)
```

后续每个 CSF 最多只加 1：

```rust
if truncated {
    truncated_count += 1;
}
```

也就是说：

- 顺序版本：一个 CSF 的 3 行都被截断，计数为 3。
- 并行版本：同样情况计数为 1。

**影响:**

`ConversionStats.truncated_count` 对用户暴露为同一个字段，但不同实现返回值含义不同。测试或下游质量控制逻辑如果依赖该字段，会得到不一致结果。

**建议修复:**

统一成“被截断的行数”，因为字段注释和输出信息都更像 line-level 统计。

并行版本可以返回截断行数：

```rust
let (line1, t1) = process_line(&chunk[0], max_line_len);
let (line2, t2) = process_line(&chunk[1], max_line_len);
let (line3, t3) = process_line(&chunk[2], max_line_len);
let truncated_lines = t1 as usize + t2 as usize + t3 as usize;
```

然后累加 `truncated_lines`。

---

### 2.3 `line.len()` 与 `chars().take()` 混用导致长度限制语义不清

**位置:** `src/csfs_conversion.rs:290`、`src/csfs_conversion.rs:291`、`src/csfs_conversion.rs:490`、`src/csfs_conversion.rs:505`

**问题描述:**

转换代码用 `line.len()` 判断是否超过 `max_line_len`，但截断时用 `line.chars().take(max_line_len)`。

在 Rust 中：

- `String::len()` 返回字节数。
- `chars().take(n)` 按 Unicode scalar value 数量截断。

对于 ASCII CSF 文件，两者通常一致。对于包含非 ASCII 字符的行，两者不一致。例如一个中文字符通常占 3 个 UTF-8 字节：

```rust
let line = "测试";
assert_eq!(line.len(), 6);
assert_eq!(line.chars().count(), 2);
```

当前行为会出现两个问题：

1. `max_line_len` 的含义不明确：到底是最大字节数，还是最大字符数？
2. StringBuilder 的容量预估使用 `num_full_csfs * max_line_len`，但实际截断后字符串字节数可能超过 `max_line_len`。

**影响:**

如果输入文件严格是 ASCII，这个问题不会实际触发。但代码没有显式校验 ASCII，因此遇到非 ASCII 内容时可能出现统计、截断和容量预估不一致。

**建议修复:**

如果 CSF 格式本身就是固定宽度 ASCII 文本，建议明确把输入约束为 ASCII：

```rust
if !line.is_ascii() {
    return Err(anyhow::anyhow!("CSF input must be ASCII"));
}
```

并把 `max_line_len` 明确定义为最大字节数，使用字节截断。

如果需要支持 UTF-8，则需要把所有固定宽度解析和容量估算都改成字符语义，但这和 CSF 固定宽度格式通常不匹配。

---

### 2.4 固定宽度 descriptor 解析按字节切片，但没有 ASCII 校验

**位置:** `src/csfs_descriptor.rs:877`、`src/csfs_descriptor.rs:878`、`src/csfs_descriptor.rs:993`、`src/csfs_descriptor.rs:994`

**问题描述:**

`chunk_string` 使用字节 chunk 把字符串切成固定 9 字节块：

```rust
s.as_bytes()
    .chunks(chunk_size)
    .map(|chunk| std::str::from_utf8(chunk).unwrap_or(""))
    .collect()
```

如果 chunk 切在 UTF-8 多字节字符中间，`from_utf8` 会失败，然后当前代码返回空字符串 `""`。这会导致解析静默丢失字段，而不是报错。

另外，电子数解析使用字节区间：

```rust
subshell_charges[6..8].trim().parse().unwrap_or(0)
```

这个操作要求索引 6 和 8 都位于 UTF-8 字符边界。对于非 ASCII 文本，可能 panic，也可能解析错误。

**影响:**

这类问题最危险的地方在于静默失败：descriptor 可能被生成出来，但内容已经错误。对 ML 数据集来说，这比直接报错更难排查。

**建议修复:**

固定宽度 CSF 格式建议使用 ASCII/bytes 解析，并在入口处校验：

```rust
if !line1.is_ascii() || !line2.is_ascii() || !line3.is_ascii() {
    return Err(anyhow::anyhow!("CSF lines must be ASCII fixed-width text"));
}
```

随后将 `chunk_string` 改成返回字节 slice 或明确不会失败的 ASCII slice。不要在 `from_utf8` 失败时返回空字符串。

---

## 3. 中优先级问题

### 3.1 `ParquetFileGuard::finish` 在 close 失败时不会清理不完整文件

**位置:** `src/csfs_conversion.rs:43`、`src/csfs_conversion.rs:44`、`src/csfs_conversion.rs:45`、`src/csfs_conversion.rs:46`

**问题描述:**

当前 `finish` 逻辑是：

```rust
fn finish(mut self) -> Result<(), ParquetError> {
    self.cleanup_on_drop = false;
    if let Some(writer) = self.writer.take() {
        writer.close()?;
    }
    Ok(())
}
```

`cleanup_on_drop` 在 `writer.close()` 之前就被设置为 `false`。如果 close 失败，函数会返回错误，但 drop 时不会删除输出文件。

**影响:**

转换失败时可能留下一个损坏或不完整的 Parquet 文件。用户下次看到文件存在，可能误以为转换成功。

**建议修复:**

只有 close 成功后才关闭清理：

```rust
fn finish(mut self) -> Result<(), ParquetError> {
    if let Some(writer) = self.writer.take() {
        writer.close()?;
    }
    self.cleanup_on_drop = false;
    Ok(())
}
```

---

### 3.2 Descriptor Parquet 写入没有 RAII 清理保护

**位置:** `src/csfs_descriptor.rs:200`、`src/csfs_descriptor.rs:210`、`src/csfs_descriptor.rs:370`、`src/csfs_descriptor.rs:473`、`src/csfs_descriptor.rs:482`、`src/csfs_descriptor.rs:773`

**问题描述:**

`csfs_conversion.rs` 为转换输出实现了 `ParquetFileGuard`，失败时会删除不完整文件。但 `csfs_descriptor.rs` 里 descriptor 输出没有类似保护。

顺序 descriptor 生成：

```rust
let output_file_handle = std::fs::File::create(output_file)?;
let mut writer = ArrowWriter::try_new(...)?;
...
writer.finish()?;
```

并行 descriptor 生成：

```rust
let output_file_handle = std::fs::File::create(output_file)?;
let mut writer = ArrowWriter::try_new(...)?;
...
writer.close()
```

如果中途 parse、normalize、RecordBatch 构造或写入失败，输出路径上可能残留半成品。

**影响:**

生成 descriptor 失败后，用户可能拿到一个存在但不完整的 Parquet 文件。

**建议修复:**

抽出一个通用 `ParquetFileGuard`，或者在 descriptor 模块实现类似 guard。注意 close 成功后再禁用 cleanup。

也可以考虑先写到临时文件，成功后原子 rename 到目标路径。这对避免半成品文件更稳健。

---

### 3.3 归一化缺少电子数合法性校验

**位置:** `src/descriptor_normalization.rs:261`、`src/descriptor_normalization.rs:263`、`src/descriptor_normalization.rs:288`、`src/descriptor_normalization.rs:298`

**问题描述:**

归一化函数从 descriptor 中提取电子数：

```rust
let n_elec: Vec<i32> = (0..n).map(|i| descriptor[3 * i]).collect();
let u: Vec<i32> = (0..n).map(|i| n_elec[i] * (g[i] - n_elec[i])).collect();
```

这里默认 `0 <= n_i <= g_i`。但函数没有校验这个条件。

如果 descriptor 中存在非法电子数，例如：

- `n_i < 0`
- `n_i > g_i`

则：

- `n_i / g_i` 可能小于 0 或大于 1。
- `u_i = n_i * (g_i - n_i)` 可能为负。
- `u_i_occ = min(prefix_i, two_j_target + suffix_i)` 也可能变成负数。
- 归一化结果可能完全失去物理意义，但函数仍返回 `Ok`。

**影响:**

如果上游解析错误或用户直接调用归一化函数传入非法 descriptor，数据质量问题会被静默放大。

**建议修复:**

在计算 `u` 之前校验：

```rust
for i in 0..n {
    if n_elec[i] < 0 || n_elec[i] > g[i] {
        return Err(anyhow::anyhow!(
            "Invalid electron count for subshell {}: {} exceeds capacity {}",
            peel_subshells[i],
            n_elec[i],
            g[i]
        ));
    }
}
```

同时建议校验用于归一化的分母必须非负。

---

### 3.4 `total_lines` 统计不包含 header，但输出文案像总行数

**位置:** `src/csfs_conversion.rs:228`、`src/csfs_conversion.rs:239`、`src/csfs_conversion.rs:609`、`src/csfs_conversion.rs:349`

**问题描述:**

转换函数先读取并跳过 5 行 header，然后才开始累加 `total_lines`。因此 `total_lines` 实际代表的是 CSF data lines，不是文件总行数。

但输出信息为：

```rust
println!("总行数: {}", total_lines);
```

`ConversionStats` 字段也叫：

```rust
pub total_lines: usize
```

**影响:**

用户可能认为 `total_lines` 是完整输入文件行数。实际值比真实文件行数少 5 行，空文件和 header 不完整文件时也更容易混淆。

**建议修复:**

有两种方案：

1. 保持当前计数方式，但字段改名或文案改成 `data_lines` / `CSF 数据行数`。
2. 让 `total_lines` 真正表示完整文件总行数，另外新增 `data_lines`。

考虑 API 兼容性，短期可先修改文案和文档，长期再考虑新增字段。

---

### 3.5 PyO3 的 descriptor API 没有暴露 header 自动检测能力

**位置:** `src/csfs_descriptor.rs:136`、`src/csfs_descriptor.rs:139`、`src/csfs_descriptor.rs:140`、`src/csfs_descriptor.rs:1086`、`rcsfs/__init__.py:203`

**问题描述:**

Rust 内部的顺序函数 `generate_descriptors_from_parquet` 支持：

```rust
peel_subshells: Option<Vec<String>>,
header_path: Option<PathBuf>,
```

也就是说它具备自动查找 header 文件并读取 peel subshells 的能力。

但 PyO3 暴露的函数是：

```rust
peel_subshells: Vec<String>,
```

Python wrapper 也要求用户必须传：

```python
peel_subshells: list[str]
```

**影响:**

Rust 内部已有的自动检测能力没有通过 Python 主 API 暴露。用户必须先调用 `read_peel_subshells()`，再把结果传给 `generate_descriptors_from_parquet()`。

这不是功能 bug，但 API 体验和内部能力不一致。

**建议修复:**

如果希望保持显式 API，则文档中说明必须传 `peel_subshells` 是设计选择。

如果希望暴露自动检测能力，可以把 Python API 改为：

```python
generate_descriptors_from_parquet(
    input_parquet,
    output_parquet,
    peel_subshells: list[str] | None = None,
    header_path: str | Path | None = None,
    ...
)
```

并在 Rust/PyO3 层接收 `Option<Vec<String>>`。

---

## 4. 性能和可维护性问题

### 4.1 `convert_csfs_to_parquet_parallel` 的实际并行收益有限

**位置:** `src/csfs_conversion.rs:270`、`src/csfs_conversion.rs:305`、`src/csfs_conversion.rs:329`、`src/csfs_conversion.rs:339`

**问题描述:**

并行转换函数主要把每个 CSF 的 3 行字符串处理放入 rayon：

```rust
chunks.into_par_iter().enumerate().map(...)
```

但主要流程仍然是：

1. 单线程读取文件。
2. 在内存中收集一批 lines。
3. 并行做轻量字符串截断。
4. 单线程构造 Arrow builders。
5. 单线程写 Parquet。

对大文件来说，瓶颈更可能在 I/O、内存分配、Arrow 构造和 Parquet 写入。单纯并行截断字符串不一定带来明显收益，反而可能增加临时 `String` 分配和调度开销。

**影响:**

函数名和文档强调 parallel，但性能收益可能取决于输入行长度和截断比例。如果大多数行不需要截断，并行部分的价值有限。

**建议优化:**

短期：在文档中更准确地描述并行范围。

中期：考虑把读取、处理、写入也改成 pipeline，类似 descriptor 生成逻辑。

长期：如果 CSF 转换只是把 3 行重组成列，可能可以减少中间 `Vec<String>` 和重复分配，直接按 batch append 到 builders。

---

### 4.2 descriptor 并行 pipeline 中归一化在 writer 线程串行执行

**位置:** `src/csfs_descriptor.rs:616`、`src/csfs_descriptor.rs:682`、`src/csfs_descriptor.rs:688`、`src/csfs_descriptor.rs:690`

**问题描述:**

worker 线程只负责：

```rust
generator_clone.parse_csf(...)
```

当 `normalize=true` 时，归一化逻辑在 writer 线程中执行：

```rust
for (idx, desc) in &descriptors {
    let two_j_target = infer_two_j_target(desc);
    let normalized = normalize_descriptor_per_csf(...);
}
```

这意味着：

- parse 是并行的。
- normalize 是串行的。
- writer 同时负责归一化、构建 Arrow arrays、写 Parquet。

**影响:**

当 descriptor 数量很大、`normalize=true`、轨道数较多时，writer 线程可能成为瓶颈。worker 可能已经完成解析，但 writer 来不及归一化和写入。

**建议优化:**

把归一化移动到 worker 阶段。可以让 `ResultItem` 根据 `normalize` 输出 enum：

```rust
enum DescriptorBatch {
    Raw(Vec<(u64, Vec<i32>)>),
    Normalized(Vec<(u64, Vec<f32>)>),
}
```

这样 writer 只负责保序和写入，CPU 计算继续并行。

---

### 4.3 默认 `chunk_size=3000000` 可能造成较高内存峰值

**位置:** `src/lib.rs:82`、`src/csfs_conversion.rs:226`、`src/csfs_conversion.rs:313`

**问题描述:**

Python `convert_csfs` 默认：

```rust
let chunk_size = chunk_size.unwrap_or(3000000);
```

这代表一次读取 3,000,000 行，也就是约 1,000,000 个 CSF。

在每个 batch 内，代码会保存：

- `batch_lines: Vec<String>`
- `lines_to_process: Vec<String>`
- 并行版本中的 `batch_results: Vec<(u64, String, String, String, bool)>`
- Arrow builders 内部 buffer

虽然代码会分批处理，不会把整个文件读入内存，但单批内的临时数据仍然可能很大。

**影响:**

对 34GB+ 文件而言，这个默认值可能适合高内存工作站，但在普通笔记本或共享服务器上可能造成内存压力。

**建议优化:**

1. 在 Python 文档中明确说明默认 chunk size 的内存成本。
2. 提供推荐配置，例如：
   - 普通机器：`chunk_size=300000` 到 `900000`
   - 高内存机器：`chunk_size=3000000`
3. 如有条件，可根据 `max_line_len` 和 `chunk_size` 估算 batch 上限并打印警告。

---

## 5. 低优先级问题和代码整洁性

### 5.1 `use toml;` 是不必要导入

**位置:** `src/csfs_conversion.rs:14`

**问题描述:**

Rust 2018+ 中可以直接使用 crate 名称，通常不需要：

```rust
use toml;
```

**建议修复:**

删除该导入即可。

---

### 5.2 `File::create(&output_path)` 存在不必要借用

**位置:** `src/csfs_conversion.rs:206`

**问题描述:**

`output_path` 已经是 `&Path`，再写 `&output_path` 形成 `&&Path`。Rust 会自动解引用，不影响功能，但风格上不够简洁。

**建议修复:**

```rust
let output_file = File::create(output_path)?;
```

---

### 5.3 `reader.lines()` 会先分配完整超长行，`max_line_len` 不能限制读取期内存

**位置:** `src/csfs_conversion.rs:66`、`src/csfs_conversion.rs:217`、`src/csfs_conversion.rs:237`

**问题描述:**

代码注释已经意识到：

```rust
/// The BufRead::lines() iterator allocates the full line before truncation.
```

这意味着 `max_line_len` 只限制写入和存储，不限制读取单行时的临时内存。

**影响:**

如果输入文件损坏，出现极长单行，程序仍然可能短时间分配大量内存。当前代码会打印 warning，但 warning 发生在完整 line 已经分配之后。

**建议修复:**

如果需要更强健的大文件输入防护，可以改用 `read_until(b'\n', &mut buffer)`，并在 buffer 超过阈值时提前报错或分段丢弃。

---

## 6. 测试建议

### 6.1 增加 `num_workers=0` 回归测试

建议增加 Rust 或 Python 测试，确保 descriptor 并行 API 对 `num_workers=0` 返回错误，而不是挂起。

建议测试点：

- `generate_descriptors_from_parquet_parallel(..., Some(0), ...)` 返回 `Err`。
- Python `generate_descriptors_from_parquet(..., num_workers=0)` 返回清晰异常或 `success=false`。

---

### 6.2 增加并行/顺序 `truncated_count` 一致性测试

建议构造一个 CSF，其中同一个 CSF 的 2-3 行都超过 `max_line_len`，分别调用：

- `convert_csfs_to_parquet`
- `convert_csfs_to_parquet_parallel`

断言两者的 `truncated_count` 相同。

---

### 6.3 增加非 ASCII 输入行为测试

如果项目决定只支持 ASCII CSF，则测试应断言非 ASCII 输入返回错误。

如果项目决定支持 UTF-8，则测试应覆盖：

- 非 ASCII header。
- 非 ASCII data line。
- 超过 `max_line_len` 的多字节字符。
- 固定宽度 chunk 不会静默返回空字符串。

---

### 6.4 增加非法电子数归一化测试

建议增加测试：

```rust
let descriptor = vec![3, 0, 0];
let subshells = vec!["s ".to_string()];
assert!(normalize_descriptor_per_csf(&descriptor, &subshells, 0).is_err());
```

因为 `s` 轨道最大电子数为 2，`3` 应该是非法输入。

---

### 6.5 增加失败写入清理测试

建议为 CSF 转换和 descriptor 生成都增加测试：

- 人为触发写入失败或 normalization 失败。
- 断言目标输出文件不存在，或只有成功完成后才出现。

如果改用临时文件 + rename，则测试应断言失败时最终目标路径不存在。

---

## 7. 合并补充问题

以下问题来自 `docs/CODE_REVIEW_1.2.1-beta2.md`，原 Rust 专项审查未覆盖或只部分覆盖。为了让本文档成为单一问题清单，这里按主题补齐。

### 7.1 `get_parquet_info` 把 `created_by` 错报为 `compression`

**位置:** `src/csfs_conversion.rs:682`、`src/csfs_conversion.rs:684`

**问题描述:**

`get_parquet_info` 当前把 Parquet metadata 的 `created_by` 写入 `"compression"` 字段：

```rust
dict.set_item(
    "compression",
    metadata.file_metadata().created_by().unwrap_or("Unknown"),
)?;
```

`created_by` 是 writer 标识，例如 `parquet-rs version 58.0.0`，不是压缩算法。Rust doc comment 和 Python docstring 都把该字段描述为压缩方法。

**影响:**

这是用户可见的错误数据。任何调用方如果根据 `info["compression"]` 判断压缩方式，都会拿到错误语义。

**建议修复:**

从 row group 的 column metadata 读取压缩算法：

```rust
let compression = metadata
    .row_group(0)
    .column(0)
    .compression();
dict.set_item("compression", format!("{:?}", compression))?;
```

需要为空文件或无 row group 的情况加保护。如果仍想暴露 writer 信息，应新增独立字段，例如 `created_by`。

---

### 7.2 Python `__version__` fallback 不可达且版本错误

**位置:** `rcsfs/__init__.py:54`、`rcsfs/__init__.py:62`

**问题描述:**

当前代码：

```python
try:
    from importlib.metadata import PackageNotFoundError, version
    try:
        __version__ = version("rcsfs")
    except PackageNotFoundError:
        __version__ = "0.1.0"
except ImportError:
    __version__ = "1.2.1-beta.2"
```

在受支持的 Python 版本中，`importlib.metadata` 是标准库，外层 `ImportError` 分支基本不可达。editable/dev 场景下会进入 `PackageNotFoundError` 分支，并返回旧版本 `"0.1.0"`。

**影响:**

开发安装或未安装 metadata 的环境中，包版本会显示错误，容易误导发布验证。

**建议修复:**

删除外层 `try/except ImportError`，并把正确版本放到 `PackageNotFoundError` 分支：

```python
try:
    __version__ = version("rcsfs")
except PackageNotFoundError:
    __version__ = "1.2.1-beta.2"
```

---

### 7.3 `_rcsfs.pyi` 声明了运行时未暴露的类

**位置:** `rcsfs/_rcsfs.pyi:50`、`rcsfs/_rcsfs.pyi:113`、`src/csfs_descriptor.rs:878`、`src/csfs_descriptor.rs:1123`、`src/csfs_descriptor.rs:1133`、`CLAUDE.md`

**问题描述:**

类型 stub 声明了：

```python
class CSFDescriptorGenerator:
    ...

class CSFProcessor:
    ...
```

但运行时并没有暴露这两个类：

- `CSFDescriptorGenerator` 在 Rust 中只是普通 struct，没有 `#[pyclass]` 和 `#[pymethods]`。
- `register_descriptor_module` 没有 `m.add_class::<CSFDescriptorGenerator>()?`。
- `CSFProcessor` 没有对应 Rust 类型。
- `CLAUDE.md` 也声称这两个类可以直接从 `rcsfs._rcsfs` 使用。

**影响:**

类型检查显示可用，但运行时 `from rcsfs._rcsfs import CSFDescriptorGenerator` 会失败。这是类型 stub 与实际 API 不一致的高风险问题。

**建议修复:**

短期建议删除 `_rcsfs.pyi` 中这两个 class 声明，并更新 `CLAUDE.md`。如果确实要暴露类，则需要补齐 PyO3 `#[pyclass]`、`#[pymethods]` 和模块注册。

---

### 7.4 `normalize=true` 对坏 CSF 的容忍度低于 `normalize=false`

**位置:** `src/csfs_descriptor.rs:620`、`src/csfs_descriptor.rs:629`、`src/csfs_descriptor.rs:687`、`src/csfs_descriptor.rs:698`

**问题描述:**

非归一化路径中，单个 CSF 解析失败会 warning 并输出零 descriptor，任务继续：

```rust
match generator_clone.parse_csf(line1, line2, line3) {
    Ok(desc) => desc,
    Err(e) => {
        eprintln!("Warning: Failed to parse CSF at index {}: {}", idx, e);
        vec![0i32; descriptor_size]
    }
}
```

归一化路径中，writer 线程对每个 descriptor 调用 `normalize_descriptor_per_csf(...) ?`。任何单个 descriptor 的归一化失败都会让整个任务失败。

**影响:**

同一批输入在 `normalize=false` 时可以完成，在 `normalize=true` 时可能中止。对长批处理任务来说，这种错误策略不一致很难预期。

**建议修复:**

统一两条路径的错误语义。默认建议保持容错：记录 CSF index，输出 `vec![0.0f32; descriptor_size]`，继续处理。若需要严格模式，应增加显式参数，并对 raw/normalized 两条路径同时生效。

---

### 7.5 `cargo test` 因 doctest 解析说明文字而失败

**位置:** `src/csfs_conversion.rs:159`、`src/csfs_conversion.rs:160`

**问题描述:**

Rust doc comment 中包含未标注语言的 fenced code block：

```rust
/// ```
/// File → [Read batch] → [Rayon parallel process] → [Write ordered] → repeat
/// ```
```

`cargo test` 会把它当作 doctest 编译。内容不是 Rust 代码，且包含 Unicode 箭头 `→`，导致 doctest 失败。

**影响:**

Rust 单元测试和集成测试即使通过，`cargo test` 仍会非零退出。该问题会直接破坏项目文档中的主要验证命令。

**建议修复:**

把代码块标记为 `text` 或 `ignore`：

```rust
/// ```text
/// File -> [Read batch] -> [Rayon parallel process] -> [Write ordered] -> repeat
/// ```
```

也可以改成普通列表，不使用代码围栏。

---

### 7.6 `num_workers` 在转换路径第一次调用后静默固定

**位置:** `src/csfs_conversion.rs:180`、`src/csfs_conversion.rs:195`

**问题描述:**

转换路径使用：

```rust
rayon::ThreadPoolBuilder::new()
    .num_threads(n)
    .build_global()
```

`build_global()` 在同一进程中只能成功一次。第一次调用之后，即使用户传入不同的 `num_workers`，后续调用也会复用第一次的全局 rayon pool。

**影响:**

Python 用户可能认为每次调用都能控制线程数，但实际第二次及之后的 `num_workers` 会被忽略。这是静默行为差异。

**建议修复:**

库代码中建议使用 per-call scoped pool：

```rust
let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(num_workers.unwrap_or_else(num_cpus::get))
    .build()?;
pool.install(|| {
    // batch work
});
```

---

### 7.7 `safe_parent_dir` 文档夸大了目录穿越防护

**位置:** `src/csfs_conversion.rs:75`、`src/csfs_conversion.rs:83`

**问题描述:**

`safe_parent_dir` 的 doc comment 声称可以 prevent directory traversal，但函数实际只是 canonicalize 用户路径的父目录：

```rust
fn safe_parent_dir(path: &Path) -> PathBuf {
    path.parent()
        .and_then(|p| p.canonicalize().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}
```

它没有校验 against allowed base directory，输出文件创建也仍使用原始 `output_path`。

**影响:**

实际行为对信任 Python 调用者的库来说可以接受，但文档会误导安全审计。

**建议修复:**

将注释改为描述真实行为：返回 `path` 父目录的 canonical absolute path，失败时 fallback 到 `.`。删除“防目录穿越攻击”的表述。

---

### 7.8 `tests/rcsfs_test.py` 已损坏

**位置:** `tests/rcsfs_test.py`

**问题描述:**

该测试导入了当前公共 API 中不存在的符号：

```python
from rcsfs import (
    convert_csfs_parallel,
    export_descriptors_with_polars_parallel,
    generate_descriptors_from_parquet_parallel,
    read_peel_subshells,
)
```

同时还硬编码了本机路径 `/Users/yiqin/Documents/PythonProjects/as3_odd4/cv4odd1as3_odd4_1.c`，并依赖未声明的 `graspkit`。

**影响:**

`pytest -q` 在干净环境中会 collection 失败。该文件让项目看起来有 Python 集成覆盖，但实际上不可运行。

**建议修复:**

删除该测试，或用当前公共 API、`tmp_path` 和仓库内小型 fixture 重写。

---

### 7.9 缺少 `normalize=true` 的端到端集成覆盖

**位置:** `tests/`

**问题描述:**

`descriptor_normalization.rs` 的单元测试覆盖了归一化函数的局部逻辑，但缺少 `generate_descriptors_from_parquet_parallel(..., normalize=true)` 的端到端测试。

**影响:**

无法捕获问题 7.4 这类 pipeline 级别的不对称错误，也无法验证 normalized descriptor 写入 schema、排序和失败策略。

**建议修复:**

增加一个小型 Parquet fixture 或由 CSF fixture 临时生成输入，分别跑：

- `normalize=false`
- `normalize=true`

并覆盖正常输入和包含单个 malformed CSF 的输入。

---

### 7.10 Rust 集成测试共享 `target/test_outputs/`

**位置:** `tests/integration_test.rs:18`、`tests/integration_test.rs:22`

**问题描述:**

所有 Rust 集成测试都写入 `target/test_outputs/`。当前文件名唯一，单独运行通常没问题，但该目录是共享状态。

**影响:**

未来如果测试复用文件名、并发运行或失败后留下输出，可能产生跨测试干扰。

**建议修复:**

使用 `tempfile::tempdir()`，让每个测试拥有隔离输出目录。

---

### 7.11 低优先级代码清理项

以下问题不影响主要行为，但可以在清理轮次中处理：

- `src/csfs_conversion.rs:261-275`：`num_full_csfs == 0` 分支中的 `continue` 不可达。
- `src/csfs_descriptor.rs:939`、`src/csfs_descriptor.rs:1037`：`occupied_orbitals` 被填充但从未读取。
- `src/csfs_conversion.rs:462`：`chunk_size as usize` 是无意义 cast。
- `src/csfs_conversion.rs:345`、`src/csfs_conversion.rs:578`：`writer_guard.writer.as_mut().unwrap()` 可用 `writer_mut()` 小 accessor 隐藏不变量。
- `rcsfs/__init__.py:52`、`rcsfs/__init__.py:64`：重复的 “Import from the Rust extension module” 注释。
- `src/csfs_descriptor.rs:840`、`src/csfs_descriptor.rs:858`：`j_to_double_j` 遇到 `"4/3"` 会返回 4，建议校验分母必须为 2。

---

### 7.12 额外性能建议

以下性能问题来自 1.2.1-beta2 审查，和第 4 节的性能问题互补：

- `src/csfs_descriptor.rs:557-566`：reader 线程把每行转成 4 个 `Arc<str>`，65,536 行批次约产生 260K 次小堆分配。可考虑发送 `RecordBatch` 本身给 worker。
- `src/csfs_descriptor.rs:603-644`：外层 OS worker 线程内部再调用全局 rayon `into_par_iter()`，可能造成嵌套并行和 oversubscription。可以让 worker 内部串行解析，或取消外层 worker，把并行性统一交给 rayon。
- `src/csfs_conversion.rs:215-216`：CSF 转 Parquet 默认未压缩，吞吐优先没问题；如果下游更关心存储空间，可向 Python 暴露 `compression` 参数。

---

## 8. 推荐修复顺序

### 第一批：正确性和死锁

1. 在 descriptor 并行入口拒绝 `num_workers=0`。
2. 修复 `get_parquet_info` 的 `compression` 字段语义。
3. 修复导致 `cargo test` 失败的 doctest。
4. 统一 `truncated_count` 统计语义。
5. 明确 CSF 输入是否必须 ASCII，并添加校验。
6. 归一化增加电子数合法性校验，并统一 `normalize=true/false` 的坏 CSF 处理策略。

### 第二批：文件完整性

1. 修复 `ParquetFileGuard::finish` 的 cleanup 时机。
2. 给 descriptor 输出增加 RAII guard 或临时文件写入策略。

### 第三批：Python API、类型和测试

1. 修复 Python `__version__` fallback。
2. 删除或实现 `_rcsfs.pyi` 中未注册的 `CSFDescriptorGenerator` / `CSFProcessor`。
3. 删除或重写损坏的 `tests/rcsfs_test.py`。
4. 增加 `normalize=true`、`num_workers=0`、非 ASCII、`truncated_count` 和失败清理回归测试。

### 第四批：API 和性能

1. 决定是否暴露 header 自动检测能力到 Python API。
2. 将 normalized descriptor 的归一化移动到 worker 阶段。
3. 调整或文档化默认 `chunk_size` 的内存成本。
4. 处理 `num_workers` 使用全局 rayon pool 后静默固定的问题。
5. 评估 `Arc<str>` 分配、嵌套并行和 CSF→Parquet 压缩参数。

### 第五批：文档和代码清理

1. 删除不必要的 `use toml;`。
2. 清理不必要的引用，例如 `File::create(&output_path)`。
3. 修正 `safe_parent_dir` 的过度安全声明。
4. 清理不可达 `continue`、无意义 cast、未使用变量、重复注释和 `j_to_double_j` 分母校验。
5. 补充与行为一致的注释和文档。

---

## 9. 总结

当前核心代码已经具备较好的大文件处理基础，尤其是 descriptor 生成 pipeline 的设计比较完整。合并两份审查后，最需要优先处理的是死锁风险、用户可见的元数据错误、`cargo test` 失败、统计语义不一致、固定宽度解析的 ASCII 假设没有显式化、归一化错误策略不一致，以及失败时半成品文件清理不一致。

这些问题大多可以通过局部修改解决，不需要推翻现有架构。建议先补齐输入参数校验、发布阻塞修复和回归测试，再逐步优化 pipeline 性能、Python API 一致性与文档准确性。
