# 修改历史 (Changelog)

本文档记录 rCSFs 项目的重要修改和改进。

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
