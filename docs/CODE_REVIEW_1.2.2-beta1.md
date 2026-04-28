# Code Review — branch `1.2.2-beta1`

**Date:** 2026-04-29
**Base:** `main`
**Scope:** focused review of source code changes (Rust / Python / tests). Docs, changelog, and the bundled `.whl` were skipped per request.
**Focus areas:** correctness & safety · performance · API & compatibility.

## Diff summary

```
 rcsfs/__init__.py                      |  18 +-
 rcsfs/_rcsfs.pyi                       |  39 -----
 scripts/compare_descriptor_outputs.py  | 213 +++++++++++++++++++++++
 src/csfs_conversion.rs                 | 207 ++++++++++++----------
 src/csfs_descriptor.rs                 | 307 ++++++++++++++++++++++++---------
 src/descriptor_normalization.rs        |  17 ++
 src/lib.rs                             |   3 +
 tests/csfs_descriptor_test.rs          |  15 ++
 tests/descriptor_normalization_test.rs |  22 +++
 tests/integration_test.rs              | 249 ++++++++++++++++++++++++++
 tests/rcsfs_test.py                    | 102 ++++++-----
 11 files changed, 921 insertions(+), 271 deletions(-)
```

The headline change is a producer/Rayon-workers/writer pipeline for descriptor generation in `csfs_descriptor.rs:500-946`, with order preserved via a `BTreeMap` in the writer thread. Output schema switched to a multi-column `col_0..col_N` Parquet layout (`Int32` or `Float32` depending on `normalize`). Validation was tightened in `descriptor_normalization.rs:249-279`. `_rcsfs.pyi` lost the `CSFProcessor` and `CSFDescriptorGenerator` stubs (which were never re-exported through `rcsfs/__init__.py`'s `__all__`).

Overall the branch is in good shape. Nothing here would block a release. Findings below are ordered by category and rough severity.

---

## Correctness / safety

### 1. `parse_csf` warning spam under bad `peel_subshells`

`src/csfs_descriptor.rs:1170` prints
```
Warning: <subshell> not found in orbs list
```
once per orbital block, per CSF, whenever a parsed subshell name is not in `peel_subshells`. The new test `test_normalize_path_tolerates_normalization_errors` (`tests/rcsfs_test.py:55-77`) intentionally passes `bad_subshells = ["xyz"]`, which means every orbital in every CSF triggers the eprintln — producing hundreds of stderr lines during a single test run.

The behaviour itself is correct (the test verifies graceful degradation), but in production this could amount to millions of lines for a malformed input. Recommend either:
- Rate-limiting the warning (e.g. first 5 + summary count, the same pattern already in the sequential converter at `csfs_conversion.rs:514-522`), or
- Demoting it to a counter that's reported once at the end of `generate_descriptors_from_parquet*`.

### 2. Asymmetric error handling: sequential vs. parallel descriptor with `normalize=true`

- Sequential path (`csfs_descriptor.rs:326-334`) treats `parse_csf` failure as direct zeros and **skips** normalization.
- Parallel path (`csfs_descriptor.rs:716-742`) constructs a zero `Vec<i32>` for the failed CSF and **still** calls `normalize_descriptor_per_csf` on it.

For a valid `peel_subshells` the outputs match (zeros either way). For an invalid `peel_subshells` (where normalization itself errors), the parallel path emits an extra warning per failed CSF that the sequential path does not. Not a correctness bug — the on-disk values are equivalent — but the streams of warnings will differ.

`test_descriptor_parallel_matches_sequential_outputs` (`tests/integration_test.rs`) only exercises the happy path, so this divergence is invisible to the suite.

Consider mirroring the sequential pattern in the parallel worker: on `parse_csf` error, append zeros directly without invoking the normalizer.

### 3. Worker error leaves writer + remaining workers running

`csfs_descriptor.rs:912-917`:

```rust
for (i, handle) in worker_handles.into_iter().enumerate() {
    handle.join()
        .map_err(|e| anyhow::anyhow!("Worker thread {} panicked: {:?}", i, e))?
        .with_context(|| format!("Worker thread {} failed", i))?;
}
```

The first failing worker short-circuits the join loop. `writer_handle` is then dropped without `.join()`, and the other workers are never joined either. Threads still exit cleanly via channel closure (workers drop `result_tx`, writer's `recv()` returns `Err`), but the function returns to Python while the writer is still finalizing the file.

Implications:
- If a caller immediately retries on error, the writer may still be flushing the previous output. The previous file's `ParquetFileGuard::drop` (line 48-54) will call `writer.close()` and `remove_file()` from the still-running thread; the new run may race against the cleanup of the old file.
- The error returned to Python may not reflect the full picture — a worker further down the loop could also have failed.

Recommendation: collect all worker results, then always join the writer before propagating any error.

### 4. Parallel conversion silently drops trailing data when `chunk_size < 3`

`src/csfs_conversion.rs:286-301`:

```rust
if batch_lines.is_empty() { break; }
let num_full_csfs = batch_lines.len() / 3;
if num_full_csfs == 0 {
    if lines_read == 0 { break; }
    eprintln!("警告: 文件末尾有 {} 行不完整的数据，将被忽略", batch_lines.len());
    break;
}
```

If a read iteration returns 1 or 2 lines and `num_full_csfs == 0`, the loop breaks regardless of whether more lines remain. Default `chunk_size = 3000000` is safe; the Python guard only enforces `chunk_size > 0`. The sequential path (`csfs_conversion.rs:539-553`) already buffers leftovers correctly with `continue`.

Recommendation: either reject `chunk_size < 3` in the Python wrapper, or mirror the sequential buffering logic (carry leftovers to the next iteration unless EOF was reached).

---

## Performance

### 5. Misleading "zero-copy" comment in the reader thread

`src/csfs_descriptor.rs:644-654`:

```rust
// Extract rows as Arc<str> for zero-copy sharing across threads
let rows: Vec<(u64, Arc<str>, Arc<str>, Arc<str>)> = (0..batch_size)
    .map(|i| (
        idx_col.value(i),
        line1_col.value(i).into(),  // allocates new Arc<str>
        line2_col.value(i).into(),
        line3_col.value(i).into(),
    ))
    .collect();
```

`StringArray::value(i)` returns `&str`; `.into()` produces a freshly-allocated `Arc<str>`. With the default batch size of 65 536 rows × 3 strings, that's ~200 000 allocations per batch — not zero-copy.

The actual zero-copy approach would be to keep the `RecordBatch` alive (already `Arc`-shared internally) and ship `(batch: Arc<RecordBatch>, row_idx: usize)` to the workers, letting them call `value(i)` on the original Arrow buffers. That's a bigger refactor; at minimum the comment should be corrected.

### 6. Row → column transpose is redundant

`transpose_i32_rows` / `transpose_f32_rows` (`csfs_descriptor.rs:453-481`) walk an already-allocated row-major `Vec<Vec<i32>>` into column-major `Vec<Vec<i32>>`. Each `parse_csf` returns a fresh `Vec<i32>` per row, and then transpose pushes every element a second time.

For 65 536 rows × 9 columns that's ~590 000 `Vec::push` calls per batch on top of the parsing. You can have workers push directly into per-column `Vec`s in a single pass through the rows, eliminating both the intermediate row vectors and the transpose.

Concretely:

```rust
let mut cols: Vec<Vec<i32>> = (0..descriptor_size)
    .map(|_| Vec::with_capacity(batch_size))
    .collect();
for (idx, l1, l2, l3) in work_item.rows {
    let row = generator.parse_csf(&l1, &l2, &l3).unwrap_or_else(/* zeros */);
    for (c, v) in cols.iter_mut().zip(&row) { c.push(*v); }
}
```

Loses the `into_par_iter` parallelism on the rows of a single batch, but workers are already parallelized at the batch level. Most batches will see lower wall time because of reduced allocator pressure.

### 7. `parse_csf` hot-path allocates two `String`s per CSF

`csfs_descriptor.rs:1077` and `1084`:

```rust
let middle_line = format!("{:<width$}", line2.trim_end(), width = line_length);
let coupling_line = format!("{:<width$}", coupling_trimmed, width = line_length);
```

These allocate a fresh `String` on every call. For 10⁸+ CSFs this is meaningful allocator pressure on a hot path. Options:
- Use a thread-local reusable `String` buffer that's cleared and `write!`d into.
- Skip the padding entirely: `chunk_string` could iterate with bounds-checked slicing, treating short lines as zero-padded.

### 8. Sequential vs parallel writer config drift

Sequential converter sets `set_write_batch_size(chunk_size)` (`csfs_conversion.rs:486`); parallel does not (`csfs_conversion.rs:245-247`). Probably immaterial because the parallel path already controls batch sizing through the read loop, but worth aligning to keep the two writers configured identically (or document why they differ).

---

## API / compatibility

### 9. `.pyi` cleanup is non-breaking ✅

`rcsfs/_rcsfs.pyi` lost the `CSFProcessor` and `CSFDescriptorGenerator` class stubs and the `num_workers` field on `ConversionStats`. None of these were re-exported through `rcsfs/__init__.py`'s `__all__`, and `lib.rs:107-117` never actually populated `num_workers` in the returned dict — the stubs were drifting from reality. Removing them is correct.

### 10. Python `__version__` fallback narrowed ✅

`rcsfs/__init__.py:55-58` dropped the outer `try/except ImportError` around `importlib.metadata`. Python ≥3.14 (the minimum per `pyproject.toml`) always has it, so the dead branch was correctly removed.

### 11. Public Python surface is otherwise unchanged

`__all__` still exports `convert_csfs`, `get_parquet_info`, `generate_descriptors_from_parquet`, `read_peel_subshells`, plus the two `TypedDict` types. The wrapper signatures match `main`. Existing callers will not need changes.

---

## Test coverage

### 12. Order preservation only tested with 2 workers

`test_descriptor_parallel_matches_sequential_outputs` (`tests/integration_test.rs:438-512`) uses `Some(2)` workers. Race conditions in the `BTreeMap` write-ordering path are unlikely to manifest at N=2 because there's effectively no contention. A second invocation at, say, `num_workers=8` over a larger fixture (≥1M CSFs, multiple worker batches each) would harden the guarantee.

### 13. No test pins `infer_two_j_target` ↔ `parse_csf` contract

`csfs_descriptor.rs:1088-1094` (parser extracts `final_double_j` from the last 5 chars of `line3`) and `descriptor_normalization.rs:331-338` (helper recovers `2J_target` by reverse-scanning the descriptor for the last occupied chunk) compute the target J independently. They agree today but nothing pins the relationship — a parser change could silently desync them.

A small unit test that runs `parse_csf` over a fixture row and asserts `infer_two_j_target(&descriptor) == manually_extracted_2J` would catch future drift.

### 14. New tests are otherwise solid

Added Rust tests cover:
- non-ASCII rejection in the descriptor generator (`tests/csfs_descriptor_test.rs:91-103`)
- invalid electron counts and negative `two_j_target` in the normalizer (`tests/descriptor_normalization_test.rs:319-339`)
- sequential vs parallel truncation count parity (`tests/integration_test.rs:277-307`)
- non-ASCII rejection cleans up incomplete parquet output (`tests/integration_test.rs:309-333`)
- zero-worker rejection in both pipelines (`tests/integration_test.rs:392-435`)
- sequential vs parallel descriptor parity, raw and normalized (`tests/integration_test.rs:437-516`)

The Python suite (`tests/rcsfs_test.py`) was rewritten away from a notebook-style script with hard-coded user paths into two proper `pytest` functions exercising the public wrapper API end-to-end. Big improvement.

---

## Misc / nothing-to-do

- `ParquetFileGuard` correctly cleans up incomplete output on drop in both modules. The duplicated implementation is fine.
- `descriptor_normalization.rs` validation additions (`two_j_target >= 0`, `0 <= n_elec[i] <= g[i]`) are sensible and well-tested.
- `scripts/compare_descriptor_outputs.py` is a debugging utility, not a test. Note that it declares `pl: Any = None` at module scope with no actual `import polars` shown in the diff — if the rest of the script uses `pl.DataFrame` without ever rebinding `pl`, it will `AttributeError` at runtime. Worth checking the file end-to-end before relying on it.
- The Chinese log strings (`开始并行转换 CSF 文件`, `[读取进度]`, etc.) are consistent with the existing style in this codebase.

---

## Recommended follow-ups, in priority order

1. Rate-limit or aggregate the per-row "subshell not in orbs list" warning (`csfs_descriptor.rs:1170`).
2. Always join the writer thread before propagating worker errors (`csfs_descriptor.rs:912-922`).
3. Either reject `chunk_size < 3` in the parallel converter or mirror the sequential leftover-buffering logic.
4. Fix the misleading "zero-copy" comment, or actually share `Arc<RecordBatch>` instead of allocating per-row `Arc<str>`.
5. Fuse the row → column transpose into a single pass during parsing.
6. Add an order-preservation stress test with `num_workers >= 8` and >1M CSFs.
7. Add a small unit test pinning the `infer_two_j_target` ↔ `parse_csf` contract.
