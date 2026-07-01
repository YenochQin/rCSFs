# Descriptor Fast Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce descriptor-generation runtime by removing per-CSF descriptor allocation, avoid batch-wide row-to-column transposition, and keep output identical for raw and normalized descriptor Parquet files.

**Architecture:** Add an allocation-light parser path to `CSFDescriptorGenerator` that writes into caller-owned descriptor buffers. Replace the current parallel batch compute path in `generate_descriptors_from_parquet_parallel()` with per-thread column chunks that are merged into Arrow columns without allocating one `Vec<i32>` per CSF row. Keep the existing `parse_csf()` API as a compatibility wrapper so existing tests and callers continue to work.

**Tech Stack:** Rust 2024, Rayon, Arrow/Parquet 58, existing PyO3 wrapper, pytest for Python API regression tests.

---

## Code Review Findings

Current descriptor generation is dominated by compute-stage memory work, not by writer backpressure. The latest large run reported:

```text
[计算完成] batches: 634 | rows: 41520968 | compute: 294.95s | wait_writer: 3.61ms
耗时: total 295.07s | read 291.14s | write+close 295.05s
```

`wait_writer` is effectively zero, so the writer channel is not the reason `--num-workers 46` only beats `--num-workers 8` slightly. The misleading `read` and `write+close` labels currently measure thread lifetimes, not isolated I/O time; they should not be used as pure read/write timings.

The hot path in `src/csfs_descriptor.rs` currently does expensive allocation and memory movement:

- `CSFDescriptorGenerator::parse_csf()` allocates `Vec<i32>` for every CSF.
- `parse_csf()` allocates padded `String`s via `format!("{:<width$}", ...)` for `line2` and `line3`.
- `parse_csf()` allocates a subshell `String` for each 9-character orbital block before HashMap lookup.
- The parallel raw path builds `Vec<Vec<i32>>` for a batch, then transposes rows into columns.
- The normalized path builds `Vec<Vec<f32>>` with the same row-first shape, then transposes.

For a 41M-row, 168-value descriptor workload, this means tens of millions of small vectors and many batch-sized copies. More threads mostly increase allocator and memory-bandwidth pressure.

## File Structure

- Modify `src/csfs_descriptor.rs`: add allocation-light parsing helpers, replace row-first batch construction in the parallel path, fix timing labels, and add focused Rust tests.
- Modify `tests/csfs_descriptor_test.rs`: add public behavior tests for the new parser wrapper if the helper is public or crate-visible.
- Keep `rcsfs/__init__.py`, `src/lib.rs`, and CLI files unchanged unless existing signatures need documentation updates. The Python API should remain unchanged.

## Task 1: Add Parser Buffer Tests

**Files:**
- Modify: `src/csfs_descriptor.rs`
- Modify: `tests/csfs_descriptor_test.rs`

- [ ] **Step 1: Write a unit test for parsing into a reused raw buffer**

Add this test to the existing `#[cfg(test)] mod tests` in `src/csfs_descriptor.rs`:

```rust
#[test]
fn parse_csf_into_reuses_caller_buffer_and_matches_parse_csf() {
    let subshells = vec![
        "5s".to_string(),
        "4d-".to_string(),
        "4d".to_string(),
        "5p-".to_string(),
        "5p".to_string(),
        "6s".to_string(),
    ];
    let generator = CSFDescriptorGenerator::new(subshells);
    let line1 = "  5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)";
    let line2 = "                   3/2               2        ";
    let line3 = "                                           4-  ";

    let expected = generator.parse_csf(line1, line2, line3).unwrap();
    let mut descriptor = vec![99i32; generator.orbital_count() * 3];

    generator
        .parse_csf_into(line1, line2, line3, &mut descriptor)
        .unwrap();

    assert_eq!(descriptor, expected);
}
```

- [ ] **Step 2: Write a unit test for output buffer size validation**

Add this test to the same module:

```rust
#[test]
fn parse_csf_into_rejects_wrong_buffer_size() {
    let generator = CSFDescriptorGenerator::new(vec!["5s".to_string()]);
    let mut too_short = vec![0i32; 2];

    let result = generator.parse_csf_into("  5s ( 2)", "", "      0  ", &mut too_short);

    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("descriptor buffer length"),
        "error should explain buffer length mismatch"
    );
}
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
uv run cargo test parse_csf_into
```

Expected: compile failure because `parse_csf_into()` does not exist.

## Task 2: Implement Allocation-Light Parser Wrapper

**Files:**
- Modify: `src/csfs_descriptor.rs`

- [ ] **Step 1: Add fixed-width helper functions**

Add helper functions near `chunk_string()`:

```rust
fn fixed_width_field(line: &str, start: usize, width: usize) -> &str {
    line.get(start..start.saturating_add(width)).unwrap_or("")
}

fn fixed_width_trimmed_field(line: &str, start: usize, width: usize) -> &str {
    fixed_width_field(line, start, width).trim()
}
```

These helpers avoid building padded strings. Missing chunks are treated as empty fields.

- [ ] **Step 2: Add `parse_csf_into()`**

Add this method to `impl CSFDescriptorGenerator`:

```rust
pub fn parse_csf_into(
    &self,
    line1: &str,
    line2: &str,
    line3: &str,
    descriptor: &mut [i32],
) -> Result<()> {
    if descriptor.len() != 3 * self.orbital_count {
        return Err(anyhow::anyhow!(
            "descriptor buffer length {} does not match expected {}",
            descriptor.len(),
            3 * self.orbital_count
        ));
    }
    descriptor.fill(0);

    if !line1.is_ascii() || !line2.is_ascii() || !line3.is_ascii() {
        return Err(anyhow::anyhow!("CSF lines must be ASCII fixed-width text"));
    }

    let subshells_line = line1.trim_end();
    let line_length = subshells_line.len();
    let coupling_line_raw = line3.trim_end();
    let coupling_start = 4usize;
    let coupling_end = coupling_line_raw.len().saturating_sub(5);
    let coupling_line = if coupling_start < coupling_end {
        &coupling_line_raw[coupling_start..coupling_end]
    } else {
        coupling_line_raw
    };

    let final_j_str = coupling_line_raw
        .get(
            coupling_line_raw.len().saturating_sub(5)
                ..coupling_line_raw.len().saturating_sub(1),
        )
        .unwrap_or("");
    let final_double_j = j_to_double_j(final_j_str)?;

    let block_count = line_length.div_ceil(9);
    for i in 0..block_count {
        let start = i * 9;
        let subshell_charges = fixed_width_field(subshells_line, start, 9);
        let subshell = fixed_width_trimmed_field(subshell_charges, 0, 5);
        if subshell.is_empty() {
            continue;
        }

        let subshell_electron_num: i32 = fixed_width_trimmed_field(subshell_charges, 6, 2)
            .parse()
            .unwrap_or(0);
        let is_last = i == block_count - 1;

        let middle_item = fixed_width_field(line2.trim_end(), start, 9);
        let coupling_item = fixed_width_field(coupling_line, start, 9);

        let mut temp_middle_item = 0i32;
        if !middle_item.trim().is_empty() {
            let middle_value = if let Some(semi_pos) = middle_item.find(';') {
                &middle_item[semi_pos + 1..]
            } else {
                middle_item
            };
            temp_middle_item = j_to_double_j(middle_value).unwrap_or(0);
        }

        let mut temp_coupling_item = 0i32;
        if !coupling_item.trim().is_empty() {
            temp_coupling_item = j_to_double_j(coupling_item).unwrap_or(0);
        } else if !middle_item.trim().is_empty() {
            temp_coupling_item = temp_middle_item;
        }
        if is_last {
            temp_coupling_item = final_double_j;
        }

        if let Some(&orbital_idx) = self.orbital_index_map.get(subshell) {
            let base_idx = orbital_idx * 3;
            descriptor[base_idx] = subshell_electron_num;
            descriptor[base_idx + 1] = temp_middle_item;
            descriptor[base_idx + 2] = temp_coupling_item;
        } else {
            let warning_index = self
                .missing_subshell_warning_count
                .fetch_add(1, Ordering::Relaxed);
            if warning_index < 10 {
                eprintln!(
                    "Warning: subshell '{}' not found in peel_subshells",
                    subshell
                );
            } else if warning_index == 10 {
                eprintln!("Warning: further missing-subshell warnings suppressed");
            }
        }
    }

    Ok(())
}
```

- [ ] **Step 3: Make `parse_csf()` delegate to `parse_csf_into()`**

Replace the body of `parse_csf()` with:

```rust
pub fn parse_csf(&self, line1: &str, line2: &str, line3: &str) -> Result<Vec<i32>> {
    let mut descriptor = vec![0i32; 3 * self.orbital_count];
    self.parse_csf_into(line1, line2, line3, &mut descriptor)?;
    Ok(descriptor)
}
```

- [ ] **Step 4: Run parser tests**

Run:

```bash
uv run cargo test parse_csf
uv run cargo test test_descriptor_generator_parse_csf_basic
```

Expected: all parser tests pass.

## Task 3: Replace Raw Batch Row Allocation With Column Chunks

**Files:**
- Modify: `src/csfs_descriptor.rs`

- [ ] **Step 1: Add raw batch helper test**

Add a test in `src/csfs_descriptor.rs` for a new helper named `build_raw_descriptor_columns_parallel()`:

```rust
#[test]
fn build_raw_descriptor_columns_parallel_matches_parse_csf_rows() {
    let generator = std::sync::Arc::new(CSFDescriptorGenerator::new(vec![
        "5s".to_string(),
        "4d-".to_string(),
        "4d".to_string(),
    ]));
    let rows = vec![
        (
            0u64,
            std::sync::Arc::<str>::from("  5s ( 2)  4d-( 4)  4d ( 6)"),
            std::sync::Arc::<str>::from("                   3/2      "),
            std::sync::Arc::<str>::from("                        4-  "),
        ),
        (
            1u64,
            std::sync::Arc::<str>::from("  5s ( 0)  4d-( 4)  4d ( 6)"),
            std::sync::Arc::<str>::from("                   5/2      "),
            std::sync::Arc::<str>::from("                        4-  "),
        ),
    ];
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

    let columns = build_raw_descriptor_columns_parallel(&pool, generator.clone(), rows.clone());

    let expected_rows: Vec<Vec<i32>> = rows
        .iter()
        .map(|(_, line1, line2, line3)| generator.parse_csf(line1, line2, line3).unwrap())
        .collect();
    let expected_columns = transpose_i32_rows(expected_rows, generator.orbital_count() * 3);
    assert_eq!(columns, expected_columns);
}
```

- [ ] **Step 2: Run helper test and verify it fails**

Run:

```bash
uv run cargo test build_raw_descriptor_columns_parallel_matches_parse_csf_rows
```

Expected: compile failure because helper does not exist.

- [ ] **Step 3: Implement `build_raw_descriptor_columns_parallel()`**

Add this helper inside `parquet_batch`:

```rust
fn build_raw_descriptor_columns_parallel(
    pool: &rayon::ThreadPool,
    generator: Arc<super::CSFDescriptorGenerator>,
    rows: Vec<(u64, Arc<str>, Arc<str>, Arc<str>)>,
) -> Vec<Vec<i32>> {
    use rayon::prelude::*;

    let descriptor_size = 3 * generator.orbital_count();
    let batch_size = rows.len();
    let chunk_size = (batch_size / pool.current_num_threads()).clamp(1024, 8192);

    let chunk_columns: Vec<(usize, Vec<Vec<i32>>)> = pool.install(|| {
        rows.par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut columns: Vec<Vec<i32>> = (0..descriptor_size)
                    .map(|_| Vec::with_capacity(chunk.len()))
                    .collect();
                let mut descriptor = vec![0i32; descriptor_size];

                for (idx, line1, line2, line3) in chunk {
                    if let Err(e) = generator.parse_csf_into(line1, line2, line3, &mut descriptor) {
                        eprintln!("Warning: Failed to parse CSF at index {}: {}", idx, e);
                        descriptor.fill(0);
                    }
                    for col_idx in 0..descriptor_size {
                        columns[col_idx].push(descriptor[col_idx]);
                    }
                }

                (chunk_idx, columns)
            })
            .collect()
    });

    let mut sorted_chunk_columns = chunk_columns;
    sorted_chunk_columns.sort_by_key(|(chunk_idx, _)| *chunk_idx);

    let mut columns: Vec<Vec<i32>> = (0..descriptor_size)
        .map(|_| Vec::with_capacity(batch_size))
        .collect();
    for (_, chunk) in sorted_chunk_columns {
        for col_idx in 0..descriptor_size {
            columns[col_idx].extend(chunk[col_idx].iter().copied());
        }
    }
    columns
}
```

- [ ] **Step 4: Replace the raw branch in `generate_descriptors_from_parquet_parallel()`**

In the raw branch inside the compute thread, replace:

```rust
let descriptor_rows: Vec<Vec<i32>> = work_item.rows.into_par_iter()...
DescriptorColumns::Raw(transpose_i32_rows(descriptor_rows, descriptor_size))
```

with:

```rust
DescriptorColumns::Raw(build_raw_descriptor_columns_parallel(
    &rayon_pool,
    generator_clone.clone(),
    work_item.rows,
))
```

- [ ] **Step 5: Run raw consistency tests**

Run:

```bash
uv run cargo test build_raw_descriptor_columns_parallel_matches_parse_csf_rows
uv run cargo test test_descriptor_parallel_matches_sequential_outputs
```

Expected: all tests pass.

## Task 4: Replace Normalized Batch Row Allocation With Column Chunks

**Files:**
- Modify: `src/csfs_descriptor.rs`

- [ ] **Step 1: Add normalized helper test**

Add a test in `src/csfs_descriptor.rs`:

```rust
#[test]
fn build_normalized_descriptor_columns_parallel_matches_row_path() {
    use crate::descriptor_normalization::{infer_two_j_target, normalize_descriptor_per_csf};

    let peel_subshells = std::sync::Arc::new(vec![
        "5s".to_string(),
        "4d-".to_string(),
        "4d".to_string(),
    ]);
    let generator = std::sync::Arc::new(CSFDescriptorGenerator::new((*peel_subshells).clone()));
    let rows = vec![
        (
            0u64,
            std::sync::Arc::<str>::from("  5s ( 2)  4d-( 4)  4d ( 6)"),
            std::sync::Arc::<str>::from("                   3/2      "),
            std::sync::Arc::<str>::from("                        4-  "),
        ),
        (
            1u64,
            std::sync::Arc::<str>::from("  5s ( 0)  4d-( 4)  4d ( 6)"),
            std::sync::Arc::<str>::from("                   5/2      "),
            std::sync::Arc::<str>::from("                        4-  "),
        ),
    ];
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

    let columns = build_normalized_descriptor_columns_parallel(
        &pool,
        generator.clone(),
        peel_subshells.clone(),
        rows.clone(),
    );

    let expected_rows: Vec<Vec<f32>> = rows
        .iter()
        .map(|(_, line1, line2, line3)| {
            let descriptor = generator.parse_csf(line1, line2, line3).unwrap();
            let two_j_target = infer_two_j_target(&descriptor);
            normalize_descriptor_per_csf(&descriptor, &peel_subshells, two_j_target).unwrap()
        })
        .collect();
    let expected_columns = transpose_f32_rows(expected_rows, generator.orbital_count() * 3);
    assert_eq!(columns, expected_columns);
}
```

- [ ] **Step 2: Run helper test and verify it fails**

Run:

```bash
uv run cargo test build_normalized_descriptor_columns_parallel_matches_row_path
```

Expected: compile failure because helper does not exist.

- [ ] **Step 3: Implement `build_normalized_descriptor_columns_parallel()`**

Add this helper inside `parquet_batch`:

```rust
fn build_normalized_descriptor_columns_parallel(
    pool: &rayon::ThreadPool,
    generator: Arc<super::CSFDescriptorGenerator>,
    peel_subshells: Arc<Vec<String>>,
    rows: Vec<(u64, Arc<str>, Arc<str>, Arc<str>)>,
) -> Vec<Vec<f32>> {
    use crate::descriptor_normalization::{infer_two_j_target, normalize_descriptor_per_csf};
    use rayon::prelude::*;

    let descriptor_size = 3 * generator.orbital_count();
    let batch_size = rows.len();
    let chunk_size = (batch_size / pool.current_num_threads()).clamp(1024, 8192);

    let chunk_columns: Vec<(usize, Vec<Vec<f32>>)> = pool.install(|| {
        rows.par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut columns: Vec<Vec<f32>> = (0..descriptor_size)
                    .map(|_| Vec::with_capacity(chunk.len()))
                    .collect();
                let mut descriptor = vec![0i32; descriptor_size];
                let mut normalized = vec![0.0f32; descriptor_size];

                for (idx, line1, line2, line3) in chunk {
                    match generator.parse_csf_into(line1, line2, line3, &mut descriptor) {
                        Ok(()) => {
                            let two_j_target = infer_two_j_target(&descriptor);
                            match normalize_descriptor_per_csf(
                                &descriptor,
                                &peel_subshells,
                                two_j_target,
                            ) {
                                Ok(values) => normalized.copy_from_slice(&values),
                                Err(e) => {
                                    eprintln!(
                                        "Warning: Failed to normalize CSF at index {}: {}",
                                        idx, e
                                    );
                                    normalized.fill(0.0);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to parse CSF at index {}: {}", idx, e);
                            normalized.fill(0.0);
                        }
                    }
                    for col_idx in 0..descriptor_size {
                        columns[col_idx].push(normalized[col_idx]);
                    }
                }

                (chunk_idx, columns)
            })
            .collect()
    });

    let mut sorted_chunk_columns = chunk_columns;
    sorted_chunk_columns.sort_by_key(|(chunk_idx, _)| *chunk_idx);

    let mut columns: Vec<Vec<f32>> = (0..descriptor_size)
        .map(|_| Vec::with_capacity(batch_size))
        .collect();
    for (_, chunk) in sorted_chunk_columns {
        for col_idx in 0..descriptor_size {
            columns[col_idx].extend(chunk[col_idx].iter().copied());
        }
    }
    columns
}
```

- [ ] **Step 4: Replace normalized branch in `generate_descriptors_from_parquet_parallel()`**

Replace the normalized row-first branch with:

```rust
DescriptorColumns::Normalized(build_normalized_descriptor_columns_parallel(
    &rayon_pool,
    generator_clone.clone(),
    peel_subshells_for_normalization.clone(),
    work_item.rows,
))
```

- [ ] **Step 5: Run normalized consistency tests**

Run:

```bash
uv run cargo test build_normalized_descriptor_columns_parallel_matches_row_path
uv run cargo test test_descriptor_parallel_matches_sequential_outputs
```

Expected: all tests pass.

## Task 5: Fix Timing Diagnostics

**Files:**
- Modify: `src/csfs_descriptor.rs`

- [ ] **Step 1: Replace misleading lifetime labels**

Change the final timing print from:

```rust
println!(
    "耗时: total {:.2?} | read {:.2?} | write+close {:.2?}",
    total_start.elapsed(),
    reader_elapsed,
    writer_elapsed
);
```

to:

```rust
println!(
    "耗时: total {:.2?} | reader_thread_lifetime {:.2?} | writer_thread_lifetime {:.2?}",
    total_start.elapsed(),
    reader_elapsed,
    writer_elapsed
);
```

- [ ] **Step 2: Add pure reader conversion timing**

Inside the reader thread, measure only batch-to-row copy time:

```rust
let mut row_copy_elapsed = Duration::ZERO;
...
let copy_start = Instant::now();
let rows: Vec<(u64, Arc<str>, Arc<str>, Arc<str>)> = ...
row_copy_elapsed += copy_start.elapsed();
```

Return it in the reader result and print it:

```rust
println!(
    "耗时: total {:.2?} | row_copy {:.2?} | compute {:.2?} | wait_writer {:.2?} | writer_thread_lifetime {:.2?}",
    total_start.elapsed(),
    row_copy_elapsed,
    compute_elapsed_total,
    send_elapsed_total,
    writer_elapsed
);
```

- [ ] **Step 3: Aggregate compute stats**

Replace per-worker-only printing with accumulated totals:

```rust
let mut total_compute_elapsed = Duration::ZERO;
let mut total_send_elapsed = Duration::ZERO;
let mut total_compute_rows = 0usize;
...
total_compute_elapsed += stats.compute_elapsed;
total_send_elapsed += stats.send_elapsed;
total_compute_rows += stats.rows_processed;
```

Print one final compute summary after all worker handles join.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run cargo test descriptor_pipeline_channel_capacity_limits_high_worker_memory_pressure
uv run cargo test test_descriptor_parallel_matches_sequential_outputs
```

Expected: all tests pass.

## Task 6: End-to-End Validation

**Files:**
- Existing Rust and Python package

- [ ] **Step 1: Run Rust descriptor tests**

Run:

```bash
uv run cargo test test_descriptor_generator_parse_csf_basic
uv run cargo test test_descriptor_parallel_matches_sequential_outputs
```

Expected: all selected Rust tests pass.

- [ ] **Step 2: Run Python API and CLI tests**

Run:

```bash
uv run pytest tests/cli_test.py tests/rcsfs_test.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run formatting**

Run:

```bash
cargo fmt --check
uv run --group lint ruff check rcsfs/cli.py tests/cli_test.py
```

Expected: no formatting or lint errors.

- [ ] **Step 4: Build local extension**

Run:

```bash
uv run maturin develop
```

Expected: local wheel builds and installs successfully.

- [ ] **Step 5: Benchmark the known large dataset**

From the environment that owns the dataset, run:

```bash
rcsfs gen-descriptors 3d8_4s2_j2as6raw.parquet 3d8_4s2_desc_fast_8.parquet \
  --header 3d8_4s2_j2as6raw_header.toml \
  --num-workers 8

rcsfs gen-descriptors 3d8_4s2_j2as6raw.parquet 3d8_4s2_desc_fast_46.parquet \
  --header 3d8_4s2_j2as6raw_header.toml \
  --num-workers 46
```

Expected: the new diagnostic output shows lower compute time than the current baseline of about `294.95s` for 41.5M rows. If `--num-workers 46` still does not improve materially over `--num-workers 8`, use the new row-copy, compute, and writer diagnostics to decide the next bottleneck.

## Self-Review

- Spec coverage: The plan addresses the observed compute bottleneck, preserves public Python API behavior, keeps CLI unchanged, and adds diagnostics to avoid misleading read/write interpretation.
- Placeholder scan: No `TBD`, `TODO`, or vague "add tests" instructions remain. Each task includes concrete test commands and implementation snippets.
- Type consistency: New APIs are `parse_csf_into()`, `build_raw_descriptor_columns_parallel()`, and `build_normalized_descriptor_columns_parallel()`. Later tasks use those exact names.
