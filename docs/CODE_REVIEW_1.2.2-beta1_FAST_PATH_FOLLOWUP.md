# Code Review Follow-up: Descriptor Fast Path

**Date:** 2026-07-02
**Scope:** Follow-up review of `src/csfs_descriptor.rs` after commit `056f249` (`descriptor: add chunked fast path`).
**Verdict:** The review findings are mostly accurate. The batch-level parallelism concern is structurally correct, but its throughput impact requires benchmark evidence.

## Findings

### 1. Batch-level pipeline parallelism was reduced

Current code creates one Rayon pool with `num_workers` threads, then starts a single compute OS thread that receives `WorkItem`s from `work_rx` and processes one batch at a time. Inside that batch, rows are split with `par_chunks()` on the Rayon pool.

Evidence:

- `src/csfs_descriptor.rs:943` builds the bounded Rayon pool.
- `src/csfs_descriptor.rs:955` pushes only one compute thread.
- `src/csfs_descriptor.rs:961` receives batches in a single `while let Ok(work_item) = work_rx_clone.recv()` loop.
- `src/csfs_descriptor.rs:510` and `src/csfs_descriptor.rs:560` use Rayon `par_chunks()` inside the active batch.

Older code in `9c4e9a6` spawned `for worker_id in 0..num_workers`, cloned `work_rx`, and let every worker compete on `work_rx.recv()`. That allowed multiple batches to be in compute at once, although each worker also used Rayon internally.

Impact:

- The structural change is real: a slow batch now delays dispatch of later batches to compute.
- Throughput regression is not proven by code inspection alone. It depends on batch size, CSF complexity skew, writer backpressure, memory bandwidth, and allocator pressure.

Recommendation:

- Benchmark before and after on realistic data before reverting the architecture.
- Include at least one large multi-batch dataset and record total time, compute time, row-copy time, writer wait time, CPU utilization, and output equivalence.

### 2. `descriptor_chunk_size` uses a vague `thread_count` name

`descriptor_chunk_size(batch_size, thread_count)` receives `pool.current_num_threads()`. The value is the Rayon pool thread count, not the old OS consumer thread count.

Impact:

- Low severity. The value is still a thread count, but the name is ambiguous after the architecture change.

Recommendation:

- Rename the parameter to `rayon_thread_count` or `pool_thread_count`.

### 3. Fixed-width padding equivalence is implicit

The old parser explicitly right-padded `middle_line` and `coupling_line` to `line1.trim_end().len()` before chunking. The current parser skips allocation and relies on `fixed_width_field()` returning `""` for out-of-range access and a shorter slice for partially available data.

Evidence:

- `src/csfs_descriptor.rs:1285` returns `""` when `start >= line.len()`.
- `src/csfs_descriptor.rs:1293` trims the field returned by `fixed_width_field()`.
- `src/csfs_descriptor.rs:1430` and `src/csfs_descriptor.rs:1431` read middle and coupling fields directly from the original short strings.

Impact:

- Current behavior is equivalent for the existing empty-field and J parsing paths.
- The dependency chain is fragile because the padding rule is no longer explicit at the call site.

Proof test:

- `parse_csf_into_treats_short_coupling_lines_as_right_padded` documents that short `line2` and `line3` currently produce the same descriptor as explicitly padded inputs.

Recommendation:

- Keep the regression test.
- If this path is refactored again, preserve the "short fixed-width line equals right-padded line" contract explicitly.

### 4. Truncated electron number parsing is more permissive

The old parser only attempted electron-number parsing when the 9-byte block had at least 8 bytes. The current parser calls `fixed_width_trimmed_field(subshell_charges, 6, 2)`, which can return a one-character slice from a truncated block. That one character can parse as a nonzero electron count.

Evidence:

- `src/csfs_descriptor.rs:1423` parses the trimmed field from start `6`, width `2`.
- For a truncated block like `"  5s (2"`, the current field is `"2"`, so the descriptor electron count becomes `2`.
- The older `subshell_charges.len() >= 8` guard would have returned `0` for the same block.

Impact:

- This is a real behavior change for malformed or truncated CSF lines.
- It is not necessarily wrong for valid GRASP fixed-width input, but it is more permissive than the previous implementation.

Proof test:

- `parse_csf_into_accepts_partial_truncated_electron_field` documents that the current parser accepts a one-character truncated electron field.

Recommendation:

- Decide whether compatibility or stricter malformed-input handling is desired.
- If strict compatibility is required, require the full electron field width before parsing.
