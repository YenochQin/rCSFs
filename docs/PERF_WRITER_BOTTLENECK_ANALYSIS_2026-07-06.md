# Descriptor Pipeline Throughput Plateau Analysis

**Date:** 2026-07-06
**Scope:** Throughput investigation of `generate_descriptors_from_parquet_parallel` on a 48-physical-core dual-socket host.
**Dataset:** `3d8_4s2_j0as6raw.parquet` — 14,585,607 CSFs, 56 orbitals, descriptor size 168, 222 full batches × 65,536 rows + 1 partial batch × 36,615 rows.
**Host:** Dual-socket, 2 × 24-core CPUs (48 physical cores total, no hyperthreading).

## TL;DR

Throughput plateaus around 17–18 s once `--num-workers` reaches ~24 on this host. **ZSTD compression is not the dominant factor**: removing compression (`--compression none`) did not improve total time and did not reduce writer backpressure at 48 workers. Current evidence does **not** prove the writer thread is the hard throughput floor, because `writer_thread_lifetime` measures thread lifetime, including time blocked on `result_rx.recv()`, not active write time. The likely bottleneck is a combination of compute-stage memory traffic, single-batch compute dispatch/merge, reader-side `row_copy` allocations, and single-threaded Parquet encoding/write. More instrumentation is needed before choosing the highest-leverage fix.

## Background

The descriptor pipeline in `src/csfs_descriptor.rs` is a three-stage streaming design:

1. **Reader thread** — reads Parquet in 65,536-row batches, copies each row into a `DescriptorRow = (u64, Arc<str>, Arc<str>, Arc<str>)` tuple, sends via bounded channel.
2. **Compute dispatch thread (single)** — receives one batch at a time and runs Rayon `par_chunks()` across a fixed worker pool. After the parallel compute returns, it sequentially merges chunk columns into final columns, then sends the result to the writer.
3. **Writer thread (single)** — receives result columns, maintains order via `BTreeMap<batch_idx>`, builds Arrow `RecordBatch` from `Vec<Vec<f32>>` / `Vec<Vec<i32>>`, writes to `ArrowWriter` with ZSTD level 3.

Observation on a 48-core host: one CPU at ~100%, the rest at ~30%. Total time plateaus around 17–18 s beyond roughly 24 workers. At 16/24 workers, `wait_writer` is near zero, so the writer is not visibly applying backpressure. At 48 workers, compute gets faster but starts waiting on the result channel, which means writer-side work becomes a contributor at high worker counts.

## Initial hypothesis (incorrect)

The single-threaded writer runs ZSTD level 3 compression per column chunk. Compression is CPU-bound, so we hypothesized:

> ZSTD compression is the writer bottleneck. Removing it should eliminate `wait_writer` at high worker counts.

To test this, a `--compression` CLI option and `compression` Python parameter were added (commit `e4131af`, allowing `none`/`snappy`/`zstd`/`zstd-N`). The experiment then re-ran the same dataset with `compression="none"`.

## Experiment

Command shape:

```bash
uv run rcsfs gen-descriptors \
    --header 3d8_4s2_j0as6raw_header.toml \
    --num-workers {16,24,48} \
    --normalize \
    --compression {zstd-3 (default), none} \
    3d8_4s2_j0as6raw.parquet 3d8_4s2_desc.parquet
```

## Results

| workers | compression | total    | compute  | row_copy | wait_writer | reader_lifetime | writer_lifetime |
|---------|-------------|----------|----------|----------|-------------|-----------------|-----------------|
| 16      | zstd-3      | 21.89 s  | 21.77 s  | 4.19 s   | 1.89 ms     | 20.95 s         | 21.88 s         |
| 16      | none        | 20.67 s  | 20.55 s  | 4.90 s   | 1.77 ms     | 19.78 s         | 20.66 s         |
| 24      | zstd-3      | 17.93 s  | 17.71 s  | 5.27 s   | 0.37 ms     | 17.06 s         | 17.92 s         |
| 24      | none        | 17.75 s  | 17.27 s  | 4.57 s   | 0.29 ms     | 16.62 s         | 17.74 s         |
| 48      | zstd-3      | 17.79 s  | 14.49 s  | 3.73 s   | **2.57 s**  | 16.39 s         | 17.78 s         |
| 48      | none        | 18.28 s  | 14.47 s  | 5.13 s   | **3.05 s**  | 16.84 s         | 18.27 s         |

### What the data says

1. **`wait_writer` did not improve with `compression="none"`.** At 48 workers it went 2.57 s → 3.05 s (slightly worse). If ZSTD were the bottleneck, removing it should have driven `wait_writer` toward zero.
2. **`total` did not improve either.** At 48 workers, `none` (18.28 s) is marginally slower than `zstd-3` (17.79 s).
3. **`compute` is unchanged** across compression modes (14.49 s → 14.47 s) — expected, since compression lives in the writer.
4. **`compute` scales with worker count** (20.55 s → 17.27 s → 14.47 s for 16 → 24 → 48 workers), so the parallel compute side is healthy.
5. **`writer_thread_lifetime` equals `total`** in every run, but this does **not** prove the writer is busy for the entire duration. The timer starts when the writer thread is spawned and includes time blocked on `result_rx.recv()`.
6. **`row_copy` is consistently 4–5 s** regardless of workers or compression. This is reader-side allocation/copy work, but because the pipeline overlaps stages, it cannot be subtracted directly from `total` without measuring reader/compute wait time.

### Conclusion

The compression-specific hypothesis is rejected: ZSTD level 3 is not the main reason throughput plateaus. The data is still insufficient to name one primary bottleneck. The next step should be finer-grained timing of active writer work, compute-stage merge work, and reader/compute waiting.

## Revised root cause analysis

### Candidate 1: compute-stage memory traffic and batch merge

The reported `compute` time includes more than parsing and normalization. For each batch, `build_normalized_descriptor_columns_parallel()`:

1. Splits rows into Rayon chunks.
2. Builds per-chunk `Vec<Vec<f32>>` column buffers.
3. Sequentially merges chunk columns into one batch-level `Vec<Vec<f32>>`.

For this dataset, each full batch contains:

```text
65,536 rows × 168 Float32 values ≈ 11.0 M values
```

Across the full run:

```text
14,585,607 rows × 168 Float32 values ≈ 2.45 B values
```

That is a large amount of memory write traffic before the writer sees the batch. This explains why scaling from 24 to 48 workers is weak: extra cores can reduce parsing/normalization time, but memory bandwidth, allocator pressure, and single-batch merge work do not scale linearly.

### Candidate 2: single-threaded Arrow + Parquet encoding/write

The writer thread does this per batch (`src/csfs_descriptor.rs` Phase 5, around line 916+):

1. Convert `Vec<Vec<f32>>` (168 independent `Vec<f32>`, each ~65,536 elements) into 168 `Float32Array` values.
2. Assemble a `RecordBatch`.
3. Hand to `ArrowWriter`, which per column chunk performs:
   - Parquet page encoding
   - optional compression (ZSTD, SNAPPY, etc.)
   - page flushing

With `compression="none"`, steps 1–2 and the encoding/write work in step 3 still run single-threaded. Uncompressed pages can also increase output size and I/O pressure, which explains why `none` can be slightly slower than `zstd-3` even when compression CPU work is removed.

At 48 workers, `wait_writer` rises to 2.57–3.05 s, so writer-side work is contributing to the plateau once compute is fast enough. However, current diagnostics do not separate:

- time waiting in `result_rx.recv()`
- Arrow array construction time
- `RecordBatch::try_new()` time
- `writer.write()` time
- `writer.close()` / footer flush time

Without those numbers, the writer should be treated as a candidate bottleneck, not a proven hard floor.

### Candidate 3: reader-side `row_copy` allocation pressure

`src/csfs_descriptor.rs` around line 798:

```rust
let rows: Vec<DescriptorRow> = (0..batch_size)
    .map(|i| {
        (
            idx_col.value(i),
            line1_col.value(i).into(),   // &str -> Arc<str>: heap allocation
            line2_col.value(i).into(),   // heap allocation
            line3_col.value(i).into(),   // heap allocation
        )
    })
    .collect();
```

For 14.58 M rows × 3 string columns = **~43.8 M `Arc<str>` heap allocations**, single-threaded. This shows up as `row_copy ≈ 4–5 s`. The strings are consumed once by compute and dropped; the copy exists only to give the compute thread owned data.

This is real overhead, but the expected effect on total runtime is unknown because reader, compute, and writer stages overlap. Removing `row_copy` may save close to 4–5 s of reader work, but total runtime will improve by that amount only if compute or writer is waiting on the reader.

### Tertiary (structural, from prior review)

The single compute-dispatch thread processes one batch at a time (see `docs/CODE_REVIEW_1.2.2-beta1_FAST_PATH_FOLLOWUP.md` finding 1). The old architecture let multiple OS threads compete on `work_rx.recv()`, allowing multiple batches to be in compute concurrently. The current model limits in-flight compute to one batch.

This may cap scaling at high worker counts, especially if one batch has enough memory traffic that more threads mainly contend for bandwidth. It should be investigated after instrumentation identifies whether compute is waiting for reader input or writer output.

## Proposed mitigations

### Option 0 (recommended first): add active-stage instrumentation

**Expected value:** identify the real limiting stage before changing architecture.

Add timings for:

- reader: Parquet read/decode time, row-copy time, `work_tx.send()` wait time
- compute dispatch: `work_rx.recv()` wait time, Rayon parse/normalize time, chunk-column merge time, `result_tx.send()` wait time
- writer: `result_rx.recv()` wait time, Arrow array construction time, `RecordBatch::try_new()` time, `writer.write()` time, `writer.finish()` time

This will distinguish "thread lifetime" from "active work". It also makes future benchmark comparisons safer.

### Option 1: reduce reader `row_copy` by passing batch references

**Expected saving:** up to ~4–5 s of reader work; total-runtime saving depends on whether downstream stages wait on the reader.

Change `DescriptorRow` from an owned `(u64, Arc<str>, Arc<str>, Arc<str>)` tuple to a borrow-based view that holds an `Arc<RecordBatch>` (or the underlying `Arc<Ref<...>>`) plus row index. The reader thread sends `Arc<RecordBatch>` directly without per-row string copies. Compute workers index into the batch columns.

Trade-off: more complex lifetime bookkeeping and possible cache-locality changes. It removes 43.8 M allocations from the reader path.

### Option 2: reduce compute-stage memory traffic

**Expected saving:** unknown until merge timing is measured; likely important because compute dominates total time at 16/24 workers.

Possible approaches:

- Split compute timing into parse/normalize vs chunk merge.
- Add `normalize_descriptor_into()` so normalization can reuse caller-owned buffers instead of allocating a new vector per CSF.
- Avoid the sequential chunk-column merge by preallocating final batch columns and letting Rayon chunks write into disjoint column ranges, if Rust aliasing and safety can be handled cleanly.

### Option 3: investigate writer active work

**Expected saving:** unknown; only justified if active writer timing is high.

Sub-options:

- **3A (low risk)** — time Arrow array construction separately. `Float32Array::from(Vec<f32>)` may mostly wrap the vector as an Arrow buffer rather than copying every element, so parallelizing this may have little benefit.
- **3B (medium)** — tune Parquet writer settings after measuring `writer.write()` and `finish()` time: row group/page sizes, encoding options, and compression level.
- **3C (large)** — write temporary shard files in parallel and merge/consume them later. Multiple writer threads cannot safely write independent row groups into one `ArrowWriter<File>` without a deeper Parquet writer redesign.

### Option 4 (deferred): restore batch-level pipeline parallelism

Revert the single compute-dispatch thread to the older multi-worker model where N OS threads compete on `work_rx.recv()`. This was flagged in the prior follow-up review. It is worth doing only after instrumentation shows that one-batch-at-a-time dispatch is limiting throughput and will not simply increase result-channel backpressure.

## Open questions

1. **Is disk I/O a factor?** The host's storage subsystem was not characterized. If writes go to a network filesystem, uncompressed output could be substantially slower. Recommend checking `iostat` during a run.
2. **NUMA effects.** Dual-socket means Rayon threads span both sockets. Memory allocated on socket 0 and touched by a worker on socket 1 pays UPI latency. `numactl --cpunodebind --membind` experiments would isolate this.
3. **Allocator choice.** With 43.8 M small allocations in `row_copy`, switching from the system allocator to `jemalloc` or `mimalloc` may move `row_copy` materially. Worth a one-line `#[global_allocator]` A/B test before restructuring.
4. **Physical-core vs logical-core default.** This host has no hyperthreading, but on other hosts `num_cpus::get()` may return logical CPUs. If the pipeline is memory-bandwidth-bound, defaulting to physical cores may be more stable than using all logical CPUs.

## Artifacts

- Compression feature commit: `e4131af` (`descriptor: add configurable parquet compression`)
- Workspace gitlink update: `c634c77` (`workspace: update rCSFs submodule`)
- Relevant source: `src/csfs_descriptor.rs`
  - `parse_compression`: lines ~58–108
  - reader `row_copy` block: lines ~796–808
  - writer thread: Phase 5, lines ~916+
- Prior review context: `docs/CODE_REVIEW_1.2.2-beta1_FAST_PATH_FOLLOWUP.md`
