# Descriptor Pipeline Throughput Plateau Analysis

**Date:** 2026-07-06
**Scope:** Throughput investigation of `generate_descriptors_from_parquet_parallel` on a 48-physical-core dual-socket host.
**Dataset:** `3d8_4s2_j0as6raw.parquet` — 14,585,607 CSFs, 56 orbitals, descriptor size 168, 222 full batches × 65,536 rows + 1 partial batch × 36,615 rows.
**Host:** Dual-socket, 2 × 24-core CPUs (48 physical cores total, no hyperthreading).

## TL;DR

Throughput plateaus around 17–18 s once `--num-workers` reaches ~24 on this host. Active-stage instrumentation identified the root cause: **Parquet dictionary encoding** in `ArrowWriter::write()` consumed ~11.5 s per run (65 % of writer active work), independent of worker count. ZSTD compression was only ~0.4 s of the 17.7 s write time. Disabling dictionary encoding (`set_dictionary_enabled(false)`, commit `034f760`) dropped `writer.write` from 17.71 s to 6.20 s and total from 18.03 s to 14.12 s. The bottleneck has now shifted to the compute stage (7.74 s active work: parallel 6.10 s + sequential merge 1.64 s).

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

### Conclusion (pre-instrumentation)

The compression-specific hypothesis is rejected: ZSTD level 3 is not the main reason throughput plateaus. At this point the data was insufficient to name one primary bottleneck. The next step was finer-grained timing of active writer work, compute-stage merge work, and reader/compute waiting. This was subsequently carried out — see [Instrumentation results](#instrumentation-results) below for the resolved findings.

## Instrumentation results

Option 0 was implemented (commits `b79aff1`, `f71b03b`). All three pipeline stages now report active-work and channel-wait timings separately. The experiments below were run on the same dataset and host described above.

### Full timing breakdown (dictionary encoding ON, zstd-3)

| Metric | 16 workers | 24 workers | 48 workers |
|--------|-----------|-----------|-----------|
| **total** | 20.94 s | 18.06 s | 18.03 s |
| **reader** | | | |
| lifetime | 20.04 s | 16.87 s | 16.60 s |
| read_decode | 1.97 s | 2.03 s | 1.59 s |
| row_copy | 5.36 s | 5.29 s | 4.02 s |
| send_wait | 12.70 s | 9.54 s | 10.97 s |
| **compute** | | | |
| parallel | 14.94 s | 10.25 s | 6.16 s |
| merge | 1.32 s | 2.08 s | 2.69 s |
| wait_reader | 31 ms | 32 ms | 30 ms |
| wait_writer | 1.89 ms | 239 µs | 2.72 s |
| **writer** | | | |
| lifetime | 20.93 s | 18.05 s | 18.02 s |
| recv_wait | 3.25 s | 282 ms | 156 ms |
| array_build | 1.77 ms | 1.25 ms | 1.31 ms |
| batch_build | 2.05 ms | 1.82 ms | 1.82 ms |
| write | 17.55 s | 17.61 s | 17.71 s |
| finish | 17.88 ms | 18.42 ms | 18.52 ms |

### Key observations from instrumentation

1. **`writer.write` is ~17.6 s regardless of worker count** (17.55 → 17.61 → 17.71 for 16 → 24 → 48 workers). This is the throughput floor — it does not scale with compute parallelism because it is single-threaded.
2. **`array_build` is ~1.3 ms total** across all 223 batches. `Float32Array::from(Vec<f32>)` is effectively zero-copy (wraps the vector as an Arrow buffer). Option 3A (parallelizing array construction) is closed — no benefit.
3. **`batch_build` and `finish` are negligible** (~1.8 ms and ~18 ms respectively).
4. **`compute.merge` grows with worker count** (1.32 → 2.08 → 2.69 s) because more workers produce more chunks that must be sequentially merged. At 48 workers it is 30 % of `compute.parallel`.
5. **`reader.send_wait` is consistently 9–13 s** — the reader finishes its actual work (read_decode + row_copy ≈ 5–7 s) quickly, then blocks because the bounded channel to compute is full.

### Isolation test: compression only (test_d)

Ran with `--compression none` at 48 workers (dictionary encoding still ON) to isolate ZSTD cost inside `write`:

| Metric | zstd-3 (test_c) | none (test_d) | Delta |
|--------|----------------|---------------|-------|
| writer.write | 17.71 s | 17.30 s | **-0.41 s** |

Removing ZSTD compression improved `writer.write` by only ~0.4 s in this run (17.71 s → 17.30 s). Note that this is a net difference, not a strict attribution: disabling compression also changes page sizes and I/O patterns, so the true compression CPU cost may differ slightly. Either way, compression is clearly not the dominant cost — the remaining ~17.3 s is Parquet encoding overhead (dictionary building, RLE, page formatting).

### Breakthrough: disable dictionary encoding (test_e)

Ran at 48 workers with `set_dictionary_enabled(false)` in both `WriterProperties` builders (commit `034f760`):

| Metric | dictionary ON (test_c) | dictionary OFF (test_e) | Delta |
|--------|----------------------|------------------------|-------|
| **total** | 18.03 s | **14.12 s** | **-3.91 s** |
| **writer.write** | 17.71 s | **6.20 s** | **-11.51 s** |
| writer.recv_wait | 156 ms | 7.79 s | +7.64 s |
| compute.wait_writer | 2.72 s | 1.30 ms | -2.72 s |

**Dictionary encoding was consuming 11.5 s of writer time.** For normalized Float32 descriptor columns (168 columns of near-unique float values), the dictionary encoder builds a hash table per column chunk, finds most values are unique, and falls back to PLAIN encoding — but the dictionary-building cost is already paid.

### Bottleneck shift

With dictionary encoding disabled, the bottleneck has moved from writer to compute:

| Stage | Active work (dictionary OFF) | Was (dictionary ON) |
|-------|------------------------------|---------------------|
| reader | 5.42 s (read_decode 1.67 + row_copy 3.75) | 5.61 s |
| **compute** | **7.74 s** (parallel 6.10 + merge 1.64) | 8.85 s |
| writer | 6.22 s (write 6.20 + overhead 0.02) | 17.73 s |

Writer `recv_wait` jumped to 7.79 s, confirming the writer now spends 55 % of its lifetime idle, waiting for compute to produce results. The pipeline is now **compute-bound**.

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

At 48 workers, `wait_writer` rises to 2.57–3.05 s, so writer-side work is contributing to the plateau once compute is fast enough. However, at the time of writing the diagnostics did not separate:

- time waiting in `result_rx.recv()`
- Arrow array construction time
- `RecordBatch::try_new()` time
- `writer.write()` time
- `writer.close()` / footer flush time

**Update:** the active-stage instrumentation subsequently resolved this. The dominant cost inside the writer is Parquet dictionary encoding inside `writer.write()` (~11.5 s), not Arrow array construction or RecordBatch creation. See [Instrumentation results](#instrumentation-results).

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

### Option 0: add active-stage instrumentation — DONE

Implemented in commits `b79aff1` and `f71b03b`. All three stages now report active-work and channel-wait timings. Results are recorded in the [Instrumentation results](#instrumentation-results) section above.

### Option 3B: disable dictionary encoding — DONE (hardcoded)

Implemented in commit `034f760` as `set_dictionary_enabled(false)` in both `WriterProperties` builders. Confirmed as the primary writer bottleneck: `writer.write` dropped from 17.71 s to 6.20 s (-65 %), total dropped from 18.03 s to 14.12 s (-22 %).

**Follow-up:** parameterize as `--encoding {dictionary,plain}` CLI option so users can choose. Default should be `plain` for descriptor output (numeric columns with near-unique values). For raw Int32 descriptors with significant value repetition, dictionary encoding may still be worth offering.

### Option 3A: parallelize Arrow array construction — CLOSED

Instrumentation proved `array_build` is ~1.3 ms total (zero-copy). No action needed.

### Option 2: reduce compute-stage memory traffic — NEXT

With the writer bottleneck resolved, compute is now the limiting stage at 7.74 s active work. The `merge` sub-step (1.64 s, 21 % of compute) is sequential and grows with worker count. Approaches:

- Parallelize the chunk-column merge using Rayon.
- Eliminate the merge entirely by pre-allocating batch-level column buffers and having each Rayon chunk write into its own disjoint row range.
- Add `normalize_descriptor_into()` to reuse caller-owned buffers.

### Option 1: reduce reader `row_copy` — DEFERRED

`row_copy` is 3.75 s of reader active work, but the reader currently finishes early and blocks on `send_wait` (8.09 s). Eliminating `row_copy` will only reduce `total` once compute drops below ~5.4 s (reader's current active-work time). Worth doing after Option 2 if compute optimization makes the reader the new limiting stage.

### Option 3C: parallel writer threads — CLOSED (for now)

Writer active work is now 6.22 s, below compute's 7.74 s. Parallelizing the writer is not justified until compute is also optimized.

### Option 4: restore batch-level pipeline parallelism — DEFERRED

Still worth investigating if one-batch-at-a-time dispatch limits compute throughput after the merge optimization.

## Open questions

1. **Is disk I/O a factor?** The host's storage subsystem was not characterized. If writes go to a network filesystem, uncompressed output could be substantially slower. Recommend checking `iostat` during a run.
2. **NUMA effects.** Dual-socket means Rayon threads span both sockets. Memory allocated on socket 0 and touched by a worker on socket 1 pays UPI latency. `numactl --cpunodebind --membind` experiments would isolate this.
3. **Allocator choice.** With 43.8 M small allocations in `row_copy`, switching from the system allocator to `jemalloc` or `mimalloc` may move `row_copy` materially. Worth a one-line `#[global_allocator]` A/B test before restructuring.
4. **Physical-core vs logical-core default.** This host has no hyperthreading, but on other hosts `num_cpus::get()` may return logical CPUs. If the pipeline is memory-bandwidth-bound, defaulting to physical cores may be more stable than using all logical CPUs.

## Artifacts

- Compression feature: `e4131af` (`descriptor: add configurable parquet compression`)
- Active-stage instrumentation: `b79aff1` (`descriptor: add active-stage pipeline instrumentation`)
- read_decode / batch_build timing split: `f71b03b` (`descriptor: add read_decode and batch_build timing split`)
- Dictionary encoding disabled: `034f760` (`descriptor: disable dictionary encoding in parquet writer`)
- Relevant source: `src/csfs_descriptor.rs`
  - `parse_compression`: lines ~58–108
  - reader `row_copy` block: lines ~850–865
  - `build_*_descriptor_columns_parallel`: lines ~552–690 (return `StageTimings`)
  - writer thread: Phase 5, lines ~1020+
- Prior review context: `docs/CODE_REVIEW_1.2.2-beta1_FAST_PATH_FOLLOWUP.md`
