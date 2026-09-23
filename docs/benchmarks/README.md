# Benchmark registry

Measurements recorded here back the performance work in
[CSF_V2_GENERATION_PERFORMANCE_PLAN.md](../CSF_V2_GENERATION_PERFORMANCE_PLAN.md).
They are not part of the test suite: no file in this directory is run by
`cargo test` or `pytest`.

The GRASP occupation-branch compatibility fix of 2026-09-23 changes B1 and B3
counts without changing their transcript bytes. Reports recorded before that fix
remain historical measurements of their named source revisions, **not** current
generator baselines. The current expected counts are registered in
`tests/fixtures/transcripts.toml`: B1 19,454 configurations / 2,693,941 records,
B2 19,243 / 560,351, and B3 7,027,846 / 5,811,925,522. New performance
comparisons require new clean-source measurements; no old timing is rebased.

## Conventions

- One `.md` interpretation per run or campaign, with the raw aggregated
  measurements beside it as `.json`. Never record only the derived table.
- A report states the input, its registered hash, the hardware, the thread
  count, the managed-memory budget, the cache state, the timed region and what
  was **not** measured. A number without those is not comparable with anything.
- No machine-specific absolute paths and no generated CSF payloads. Inputs are
  referenced by their registered file name and SHA-256.
- Percentiles are taken from real samples, never interpolated. Performance
  claims name the code revision they were measured at.
- A report names the source it was built from, not only a commit: `git.tree`,
  `environment.extension.module_sha256` for the binary the process actually
  loaded, and, when the tree was dirty, `git.dirty_paths` plus
  `git.dirty_diff_sha256`, a fingerprint over the tracked diff and the name and
  content of every untracked file. Untracked directories are listed file by file
  (`--untracked-files=all`), since a collapsed `?? dir/` row would hide an edit
  inside it; a directory row that arrives anyway is hashed recursively, and an
  unreadable path or `git status` warning refuses the report rather than
  contributing a constant. Symbolic links are hashed as Git objects (type plus
  link-target text) and are never followed outside the repository. A reader can
  rebuild the recorded tree and compare binaries.
- The scripts **refuse to run against a dirty tree** unless
  `--allow-dirty-source` is given, and check it before doing any work: a
  measurement taken while the source is being edited is not a baseline, and one
  of the 2026-09-22 rounds was discarded for exactly that reason. The dirty
  check, and the fingerprint, both ignore `docs/benchmarks`, because writing one
  report must not make the next one claim a dirty source.
- The identity is captured **before** the run, and the report carries that
  snapshot, because the code and extension a process loaded cannot change under
  it. It is captured again afterwards and compared: a commit, checkout or
  rebuild during a long run would otherwise let the report name a revision, or a
  binary, that was never measured. `--allow-dirty-source` accepts a stable dirty
  tree, not a moving one.

## Registered inputs

Benchmark inputs live in `tests/fixtures/` and are bound to their exact bytes
and expected counts by `tests/fixtures/transcripts.toml`:

| Name | File | Purpose |
| --- | --- | --- |
| B1 | `b1_cc1_5spdfg_3exc.rcsfgenerate` | 2.7M CSFs; the stage-mix baseline |
| B2 | `b2_cc1_fullas_2exc.rcsfgenerate` | 560k CSFs; fast turnaround, wide active space |
| B3 | `b3_cc1_9spdfg_4exc.rcsfgenerate` | 7.03M configurations / 5.81G CSFs; capacity acceptance only |

`tests/benchmark_transcripts_test.rs` checks the hashes and the enumerated
configuration counts on every `cargo test`. The 2J range, excitation count and
record totals are cross-checked by `scripts/benchmark_v2_generation.py` against
a real run, because generating millions of records does not belong in the
default suite. B3 is marked `manual = true`: its hash is still checked, but its
enumeration is too expensive for the default suite and its counts are verified
by `scripts/estimate_v2_generation.py`, which counts without generating.

## Capacity reports

```bash
source ../graspkit-tools/.venv/bin/activate
python scripts/estimate_v2_generation.py tests/fixtures/b3_cc1_9spdfg_4exc.rcsfgenerate \
  --threads 8 --destination . \
  --output docs/benchmarks/<name>.json
```

The script verifies the transcript against the manifest, cross-checks the
counted configuration and record totals against the registered ones, and reports
the size of every path plus the free space of the destinations it is given. It
creates no scratch and writes no artifact, so a full-scale input can be assessed
without producing 5.8 billion CSFs.

## Scripts and report hygiene

`scripts/benchmark_support.py` holds what both scripts must agree on: the
manifest contract and its verification, the environment and filesystem
metadata, and path normalization. A report is normalized before it is written,
so a committed report carries `<system-temp>`, `<repo-root>` or `<path>/<name>`
instead of the directory layout of the machine that produced it; the filesystem
type and capacity stay. `tests/benchmark_reports_test.py` fails if any
registered report contains an absolute path.

A run that is refused by the managed-memory budget or the space pre-flight is
recorded as a measurement with `outcome: "rejected"` and its error, and is
excluded from the timing summary: a rejection is a result the matrix exists to
record, and it must not be averaged into a latency.

## Running the disk-generation benchmark

```bash
source ../graspkit-tools/.venv/bin/activate
python scripts/benchmark_v2_generation.py tests/fixtures/b1_cc1_5spdfg_3exc.rcsfgenerate \
  --threads 1 2 4 8 --memory-budget-mib 1024 8192 --repeats 3 --warmup 1 \
  --output docs/benchmarks/<name>.json
```

`--segment-codec` (P2a) and `--deduplication` (P6b) measure internal knobs the
same way. The extension reads each from the environment (`RCSFS_SEGMENT_CODEC`,
`RCSFS_DEDUPLICATION`) and reports back what it actually used, and the script
refuses a run whose report disagrees with the request: a report that named a
codec or a strategy the run did not use would make the measurement fiction.
Both are storage or checking choices, never output choices — the published
bytes and counts are unchanged. The harness records SHA-256 for the CSF text,
CSF Parquet, descriptor and header, compares the two de-duplication strategies
within one invocation, and refuses to write a report when their counts or any
published artifact differ.

Every successful measurement records `stage_seconds` and the residual
`unattributed_seconds`, and the script **refuses to write a report** when the
residual exceeds 10% (or 50 ms) of the call: a phase that grew without a timer
would otherwise turn the reported stage numbers into a claim that cannot be
checked against the end-to-end time.

The script verifies the transcript against the manifest and runs every warm-up
and measured combination in a fresh spawned process. Process creation and
artifact hashing are outside the timed generation call; this keeps `ru_maxrss`
scoped to one run instead of inheriting the high-water mark of earlier matrix
entries. It records execution order, wall-clock per stage, peak RSS, managed
memory, a sampled scratch peak, file counts and physical process I/O where the
platform exposes it. A measurement whose registry hash or record count does not
match is rejected rather than silently recorded.

Reports written before this isolated-process harness do not contain artifact
digests, and their `rss_peak_bytes` is the monotonically increasing peak of the
whole benchmark process. The codec, dedup and final-encoding campaigns dated
2026-09-22 were measured with the current harness (or re-measured at `47e04ea`)
— their committed JSONs carry per-run RSS and the four published-artifact
digests — while the older campaigns (threads, budget, 2026-09-21 baselines)
remain valid for timings, logical I/O, scratch and managed memory but not for
per-combination RSS or content differential claims.

The historical final-encoding campaign also demonstrates what the harness's
digests are for: at `45618fc`, the one-pass tail reproduced the two-pass tail's
CSF text, descriptor and header digests exactly while the CSF Parquet digest
changed. The later bounded-row-group fix intentionally changes descriptor's
physical Parquet layout; current correctness is therefore checked by live
logical-row differentials plus stable CSF/header bytes, not by freezing that
historical descriptor digest.

## Registered campaigns

| Report | What it measures |
| --- | --- |
| `v2_disk_generation_threads_b1/b2_20260922.json` | 1/2/4/8 threads at the derived task size, revision `1d1f29a` |
| `v2_disk_generation_budget_b1/b2_20260922.json` | Low (rejected), mid and high `memory_budget_mib` at 8 threads |
| `v2_disk_generation_codec_b1/b2_20260922.json` | Uncompressed / LZ4 / ZSTD temporary segments at 8 threads; first round `3588df1`, re-measured with per-run RSS and digests at `47e04ea` |
| `v2_disk_generation_dedup_b1/b2_20260922.json` | Verified-unique vs exact de-duplication, each with and without zstd segments; first round `bbcd714`, re-measured with content digests at `47e04ea` |
| `v2_disk_generation_final_encoding_b1/b2_20260922.json` | One-pass final encoding vs the two-pass merge+restore tail at 8 threads; seven-phase/bounded-row-group clean report `fbcf884` |
| `v2_disk_generation_parallel_descriptor_b1/b2_20260923.json` | Parallel Parquet descriptor column encoding at 8 threads; clean report `c91e51e`, none/zstd × verified/exact |
| `v2_disk_generation_p0b_*_20260921.json` | (legacy) measured before source identity was captured |
| `v2_disk_generation_b3_capacity_20260921.json` | Full-scale capacity, workload and time-range estimate (no generation) |
| `v2_disk_generation_matrix_20260922.md` | What the three 2026-09-22 campaigns show |
| `v2_disk_generation_codec_20260922.md` | What the P2a codec experiment shows |
| `v2_disk_generation_dedup_20260922.md` | What the P6b fast path saves, and what the P2a+P6b combination costs |
| `v2_disk_generation_final_encoding_20260922.md` | What the P4 one-pass tail saves, and the cumulative gain against the `7ad18b1` baseline |
| `v2_disk_generation_parallel_descriptor_20260923.md` | Parallel descriptor encoding speedup, CPU/墙钟并行度、受管内存与新物理摘要 |

A rejected configuration is part of the matrix, not a hole in it: the plan asks
for a low budget to be reported as a resource rejection rather than being
presented as a slow baseline.

## Other campaigns

- `rcsfgenerate_serial_20260910.*`, `rcsfgenerate_parallel_20260912.*` — single-J
  in-memory generation, with the original Fortran `rcsfgenerate` as a
  byte-for-byte correctness reference. Produced by
  `scripts/benchmark_single_j_generation.py`; they need a GRASP checkout and are
  therefore not reproducible from this repository alone.
