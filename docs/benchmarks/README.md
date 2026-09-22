# Benchmark registry

Measurements recorded here back the performance work in
[CSF_V2_GENERATION_PERFORMANCE_PLAN.md](../CSF_V2_GENERATION_PERFORMANCE_PLAN.md).
They are not part of the test suite: no file in this directory is run by
`cargo test` or `pytest`.

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
  content of every untracked file. A reader can rebuild the recorded tree and
  compare binaries.
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

The script verifies the transcript against the manifest, discards a warm-up run
per combination, records execution order, and reports wall-clock per stage
(the temporary output set's deletion is timed separately), peak RSS, managed
memory, a sampled scratch peak, file counts and physical process I/O where the
platform exposes it. A measurement whose registry hash or record count does not
match is rejected rather than silently recorded.

## Registered campaigns

| Report | What it measures |
| --- | --- |
| `v2_disk_generation_threads_b1/b2_20260922.json` | 1/2/4/8 threads at the derived task size, revision `1d1f29a` |
| `v2_disk_generation_budget_b1/b2_20260922.json` | Low (rejected), mid and high `memory_budget_mib` at 8 threads |
| `v2_disk_generation_p0b_*_20260921.json` | (legacy) measured before source identity was captured |
| `v2_disk_generation_b3_capacity_20260921.json` | Full-scale capacity, workload and time-range estimate (no generation) |
| `v2_disk_generation_matrix_20260922.md` | What the three 2026-09-22 campaigns show |

A rejected configuration is part of the matrix, not a hole in it: the plan asks
for a low budget to be reported as a resource rejection rather than being
presented as a slow baseline.

## Other campaigns

- `rcsfgenerate_serial_20260910.*`, `rcsfgenerate_parallel_20260912.*` — single-J
  in-memory generation, with the original Fortran `rcsfgenerate` as a
  byte-for-byte correctness reference. Produced by
  `scripts/benchmark_single_j_generation.py`; they need a GRASP checkout and are
  therefore not reproducible from this repository alone.
