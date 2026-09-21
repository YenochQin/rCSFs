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

## Registered inputs

Benchmark inputs live in `tests/fixtures/` and are bound to their exact bytes
and expected counts by `tests/fixtures/transcripts.toml`:

| Name | File | Purpose |
| --- | --- | --- |
| B1 | `b1_cc1_5spdfg_3exc.rcsfgenerate` | 2.7M CSFs; the stage-mix baseline |
| B2 | `b2_cc1_fullas_2exc.rcsfgenerate` | 560k CSFs; fast turnaround, wide active space |

`tests/benchmark_transcripts_test.rs` checks the hashes and the enumerated
configuration counts on every `cargo test`. The 2J range, excitation count and
record totals are cross-checked by `scripts/benchmark_v2_generation.py` against
a real run, because generating millions of records does not belong in the
default suite.

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

## Other campaigns

- `rcsfgenerate_serial_20260910.*`, `rcsfgenerate_parallel_20260912.*` — single-J
  in-memory generation, with the original Fortran `rcsfgenerate` as a
  byte-for-byte correctness reference. Produced by
  `scripts/benchmark_single_j_generation.py`; they need a GRASP checkout and are
  therefore not reproducible from this repository alone.
