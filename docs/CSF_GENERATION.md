# Fixed-configuration CSF generation

For the planned V2 disk-generation performance work, see the
[V2 descriptor and CSF generation performance implementation plan](CSF_V2_GENERATION_PERFORMANCE_PLAN.md).
It records the current measurements, staged changes, and acceptance criteria.
The disk path now accepts a managed memory budget and reports its accounting
alongside the stage timings; the budget is not an operating-system RSS limit.

`csf_generation::generate_csfs` is the first serial implementation in phase C.
It creates new CSFs for **one explicit relativistic occupation configuration**.
The implementation runs entirely in Rust and returns `CompleteCsfFile` directly;
it does not call GRASP or construct intermediate CSF text.

This interface covers the subshell-state and coupling enumeration performed by
`rcsfgenerate90/GEN`. The occupation stage above it — reference-configuration
excitation rules, splitting an `nl` occupation between its two relativistic
subshells, and multiple-reference merging — is `enumerate_occupations`.
Expansion of an existing list remains to be implemented. This fixed-configuration
Rust API remains separate from the transcript CLI's disk pipeline, which now
provides parallel range generation, V2 descriptor derivation, stage statistics,
and an optional managed-memory budget as described below.

## Run the development example

From the `rCSFs` repository, with Rust and the uv environment installed:

```bash
uv run cargo run --release --example generate_csfs -- \
  examples/fixed_configuration.toml /path/to/new-output.c
```

The example requires a new output path. It rejects existing paths, including
links, and does not create an output when generation fails or finds zero CSFs.
Generation has no fixed record-count cap; results remain in memory.

The supplied request has a filled `1s` core and two electrons in `2p_{3/2}`:

```toml
core_subshells = ["1s"]
min_two_j = 0
max_two_j = 4

[[occupations]]
subshell = "2p"
electrons = 2
```

It produces two records, in the `J=0` and `J=2` blocks. `2p-` denotes
`2p_{1/2}`; `2p` denotes `2p_{3/2}`. These are separate occupations. The request
does not mean a nonrelativistic `2p^2` reference distributed across both.

## Rust interface

```rust
use _rcsfs::csf_generation::{GenerationRequest, SubshellOccupation, generate_csfs};

fn main() -> anyhow::Result<()> {
    let request = GenerationRequest {
        core_subshells: vec!["1s".parse()?],
        configuration: vec![SubshellOccupation {
            subshell: "2p".parse()?,
            electrons: 2,
        }],
        min_two_j: 0,
        max_two_j: 4,
    };
    let csfs = generate_csfs(&request)?;
    eprintln!("{} CSFs in {} blocks", csfs.records.len(), csfs.blocks.len());
    // Inspect the integer records, or export to any std::io::Write:
    csfs.write_to(std::io::stdout().lock())?;
    Ok(())
}
```

Subshell labels can also be constructed as `Subshell::new(n, kappa)`. Positive
`kappa=l` is the lower-j partner, and negative `kappa=-(l+1)` is the upper one.
Closed core subshells are fully filled; they are excluded from the coupling
chain. Occupations follow the supplied coupling order, and zero occupations
are omitted from the peel header and records.

The inclusive target range advances by two in `2J`. Both endpoints must have
the same parity as the electron count. Records are grouped by ascending total
`2J`, preserving the Fortran state-table order and ascending intermediate
couplings within each block. The writer reproduces `kopp1`'s state suppression
and `kopp2`'s sparse printed couplings, including explicitly printed zero states
and distinguishing seniority labels. Dense couplings are used during generation;
the returned codec representation retains the printed sparse couplings.

## Limits and failure behavior

- Orbitals must satisfy `1 <= n <= 15`, `0 <= l < n`, and `l <= 10`.
- At most 20 peel subshells may have nonzero occupations, matching `GEN`.
- Every occupation must be within the relativistic subshell capacity. Core
  duplicates, peel duplicates and core/peel overlaps are errors.
- The populated Fortran state tables are supported: all electron/hole
  occupations for `j <= 9/2`; at higher j, at most two electrons or two holes,
  including filled subshells. Other occupations fail explicitly.
- Requested and printed J values must fit the original writer's decimal
  fields: integer J at most 99, half-integer numerator at most 99.
- Generation has no fixed record-count cap. Available process memory limits the problem size.
  This limits record count; it is not a process memory or work/time budget.
  Large state products and coupling spaces can still take substantial time.
- An empty peel configuration or an unreachable target returns an empty
  `CompleteCsfFile`. Its records can be inspected, but the strict writer rejects
  empty lists. A core-only configuration currently follows `GEN`'s zero-peel
  behavior and produces no records.

## Test scope

The maintained test suite is responsible only for this repository's own code:
its Rust algorithms, Python APIs, CLI behavior, and file formats. Tests must be
self-contained, using repository fixtures and normal project dependencies; they
must not require an external GRASP source checkout, executable, or private
baseline dataset. Stored fixtures may encode expected compatibility behavior.

Put temporary test code, exploratory probes, and one-off external comparisons
under `temp/`, not `tests/`, `src/`, or `examples/`. Temporary code is outside the
maintained suite and must not be added to Cargo or pytest test discovery. Keep
local inputs and generated outputs there untracked. This rule concerns temporary
test code; maintained tests may still use standard temporary directories for I/O.

The three optional external comparison tests were removed on 2026-09-14.
Earlier GRASP comparisons are historical development evidence, not checks run
by the maintained suite. See the dated benchmark reports for those results.
The repository retains self-contained generation, parsing, block-count,
serialization, and serial/parallel consistency regressions.

The transcript parser accepts default orbital order (`*`) and one list terminated
by `n`. Nondefault order, continuation (`y`), missing termination, and trailing
input fail explicitly. Existing-list expansion remains unfinished.

## Transcript input normalization

`ExcitationRequest::from_transcript` normalizes user core 5 to low-level core 4
plus closed `4d/5s/5p`, and user core 6 to low-level core 5 plus closed
`4f/5d/6s/6p`. Explicit closed shells are included in the returned core metadata.
Directly constructed requests use the low-level core selector. Closed shells
must be full; changes in the closed set across references are unsupported.

The active-limit patches follow `rcsfexcitation`, including its nonphysical
`3f` sentinel for core 6; the sentinel never becomes a subshell. One upstream
edge case is deliberately preserved: core 6 with active `7s` becomes `3f,7s`,
and the wrapper's last-l rule omits closed `4f`, reducing the core by 14 electrons.
Core 6 with `7s,6d` or `7s,6g` retains it. This compatibility behavior is tested;
it is not a correction to the original physics input. Other active lists must
end with their highest l symmetry.

Negative excitation counts become positive limits and insert zero-population
`d` entries only for missing active reference shells. Explicit zero-population
selectors remain unchanged. Nonempty `d` shells fail. Excitation limits must fit
`u8`, including after taking the absolute value. Active shells with l >= 5 use
Fortran's four-electron enumeration cap, independently of physical capacity.

The parser also accepts log comments and whitespace-separated J ranges.
Self-contained tests cover these input-normalization rules using repository
fixtures and explicit expected values.

## Serial performance baseline

`examples/benchmark_generation.rs` composes the existing serial stages for
measurement. It retains integer chunks per occupation and exports their blocks
in final order through `CompleteCsfFile::write_record_to`, without copying
record payloads or writing intermediate CSF files. There is no fixed record-count
cap or process memory budget. The Python product entry is `uv run rcsfs csfsgenerate`.

Both registered full outputs, including their headers, match byte for byte.
The [2026-09-10 baseline report](benchmarks/rcsfgenerate_serial_20260910.md)
contains reproducible commands, per-stage timings, RSS, logical record I/O,
raw measurements and limitations. The measurement harness compares unmodified
Fortran binaries separately from an instrumented copy built outside the GRASP
checkout. No parallel implementation is included in these results.

## Deterministic parallel batch generation

`generate_csfs_parallel` expands independent occupation tasks with Rayon and
collects indexed results in input order. An optional thread count creates an
isolated pool; omitting it uses Rayon defaults. Record-count arithmetic is checked
for overflow, without a fixed cap. Tasks with at least 64 state combinations
can split into ordered state-prefix subtrees within the pool. The
`benchmark_generation` example accepts `RCSFS_THREADS` to exercise this path.
Per-task generation and final block organization remain unchanged, enabling
byte-for-byte serial/parallel comparisons.

## Transcript CLI

The product entry is `uv run rcsfs csfsgenerate [output.c]`, defaulting to
`rcsf.out`. The `generate_transcript_csfs` example remains a Rust development
regression tool for transcript inputs:

```bash
RCSFS_THREADS=4 uv run cargo run --release --example generate_transcript_csfs -- \
  input.rcsfgenerate output.c
```

The transcript is parsed in memory and CSFs are written in deterministic J/parity
block order. The Rust example accepts only input and output paths. Use the Python
CLI's `--generate-descriptors` option for CSF and descriptor Parquet output, and
`--normalize` for normalized descriptors. TOML `[output]` settings offer the same
options (see the configuration example above).

All destinations must be new and distinct: CSF text, CSF Parquet, its
`{csf_stem}_header.toml` sidecar, descriptor Parquet, and its same-stem `.toml`
sidecar. Outputs cannot alias the configuration file. The descriptor pipeline
stages files in a temporary directory before publishing through exclusively
created destination handles; existing files are never truncated. Staging requires
temporary disk space for the complete output set.

The descriptor sidecar schema is:

```toml
format_version = 2
encoding = "parquet"
normalized = false
record_count = 1
subshells = ["1s"]
```

For descriptor-producing TOML/config `csfsgenerate` runs (which default to the
disk backend), `subshells` lists the final header's ordered peel
subshells. Each contributes four consecutive V2 columns:
`n` (occupation), `2j` (printed subshell state), `v` (seniority), and `2k`
(printed intermediate coupling). The row then carries the global
`total_two_j` and `parity` columns. An unoccupied slot is `[0, -1, -1, -1]`;
`-1` is `MISSING` and distinguishes a value that GRASP did not print from an
explicit zero. `record_count` is the descriptor Parquet row count. All V2
columns are `Int32`, compressed with ZSTD level 3. V2 descriptors are
reversible and normalization is rejected.

For descriptor-producing TOML/config `csfsgenerate` runs, add the optional
setting below to the `[generate]` table to reject runs whose managed
occupation, batch, bucket, bitset, or writer reservations would exceed the
requested budget. The interactive in-memory compatibility path is separate
and does not use this disk budget:

```toml
memory_budget_mib = 8192
```

The CLI flag `--memory-budget-mib` overrides the TOML value. The returned JSON
contains `stage_stats` for enumeration, workload planning, generation,
de-duplication, descriptor merge, and CSF restoration, plus
`resource_stats.memory_budget_mib`, `budget_bytes`, `peak_managed_bytes`,
`current_managed_bytes`, and `occupation_bytes`. The budget accounts for
selected internal data structures; it is not an operating-system RSS limit.

### Counted workload planning

The disk path counts what each enumerated occupation configuration will produce
before it generates anything, then divides the work into tasks of comparable
estimated size. The count is the number of `(state selection, coupling chain)`
pairs that reach a requested `2J`; it is computed by dynamic programming over
the cumulative `2J` values rather than by walking the chains, and it is checked
for overflow rather than truncated. Its input contract is exactly the
generator's: the same occupation validation, the same state tables, and the same
`2J` range parity rule. When a reachable intermediate coupling could exceed
GRASP's output field, the configuration is counted by running the generator
into a counting sink instead, so a configuration the generator would reject is
rejected here too.

Two invariants hold for the result:

- The plan partitions the configuration list and each configuration's generation
  order, so concatenating its tasks in ordinal order reproduces the unsplit
  record order. Task sizes are execution details; `RCSFS_RECORDS_PER_TASK`
  exists only to hold the schedule fixed in a benchmark.
- Every counted record is scheduled. Work that cannot be divided far enough to
  reach the size target is still generated as one task and reported separately
  in `plan_stats.unsplittable_tasks`, rather than being dropped or hidden.

`plan_stats` reports `task_count`, `target_records_per_task`, the per-task
estimate distribution (`minimum`/`p50`/`p95`/`maximum`), `unique_occupations`,
`zero_record_configurations`, `unsplittable_tasks`, and `unsplittable_records`.
`estimated_total_records` is a pre-de-duplication estimate and is expected to
equal `generated_count`; the benchmark fails rather than records the run when
they disagree.

### De-duplication

The disk path still runs the exact de-duplication stage, and its
`duplicate_count` is a real measurement, but for this generation path it can
never remove a row: the enumeration emits pairwise distinct occupations, a
row's occupation columns identify its configuration, and one traversal of the
state table produces at most one record whose row identifies that traversal.
[V2_GENERATION_UNIQUENESS.md](V2_GENERATION_UNIQUENESS.md) proves this and
`tests/p6a_uniqueness_test.rs` searches for a counterexample by exhaustive
differential over bounded systems. The stage exists as the safety net for
inputs outside that argument — most importantly external CSF text, and any
future path that can schedule the same configuration twice — so it is not
removed; what the proof rules out is paying to *optimize* it for this path.

### Capacity pre-flight

The same counts size the run. Before the first segment is written, the disk path
estimates the Arrow segments, root and recursive de-duplication buckets, the
survivor bitsets and the published artifacts and applies a 25% safety margin.

Space is then checked **per volume, not per path**: requirements that share a
filesystem are added together, because scratch and the staged set coexist during
generation and the staged set coexists with its published copies during
publication. Each volume is required to hold the larger of those two phases. The
generation call checks the scratch and staging volumes; the CLI passes its final
destinations to the estimate so one model covers all of them, since publication
copies the staged set into paths only the CLI knows. Each artifact is named by
kind (`csf_text`, `csf_parquet`, `descriptor`, `header`, `descriptor_metadata`),
so the header and the descriptor sidecar are charged to their own volumes even
when those differ. A run that cannot fit is
refused while the scratch directory is still empty.

A volume the platform cannot measure fails the run rather than being skipped:
`allow_unchecked_space=True` (CLI `--allow-unchecked-space`) is the explicit way
to accept an unchecked pre-flight. This platform check is the only automatic
probe; there is no Windows implementation, so a Windows run must pass the
opt-out or run where free space can be measured.

Every ratio comes from the registered B1/B2 runs and is reported in the result's
`assumptions`, together with the quantities that were never measured — most
importantly one full recursive repartition, which is assumed rather than assumed
away. The estimate is therefore an upper bound with a visible derivation: on B2
it lands 1.01x-1.62x above the measured run, the larger end being that recursive
allowance. Ratios and their measured ranges are registered in
[docs/benchmarks](benchmarks/README.md).

`estimate_disk_generation` performs the same enumeration, counting and
scheduling without creating scratch or writing anything, charges the same
`memory_budget_mib` to the occupation arena (so a low budget fails the estimate
exactly as it fails the run), and reports the space checks for the paths it is
given. The CLI exposes it as `csfsgenerate --estimate-only` (requires the disk
descriptor path). The report's `failure_recovery` is `"restart"`: scratch is not
bound to an input hash or a format version, so a failed run is restarted rather
than resumed, and a pre-flight cannot promise otherwise. The CLI removes the
scratch directory it created whether generation succeeds or fails, so a retry
does not collide with its own leftovers.

The plan is also a partition of the work: a run asserts that the number of
generated records equals the counted total, and every oversized configuration
asserts that its tasks account for exactly the records counted for it. A
scheduling mistake that drops work therefore fails the run instead of producing
a shorter output file.

Managed memory and RSS are reported separately. On B1/B2 the managed peak was
29-88 MiB against 371-660 MiB of process RSS: the accounting covers the
structures the pipeline owns, not the allocator, thread stacks, Arrow/Parquet
runtime buffers or the page cache. A reservation that would exceed the budget
fails immediately with a resource error instead of waiting, so a brief
concurrent overshoot produces one clear failure rather than a stall whose timing
depends on which worker finishes first.


Descriptor columns and sidecar subshells follow the final CSF header. The
standalone `gen-descriptors` command still supports the legacy V1 layout only
when `descriptor_version=1` is explicit; V1 normalization is also explicit.
Raw V2 values are checked against the existing text-to-Parquet descriptor
pipeline and can be restored to CSF text with the exact source header. See
[local parallel measurements](benchmarks/rcsfgenerate_parallel_20260912.md).
