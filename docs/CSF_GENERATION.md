# Fixed-configuration CSF generation

`csf_generation::generate_csfs` is the first serial implementation in phase C.
It creates new CSFs for **one explicit relativistic occupation configuration**.
The implementation runs entirely in Rust and returns `CompleteCsfFile` directly;
it does not call GRASP or construct intermediate CSF text.

This interface covers the subshell-state and coupling enumeration performed by
`rcsfgenerate90/GEN`. The occupation stage above it — reference-configuration
excitation rules, splitting an `nl` occupation between its two relativistic
subshells, and multiple-reference merging — is `enumerate_occupations`.
Expansion of an existing list remains to be implemented. Python bindings,
parallel execution and descriptor derivation belong to later stages.

## Run the development example

From the `rCSFs` repository, with Rust and the uv environment installed:

```bash
uv run cargo run --release --example generate_csfs -- \
  examples/fixed_configuration.toml /path/to/new-output.c
```

The example requires a new output path. It rejects existing paths, including
links, and does not create an output when generation fails or finds zero CSFs.
`max_records` is required so a request has an explicit record limit.

The supplied request has a filled `1s` core and two electrons in `2p_{3/2}`:

```toml
core_subshells = ["1s"]
min_two_j = 0
max_two_j = 4
max_records = 10000

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
        max_records: 10000,
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
- Exceeding `max_records` returns an error and discards the partial result.
  This limits record count; it is not a process memory or work/time budget.
  Large state products and coupling spaces can still take substantial time.
- An empty peel configuration or an unreachable target returns an empty
  `CompleteCsfFile`. Its records can be inspected, but the strict writer rejects
  empty lists. A core-only configuration currently follows `GEN`'s zero-peel
  behavior and produces no records.

## Differential verification

The implementation was checked against `genb.f90`, `kopp1.f90` and `kopp2.f90`
from GRASP commit `9006157730a82ac839f2b4ff4e938bcba63a539e`. The optional test
compiles these unmodified sources using `gfortran` in an isolated temporary
directory, with `tests/fixtures/gen_reference.f90` as the driver:

```bash
GRASP_SOURCE=/path/to/grasp uv run cargo test --test csf_generation_test \
  generated_records_match_unmodified_fortran_gen -- --ignored --nocapture
```

The comparison covers every populated subshell table and its particle-hole
mirror, mixed configurations with seniority, suppressed coupling fields, and
the 20-subshell limit (121 configurations passed with GNU Fortran 16.2.0).
It compares the three record lines in exact order after
stable grouping by total J and the trailing-space trimming done by `rcsfblock`.
It does not compare the full `rcsfgenerate` wrapper or header generation by
`fivefirst`. Normal integration
tests separately check generated headers through the strict codec round trip.

This is a substage differential check. Occupation enumeration is covered
separately: the two registered transcripts in `tests/fixtures/` reproduce the
per-symmetry-block record counts of the registered baselines exactly (452,373
records in 7 even blocks for `e1_cc1as1`, 89,786 in 2 odd blocks for
`o1_cc1as1`). Phase C acceptance additionally requires the existing-list
expansion mode.

## Registered full-output regression

The transcript parser currently accepts only default orbital order (`*`) and
one list terminated by `n`. Nondefault order, continuation (`y`), missing
termination and extra input after termination fail explicitly. Core rewrites and negative excitation counts are implemented. Existing-list
expansion remains unfinished.

Run the record-level differential test against the external registered files:

```bash
RCSFS_BASELINE_DIR=/path/to/baselines uv run cargo test --test csf_generation_test \
  registered_inputs_match_every_baseline_record_in_order -- --ignored --nocapture
```

On 2026-09-10 both files passed: 452,373 even and 89,786 odd records. The test
compares every occupied subshell identity, population, state (including seniority),
coupling, total J and parity in order and checks block boundaries. Failures identify
the file, occupation task and record. Missing files fail rather than silently skip.
This is semantic record equality; full generated headers and byte-level text
formatting are not checked by this test.

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

The parser also accepts the actual log's comments and whitespace-separated J
range. To reproduce the wrapper-level comparison with the registered executable:

```bash
GRASP_RCSFGENERATE=/path/to/grasp/build-debug/bin/rcsfgenerate \
  uv run cargo test --test csf_generation_test \
  input_rewrites_match_unmodified_rcsfgenerate -- --ignored --nocapture
```

All 18 cases passed on 2026-09-10, covering cores 0–6, active-limit patches,
negative/positive/zero excitation counts, explicit zero selectors, and multiple
references. The test runs each original calculation in its own temporary directory,
checks core labels and every three-line record in order, and replays the generated
log. The two registered full baselines also continue to pass record-level equality.

## Serial performance baseline

`examples/benchmark_generation.rs` composes the existing serial stages for
measurement. It retains integer chunks per occupation and exports their blocks
in final order through `CompleteCsfFile::write_record_to`, without copying
record payloads or writing intermediate CSF files. It requires an explicit global
record limit; it does not yet implement a process memory budget or a public
transcript-generation API.

Both registered full outputs, including their headers, match byte for byte.
The [2026-09-10 baseline report](benchmarks/rcsfgenerate_serial_20260910.md)
contains reproducible commands, per-stage timings, RSS, logical record I/O,
raw measurements and limitations. The measurement harness compares unmodified
Fortran binaries separately from an instrumented copy built outside the GRASP
checkout. No parallel implementation is included in these results.

## Deterministic parallel batch generation

`generate_csfs_parallel` expands independent occupation tasks with Rayon and
collects indexed results in input order. An optional thread count creates an
isolated pool; omitting it uses Rayon defaults. The API applies a global record
limit and rejects batches whose combined output exceeds it. The
`benchmark_generation` example accepts `RCSFS_THREADS` to exercise this path.
Per-task generation and final block organization remain unchanged, enabling
byte-for-byte serial/parallel comparisons.

## Transcript CLI

The `generate_transcript_csfs` example is the supported command-line path for
transcript inputs:

```bash
RCSFS_THREADS=4 uv run cargo run --release --example generate_transcript_csfs -- \
  input.rcsfgenerate output.c 500000 descriptors.csv --normalize
```

The transcript is parsed and expanded in memory; `output.c` is written in
deterministic J/parity block order. The optional CSV contains one dense
descriptor row per CSF, and `--normalize` applies the existing normalization
rules. Both output paths must be new files. When a descriptor CSV is requested,
the CLI also writes a same-stem `descriptors.toml` sidecar with
`format_version`, `encoding`, `normalized`, `record_count`, and the ordered
`subshells` list. The sidecar is versioned and must also be a new file.
