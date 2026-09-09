# Fixed-configuration CSF generation

`csf_generation::generate_csfs` is the first serial implementation in phase C.
It creates new CSFs for **one explicit relativistic occupation configuration**.
The implementation runs entirely in Rust and returns `CompleteCsfFile` directly;
it does not call GRASP or construct intermediate CSF text.

This interface covers the subshell-state and coupling enumeration performed by
`rcsfgenerate90/GEN`. Reference-configuration excitation rules, splitting an `nl`
occupation between its two relativistic subshells, multiple-reference merging,
and expansion of an existing list remain to be implemented. Python bindings,
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
It does not compare the full `rcsfgenerate` wrapper, excitation enumeration,
header generation by `fivefirst`, or multi-reference merging. Normal integration
tests separately check generated headers through the strict codec round trip.

This is a substage differential check. It does not complete phase C acceptance
against the two registered large calculation outputs; those still require their
original generation inputs and the occupation-enumeration stage.
