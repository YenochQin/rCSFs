# rCSFs

[中文说明](README.zh-CN.md)

High-performance Rust-powered tools for working with atomic-physics CSF (Configuration State Function) data in Python.

rCSFs focuses on two jobs:

1. Convert large CSF text files into Parquet for downstream analysis.
2. Generate fixed-width descriptor tables for machine learning workflows.

The current Python API is function-based and centered around streaming, parallel processing, and Parquet-first data pipelines.

## What It Does

rCSFs is a Rust + PyO3 library for CSF datasets used in atomic-structure and spectroscopy workflows.

It helps you:

- Convert CSF text files into columnar Parquet files.
- Preserve the original CSF ordering during conversion.
- Extract peel subshell definitions from the generated header TOML.
- Generate descriptor Parquet files from converted CSFs.
- Optionally normalize descriptors for ML-oriented downstream use.

## Why rCSFs

- Rust core for throughput and predictable memory behavior.
- Streaming batch processing for large files.
- Parallel execution via `rayon`.
- Python-friendly API with `Path` support.
- Parquet output that works well with tools like Polars and PyArrow.

## Installation

`rcsfs` currently targets Python `3.14+`.

Build a wheel from source and install it wherever you need it:

```bash
git clone https://github.com/YenochQin/rCSFs.git
cd rCSFs
uvx --from 'maturin>=1.14,<2.0' maturin build --release
pip install target/wheels/rcsfs-*.whl
```

Maturin is only required for this packaging step, so running it with `uvx` avoids
adding a virtual environment to the checkout. Any PEP 517 front end works too —
`pip install .` fetches Maturin into an isolated build environment on its own.

### Inside the GraspKit workspace

When this repository sits beside `graspkit-tools/`, it is consumed as a
Maturin-backed path dependency and there is a single shared environment at
`../graspkit-tools/.venv`. Rebuild and reinstall the extension by syncing that
environment — `[tool.uv] cache-keys` tracks `src/**/*.rs`, so Rust edits are
picked up automatically:

```bash
cd ../graspkit-tools && uv sync
source .venv/bin/activate
```

Do not create `rCSFs/.venv`, and do not install Maturin into the shared
environment; build isolation provides it.

## Quick Start

```python
from pathlib import Path

import polars as pl
from rcsfs import (
    convert_csfs,
    generate_descriptors_from_parquet,
    get_parquet_info,
    read_csfs,
    read_peel_subshells,
    select_interacting_csfs,
)

input_csf = Path("tests/fixtures/sample.csf")
csf_parquet = Path("sample.parquet")
desc_parquet = Path("sample_descriptors.parquet")

# Read header metadata and CSF rows directly (no intermediate Parquet file)
header, csf_df = read_csfs(input_csf, num_workers=8)
print(header["block_info"])
print(csf_df.head())

# Add include_block_id=True when J^P block membership is needed
blocked_header, blocked_csf_df = read_csfs(
    input_csf, num_workers=8, include_block_id=True
)

# 1. Convert CSF text to parquet for persistent workflows
stats = convert_csfs(input_csf, csf_parquet)
print(stats)

# 2. Inspect parquet metadata
info = get_parquet_info(csf_parquet)
print(info)

# 3. Read peel subshells from the generated header TOML
peel_subshells = read_peel_subshells(stats["header_file"])
print(peel_subshells[:6])

# 4. Generate descriptor parquet (descriptor_version=2 by default)
desc_stats = generate_descriptors_from_parquet(
    csf_parquet,
    desc_parquet,
    peel_subshells=peel_subshells,
    header_path=stats["header_file"],
)
print(desc_stats)

# 5. Load the descriptor table
df = pl.read_parquet(desc_parquet)
print(df.head())

# 6. Select a conservative (non-exact) interaction upper bound
interaction_stats = select_interacting_csfs(
    "reference.csf", "candidates.csf", "selected.csf", num_workers=8
)
assert interaction_stats["exact"] is False
```

## Workflow

### 1. Convert CSF text to Parquet

`convert_csfs(...)` reads a CSF file, skips the 5-line header, skips GRASP block
separator lines containing only `*`, and writes the CSF data as ordered triples:

- `idx`
- `line1`
- `line2`
- `line3`

It also writes a companion TOML file named:

```text
<input_stem>_header.toml
```

That file contains:

- the original 5 header lines
- block metadata, including `block_info.block_lengths`
- conversion statistics

Example:

```python
from rcsfs import convert_csfs

stats = convert_csfs(
    "input.csf",
    "output.parquet",
    max_line_len=256,
    chunk_size=3_000_000,
    num_workers=None,
)
```

Returned stats include:

- `success`
- `input_file`
- `output_file`
- `header_file`
- `max_line_len`
- `chunk_size`
- `csf_count`
- `total_lines`
- `truncated_count`

### 2. Read peel subshells

`read_peel_subshells(...)` extracts the peel subshell sequence from the generated header TOML.

```python
from rcsfs import read_peel_subshells

peel_subshells = read_peel_subshells("output_header.toml")
```

Typical output:

```python
["5s", "4d-", "4d", "5p-", "5p", "6s"]
```

### 3. Generate descriptor Parquet

`generate_descriptors_from_parquet(...)` reads the converted CSF Parquet file and writes a
descriptor table. There are two descriptor formats, selected with `descriptor_version`:

#### V2 (default)

Four integer channels per peel subshell, in named columns:

```text
sub{i}_n, sub{i}_2j, sub{i}_v, sub{i}_2k   (occupation, printed 2J, seniority, printed coupling 2K)
```

plus two global columns:

```text
total_two_j, parity   (parity is +1/-1)
```

A value GRASP never printed for that record is `-1` (`MISSING`), distinct from a printed `0`.
V2 always writes `Int32` columns and does **not** support `normalize=True`.

When `header_path` is given (or auto-detected next to `input_parquet`), its SHA-256 is recorded
in the output Parquet's key-value metadata as `source_header_sha256`, binding the descriptor
file to the exact header it was generated from. `get_parquet_info(...)` returns this and the
rest of the format contract (`descriptor_version`, `channels_per_subshell`, `peel_subshells`,
`feature_columns`, `global_columns`, `missing_sentinel`, `normalized`) under
`key_value_metadata`.

#### V1 (legacy)

Positional columns `col_0, col_1, ..., col_N`, flattened by orbital as a dense triplet:

```text
[n_i, 2Q_i, 2J_cum,i] for each peel subshell
```

Pass `descriptor_version=1` explicitly to get this format. Raw V1 descriptors are `Int32`;
`normalize=True` (V1-only) writes `Float32` columns instead.

Output Parquet uses ZSTD compression in both formats.

Example:

```python
from rcsfs import generate_descriptors_from_parquet

# V2 (default)
stats = generate_descriptors_from_parquet(
    "output.parquet",
    "descriptors.parquet",
    peel_subshells=["5s", "4d-", "4d", "5p-", "5p", "6s"],
    num_workers=8,
    header_path="output_header.toml",
)

# V1, with normalization
stats_v1 = generate_descriptors_from_parquet(
    "output.parquet",
    "descriptors_v1.parquet",
    peel_subshells=["5s", "4d-", "4d", "5p-", "5p", "6s"],
    num_workers=8,
    normalize=True,
    descriptor_version=1,
)
```

### 3a. Restore CSFs from a V2 descriptor file

`restore_csfs_from_descriptors(...)` rebuilds a CSF text file from a V2 descriptor Parquet file
and its source `{stem}_header.toml`. If the descriptor file recorded a `source_header_sha256`,
the header is verified against it before anything is written — a mismatch means the header was
regenerated or edited since the descriptors were produced.

```python
from rcsfs import restore_csfs_from_descriptors

stats = restore_csfs_from_descriptors(
    "descriptors.parquet",
    "output_header.toml",
    "restored.c",
)
# Restore only a subset, in a given order:
stats = restore_csfs_from_descriptors(
    "descriptors.parquet", "output_header.toml", "subset.c", indices=[0, 5, 12],
)
```

### 4. Inspect Parquet metadata

```python
from rcsfs import get_parquet_info

info = get_parquet_info("output.parquet")
```

Returned metadata includes:

- `file_path`
- `file_size`
- `num_rows`
- `num_columns`
- `compression`
- `created_by`
- `key_value_metadata` — a `dict[str, str | None]` of the Parquet file's key-value metadata.
  For a V2 descriptor file this carries the full format contract: `descriptor_version`,
  `channels_per_subshell`, `subshell_count`, `peel_subshells`, `missing_sentinel`,
  `normalized`, `feature_columns`, `global_columns`, `source_header_sha256`,
  `source_header_filename`. Empty for files that carry no key-value metadata (e.g. plain
  CSF Parquet or V1 descriptor files).

### 5. Parse and reproduce a CSF with the complete integer representation

The development API in `complete_csf` reads a complete GRASP CSF file into a
compact in-memory integer representation. Unlike the ML descriptor, it keeps
subshell occupations, explicitly printed zero states, seniority labels, the
printed intermediate couplings, total `2J`, parity, block boundaries, and record
order.

Use the round-trip tool to validate a CSF file:

```bash
uv run cargo run --release --example roundtrip_csf -- \
  /path/to/input.c \
  /path/to/output.c
```

Example output:

```text
records=225157 blocks=7 occupied_entries=1974490 coupling_entries=670869 allocated_bytes=27263731 byte_identical=true
```

The command parses the input, writes it back from the integer representation,
and compares both files without loading the two text files into memory for the
comparison. It exits with an error if the output is not byte-identical. The output path must not already exist; existing files (including links to the
input) are rejected before writing, preserving the original baseline.

The same representation is available from Rust:

```rust
use _rcsfs::complete_csf::CompleteCsfFile;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let csfs = CompleteCsfFile::parse_path(Path::new("input.c"))?;
    println!("CSFs: {}", csfs.records.len());
    println!("J/P blocks: {}", csfs.blocks.len());
    println!("allocated bytes: {}", csfs.allocated_bytes());

    let first = &csfs.records[0];
    let occupied = csfs.occupied(first)?;
    let intermediate_couplings = csfs.couplings(first)?;
    println!(
        "occupied={} intermediate_couplings={}",
        occupied.len(),
        intermediate_couplings.len()
    );

    csfs.write_path(Path::new("output.c"))?;
    Ok(())
}
```

`allocated_bytes()` reports the heap capacity owned by the integer structure.
It excludes allocator bookkeeping and temporary parser/formatter buffers, so
it is not the process peak RSS.

`couplings()` returns only the intermediate couplings GRASP actually prints, not
a dense coupling chain. The `first` flag in `kopp2.f90` suppresses leading
couplings, so the result is sparse and must be indexed through the `boundary`
field — on `e1_cc1as1.c` only 670,869 of 1,524,176 interior boundaries are
printed. Consumers that need a value at every boundary, such as the cumulative
`2J` column of the ML descriptor, have to reconstruct the suppressed prefix
themselves.

Parsing is strict, and only the spellings the GRASP writers emit are accepted.
The block separator must be exactly `" *"`, empty symmetry blocks are rejected,
the three header labels are verified, seniority must occupy field offsets 3 and
4, and J fields must be bare decimals in reduced form — `"+4"` and `"8/2"` are
errors rather than fields that would be silently rewritten on output. Every
record is checked against its canonical formatting, including padding and unused
columns. Lines must end with LF; CRLF and a missing final LF are rejected. This keeps
parsing and formatting mutually inverse, so a successful round-trip is
byte-identical.

This is currently a Rust development API. The serial generator now uses this
representation to enumerate states and couplings for one explicit relativistic
occupation configuration. Python bindings for this representation are not
implemented yet.

### 6. Generate CSFs for a single fixed relativistic configuration (Rust dev API)

```bash
uv run cargo run --release --example generate_csfs -- \
  examples/fixed_configuration.toml /path/to/new-output.c
```

The example produces the two allowed CSFs of `2p_{3/2}^2` with a filled `1s`
core. It accepts a TOML request with explicit subshell occupations, an inclusive
`2J` range; output must be a new file. This is a Rust
development entry point (`csf_generation::generate_csfs(&GenerationRequest)`),
not the product CLI — see [Command Line Interface](#command-line-interface)
below for generating a full CSF list from Python.

## Command Line Interface

Installing `rcsfs` also installs an `rcsfs` console script (`uv run rcsfs ...`)
with five subcommands.

### `rcsfs csfsgenerate` — interactively generate a new CSF list

Replicates GRASP2018's `rcsfgenerate` interactive dialog (orbital order, core,
reference configurations, active orbitals, `2J` range, excitation count) and
generates the resulting CSF list with the Rust generator described above,
composed via `csf_generation::enumerate_occupations` +
`csf_generation::generate_csfs_parallel`. Options with no equivalent question
in the original dialog — output path, descriptor export, thread
count — are plain CLI flags instead:

```text
$ uv run rcsfs csfsgenerate out.c
Default, reverse, symmetry or user specified ordering? (*/r/s/u) *
Select core
 0  No core
 1  He (2)
 2  Ne (10)
 3  Ar (18)
 4  Kr (36)
 5  Xe (54)
 6  Rn (86)
Core? (0-6) 3
Enter list of (maximum 100) configurations. End list with a blank line or an asterisk (*)
Give configuration 1: 3d(10,i)4s(2,*)4p(6,*)4d(6,*)
Give configuration 2: 3d(10,*)4s(2,i)4p(6,i)4d(6,*)
Give configuration 3:
Give set of active orbitals, as defined by the highest principal quantum number per l-symmetry, in a comma delimited list in s,p,d etc order, e.g. 5s,4p,3d: 5s,5p,5d,4f
Resulting 2*J-number? lower, higher (J=1 -> 2*J=2 etc.): 0,12
Number of excitations (if negative number e.g. -2, correlation orbitals will always be doubly occupied): 2
Generate more lists ? (y/n) n
Generated CSFs: out.c
record_count: 452373
block_count: 7
```

Flags: `--generate-descriptors`, `--normalize`, `--threads N`,
`--memory-budget-mib MiB`, `--json`.

For reproducible batch runs, use a TOML configuration instead of the interactive
dialog. `generate_descriptors` defaults to `false`; when enabled, generation
writes the CSF text, the CSF Parquet plus its header TOML, and descriptor
Parquet plus its metadata sidecar:

```toml
[generate]
order = "*"
core = 2
references = ["3s(2,i)3p(6,5)3d(6,5)4s(2,i)", "3s(2,i)3p(6,5)3d(6,i)4s(2,*)"]
active_orbitals = "4s,4p,3d"
j_min = 0
j_max = 12
excitations = 2
continue_lists = false

[output]
generate_descriptors = true
csf = "calculation.c"
parquet = "calculation.parquet"
descriptor_parquet = "calculation_descriptors.parquet"
normalize = false
```

Run it with:

```bash
uv run rcsfs csfsgenerate --config calculation.toml
```


For reproducible scripted runs, use a TOML configuration. This avoids
transcript files and keeps generation settings with the output names:

```toml
[generate]
order = "*"
core = 3
references = [
  "3d(10,i)4s(2,*)4p(6,*)4d(6,*)",
  "3d(10,*)4s(2,i)4p(6,i)4d(6,*)",
]
active_orbitals = "5s,5p,5d,4f"
j_min = 0
j_max = 12
excitations = 2
continue_lists = false

[output]
generate_descriptors = true
csf = "out.c"
parquet = "out.parquet"
descriptor_parquet = "out_descriptors.parquet"
normalize = false
```

Run it with `uv run rcsfs csfsgenerate --config generation.toml`. By default
only the CSF text is written. With `generate_descriptors = true`, the command
also writes the CSF Parquet and header TOML, followed by descriptor Parquet and
its TOML sidecar. Descriptor CSV output is not supported. TOML/config
descriptor runs select the disk backend by default and write reversible **V2**
descriptors. Add
`memory_budget_mib = <MiB>` under `[generate]`, or pass
`--memory-budget-mib`, to enforce the managed-memory budget for disk generation.
With `--json`, the result also contains per-stage wall/CPU timing and logical
byte counters in `stage_stats`, together with managed-memory counters in
`resource_stats`. The memory budget does not cap RSS, allocator overhead, or
thread stacks.
The interactive in-memory compatibility path remains separate and still emits
legacy V1 descriptors; it is not the bounded disk path described here.

### `rcsfs gen-descriptors` — descriptor Parquet from a CSF Parquet file

```bash
# V2 (default)
uv run rcsfs gen-descriptors csf.parquet descriptors.parquet --header csf_header.toml

# V1, with normalization
uv run rcsfs gen-descriptors csf.parquet descriptors.parquet \
  --header csf_header.toml --descriptor-version 1 --normalize
```

`--descriptor-version {1,2}` selects the format (default: `2`); `--normalize` is V1-only and
errors if combined with `--descriptor-version 2`. The TOML/config transcript
generation path uses V2 and does not silently fall back to V1. This command also writes a
`{output_stem}.toml` sidecar mirroring the descriptor version and subshell list, for tools
that read TOML without opening the Parquet file.

### `rcsfs restore-csfs` — rebuild a CSF text file from V2 descriptors

```bash
uv run rcsfs restore-csfs --descriptors descriptors.parquet --header csf_header.toml \
  --output restored.c

# Restore only a subset, in a given order
uv run rcsfs restore-csfs --descriptors descriptors.parquet --header csf_header.toml \
  --output subset.c --indices 0 5 12
```

`--header` must be the exact `{stem}_header.toml` the descriptors were generated from. If the
descriptor file recorded `source_header_sha256`, it is verified against this file before
anything is written.

### `rcsfs zero-first` — reorder a CSF list into zero-order + first-order space

Mirrors GRASP2018's `rcsfzerofirst`: within each symmetry block, the
zero-order reference CSFs are locked to the head, followed by the
first-order complement.

```bash
uv run rcsfs zero-first zero.csf full.csf out.csf
```

### `rcsfs interacting` — conservative interaction candidates

This first implementation writes each reference block followed by candidate
CSFs that pass the two-electron occupation bound. It preserves input order and
parallel results are deterministic, but it does **not** yet implement GRASP's
recoupling, Coulomb angular-factor, or Breit/SNRC tests. The result is therefore
a structural upper bound (`exact=False`) that can retain false positives.
Dirac–Coulomb and Dirac–Coulomb–Breit currently share this same bound.

The CLI writes `rcsf.out` and uses 8 worker threads by default. Use
`--output PATH` and `--threads N` to override either default.

```bash
rcsfs interacting rcsfsmr.inp rcsf.inp --hamiltonian dc

# Optional overrides
rcsfs interacting rcsfsmr.inp rcsf.inp --hamiltonian dc \
  --threads 4 --output selected.csf
```

#### Input requirements

**Orbital basis — prefix, not equality.** The core header lines must be
byte-identical, and the reference peel subshell list must be a *prefix* of the
candidate peel subshell list in the same order. GRASP's own `rcsfinteract`
requires the two peel lists to be identical; rCSFs deliberately relaxes this to
a prefix so one reference space can be reused against a candidate space that
appends further correlation orbitals. Consequences:

- the candidate header, including the appended subshells, becomes the output
  header, so reference records are re-emitted under the wider peel declaration;
- the appended subshells participate in the occupation comparison;
- block counts and each corresponding block's `J/P` must still match exactly.

**The two inputs must be distinct files.** Equal paths, symlinks resolving to a
common target, and Unix hard links are all rejected. The output path likewise
must not alias either input, so a run can never overwrite the data it is
reading. Passing the same file as both reference and candidate is refused
rather than treated as an identity selection.

## Public Python API

| Function | Description |
| --- | --- |
| `read_csfs(input_path, max_line_len=256, num_workers=None, *, include_block_id=False, include_coupling_signature=False, strict=True)` | Return `(header, dataframe)` without a Parquet round trip; optional columns preserve block membership and fixed-width coupling signatures, while strict mode rejects incomplete final CSFs |
| `convert_csfs(input_path, output_path, max_line_len=256, chunk_size=3000000, num_workers=None)` | Convert CSF text to Parquet |
| `get_parquet_info(input_path)` | Inspect Parquet metadata |
| `read_peel_subshells(header_path)` | Read peel subshells from header TOML |
| `generate_descriptors_from_parquet(input_parquet, output_parquet, peel_subshells, num_workers=None, normalize=False, compression=None, *, descriptor_version=2, header_path=None)` | Generate descriptor Parquet from converted CSFs; `descriptor_version=2` (default) writes named columns, `1` writes the legacy `col_{i}` layout and is required for `normalize=True` |
| `restore_csfs_from_descriptors(descriptor_parquet, header_path, output, indices=None)` | Rebuild a CSF text file from a V2 descriptor Parquet file and its source header TOML |
| `partition_csfs(zero_parquet, zero_header, full_parquet, full_header, output_csf)` | Reorder a CSF list into zero-order + first-order space per symmetry block |
| `split_csfs_by_active_spaces(input_parquet, header_path, targets)` | Split one Parquet CSF list into independently selected, possibly overlapping active-space CSF text files; `targets` maps output path to orbital limits |
| `select_interacting_csfs(reference_csf, candidate_csf, output_csf, *, hamiltonian="dirac_coulomb", method="structural_upper_bound", num_workers=None, overwrite=False)` | Write a conservative, non-exact upper bound of interacting candidates; returned stats always include `exact=False` |
| `generate_csfs_from_transcript(transcript, output_path, normalize=False, threads=None)` | Generate CSFs from an in-memory `rcsfgenerate.log`-format transcript; backs `rcsfs csfsgenerate` |
| `generate_disk_outputs_from_transcript(transcript, csf_output, csf_parquet_output, descriptor_output, header_output, scratch_dir, threads=None, *, memory_budget_mib=None)` | Generate staged CSF and reversible V2 outputs with managed-memory accounting |

### Split CSFs by active space

Use the CSF Parquet file and matching `*_header.toml` produced by conversion or
disk generation. For example, from a large source space, request two nested
spaces in one pass:

```bash
rcsfs split-active source.parquet --header source_header.toml \
  --output-dir results --space '_small=5s,4p,3d' \
  --space '_large=7s,6p,5d,4f'
```

This writes `results/source_small.c` and `results/source_large.c`.
`rcsfs rcsfsplit` is an alias for `rcsfs split-active`. Each output is filtered
independently, so a CSF belonging to both spaces appears in both.
The five-line header is retained with its peel list filtered to the target;
CSF order and symmetry-block separators are preserved. Occupation lines are
parsed as exact orbital tokens (`5g` cannot accidentally match `15g`), and
relativistic partners such as `5g-` and `5g` share the `5g` maximum. Even an
explicitly zero-occupied orbital is checked, matching GRASP's line-based
filter and keeping every listed orbital within the output header. Input is
read in bounded Parquet batches; a 77-million-row source is not loaded into
RAM. Each output needs an existing parent directory and must not already
exist. A completed output is published atomically, but a group of outputs is
not an all-or-nothing transaction: on a late publication failure, earlier
completed outputs remain and are named in the error.

## Input Format

rCSFs expects CSF text files in this layout:

- first 5 lines: header / metadata
- remaining lines: CSFs in groups of 3
- line 1: subshell occupations
- line 2: intermediate coupling values
- line 3: final coupling and total `J`

## Typical Use Cases

- Preparing CSF datasets for analytics pipelines.
- Moving large text-based CSF collections into Parquet.
- Building ML-ready descriptor matrices from CSF data.
- Creating normalized descriptor datasets for model training.

## Performance Tips

- For dedicated machines, leave `num_workers=None` to use the default rayon worker configuration.
- For shared servers, set `num_workers` explicitly to avoid CPU contention.
- Keep the default `chunk_size=3_000_000` unless you have a measured reason to tune it.
- Increase `max_line_len` if your input has unusually long CSF lines and you want to avoid truncation.

## Development

Useful local commands:

```bash
uv run cargo test
uv run pytest
uv run ruff check .
uv run basedpyright rcsfs/
```

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

Run Cargo tests through `uv run` so PyO3 links against the project Python 3.14 environment. Bare `cargo test` can pick up a system Python and fail at link time, for example with `library 'python3.9' not found` on macOS.

## License

MIT. See [LICENSE](LICENSE).

`uv run rcsfs csfsgenerate` defaults to `rcsf.out`; the output path is optional.
