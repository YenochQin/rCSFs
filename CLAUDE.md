# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rCSFs is a high-performance Rust/Python hybrid library for processing CSF (Configuration State Function) data from atomic physics calculations (e.g., GRASP). It provides:
1. **CSF-to-Parquet conversion**: Convert CSF text files to Parquet format
2. **CSF descriptor generation**: Convert CSF data into fixed-length descriptor arrays for ML applications
3. **Descriptor normalization**: Normalize descriptors using relativistic subshell physics properties

**Key Implementation Details:**
- Rust edition: 2024
- Python support: 3.14 only (`requires-python = ">=3.14"` in pyproject.toml)
- Extension module name: `_rcsfs` (compiled Rust library, defined as `module-name = "rcsfs._rcsfs"`)
- Public package name: `rcsfs` (Python wrapper in `rcsfs/` at project root)
- Uses the shared uv environment at `../graspkit-tools/.venv`

## Build Commands

There is exactly one Python environment, `../graspkit-tools/.venv`, because every
consumer of this package lives in the parent workspace and imports it from there.
Do not run `uv sync` or `uv run` here, and do not create or use `rCSFs/.venv`.

After a Rust change there are two distinct places the extension may need
refreshing, because `[tool.maturin] python-source = "."` makes the repository
root itself the package root. Running anything from this directory imports the
in-tree `rcsfs/`, which shadows the copy installed in the shared environment.

**1. For parent-workspace code (`graspkit`, `graspkit-tools`, scripts) — sync:**

```bash
cd ../graspkit-tools && uv sync
```

`graspkit-tools` consumes this repository as a path dependency whose build
backend is Maturin, and `[tool.uv] cache-keys` in `pyproject.toml` lists
`src/**/*.rs`. `uv sync` therefore detects Rust edits, rebuilds the wheel through
PEP 517 build isolation — fetching Maturin into a throwaway build environment
itself, and using the release profile — and reinstalls it.

**2. For this repository's own `pytest` — build the in-tree extension:**

```bash
source ../graspkit-tools/.venv/bin/activate
cargo build --release --features pyo3/extension-module
cp target/release/lib_rcsfs.so rcsfs/_rcsfs.cpython-314-x86_64-linux-gnu.so
```

`pytest` runs from this directory, so it loads `rcsfs/_rcsfs*.so` rather than the
installed wheel; `uv sync` alone will not update what the tests import. The
Cargo build needs no Maturin and writes nothing into the shared environment.
Adjust the destination filename for your platform ABI tag.

**Maturin is deliberately absent from the shared environment and must not be
added to it.** Do not run `maturin develop`: it installs Maturin and this
repository's dev dependencies into whatever environment is active and converts
the `rcsfs` wheel install into an editable one, which silently desynchronizes the
shared environment from `graspkit-tools/uv.lock`. Recover with `uv sync` in
`graspkit-tools/`.

Maturin is declared only in this repository's `dev` dependency group, because
wheel packaging is the one task scoped to this repository alone. Run it as a
one-off tool so no second virtual environment appears:

```bash
# Produce a distributable wheel in target/wheels/
uvx --from 'maturin>=1.14,<2.0' maturin build --release \
  --interpreter ../graspkit-tools/.venv/bin/python
```

For Rust-only work, `cargo build` and `cargo test` need no wheel at all —
activate the Tools venv first so PyO3 links against its Python 3.14 runtime.

## Testing

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

```bash
# Run all Python tests
pytest

# Run a single Python test file
pytest tests/rcsfs_test.py

# Run all Rust tests (unit + integration)
cargo test

# Run a single Rust test by name
cargo test test_descriptor_generator_parse_csf_basic

# Run tests with speed benchmarking
pytest --speed

# Lint and type-check Python
ruff check .
ruff format .
basedpyright rcsfs/
```

**Note:** `tests/rcsfs_test.py` exercises the canonical API from `rcsfs/__init__.py`.

**Rust test note:** Activate `../graspkit-tools/.venv` before Cargo tests so PyO3 links against its Python 3.14 runtime. Cargo without that environment may pick up an incompatible system Python.

## Code Architecture

### Module Structure

**Rust Backend (`src/`):**
- `lib.rs` — PyO3 module entry point: registers `convert_csfs`, `get_parquet_info`, and the descriptor submodule
- `csfs_conversion.rs` — CSF-to-Parquet conversion (parallel via rayon, streaming batches)
- `csfs_descriptor.rs` — descriptor parsing core (both V1 and V2 text-path parsers) and batch `generate_descriptors_from_parquet_parallel()`
- `descriptor_schema.rs` — shared format contract: `DescriptorVersion`, `DescriptorLayout` (row/column layout for both versions), `validate_record` (legality checks shared by both descriptor producers), Parquet key-value metadata construction
- `descriptor_v2.rs` — V2 encode/decode (`encode_v2`, `decode_v2_into`) and CSF restoration (`restore_file`) from decoded V2 rows
- `descriptor_normalization.rs` — V1-only normalization utilities: converts descriptor values using relativistic subshell physics (`max_electrons`, `kappa²`, cumulative `2J`); V2 does not support normalization

**Python Frontend (`rcsfs/`):**
- `__init__.py` — Public API; wraps Rust functions with `pathlib.Path` support
- `py.typed` — PEP 561 marker

**Tests (`tests/`):**
- `integration_test.rs` — Rust integration tests for `csfs_conversion`
- `csfs_descriptor_test.rs` — Rust integration tests for `CSFDescriptorGenerator`
- `descriptor_normalization_test.rs` — Rust unit tests for normalization functions
- `rcsfs_test.py` — Python integration tests for the public wrapper API

### Public Python API (`rcsfs/__init__.py`)

| Symbol | Description |
|--------|-------------|
| `convert_csfs(input_path, output_path, ...)` | CSF → Parquet, parallel via rayon |
| `get_parquet_info(input_path)` | Parquet file metadata, including `key_value_metadata` |
| `generate_descriptors_from_parquet(input, output, peel_subshells, ..., descriptor_version=2, header_path=None)` | Batch descriptor generation; `descriptor_version=2` is the default, `1` is legacy |
| `restore_csfs_from_descriptors(descriptor_parquet, header_path, output, indices=None)` | Rebuild a CSF text file from a V2 descriptor Parquet file |
| `read_peel_subshells(header_path)` | Extract subshell list from `*_header.toml` |
| `ConversionStats` | TypedDict for `convert_csfs` return |
| `DescriptorGenerationStats` | TypedDict for descriptor generation return |
| `CsfRestoreStats` | TypedDict for `restore_csfs_from_descriptors` return |

`rcsfs` only exposes function-based Python APIs today. `CSFProcessor` and `CSFDescriptorGenerator` are internal Rust types and are not importable from `rcsfs._rcsfs`.

**`csfsgenerate` still always writes V1 descriptors** regardless of the library-wide V2
default (deferred; see the design doc's migration plan). Use `gen-descriptors` for V2 output.

### Key Data Flow

**CSF Conversion:**
1. Stream CSF file in `chunk_size`-line batches (default: 3M lines = 1M CSFs)
2. Extract header (first 5 lines) → `{input_stem}_header.toml` in the output directory
3. Process remaining lines in parallel (rayon work-stealing)
4. Write Parquet with schema: `idx: UInt64, line1: Utf8, line2: Utf8, line3: Utf8` (uncompressed)

**Descriptor Generation:**
- Three-stage pipeline: Reader thread → Rayon workers → Writer thread
- Batch size: 65536 rows from Parquet
- Output schema, V2 (default): named columns `sub{i}_n, sub{i}_2j, sub{i}_v, sub{i}_2k` per peel
  subshell plus global `total_two_j, parity`; always Int32; unprinted values are `-1`
  (`MISSING`), distinct from a printed `0`; Parquet key-value metadata carries the format
  contract (`descriptor_version`, `channels_per_subshell`, `peel_subshells`,
  `source_header_sha256`, etc.)
- Output schema, V1 (legacy, `descriptor_version=1`): positional `col_0, col_1, ..., col_N`
  (Int32, or Float32 if `normalize=True`)
- ZSTD level 3 by default; order preserved via `BTreeMap` in writer thread

**Descriptor Normalization (`descriptor_normalization.rs`) — V1 only:**
- Converts subshell notation: `"2p-"` → `"p-"` (angular notation with trailing space for positive parity)
- Each descriptor triplet `[n_i, 2Q_i, 2J_cum,i]` is divided per-CSF by `[g_i, n_i*(g_i-n_i),
  min(prefix_i, 2J_target+suffix_i)]`, where `g_i = 2|kappa_i|` and `2J_target` is inferred from
  the descriptor itself (no separate parameter)
- Python API: `generate_descriptors_from_parquet(..., normalize=True, descriptor_version=1)`
  (`normalize=True` raises for the V2 default — normalization is not implemented for V2)

### CSF File Format

```
Line 1-5:  Header metadata (saved as {stem}_header.toml)
Line 6+:   CSF entries, 3 lines per CSF:
  Line 1: orbital configs  e.g. "  5s ( 2)  4d-( 4)  4d ( 6)"
  Line 2: intermediate J   e.g. "                   3/2      "
  Line 3: final coupling   e.g. "                        4-  "
```

**J-value text parsing:** fractional `"3/2"` → 3 (numerator); integer `"4"` → 8, `"4-"` → 8
(doubled, parity stripped). Both V1 and V2 store 2J as integer; V2 additionally reads the
trailing `+`/`-` byte on line 3 as a separate `parity` global column (`+1`/`-1`), and keeps the
seniority digit at line 2 offsets 3-4 that V1 discards.

### Module Naming

- **Cargo package**: `rCSFs` (`Cargo.toml`)
- **Rust lib / compiled file**: `_rcsfs` (`[lib] name = "_rcsfs"`)
- **Python install target**: `rcsfs._rcsfs` (`tool.maturin.module-name`)
- **Public Python package**: `rcsfs` (`rcsfs/` directory)

### Release Build

```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

`uv sync` in `graspkit-tools/` builds this profile, so the extension installed in
the shared environment is already LTO-optimized. Use the `uvx maturin build
--release` command above only when you need a redistributable wheel file.
