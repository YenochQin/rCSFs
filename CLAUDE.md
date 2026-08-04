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
- Uses uv for Python environment management

## Build Commands

```bash
# Set up Python environment
uv sync --group dev --group lint

# Build Rust extension in development mode (run after any Rust changes)
uv run maturin develop

# Build optimized release (for production/distribution)
uv run maturin build --release
```

`maturin` is installed in the uv-managed Python environment. Use `uv run maturin ...`, or activate `.venv` first before using bare `maturin`.

## Testing

```bash
# Run all Python tests
uv run pytest

# Run a single Python test file
uv run pytest tests/rcsfs_test.py

# Run all Rust tests (unit + integration)
uv run cargo test

# Run a single Rust test by name
uv run cargo test test_descriptor_generator_parse_csf_basic

# Run tests with speed benchmarking
uv run pytest --speed

# Lint and type-check Python
uv run ruff check .
uv run ruff format .
uv run basedpyright rcsfs/
```

**Note:** `tests/rcsfs_test.py` exercises the canonical API from `rcsfs/__init__.py`.

**Rust test note:** Run Cargo tests through `uv run` so PyO3 links against the uv-managed Python 3.14 runtime. Bare `cargo test` may pick up a system Python, for example Xcode's Python 3.9 on macOS, and fail during linking with `library 'python3.9' not found`.

## Code Architecture

### Module Structure

**Rust Backend (`src/`):**
- `lib.rs` — PyO3 module entry point: registers `convert_csfs`, `get_parquet_info`, and the descriptor submodule
- `csfs_conversion.rs` — CSF-to-Parquet conversion (parallel via rayon, streaming batches)
- `csfs_descriptor.rs` — descriptor parsing core and batch `generate_descriptors_from_parquet_parallel()`
- `descriptor_normalization.rs` — Normalization utilities: converts descriptor values using relativistic subshell physics (`max_electrons`, `kappa²`, `max_cumulative_2J`)

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
| `get_parquet_info(input_path)` | Parquet file metadata |
| `generate_descriptors_from_parquet(input, output, peel_subshells, ...)` | Batch descriptor generation |
| `read_peel_subshells(header_path)` | Extract subshell list from `*_header.toml` |
| `ConversionStats` | TypedDict for `convert_csfs` return |
| `DescriptorGenerationStats` | TypedDict for descriptor generation return |

`rcsfs` only exposes function-based Python APIs today. `CSFProcessor` and `CSFDescriptorGenerator` are internal Rust types and are not importable from `rcsfs._rcsfs`.

### Key Data Flow

**CSF Conversion:**
1. Stream CSF file in `chunk_size`-line batches (default: 3M lines = 1M CSFs)
2. Extract header (first 5 lines) → `{input_stem}_header.toml` in the output directory
3. Process remaining lines in parallel (rayon work-stealing)
4. Write Parquet with schema: `idx: UInt64, line1: Utf8, line2: Utf8, line3: Utf8` (uncompressed)

**Descriptor Generation:**
- Three-stage pipeline: Reader thread → Rayon workers → Writer thread
- Batch size: 65536 rows from Parquet
- Output schema: multi-column `col_0, col_1, ..., col_N` (Int32 or Float32 if normalized), ZSTD level 3
- Order preserved via `BTreeMap` in writer thread

**Descriptor Normalization (`descriptor_normalization.rs`):**
- Converts subshell notation: `"2p-"` → `"p-"` (angular notation with trailing space for positive parity)
- Each descriptor triplet `[n_electrons, J_middle, J_coupling]` is divided by `[max_electrons, kappa², max_2J]`
- Python API: `generate_descriptors_from_parquet(..., normalize=True, max_cumulative_doubled_j=N)`

### CSF File Format

```
Line 1-5:  Header metadata (saved as {stem}_header.toml)
Line 6+:   CSF entries, 3 lines per CSF:
  Line 1: orbital configs  e.g. "  5s ( 2)  4d-( 4)  4d ( 6)"
  Line 2: intermediate J   e.g. "                   3/2      "
  Line 3: final coupling   e.g. "                        4-  "
```

**J-value encoding in descriptors:** fractional `"3/2"` → 3 (numerator); integer `"4"` → 8, `"4-"` → 8 (doubled, parity stripped). Stores 2J as integer.

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

Always use `uv run maturin build --release` for production — the development build (`uv run maturin develop`) skips LTO.
