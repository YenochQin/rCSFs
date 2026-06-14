# Repository Guidelines

## Project Structure & Module Organization
`rCSFs` is a high-performance Rust/Python hybrid library for processing Configuration State Function (CSF) data from atomic-physics calculations such as GRASP. It provides CSF-to-Parquet conversion, fixed-length descriptor generation for ML workflows, and descriptor normalization using relativistic subshell physics.

Key implementation details:

- Rust edition: 2024.
- Python support: 3.14 only (`requires-python = ">=3.14"`).
- Cargo package: `rCSFs`.
- Rust library / compiled file: `_rcsfs`.
- Python install target: `rcsfs._rcsfs`.
- Public Python package: `rcsfs`.
- Build system: Maturin/PyO3 with uv for Python environment management.

Rust backend code lives in `src/`:

- `src/lib.rs`: PyO3 module entry point registering `convert_csfs`, `get_parquet_info`, and the descriptor submodule.
- `src/csfs_conversion.rs`: CSF-to-Parquet conversion, parallelized with rayon and streaming batches.
- `src/csfs_descriptor.rs`: descriptor parsing core and batch `generate_descriptors_from_parquet_parallel()`.
- `src/descriptor_normalization.rs`: descriptor normalization with `max_electrons`, `kappa^2`, and `max_cumulative_2J`.

The Python frontend lives in `rcsfs/`. `rcsfs/__init__.py` is the public API wrapper and supports `pathlib.Path`; `rcsfs/py.typed` marks the package as typed. Tests live in `tests/`, including Rust integration tests, Python API checks, speed tests, and fixtures such as `tests/fixtures/sample.csf`. Treat `dist/` and `target/` as build output unless a release task explicitly requires them.

## Build, Test, and Development Commands
Set up the Python environment and build the Rust extension before Python API tests:

```bash
uv sync --group dev --group lint
maturin develop
```

- `maturin develop`: build the Rust extension and install it into the active environment for local testing; run this after Rust changes.
- `maturin build --release`: build optimized production/distribution wheels.
- `cargo build --release`: produce optimized Rust artifacts.
- `cargo test`: run Rust unit and integration tests.
- `cargo test test_descriptor_generator_parse_csf_basic`: run a single Rust test by name.
- `pytest`: run all Python tests.
- `pytest tests/rcsfs_test.py`: run the canonical Python API tests.
- `pytest --speed`: run tests with speed benchmarking.
- `ruff check .`: lint Python code.
- `ruff format .`: format Python code.
- `mypy rcsfs/`: type-check the Python wrapper.

Always use `maturin build --release` for production. The development build from `maturin develop` skips LTO.

## Public Python API
`tests/rcsfs_test.py` exercises the canonical API exported from `rcsfs/__init__.py`.

| Symbol | Description |
|--------|-------------|
| `convert_csfs(input_path, output_path, ...)` | Convert CSF text to Parquet, parallelized with rayon. |
| `get_parquet_info(input_path)` | Return Parquet file metadata. |
| `generate_descriptors_from_parquet(input, output, peel_subshells, ...)` | Generate descriptor arrays from Parquet input. |
| `read_peel_subshells(header_path)` | Extract the subshell list from a `*_header.toml` file. |
| `ConversionStats` | TypedDict for `convert_csfs` results. |
| `DescriptorGenerationStats` | TypedDict for descriptor-generation results. |

`rcsfs` exposes function-based Python APIs only. `CSFProcessor` and `CSFDescriptorGenerator` are internal Rust types and are not importable from `rcsfs._rcsfs`.

## Data Flow & File Format
CSF conversion streams a CSF file in `chunk_size`-line batches. The default is 3 million lines, or 1 million CSFs. The first five lines are extracted as header metadata and saved as `{input_stem}_header.toml` in the output directory. Remaining lines are processed in parallel with rayon and written to uncompressed Parquet with schema `idx: UInt64`, `line1: Utf8`, `line2: Utf8`, `line3: Utf8`.

Descriptor generation uses a three-stage pipeline: reader thread -> rayon workers -> writer thread. It reads Parquet in 65,536-row batches and writes `col_0`, `col_1`, ..., `col_N` as `Int32`, or `Float32` when normalized, using ZSTD level 3. Output order is preserved through a `BTreeMap` in the writer thread.

Descriptor normalization converts subshell notation such as `"2p-"` to angular notation such as `"p-"`, and divides each descriptor triplet `[n_electrons, J_middle, J_coupling]` by `[max_electrons, kappa^2, max_2J]`. The Python API is `generate_descriptors_from_parquet(..., normalize=True, max_cumulative_doubled_j=N)`.

CSF files use this structure:

```text
Line 1-5:  Header metadata, saved as {stem}_header.toml
Line 6+:   CSF entries, 3 lines per CSF
  Line 1: orbital configs, e.g. "  5s ( 2)  4d-( 4)  4d ( 6)"
  Line 2: intermediate J,  e.g. "                   3/2      "
  Line 3: final coupling,  e.g. "                        4-  "
```

J-value encoding stores 2J as an integer. Fractional `"3/2"` becomes `3`; integer `"4"` and parity-marked `"4-"` both become `8`.

## Coding Style & Naming Conventions
Follow Rust 2024 idioms: 4-space indentation, `snake_case` for modules and functions, `CamelCase` for types, and small focused modules. Keep PyO3 bindings in `src/lib.rs` thin; push heavy logic into Rust modules. In Python, use PEP 8 naming and type hints for public APIs. Keep package exports aligned with `rcsfs/_rcsfs.pyi` when that stub is present. Prefer clear test filenames such as `*_test.rs`, and avoid committing exploratory notebooks or ad hoc scripts to the root.

Release builds use:

```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

Keep version changes aligned between `Cargo.toml` and packaging metadata, and document release-facing behavior in `docs/CHANGELOG.md` when appropriate.

## Testing Guidelines
Add Rust coverage for core parsing, conversion, descriptor behavior, and normalization in `tests/`. Add Python regression tests when changing the public package API or file I/O behavior. Name Rust tests `*_test.rs`; keep Python tests under `tests/` and start functions with `test_`. Reuse `tests/fixtures/` for stable sample data.

Run both Rust and Python checks before opening a PR:

```bash
cargo test
pytest
ruff check .
mypy rcsfs/
```

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects such as `update linux build` or `create win artifact`. Keep subjects brief and descriptive, and expand in the body when needed. PRs should explain the user-visible change, list validation commands run, and link related issues or docs. Include sample output or screenshots only when CLI/API behavior, generated files, or documentation rendering changes.

## Security & Configuration Tips
Do not commit generated build outputs, local data, credentials, virtual environments, or machine-specific paths. Treat CSF inputs as external data and validate file paths at the Python boundary. Build wheels intentionally with `maturin build --release`; do not rely on development artifacts for release validation.
