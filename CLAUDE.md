# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CSFs_loader is a high-performance Rust/Python hybrid library for converting CSF (Configuration State Function) text files to Parquet format. It uses PyO3 bindings to create Python extensions with Rust's performance benefits.

**Key Implementation Details:**
- Rust edition: 2024 (requires nightly/unstable Rust features)
- Python support: 3.13 only (strictly pinned in pyproject.toml)
- Extension module name: `_csfs_loader` (the compiled Rust library)
- Public package name: `csfs_loader` (Python wrapper in `python/`)
- Uses pixi for development environment management

## Build Commands

### Environment Setup
```bash
# Using pixi (recommended)
pixi install
pixi shell

# Or use virtual environment (requires Python 3.13)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
```

### Development Build
```bash
# Build Rust extension and install in development mode
maturin develop

# Build optimized release version
maturin build --release

# Verify installation
python -c "import csfs_loader; print(csfs_loader.__version__)"
```

### Testing and Linting
```bash
# Run tests
pytest

# Type checking Python code
mypy python/csfs_loader/

# Linting
ruff check python/

# Format code
ruff format python/
```

### Clean Build Artifacts
```bash
# Clean Rust artifacts
cargo clean

# Clean Python build artifacts
rm -rf build/ dist/ *.egg-info/

# Clean maturin cache
maturin clean
```

## Code Architecture

### Module Structure

The project has a two-tier architecture with Rust backend and Python frontend:

**Rust Backend (`src/`):**
- `lib.rs` - PyO3 module definitions, Python exception mappings, and function exports
- `csfs_conversion.rs` - Core conversion logic with both sequential and parallel implementations

**Python Frontend (`python/csfs_loader/`):**
- `__init__.py` - Public API wrapper that re-exports Rust functions with Python-friendly signatures
- `_csfs_loader.pyi` - Type stubs for the compiled Rust extension
- `py.typed` - Marker file indicating this is a typed package for mypy

### Key Architectural Patterns

**Dual Processing Modes:**
The library provides both sequential and parallel conversion implementations:
- Sequential (`convert_csfs`): Single-threaded processing in `convert_csfs_to_parquet()`
- Parallel (`convert_csfs_parallel`): Multi-threaded processing with producer-consumer pattern using `crossbeam-channel`

**Header/Data Separation:**
CSF files have a specific structure:
- First 5 lines: Header metadata (extracted to `{input_stem}_header.toml`)
- Remaining lines: CSF data in 3-line groups (line1, line2, line3)
- Headers are processed first and saved separately from the main Parquet data

**Parallel Processing Architecture:**
The parallel implementation uses a three-stage pipeline:
1. **Reader thread**: Chunks input data and sends work items via bounded channel
2. **Worker threads** (configurable count): Process chunks in parallel, truncating lines to `max_line_len`
3. **Writer thread**: Receives processed chunks and writes them in CSF index order using `BTreeMap` for sequencing

**Data Flow:**
```
File Reader → Work Channel → Worker Threads → Result Channel → Writer → Parquet
                      ↓
                 Bounded Queue
                      ↓
           (prevents memory explosion)
```

### Critical Dependencies

**Rust:**
- `pyo3` - Python-Rust bindings and FFI
- `arrow`/`parquet` - Columnar data format and I/O
- `crossbeam-channel` - Multi-producer multi-consumer channels for parallel processing
- `num_cpus` - CPU core detection for worker thread optimization
- `toml`/`serde` - Header file serialization

**Python:**
- `maturin` - Build tool for Rust-Python extensions
- Build outputs: `.so` files on Linux, `.pyd` on Windows

### Function/API Mapping

**Rust Functions → Python Exports:**
- `convert_csfs_to_parquet()` → `convert_csfs()` / `CSFProcessor.convert()`
- `convert_csfs_to_parquet_parallel()` → `convert_csfs_parallel()` / `CSFProcessor.convert_parallel()`
- `get_parquet_metadata()` → `get_parquet_info()` / `CSFProcessor.get_metadata()`
- `csfs_header()` → `csfs_header()` (standalone header extraction)

**Class Wrappers:**
The `CSFProcessor` class in both Rust (`lib.rs`) and Python (`__init__.py`) provides an object-oriented interface with:
- Configurable `max_line_len` and `chunk_size` properties
- Validation ensuring parameters are > 0
- Both sequential and parallel conversion methods

## File Naming Conventions

**Output Files:**
- Parquet data: User-specified path (e.g., `output.parquet`)
- Header metadata: `{input_file_stem}_header.toml` (auto-generated in same directory as output)

**Example:** Converting `data.csf` to `results/output.parquet` generates:
- `results/output.parquet` (CSF data)
- `results/data_header.toml` (header metadata)

## Threading and Performance Considerations

**Parallel Mode Optimization:**
- Default `chunk_size` is larger for parallel (50000) vs sequential (100000 in Python wrapper, 30000 in Rust default)
- Worker count defaults to `num_cpus::get()` but is configurable
- Bounded channel size = `num_workers * 2` to prevent memory issues
- Progress logging every 50,000 CSFs written

**When to Use Parallel Mode:**
- Large files (>1M CSF entries)
- Multi-core systems available
- Memory-constrained environments (bounded queue prevents OOM)

## CSF File Format Assumptions

The code assumes CSF files follow this structure:
1. **Lines 1-5**: Header (metadata about the calculation)
2. **Lines 6+**: CSF entries in groups of exactly 3 lines
   - Line 1: CSF identifier/configuration
   - Line 2: Additional parameters
   - Line 3: More parameters or coefficients

**Validation:** The code processes complete 3-line groups, discarding any incomplete final group.
