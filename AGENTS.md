# Agent Guidelines for rCSFs

This file provides build commands and code style guidelines for agentic coding assistants working on the rCSFs repository.

## Build/Test Commands

### Environment Setup
```bash
# Using pixi (recommended)
pixi install && pixi shell

# Or use Python 3.13 virtual environment
python -m venv .venv && source .venv/bin/activate
```

### Building
```bash
# Development build
maturin develop

# Optimized release build (always use for production)
maturin build --release

# Clean build artifacts
cargo clean && maturin clean
```

### Testing
```bash
# Run all tests
pytest

# Run with speed benchmarking
pytest --speed

# Run a single test file
pytest tests/rcsfs_test.py

# Run a specific test function
pytest tests/rcsfs_test.py::test_function_name
```

### Linting and Type Checking
```bash
# Type checking
mypy python/rcsfs/

# Linting
ruff check python/

# Format code
ruff format python/
```

## Code Style Guidelines

### Python Code

**Imports:** Group in order: standard library → third-party → local (Rust modules prefixed with `_`)

```python
from pathlib import Path
from typing import NotRequired, Optional, TypedDict, Union

from ._rcsfs import CSFProcessor as _CSFProcessor
```

**Type Hints:** Use `typing` module strictly. Accept `Union[str, Path]` for file paths, convert to `str` internally.

```python
def convert_csfs(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    max_line_len: Optional[int] = 256,
) -> ConversionStats:
    return _function(str(input_path), str(output_path), max_line_len)
```

**Docstrings:** Google-style with triple quotes. Include Examples section for non-trivial functions.

**Naming:**
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Private members: `_leading_underscore`
- Constants: `UPPER_SNAKE_CASE`

**Error Handling:** Raise descriptive errors with context.

```python
if value <= 0:
    raise ValueError("max_line_len must be greater than 0")
```

### Rust Code

**Edition & Features:** Rust 2024 edition with nightly/unstable features. Uses PyO3 for Python bindings.

**Error Handling:** Prefer `Result<T, String>` for Python-exposed functions, `anyhow::Error` for internal.

```rust
pub fn process_file(path: &Path) -> Result<Stats, String> {
    let content = read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    // ...
}
```

**Documentation:** Use `///` for public items, `//!` for module docs. Include `# Arguments` and `# Returns` sections.

```rust
/// Convert CSF file to Parquet format.
///
/// # Arguments
/// * `input_path` - Path to input CSF file
///
/// # Returns
/// * `Ok(ConversionStats)` - Statistics about conversion
/// * `Err(String)` - Error message on failure
pub fn convert_csfs_to_parquet(input_path: &Path) -> Result<ConversionStats, String> {
    // ...
}
```

**Naming:**
- Functions/variables: `snake_case`
- Structs/enums: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`
- Modules: `snake_case`

**PyO3 Patterns:**
- Use `#[pyo3(signature = (...))]` for optional parameters
- Expose structs with `#[pymodule]` and `#[pyclass]`
- Map Rust errors to Python exceptions (`PyValueError`, `PyIOError`)

### Performance Patterns

**Parallel Processing:** Use `rayon::par_iter` for CPU-bound work. Default to `num_cpus::get()` workers.

**Streaming:** Process files in chunks (default `chunk_size=3000000` lines) to avoid loading into memory.

**Output Formats:**
- CSF-to-Parquet: Uncompressed Parquet with `[idx, line1, line2, line3]` schema
- Descriptors: ZSTD level 3 compressed Parquet with multi-column format `col_0, col_1, ...`

### Testing

- Test files go in `tests/` directory
- Use pytest with descriptive test names
- Test both success and error paths
- For Python wrapper tests, verify Path handling: `Union[str, Path]` → `str` conversion

### Critical Constraints

- **Python version:** Strictly `>=3.13,<3.14` (pinned in pyproject.toml)
- **Rust edition:** 2024 (requires nightly/unstable, Rust >=1.92.0,<1.93)
- **Module naming:**
  - Cargo package: `rCSFs`
  - Extension module: `_rcsfs`
  - Public Python package: `rcsfs`
- **Always run** `maturin develop` after Rust code changes before testing
