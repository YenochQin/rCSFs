# rCSFs

[中文说明](README.zh-CN.md)

`rcsfs` is the Rust/PyO3 acceleration layer used by the GraspKit workflow to
process GRASP-style CSF (Configuration State Function) data from Python.

It converts large CSF text files into Parquet tables and generates fixed-width,
machine-learning-ready descriptor matrices from those tables. The public Python
surface is intentionally small and function-based, while the heavy parsing,
streaming, and parallel descriptor work runs in Rust.

## Highlights

- Converts GRASP CSF text files to Parquet while preserving CSF order.
- Writes a companion `*_header.toml` file with the original header and
  conversion statistics.
- Extracts peel subshell definitions from the generated header TOML.
- Generates descriptor Parquet files with stable `col_0..col_N` columns.
- Supports optional descriptor normalization for ML pipelines.
- Uses Rust, Arrow/Parquet, and Rayon for high-throughput batch processing.

## Repository Layout

```text
rCSFs/
├── src/                  # Rust core and PyO3 bindings
├── rcsfs/                # Python package wrapper, type hints, py.typed marker
├── tests/                # Rust integration tests and Python API tests
├── docs/                 # Design notes, changelog, review notes
├── scripts/              # Validation and comparison utilities
├── Cargo.toml            # Rust crate metadata
└── pyproject.toml        # Python/maturin build metadata
```

## Requirements

- Python `3.14+`
- Rust toolchain with Cargo
- `uv` for Python environment management
- `maturin` for building the Python extension

`rcsfs` is currently distributed as a source/build project in this repository.
The compiled Python module is named `rcsfs._rcsfs`; most users should import the
wrapper functions from `rcsfs`.

## Installation

For local development:

```bash
uv sync --group dev --group lint
uv run maturin develop
```

For an optimized local build:

```bash
uv sync --group dev --group lint
uv run maturin develop --release
```

To build distributable wheels:

```bash
uv run maturin build --release
```

The generated wheels are written under `target/wheels/` unless Maturin is given
another output directory.

## Quick Start

```python
from pathlib import Path

from rcsfs import (
    convert_csfs,
    generate_descriptors_from_parquet,
    get_parquet_info,
    read_peel_subshells,
)

input_csf = Path("tests/fixtures/sample.csf")
csf_parquet = Path("sample.parquet")
descriptor_parquet = Path("sample_descriptors.parquet")

# 1. Convert CSF text to Parquet.
conversion = convert_csfs(
    input_csf,
    csf_parquet,
    chunk_size=3_000_000,
    num_workers=None,
)
print(conversion)

# 2. Inspect the converted Parquet file.
print(get_parquet_info(csf_parquet))

# 3. Read peel subshells from the generated header TOML.
peel_subshells = read_peel_subshells(conversion["header_file"])

# 4. Generate descriptor columns.
descriptors = generate_descriptors_from_parquet(
    csf_parquet,
    descriptor_parquet,
    peel_subshells=peel_subshells,
    normalize=True,
)
print(descriptors)
```

To inspect the output table from Python, install a Parquet reader such as Polars
or PyArrow in your environment:

```python
import polars as pl

df = pl.read_parquet("sample_descriptors.parquet")
print(df.head())
```

## Workflow

### 1. Convert CSF Text to Parquet

`convert_csfs(...)` reads a CSF text file, skips the first five header lines,
then stores the remaining records as ordered three-line CSF groups:

| Column | Meaning |
| --- | --- |
| `idx` | Zero-based CSF row index |
| `line1` | Subshell occupation line |
| `line2` | Intermediate coupling line |
| `line3` | Final coupling and total `J` line |

Example:

```python
from rcsfs import convert_csfs

stats = convert_csfs(
    "input.csf",
    "csfs.parquet",
    max_line_len=256,
    chunk_size=3_000_000,
    num_workers=8,
)
```

The conversion also writes a header file next to the output Parquet file:

```text
input_header.toml
```

Returned conversion statistics include:

- `success`
- `input_file`
- `output_file`
- `header_file`
- `max_line_len`
- `chunk_size`
- `total_lines`
- `csf_count`
- `truncated_count`
- `error` when conversion fails

### 2. Read Peel Subshells

`read_peel_subshells(...)` reads the ordered peel subshell sequence from the
header TOML produced by `convert_csfs(...)`.

```python
from rcsfs import read_peel_subshells

peel_subshells = read_peel_subshells("input_header.toml")
```

Example output:

```python
["5s", "4d-", "4d", "5p-", "5p", "6s"]
```

### 3. Generate Descriptor Parquet

`generate_descriptors_from_parquet(...)` reads a converted CSF Parquet file and
writes a descriptor matrix. The output uses one Parquet column per descriptor
position:

```text
col_0, col_1, ..., col_N
```

For each peel subshell, descriptors are flattened as:

```text
[n_i, 2Q_i, 2J_cum,i]
```

Example:

```python
from rcsfs import generate_descriptors_from_parquet

stats = generate_descriptors_from_parquet(
    "csfs.parquet",
    "descriptors.parquet",
    peel_subshells=["5s", "4d-", "4d", "5p-", "5p", "6s"],
    num_workers=8,
    normalize=False,
)
```

Descriptor output details:

- Raw descriptors are written as `Int32` columns.
- Normalized descriptors are written as `Float32` columns.
- Descriptor Parquet files use ZSTD compression.
- Descriptor width is `3 * len(peel_subshells)`.

### 4. Inspect Parquet Metadata

```python
from rcsfs import get_parquet_info

info = get_parquet_info("descriptors.parquet")
```

Returned metadata includes:

- `file_path`
- `file_size`
- `num_rows`
- `num_columns`
- `compression`
- writer metadata such as `created_by`, when available

## Public Python API

| Function | Description |
| --- | --- |
| `convert_csfs(input_path, output_path, max_line_len=256, chunk_size=3000000, num_workers=None)` | Convert a GRASP CSF text file to Parquet. |
| `read_peel_subshells(header_path)` | Extract peel subshell names from a generated `*_header.toml` file. |
| `generate_descriptors_from_parquet(input_parquet, output_parquet, peel_subshells, num_workers=None, normalize=False)` | Generate descriptor columns from converted CSF Parquet data. |
| `get_parquet_info(input_path)` | Return basic Parquet file metadata. |

The exported package surface is defined in [`rcsfs/__init__.py`](rcsfs/__init__.py).
Lower-level Rust functions are implementation details and may change without
notice.

## Expected Input Format

`rcsfs` expects CSF files with this structure:

- first 5 lines: header and metadata
- remaining content: CSF records in groups of 3 lines
- line 1 of each group: subshell occupations
- line 2 of each group: intermediate coupling values
- line 3 of each group: final coupling values and total `J`

Files that do not follow this layout may fail conversion or produce invalid
descriptors.

## Performance Notes

- Leave `num_workers=None` on dedicated machines to use Rayon's default worker
  configuration.
- Set `num_workers` explicitly on shared servers to avoid excessive CPU use.
- Keep `chunk_size=3_000_000` unless profiling shows a better value for your
  workload. This default corresponds to roughly one million three-line CSFs per
  streaming batch.
- Increase `max_line_len` if valid CSF lines are longer than the default 256
  characters. Lines beyond this limit are counted in `truncated_count`.
- Descriptor generation is usually the most CPU-intensive step; normalized
  descriptors also perform per-CSF normalization in parallel.

## Development

Useful commands:

```bash
cargo test
uv run pytest tests/rcsfs_test.py
uv run ruff check .
uv run mypy rcsfs
```

Compare descriptor outputs after performance or normalization changes:

```bash
uv run python scripts/compare_descriptor_outputs.py --help
```

## Documentation

- [`docs/CHANGELOG.md`](docs/CHANGELOG.md): release-facing changes
- [`docs/CSF_DESCRIPTOR_GUIDE.md`](docs/CSF_DESCRIPTOR_GUIDE.md): descriptor notes
- [`docs/normalization_analysis.md`](docs/normalization_analysis.md): normalization details
- [`docs/performance_optimization_log.md`](docs/performance_optimization_log.md): performance notes

## Compatibility Notes

Some older repository documents mention APIs such as `convert_csfs_parallel`,
`CSFProcessor`, `CSFDescriptorGenerator`, `csfs_header`, or `j_to_double_j`.
Those names are not part of the current public Python wrapper. Prefer the four
functions listed in the public API table above.

## License

MIT. See [`LICENSE`](LICENSE).
