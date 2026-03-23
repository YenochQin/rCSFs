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

Build from source:

```bash
git clone https://github.com/YenochQin/rCSFs.git
cd rCSFs
uv sync
maturin develop --release
```

You can also use the repository's documented development flow:

```bash
uv sync --group dev --group lint
maturin develop
```

## Quick Start

```python
from pathlib import Path

import polars as pl
from rcsfs import (
    convert_csfs,
    generate_descriptors_from_parquet,
    get_parquet_info,
    read_peel_subshells,
)

input_csf = Path("tests/fixtures/sample.csf")
csf_parquet = Path("sample.parquet")
desc_parquet = Path("sample_descriptors.parquet")

# 1. Convert CSF text to parquet
stats = convert_csfs(input_csf, csf_parquet)
print(stats)

# 2. Inspect parquet metadata
info = get_parquet_info(csf_parquet)
print(info)

# 3. Read peel subshells from the generated header TOML
peel_subshells = read_peel_subshells(stats["header_file"])
print(peel_subshells[:6])

# 4. Generate descriptor parquet
desc_stats = generate_descriptors_from_parquet(
    csf_parquet,
    desc_parquet,
    peel_subshells=peel_subshells,
    normalize=True,
)
print(desc_stats)

# 5. Load the descriptor table
df = pl.read_parquet(desc_parquet)
print(df.head())
```

## Workflow

### 1. Convert CSF text to Parquet

`convert_csfs(...)` reads a CSF file, skips the 5-line header, and writes the remaining data as ordered triples:

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

`generate_descriptors_from_parquet(...)` reads the converted CSF Parquet file and writes a descriptor table with columns:

```text
col_0, col_1, ..., col_N
```

Descriptor layout is flattened by orbital:

```text
[n_i, 2Q_i, 2J_cum,i] for each peel subshell
```

Notes:

- Raw descriptors are written as `Int32` columns.
- Normalized descriptors are written as `Float32` columns.
- Output Parquet uses ZSTD compression.

Example:

```python
from rcsfs import generate_descriptors_from_parquet

stats = generate_descriptors_from_parquet(
    "output.parquet",
    "descriptors.parquet",
    peel_subshells=["5s", "4d-", "4d", "5p-", "5p", "6s"],
    num_workers=8,
    normalize=False,
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

## Public Python API

| Function | Description |
| --- | --- |
| `convert_csfs(input_path, output_path, max_line_len=256, chunk_size=3000000, num_workers=None)` | Convert CSF text to Parquet |
| `get_parquet_info(input_path)` | Inspect Parquet metadata |
| `read_peel_subshells(header_path)` | Read peel subshells from header TOML |
| `generate_descriptors_from_parquet(input_parquet, output_parquet, peel_subshells, num_workers=None, normalize=False)` | Generate descriptor Parquet from converted CSFs |

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

## Notes on Older Docs

Some older repository docs refer to APIs such as `convert_csfs_parallel`, `CSFProcessor`, `CSFDescriptorGenerator`, `csfs_header`, or `j_to_double_j`.

Those are not part of the current public Python wrapper exported by [rcsfs/__init__.py](rcsfs/__init__.py), so this README documents the current supported surface instead.

## Development

Useful local commands:

```bash
cargo test
pytest tests/rcsfs_test.py
ruff check .
mypy rcsfs
```

## License

MIT. See [LICENSE](LICENSE).
