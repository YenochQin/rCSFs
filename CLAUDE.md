# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CSFs_loader is a high-performance Rust/Python hybrid library for converting CSF (Configuration State Function) text files to Parquet format. It uses PyO3 bindings to create Python extensions with Rust's performance benefits.

## Build Commands

### Environment Setup
```bash
# Using pixi (recommended)
pixi install
pixi shell

# Or use virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

### Development Build
```bash
# Build in development mode
maturin develop

# Build with optimizations
maturin build --release

# Install with pip in editable mode
pip install -e .
```

### Testing and Linting
```bash
# Run tests
pytest

# Type checking
mypy python/csfs_loader/

# Linting and formatting
ruff check python/
ruff format python/
```

### Clean Build
```bash
# Clean Rust artifacts
cargo clean

# Clean Python build artifacts
rm -rf build/ dist/ *.egg-info/
```

## Code Architecture

### Core Components

**Rust Backend (`src/`)**:
- `lib.rs` - PyO3 module definition and Python bindings
- `convert_csfs.rs` - Core conversion logic using Arrow/Parquet
- High-performance file processing with configurable chunking

**Python Package (`python/csfs_loader/`)**:
- `__init__.py` - Python API wrapper and public interface
- Type stubs (`.pyi`) for IDE support
- `py.typed` - Marks package as typed for mypy

### Key Features

**Performance Optimizations**:
- Rust-based file processing for high throughput
- Configurable line length limits and chunk sizes
- GZIP compression for Parquet output
- Batch processing to handle large files efficiently

**Data Processing**:
- Extracts 5-line headers from CSF files
- Processes CSF data in 3-line groups (line1, line2, line3)
- Separate `.headers.txt` file for metadata
- Truncation handling for oversized lines

**Python API**:
- Functional interface: `convert_csfs_to_parquet()`, `get_parquet_metadata()`
- Object-oriented interface: `CSFProcessor` class
- Bilingual error messages (Chinese/English)

## Development Workflow

### Project Structure
```
CSFs_loader/
├── src/                    # Rust source code
│   ├── lib.rs             # PyO3 bindings
│   └── convert_csfs.rs    # Core conversion logic
├── python/csfs_loader/    # Python package
│   ├── __init__.py        # Public API
│   ├── _csfs_loader.pyi   # Type stubs
│   └── py.typed          # Type marker
├── Cargo.toml             # Rust dependencies
├── pyproject.toml         # Python packaging
└── pixi.toml             # Development environment
```

### Dependencies

**Rust Dependencies** (`Cargo.toml`):
- `pyo3` - Python bindings
- `arrow` - Columnar data format
- `parquet` - Parquet file format support

**Python Dependencies** (`pyproject.toml`):
- `maturin` - Rust extension building
- `pytest` - Testing framework
- `mypy`, `ruff` - Type checking and linting

### Configuration

**Build Configuration**:
- Rust edition: 2024
- Python support: 3.10+ (optimized for 3.12-3.14)
- Release profile: Level 3 optimizations with LTO

**PyO3 Configuration**:
- Extension module: `_csfs_loader`
- Features: `pyo3/extension-module`, `pyo3/generate-import-lib`

## Common Usage Patterns

### Basic Conversion
```python
from csfs_loader import convert_csfs_to_parquet

# Convert CSF file to Parquet
result = convert_csfs_to_parquet(
    input_path="input.csf",
    output_path="output.parquet",
    max_line_len=256,
    chunk_size=30000
)
```

### Using the Processor Class
```python
from csfs_loader import CSFProcessor

# Create processor with custom settings
processor = CSFProcessor(max_line_len=512, chunk_size=50000)

# Convert multiple files
for csf_file in csf_files:
    processor.convert(csf_file, f"{csf_file}.parquet")
```

## Input/Output Patterns

### Input Files
- CSF text files with 5-line headers followed by 3-line CSF entries
- Variable line lengths (configurable truncation handling)

### Output Files
- `.parquet` - Main data with columns: line1, line2, line3
- `.headers.txt` - Extracted 5-line header information
- GZIP compression applied by default

## Performance Guidelines

### Optimization Parameters
- `max_line_len`: Default 256, increase for long CSF lines
- `chunk_size`: Default 30000, adjust based on available memory
- Typical performance: 4-10x speedup over pure Python implementations

### Memory Usage
- Processes data in configurable chunks to control memory footprint
- Arrow memory pools for efficient string handling
- Automatic cleanup between chunks

## Error Handling

### Common Issues
- File I/O errors (missing files, permissions)
- Invalid CSF format (incorrect line counts)
- Memory constraints (reduce chunk_size)
- Line truncation warnings (increase max_line_len)

### Error Types
- `PyValueError` - Invalid parameters
- `PyIOError` - File operation failures
- Detailed error messages in both English and Chinese

## Integration with GRASP

This utility is designed to work with GRASP atomic physics calculations:
- Converts GRASP CSF output to efficient Parquet format
- Preserves header information for downstream processing
- Enables fast data access for large-scale atomic structure calculations
- Compatible with GRASP workflow tools and analysis scripts