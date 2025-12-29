"""
rCSFs - Rust-powered CSF (Configuration State Function) Processing Library

This package provides high-performance Rust implementations for processing
atomic physics CSF data.

Main Components:
- CSF file format conversion to Parquet
- CSF descriptor generation for ML applications

## Quick Start

### CSF File Conversion

```python
from rcsfs import CSFProcessor

processor = CSFProcessor()
processor.convert("input.csf", "output.parquet")
```

Or use the functional API:

```python
from rcsfs import convert_csfs

convert_csfs("input.csf", "output.parquet")

# With custom chunk size (default: 3000000 lines = 1M CSFs)
convert_csfs("input.csf", "output.parquet", chunk_size=6000000)

# Limit to 8 workers for shared servers
convert_csfs("input.csf", "output.parquet", num_workers=8)
```

### CSF Descriptor Generation

```python
from rcsfs import CSFDescriptorGenerator

# Define orbitals
peel_subshells = ['5s', '4d-', '4d', '5p-', '5p', '6s']
gen = CSFDescriptorGenerator(peel_subshells)

# Parse CSF
line1 = '  5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)'
line2 = '                   3/2               2        '
line3 = '                                           4-  '

descriptor = gen.parse_csf(line1, line2, line3)
# Returns: [2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 6.0, 3.0, 3.0, ...]
```

For detailed documentation, see:
- CSF File Conversion: See `CSFProcessor` class documentation
- CSF Descriptor Generation: See `CSFDescriptorGenerator` class documentation
"""

from pathlib import Path
from typing import NotRequired, Optional, TypedDict, Union

# Import from the Rust extension module
try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("rcsfs")
    except PackageNotFoundError:
        __version__ = "0.1.0"
except ImportError:
    __version__ = "0.1.0"

# Import from the Rust extension module
from ._rcsfs import (
    CSFDescriptorGenerator,
    CSFProcessor as _CSFProcessor,
    convert_csfs as _convert_csfs,
    get_parquet_info as _get_parquet_info,
    py_generate_descriptors_from_parquet as _generate_descriptors_from_parquet,
    py_read_peel_subshells as _read_peel_subshells,
)


#///////////////////////////////////////////////////////////////////////////////
# Type Definitions
#///////////////////////////////////////////////////////////////////////////////


class ConversionStats(TypedDict):
    """Statistics returned from CSF conversion operations."""

    success: bool
    input_file: str
    output_file: str
    header_file: NotRequired[str]
    max_line_len: int
    chunk_size: int
    error: NotRequired[str]
    total_lines: NotRequired[int]
    csf_count: NotRequired[int]
    truncated_count: NotRequired[int]


class DescriptorGenerationStats(TypedDict):
    """Statistics returned from batch descriptor generation."""

    success: bool
    input_file: str
    output_file: str
    csf_count: NotRequired[int]
    descriptor_count: NotRequired[int]
    orbital_count: NotRequired[int]
    descriptor_size: NotRequired[int]
    error: NotRequired[str]


#///////////////////////////////////////////////////////////////////////////////
# Python Wrapper Functions (with Path support)
#///////////////////////////////////////////////////////////////////////////////


def convert_csfs(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    max_line_len: Optional[int] = 256,
    chunk_size: Optional[int] = 3000000,
    num_workers: Optional[int] = None,
) -> ConversionStats:
    """
    Convert CSF text file to Parquet format using parallel processing.

    This function is optimized for large-scale data processing. It uses a streaming
    approach with rayon-based parallel processing:
    - Stream: Read file in batches to avoid loading large files into memory
    - Parallel: Process each batch with rayon's work-stealing (all cores used by default)
    - Order: Maintain CSF order in output

    Args:
        input_path: Path to input CSF file
        output_path: Path to output Parquet file
        max_line_len: Maximum line length (default: 256)
        chunk_size: Number of lines per read batch (default: 3000000)
        num_workers: Optional number of worker threads (default: CPU core count)

    Returns:
        Dictionary containing conversion statistics and status

    Examples:
        >>> # Use all CPU cores (default)
        >>> stats = convert_csfs("input.csf", "output.parquet")
        >>>
        >>> # Limit to 8 workers for shared servers
        >>> stats = convert_csfs("input.csf", "output.parquet", num_workers=8)

    Performance Considerations:
        - For single-task environments: omit num_workers (uses all cores)
        - For multi-task servers: set num_workers to avoid CPU contention
        - Typical values: num_workers=4-8 for shared servers, None for dedicated
    """
    return _convert_csfs(
        input_path=str(input_path),
        output_path=str(output_path),
        max_line_len=max_line_len,
        chunk_size=chunk_size,
        num_workers=num_workers,
    )


def get_parquet_info(input_path: Union[str, Path]) -> dict:
    """
    Get basic information and metadata from a Parquet file.

    Args:
        input_path: Path to Parquet file

    Returns:
        Dictionary containing file information:
        - file_path: File path
        - file_size: File size in bytes
        - num_rows: Number of rows in the file
        - num_columns: Number of columns
        - compression: Compression method used
    """
    return _get_parquet_info(input_path=str(input_path))


#///////////////////////////////////////////////////////////////////////////////
# Batch Descriptor Generation Functions
#///////////////////////////////////////////////////////////////////////////////


def read_peel_subshells(header_path: Union[str, Path]) -> list[str]:
    """
    Extract peel subshells from a header TOML file.

    Args:
        header_path: Path to the header TOML file

    Returns:
        List of subshell names (e.g., ['5s', '4d-', '4d', '5p-', '5p', '6s'])

    Examples:
        >>> peel_subshells = read_peel_subshells("data_header.toml")
        >>> print(peel_subshells)
        ['5s', '4d-', '4d', '5p-', '5p', '6s']
    """
    return _read_peel_subshells(str(header_path))


def generate_descriptors_from_parquet(
    input_parquet: Union[str, Path],
    output_parquet: Union[str, Path],
    peel_subshells: list[str],
    num_workers: Optional[int] = None,
) -> DescriptorGenerationStats:
    """
    Generate CSF descriptors from a parquet file using parallel processing.

    This function is optimized for large-scale descriptor generation (tens of millions
    to billions of CSFs). It uses rayon's work-stealing for automatic load balancing
    with streaming batch processing for low memory usage.

    Output Format:
        Arrow IPC (Feather) - columnar format, Polars compatible.
        Read with: `polars.read_ipc()` or `pyarrow.feather.read_table()`
        The output file will have descriptor_size columns (col_0, col_1, ..., col_N),
        where each row represents a CSF descriptor with each element in its own column.

    Args:
        input_parquet: Path to input parquet file (must have line1, line2, line3, idx columns)
        output_parquet: Path to output Arrow IPC/Feather file for descriptors
        peel_subshells: List of subshell names (e.g., ['5s', '4d-', '4d', '5p-', '5p', '6s'])
        num_workers: Number of worker threads (default: CPU core count)

    Returns:
        Dictionary containing generation statistics:
        - success: Whether generation succeeded
        - input_file: Input parquet file path
        - output_file: Output Arrow IPC/Feather file path
        - csf_count: Number of CSFs processed
        - descriptor_count: Number of descriptors generated
        - orbital_count: Number of orbitals
        - descriptor_size: Size of each descriptor (3 * orbital_count)

    Examples:
        >>> # Basic usage with peel_subshells from header
        >>> from rcsfs import read_peel_subshells, generate_descriptors_from_parquet
        >>>
        >>> peel_subshells = read_peel_subshells("data_header.toml")
        >>> stats = generate_descriptors_from_parquet(
        ...     "csfs_data.parquet",
        ...     "descriptors.feather",
        ...     peel_subshells=peel_subshells
        ... )

        >>> # Read with polars
        >>> import polars as pl
        >>> df = pl.read_ipc("descriptors.feather")
        >>> descriptors = df.to_numpy()  # Shape: (n_csfs, descriptor_size)

        >>> # With custom worker count for large files
        >>> stats = generate_descriptors_from_parquet(
        ...     "csfs_data.parquet",
        ...     "descriptors.feather",
        ...     peel_subshells=['5s', '4d-', '4d', '5p-', '5p', '6s'],
        ...     num_workers=8
        ... )

    Performance Considerations:
        - For medium files (1-10M CSFs): num_workers=4-8
        - For large files (>10M CSFs): num_workers=8+
        - More workers = higher CPU usage, faster processing
        - Rayon automatically handles work stealing for optimal load balancing

    Note:
        This implementation uses streaming batch processing to minimize memory usage:
        1. Read parquet in batches
        2. Parse CSFs to descriptors in parallel
        3. Convert descriptors to columnar format in parallel
        4. Write batch to Arrow IPC/Feather file
        5. Repeat until all data processed
    """
    return _generate_descriptors_from_parquet(
        input_parquet=str(input_parquet),
        output_file=str(output_parquet),
        peel_subshells=peel_subshells,
        num_workers=num_workers,
    )


#///////////////////////////////////////////////////////////////////////////////
# CSFProcessor Wrapper Class (with property accessors)
#///////////////////////////////////////////////////////////////////////////////


class CSFProcessor:
    """
    CSF file processor class providing an object-oriented interface.

    This class allows for easy configuration and repeated conversion operations
    with consistent settings.
    """

    def __init__(
        self, max_line_len: Optional[int] = 256, chunk_size: Optional[int] = 3000000
    ):
        """
        Create a new CSF processor instance.

        Args:
            max_line_len: Maximum line length (default: 256)
            chunk_size: Batch processing size (default: 3000000)
        """
        self._processor = _CSFProcessor(
            max_line_len=max_line_len, chunk_size=chunk_size
        )

    @property
    def max_line_len(self) -> int:
        """Get current maximum line length."""
        return self._processor.get_config()["max_line_len"]

    @max_line_len.setter
    def max_line_len(self, value: int) -> None:
        """Set maximum line length."""
        if value <= 0:
            raise ValueError("max_line_len must be greater than 0")
        self._processor.set_max_line_len(value)

    @property
    def chunk_size(self) -> int:
        """Get current chunk size."""
        return self._processor.get_config()["chunk_size"]

    @chunk_size.setter
    def chunk_size(self, value: int) -> None:
        """Set chunk size."""
        if value <= 0:
            raise ValueError("chunk_size must be greater than 0")
        self._processor.set_chunk_size(value)

    def get_config(self) -> dict:
        """
        Get current processor configuration.

        Returns:
            Dictionary containing current settings
        """
        return self._processor.get_config()

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        num_workers: Optional[int] = None,
    ) -> ConversionStats:
        """
        Convert CSF file using parallel processing.

        Args:
            input_path: Path to input CSF file
            output_path: Path to output Parquet file
            num_workers: Optional number of worker threads (default: CPU core count)

        Returns:
            Conversion statistics and status
        """
        return self._processor.convert(
            input_path=str(input_path),
            output_path=str(output_path),
            num_workers=num_workers,
        )

    def get_metadata(self, input_path: Union[str, Path]) -> dict:
        """
        Get Parquet file information.

        Args:
            input_path: Path to Parquet file

        Returns:
            File information and metadata
        """
        return self._processor.get_metadata(input_path=str(input_path))


#///////////////////////////////////////////////////////////////////////////////
# Public API
#///////////////////////////////////////////////////////////////////////////////

__all__ = [
    # Version
    "__version__",
    # CSF file conversion
    "convert_csfs",
    "get_parquet_info",
    "CSFProcessor",
    # CSF descriptor generation
    "CSFDescriptorGenerator",
    # Batch descriptor generation
    "generate_descriptors_from_parquet",
    "read_peel_subshells",
    # Type definitions
    "ConversionStats",
    "DescriptorGenerationStats",
]
