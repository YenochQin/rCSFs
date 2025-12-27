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
processor.convert_parallel("input.csf", "output.parquet")
```

Or use the functional API:

```python
from rcsfs import convert_csfs_parallel

convert_csfs_parallel("input.csf", "output.parquet")
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

### J-Value Conversion

```python
from rcsfs import j_to_double_j

j_to_double_j("3/2")  # → 3
j_to_double_j("2")    # → 4
j_to_double_j("4-")   # → 8
```

For detailed documentation, see:
- CSF File Conversion: See `CSFProcessor` class documentation
- CSF Descriptor Generation: See `CSFDescriptorGenerator` class documentation
"""

from pathlib import Path
from typing import NotRequired, Optional, TypedDict, Union

# Import from the Rust extension module
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("rcsfs")
    except PackageNotFoundError:
        __version__ = "0.1.0"
except ImportError:
    __version__ = "0.1.0"

# Re-export the Rust extension
try:
    from rcsfs._rcsfs import (
        # CSF file conversion
        convert_csfs as _convert_csfs,
        convert_csfs_parallel as _convert_csfs_parallel,
        get_parquet_info as _get_parquet_info,
        csfs_header as _csfs_header,
        CSFProcessor as _CSFProcessor,

        # CSF descriptor generation
        CSFDescriptorGenerator,
        py_j_to_double_j,

        # Batch descriptor generation
        py_generate_descriptors_from_parquet as _generate_descriptors_from_parquet,
        py_read_peel_subshells as _read_peel_subshells,
    )
except ImportError:
    # Fallback for development/testing
    from _rcsfs import (
        # CSF file conversion
        convert_csfs as _convert_csfs,
        convert_csfs_parallel as _convert_csfs_parallel,
        get_parquet_info as _get_parquet_info,
        csfs_header as _csfs_header,
        CSFProcessor as _CSFProcessor,

        # CSF descriptor generation
        CSFDescriptorGenerator,
        py_j_to_double_j,

        # Batch descriptor generation
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


class ParallelConversionStats(ConversionStats):
    """Statistics returned from parallel CSF conversion operations."""
    num_workers: NotRequired[int]


class HeaderInfo(TypedDict):
    """Header information extracted from CSF files."""
    header_lines: int
    file_path: str
    line1: NotRequired[str]
    line2: NotRequired[str]
    line3: NotRequired[str]
    line4: NotRequired[str]
    line5: NotRequired[str]


#///////////////////////////////////////////////////////////////////////////////
# Python Wrapper Functions (with Path support)
#///////////////////////////////////////////////////////////////////////////////

def convert_csfs(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    max_line_len: Optional[int] = 256,
    chunk_size: Optional[int] = 100000
) -> ConversionStats:
    """
    Convert CSF text file to Parquet format.

    Args:
        input_path: Path to input CSF file
        output_path: Path to output Parquet file
        max_line_len: Maximum line length (default: 256)
        chunk_size: Batch processing size (default: 100000)

    Returns:
        Dictionary containing conversion statistics and status
    """
    return _convert_csfs(
        input_path=str(input_path),
        output_path=str(output_path),
        max_line_len=max_line_len,
        chunk_size=chunk_size
    )


def convert_csfs_parallel(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    max_line_len: Optional[int] = 256,
    chunk_size: Optional[int] = 50000,
    num_workers: Optional[int] = None
) -> ParallelConversionStats:
    """
    Convert CSF text file to Parquet format using parallel processing.

    This function is optimized for large-scale data processing and provides
    significant performance improvements over the sequential version.

    Args:
        input_path: Path to input CSF file
        output_path: Path to output Parquet file
        max_line_len: Maximum line length (default: 256)
        chunk_size: Batch processing size (default: 50000, larger than sequential)
        num_workers: Number of worker threads (default: CPU core count)

    Returns:
        Dictionary containing conversion statistics and status

    Features:
        - Multi-threaded parallel processing for large datasets
        - Maintains original CSF order for consistent output
        - Memory-efficient processing with bounded queues
        - Real-time progress monitoring and statistics
        - Automatic worker count optimization
    """
    return _convert_csfs_parallel(
        input_path=str(input_path),
        output_path=str(output_path),
        max_line_len=max_line_len,
        chunk_size=chunk_size,
        num_workers=num_workers
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


def csfs_header(input_path: Union[str, Path]) -> HeaderInfo:
    """
    Extract header information from CSF file.

    Args:
        input_path: Path to CSF file

    Returns:
        Header information from the CSF file
    """
    return _csfs_header(input_path=str(input_path))


def j_to_double_j(j_str: str) -> int:
    """
    Convert a J-value string to its doubled integer representation (2J).

    This is a Python-friendly wrapper around the Rust implementation.

    Args:
        j_str: J value as string, e.g., "3/2", "2", "5/2", "4-"

    Returns:
        Integer value of 2J

    Examples:
        >>> j_to_double_j("3/2")
        3
        >>> j_to_double_j("2")
        4
        >>> j_to_double_j("4-")
        8
    """
    return py_j_to_double_j(j_str)


#///////////////////////////////////////////////////////////////////////////////
# Batch Descriptor Generation Functions
#///////////////////////////////////////////////////////////////////////////////

def read_peel_subshells(
    header_path: Union[str, Path]
) -> list[str]:
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


def generate_descriptors_from_parquet(
    input_parquet: Union[str, Path],
    output_parquet: Union[str, Path],
    peel_subshells: Optional[list[str]] = None,
    header_path: Optional[Union[str, Path]] = None,
) -> DescriptorGenerationStats:
    """
    Generate CSF descriptors from a parquet file and save to a new parquet file.

    This function reads CSF data directly from a parquet file, generates descriptors
    in batch, and writes the results to a new parquet file.

    Output Schema:
        The output parquet file will have descriptor_size columns (col_0, col_1, ..., col_N),
        where each row represents a CSF descriptor with each element in its own column.
        This format is convenient for use with polars and other dataframe libraries.

    Args:
        input_parquet: Path to input parquet file (must have line1, line2, line3, idx columns)
        output_parquet: Path to output parquet file for descriptors
        peel_subshells: Optional list of subshell names (auto-detected if None)
        header_path: Optional path to header TOML file for peel_subshells

    Returns:
        Dictionary containing generation statistics:
        - success: Whether generation succeeded
        - input_file: Input parquet file path
        - output_file: Output parquet file path
        - csf_count: Number of CSFs processed
        - descriptor_count: Number of descriptors generated
        - orbital_count: Number of orbitals
        - descriptor_size: Size of each descriptor (3 * orbital_count)
        - error: Error message (only if failed)

    Examples:
        >>> # Auto-detect peel_subshells from header file
        >>> stats = generate_descriptors_from_parquet(
        ...     "csfs_data.parquet",
        ...     "descriptors.parquet"
        ... )
        >>>
        >>> # Read with polars
        >>> import polars as pl
        >>> df = pl.read_parquet("descriptors.parquet")
        >>> # Each descriptor element is in its own column: col_0, col_1, ...
        >>> # You can select specific columns or convert to numpy:
        >>> descriptors = df.to_numpy()  # Shape: (n_csfs, descriptor_size)

        >>> # Manually specify peel_subshells
        >>> stats = generate_descriptors_from_parquet(
        ...     "csfs_data.parquet",
        ...     "descriptors.parquet",
        ...     peel_subshells=['5s', '4d-', '4d', '5p-', '5p', '6s']
        ... )

        >>> # Use specific header file
        >>> stats = generate_descriptors_from_parquet(
        ...     "csfs_data.parquet",
        ...     "descriptors.parquet",
        ...     header_path="custom_header.toml"
        ... )
    """
    return _generate_descriptors_from_parquet(
        input_parquet=str(input_parquet),
        output_parquet=str(output_parquet),
        peel_subshells=peel_subshells,
        header_path=str(header_path) if header_path else None,
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
        self,
        max_line_len: Optional[int] = 256,
        chunk_size: Optional[int] = 30000
    ):
        """
        Create a new CSF processor instance.

        Args:
            max_line_len: Maximum line length (default: 256)
            chunk_size: Batch processing size (default: 30000)
        """
        self._processor = _CSFProcessor(
            max_line_len=max_line_len,
            chunk_size=chunk_size
        )

    @property
    def max_line_len(self) -> int:
        """Get current maximum line length."""
        return self._processor.get_config()['max_line_len']

    @max_line_len.setter
    def max_line_len(self, value: int) -> None:
        """Set maximum line length."""
        if value <= 0:
            raise ValueError("max_line_len must be greater than 0")
        self._processor.set_max_line_len(value)

    @property
    def chunk_size(self) -> int:
        """Get current chunk size."""
        return self._processor.get_config()['chunk_size']

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
        output_path: Union[str, Path]
    ) -> ConversionStats:
        """
        Convert CSF file using sequential processing.

        Args:
            input_path: Path to input CSF file
            output_path: Path to output Parquet file

        Returns:
            Conversion statistics and status
        """
        return self._processor.convert(
            input_path=str(input_path),
            output_path=str(output_path)
        )

    def convert_parallel(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        num_workers: Optional[int] = None
    ) -> ParallelConversionStats:
        """
        Convert CSF file using parallel processing.

        Args:
            input_path: Path to input CSF file
            output_path: Path to output Parquet file
            num_workers: Number of worker threads (default: CPU core count)

        Returns:
            Conversion statistics and status
        """
        return self._processor.convert_parallel(
            input_path=str(input_path),
            output_path=str(output_path),
            num_workers=num_workers
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
    "convert_csfs_parallel",
    "get_parquet_info",
    "csfs_header",
    "CSFProcessor",

    # CSF descriptor generation
    "CSFDescriptorGenerator",
    "j_to_double_j",

    # Batch descriptor generation
    "generate_descriptors_from_parquet",
    "read_peel_subshells",

    # Type definitions
    "ConversionStats",
    "ParallelConversionStats",
    "HeaderInfo",
    "DescriptorGenerationStats",
]


