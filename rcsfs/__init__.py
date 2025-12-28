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

# With custom chunk size (default: 3000000 lines = 1M CSFs)
convert_csfs_parallel("input.csf", "output.parquet", chunk_size=6000000)
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
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("rcsfs")
    except PackageNotFoundError:
        __version__ = "0.1.0"
except ImportError:
    __version__ = "0.1.0"

# Import from the Rust extension module
from ._rcsfs import (
    # CSF file conversion
    convert_csfs as _convert_csfs,
    convert_csfs_parallel as _convert_csfs_parallel,
    get_parquet_info as _get_parquet_info,
    CSFProcessor as _CSFProcessor,

    # CSF descriptor generation
    CSFDescriptorGenerator,

    # Batch descriptor generation
    py_generate_descriptors_from_parquet as _generate_descriptors_from_parquet,
    py_generate_descriptors_from_parquet_parallel as _generate_descriptors_from_parquet_parallel,
    py_read_peel_subshells as _read_peel_subshells,
)

# Optional: Polars export function (only available if built with polars feature)
try:
    from ._rcsfs import (
        py_export_descriptors_with_polars as _export_descriptors_with_polars,
        py_export_descriptors_with_polars_parallel as _export_descriptors_with_polars_parallel,
    )
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False


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
    pass  # Inherits all fields from ConversionStats


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
    chunk_size: Optional[int] = 3000000,
) -> ParallelConversionStats:
    """
    Convert CSF text file to Parquet format using parallel processing.

    This function is optimized for large-scale data processing. It uses a streaming
    approach with rayon-based parallel processing:
    - Stream: Read file in batches to avoid loading 34GB+ into memory
    - Parallel: Process each batch with rayon's work-stealing (all cores used)
    - Order: Maintain CSF order in output

    Args:
        input_path: Path to input CSF file
        output_path: Path to output Parquet file
        max_line_len: Maximum line length (default: 256)
        chunk_size: Number of lines per read batch (default: 3000000)

    Returns:
        Dictionary containing conversion statistics and status

    Note:
        Rayon automatically uses all CPU cores. To control thread count, set the
        RAYON_NUM_THREADS environment variable.
    """
    return _convert_csfs_parallel(
        input_path=str(input_path),
        output_path=str(output_path),
        max_line_len=max_line_len,
        chunk_size=chunk_size,
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


class ParallelDescriptorGenerationStats(DescriptorGenerationStats):
    """Statistics returned from parallel descriptor generation operations."""
    pass  # Inherits all fields from DescriptorGenerationStats


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


def generate_descriptors_from_parquet_parallel(
    input_parquet: Union[str, Path],
    output_parquet: Union[str, Path],
    peel_subshells: list[str],
    num_workers: Optional[int] = None,
) -> ParallelDescriptorGenerationStats:
    """
    Generate CSF descriptors from a parquet file using parallel processing.

    This function is optimized for large-scale descriptor generation (tens of millions
    to billions of CSFs). It uses rayon's work-stealing for automatic load balancing.

    Output Schema:
        The output parquet file will have descriptor_size columns (col_0, col_1, ..., col_N),
        where each row represents a CSF descriptor with each element in its own column.
        This format is convenient for use with polars and other dataframe libraries.

    Args:
        input_parquet: Path to input parquet file (must have line1, line2, line3, idx columns)
        output_parquet: Path to output parquet file for descriptors
        peel_subshells: List of subshell names (e.g., ['5s', '4d-', '4d', '5p-', '5p', '6s'])
        num_workers: Number of worker threads (default: CPU core count)

    Returns:
        Dictionary containing generation statistics:
        - success: Whether generation succeeded
        - input_file: Input parquet file path
        - output_file: Output parquet file path
        - csf_count: Number of CSFs processed
        - descriptor_count: Number of descriptors generated
        - orbital_count: Number of orbitals
        - descriptor_size: Size of each descriptor (3 * orbital_count)

    Examples:
        >>> # Basic usage with auto-detected peel_subshells
        >>> from rcsfs import read_peel_subshells, generate_descriptors_from_parquet_parallel
        >>>
        >>> peel_subshells = read_peel_subshells("data_header.toml")
        >>> stats = generate_descriptors_from_parquet_parallel(
        ...     "csfs_data.parquet",
        ...     "descriptors.parquet",
        ...     peel_subshells=peel_subshells
        ... )

        >>> # Read with polars
        >>> import polars as pl
        >>> df = pl.read_parquet("descriptors.parquet")
        >>> descriptors = df.to_numpy()  # Shape: (n_csfs, descriptor_size)

        >>> # With custom worker count for large files
        >>> stats = generate_descriptors_from_parquet_parallel(
        ...     "csfs_data.parquet",
        ...     "descriptors.parquet",
        ...     peel_subshells=['5s', '4d-', '4d', '5p-', '5p', '6s'],
        ...     num_workers=8
        ... )

    Performance Considerations:
        - For small files (<1M CSFs): use generate_descriptors_from_parquet() instead
        - For medium files (1-10M CSFs): num_workers=4-8
        - For large files (>10M CSFs): num_workers=8+
        - More workers = higher CPU usage, faster processing
        - Rayon automatically handles work stealing for optimal load balancing

    Note:
        This implementation loads all data into memory upfront. For 280MB parquet files
        (typical for 100M+ CSFs), this fits easily in 256GB RAM. The use of rayon's
        work-stealing ensures optimal CPU utilization without manual task sizing.
    """
    return _generate_descriptors_from_parquet_parallel(
        input_parquet=str(input_parquet),
        output_parquet=str(output_parquet),
        peel_subshells=peel_subshells,
        num_workers=num_workers,
    )


def export_descriptors_with_polars(
    input_parquet: Union[str, Path],
    output_parquet: Union[str, Path],
    peel_subshells: Optional[list[str]] = None,
    header_path: Optional[Union[str, Path]] = None,
) -> DescriptorGenerationStats:
    """
    Export CSF descriptors to Parquet using Polars DataFrame (experimental).

    This is an alternative implementation that uses Polars DataFrame for descriptor
    generation and export. It provides similar functionality to generate_descriptors_from_parquet
    but uses Polars' LazyFrame API for data processing.

    **Note:** This function is only available if the package was built with the
    `polars` feature enabled. Use `rcsfs._POLARS_AVAILABLE` to check availability.

    Args:
        input_parquet: Path to input parquet file (must have line1, line2, line3 columns)
        output_parquet: Path to output parquet file for descriptors
        peel_subshells: Optional list of subshell names (auto-detected if None)
        header_path: Optional path to header TOML file for peel_subshells

    Returns:
        Dictionary containing generation statistics

    Raises:
        ImportError: If the package was not built with the `polars` feature

    Examples:
        >>> # Check if Polars export is available
        >>> import rcsfs
        >>> if rcsfs._POLARS_AVAILABLE:
        ...     stats = rcsfs.export_descriptors_with_polars(
        ...         "csfs_data.parquet",
        ...         "descriptors.parquet"
        ...     )
        ... else:
        ...     # Fallback to standard Arrow-based export
        ...     stats = rcsfs.generate_descriptors_from_parquet(
        ...         "csfs_data.parquet",
        ...         "descriptors.parquet"
        ...     )
    """
    if not _POLARS_AVAILABLE:
        raise ImportError(
            "Polars export is not available. The package was built without the 'polars' feature. "
            "Rebuild with: maturin develop --features polars"
        )
    return _export_descriptors_with_polars(
        input_parquet=str(input_parquet),
        output_parquet=str(output_parquet),
        peel_subshells=peel_subshells,
        header_path=str(header_path) if header_path else None,
    )



def export_descriptors_with_polars_parallel(
    input_parquet: Union[str, Path],
    output_parquet: Union[str, Path],
    peel_subshells: list[str],
    num_workers: Optional[int] = None,
) -> ParallelDescriptorGenerationStats:
    """
    Export CSF descriptors to Parquet using Polars DataFrame with parallel processing (experimental).

    This is a parallel version of export_descriptors_with_polars that uses rayon for
    multi-threaded CSF parsing. It is optimized for large-scale descriptor generation.

    **Note:** This function is only available if the package was built with the
    `polars` feature enabled. Use `rcsfs._POLARS_AVAILABLE` to check availability.

    Args:
        input_parquet: Path to input parquet file (must have line1, line2, line3 columns)
        output_parquet: Path to output parquet file for descriptors
        peel_subshells: List of subshell names (e.g., ["5s", "4d-", "4d", "5p-", "5p", "6s"])
        num_workers: Optional number of worker threads (default: CPU core count)

    Returns:
        Dictionary containing generation statistics

    Raises:
        ImportError: If the package was not built with the `polars` feature

    Examples:
        >>> # Check if Polars export is available
        >>> import rcsfs
        >>> if rcsfs._POLARS_AVAILABLE:
        ...     stats = rcsfs.export_descriptors_with_polars_parallel(
        ...         "csfs_data.parquet",
        ...         "descriptors.parquet",
        ...         peel_subshells=["5s", "4d-", "4d", "5p-", "5p", "6s"],
        ...         num_workers=8
        ...     )
        ... else:
        ...     # Fallback to standard Arrow-based parallel export
        ...     stats = rcsfs.generate_descriptors_from_parquet_parallel(
        ...         "csfs_data.parquet",
        ...         "descriptors.parquet",
        ...         peel_subshells=["5s", "4d-", "4d", "5p-", "5p", "6s"],
        ...         num_workers=8
        ...     )

    Performance Considerations:
        - For small files (<1M CSFs): use export_descriptors_with_polars() instead
        - For medium files (1-10M CSFs): num_workers=4-8
        - For large files (>10M CSFs): num_workers=8+
        - More workers = higher CPU usage, faster processing
    """
    if not _POLARS_AVAILABLE:
        raise ImportError(
            "Polars export is not available. The package was built without the 'polars' feature. "
            "Rebuild with: maturin develop --features polars"
        )
    return _export_descriptors_with_polars_parallel(
        input_parquet=str(input_parquet),
        output_parquet=str(output_parquet),
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
        self,
        max_line_len: Optional[int] = 256,
        chunk_size: Optional[int] = 3000000
    ):
        """
        Create a new CSF processor instance.

        Args:
            max_line_len: Maximum line length (default: 256)
            chunk_size: Batch processing size (default: 3000000)
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
    ) -> ParallelConversionStats:
        """
        Convert CSF file using parallel processing.

        Args:
            input_path: Path to input CSF file
            output_path: Path to output Parquet file

        Returns:
            Conversion statistics and status
        """
        return self._processor.convert_parallel(
            input_path=str(input_path),
            output_path=str(output_path),
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

    # Feature flags
    "_POLARS_AVAILABLE",

    # CSF file conversion
    "convert_csfs",
    "convert_csfs_parallel",
    "get_parquet_info",
    "CSFProcessor",

    # CSF descriptor generation
    "CSFDescriptorGenerator",

    # Batch descriptor generation
    "generate_descriptors_from_parquet",
    "generate_descriptors_from_parquet_parallel",
    "export_descriptors_with_polars",  # Experimental Polars-based export
    "export_descriptors_with_polars_parallel",  # Experimental Polars-based parallel export
    "read_peel_subshells",

    # Type definitions
    "ConversionStats",
    "ParallelConversionStats",
    "DescriptorGenerationStats",
    "ParallelDescriptorGenerationStats",
]


