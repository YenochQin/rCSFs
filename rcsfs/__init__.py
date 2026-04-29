"""Rust-powered CSF conversion and descriptor generation.

``rcsfs`` provides the performance-critical file conversion layer used by
GraspKit-Tools. It converts GRASP CSF text files into Parquet tables and
generates machine-learning descriptor Parquet files from those tables.

Typical workflow:
    1. Convert a GRASP ``.c``/CSF text file with :func:`convert_csfs`.
    2. Read peel subshells from the generated header TOML with
       :func:`read_peel_subshells`.
    3. Generate descriptor columns with :func:`generate_descriptors_from_parquet`.

Example:
    >>> from rcsfs import convert_csfs, read_peel_subshells
    >>> from rcsfs import generate_descriptors_from_parquet
    >>> stats = convert_csfs("input.csf", "csfs.parquet", num_workers=8)
    >>> peel = read_peel_subshells(stats["header_file"])
    >>> generate_descriptors_from_parquet("csfs.parquet", "desc.parquet", peel)
"""

from pathlib import Path
from typing import NotRequired, Optional, TypedDict, Union

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rcsfs")
except PackageNotFoundError:
    __version__ = "1.2.2"

from ._rcsfs import (
    convert_csfs as _convert_csfs,
)
from ._rcsfs import (
    get_parquet_info as _get_parquet_info,
)
from ._rcsfs import (
    py_generate_descriptors_from_parquet as _generate_descriptors_from_parquet,
)
from ._rcsfs import (
    py_read_peel_subshells as _read_peel_subshells,
)

# ///////////////////////////////////////////////////////////////////////////////
# Type Definitions
# ///////////////////////////////////////////////////////////////////////////////


class ConversionStats(TypedDict):
    """Statistics returned by :func:`convert_csfs`.

    Attributes:
        success: Whether conversion completed successfully.
        input_file: Input CSF file path.
        output_file: Output Parquet file path.
        header_file: Generated TOML header file path, when available.
        max_line_len: Maximum line length used while reading CSF records.
        chunk_size: Number of input lines processed per streaming batch.
        error: Error message when ``success`` is false.
        total_lines: Number of input lines processed.
        csf_count: Number of CSF records written.
        truncated_count: Number of lines truncated to ``max_line_len``.
    """

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
    """Statistics returned by :func:`generate_descriptors_from_parquet`.

    Attributes:
        success: Whether descriptor generation completed successfully.
        input_file: Input CSF Parquet file path.
        output_file: Output descriptor Parquet file path.
        csf_count: Number of CSFs read from the input file.
        descriptor_count: Number of descriptor rows written.
        orbital_count: Number of peel subshells used.
        descriptor_size: Descriptor width, equal to ``3 * orbital_count``.
        error: Error message when ``success`` is false.
    """

    success: bool
    input_file: str
    output_file: str
    csf_count: NotRequired[int]
    descriptor_count: NotRequired[int]
    orbital_count: NotRequired[int]
    descriptor_size: NotRequired[int]
    error: NotRequired[str]


# ///////////////////////////////////////////////////////////////////////////////
# Python Wrapper Functions (with Path support)
# ///////////////////////////////////////////////////////////////////////////////


def convert_csfs(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    max_line_len: Optional[int] = 256,
    chunk_size: Optional[int] = 3000000,
    num_workers: Optional[int] = None,
) -> ConversionStats:
    """Convert a GRASP CSF text file to Parquet.

    This function is optimized for large-scale data processing. It uses a streaming
    approach with Rayon-based parallel processing while preserving CSF order in
    the output file.

    Args:
        input_path: Path to the input CSF file.
        output_path: Path where the CSF Parquet file should be written.
        max_line_len: Maximum input line length. Longer lines are truncated.
        chunk_size: Number of input lines per streaming batch. The default
            ``3_000_000`` corresponds to roughly one million three-line CSFs.
        num_workers: Worker thread count. Use ``None`` to let Rayon use the
            available CPU cores.

    Returns:
        Conversion statistics and generated file paths.

    Examples:
        >>> stats = convert_csfs("input.csf", "output.parquet")
        >>> stats["success"]
        True
        >>> convert_csfs("input.csf", "output.parquet", num_workers=8)
    """
    return _convert_csfs(
        input_path=str(input_path),
        output_path=str(output_path),
        max_line_len=max_line_len,
        chunk_size=chunk_size,
        num_workers=num_workers,
    )


def get_parquet_info(input_path: Union[str, Path]) -> dict:
    """Return basic metadata for a Parquet file.

    Args:
        input_path: Path to the Parquet file.

    Returns:
        Dictionary with file path, file size, row count, column count,
        compression, and writer metadata.
    """
    return _get_parquet_info(input_path=str(input_path))


# ///////////////////////////////////////////////////////////////////////////////
# Batch Descriptor Generation Functions
# ///////////////////////////////////////////////////////////////////////////////


def read_peel_subshells(header_path: Union[str, Path]) -> list[str]:
    """Read peel subshell names from a generated header TOML file.

    Args:
        header_path: Path to the ``*_header.toml`` file produced by
            :func:`convert_csfs`.

    Returns:
        Subshell names in the order used by descriptor generation.

    Examples:
        >>> peel_subshells = read_peel_subshells("data_header.toml")
        >>> isinstance(peel_subshells, list)
        True
    """
    return _read_peel_subshells(str(header_path))


def generate_descriptors_from_parquet(
    input_parquet: Union[str, Path],
    output_parquet: Union[str, Path],
    peel_subshells: list[str],
    num_workers: Optional[int] = None,
    normalize: bool = False,
) -> DescriptorGenerationStats:
    """Generate descriptor Parquet columns from converted CSF Parquet data.

    This function is optimized for large-scale descriptor generation (tens of millions
    to billions of CSFs). It uses rayon's work-stealing for automatic load balancing
    with streaming batch processing for low memory usage.

    Output format:
        The output contains one column per descriptor position:
        ``col_0``, ``col_1``, ..., ``col_N``. Non-normalized descriptors are
        written as ``Int32`` columns; normalized descriptors are written as
        ``Float32`` columns. Files use ZSTD compression.

    Args:
        input_parquet: Converted CSF Parquet file with ``line1``, ``line2``,
            ``line3``, and ``idx`` columns.
        output_parquet: Destination descriptor Parquet file.
        peel_subshells: Ordered subshell names, usually from
            :func:`read_peel_subshells`.
        num_workers: Worker thread count. Use ``None`` for Rayon defaults.
        normalize: Whether to normalize descriptors using per-CSF physics-correct
            denominators. When true, descriptor triplets
            ``[n_i, 2Q_i, 2J_cum,i]`` are normalized by
            ``[g_i, n_i * (g_i - n_i), min(prefix_i, 2J_target + suffix_i)]``.

    Returns:
        Descriptor generation statistics and output file metadata.

    Examples:
        >>> # Basic usage with peel_subshells from header
        >>> from rcsfs import read_peel_subshells, generate_descriptors_from_parquet
        >>>
        >>> peel_subshells = read_peel_subshells("data_header.toml")
        >>> stats = generate_descriptors_from_parquet(
        ...     "csfs_data.parquet",
        ...     "descriptors.parquet",
        ...     peel_subshells=peel_subshells
        ... )

        >>> import polars as pl
        >>> df = pl.read_parquet("descriptors.parquet")
        >>> descriptor_cols = [c for c in df.columns if c.startswith("col_")]
        >>> descriptors = df[descriptor_cols].to_numpy()

        >>> # With normalization
        >>> stats = generate_descriptors_from_parquet(
        ...     "csfs_data.parquet",
        ...     "descriptors_normalized.parquet",
        ...     peel_subshells=['5s', '4d-', '4d', '5p-', '5p', '6s'],
        ...     normalize=True,
        ... )

        >>> # With custom worker count for large files
        >>> stats = generate_descriptors_from_parquet(
        ...     "csfs_data.parquet",
        ...     "descriptors.parquet",
        ...     peel_subshells=['5s', '4d-', '4d', '5p-', '5p', '6s'],
        ...     num_workers=8
        ... )

    Note:
        The multi-column format is intentionally used instead of a List column
        because it is faster for large descriptor matrices and easier to scan
        lazily from Polars.
    """
    return _generate_descriptors_from_parquet(
        input_parquet=str(input_parquet),
        output_file=str(output_parquet),
        peel_subshells=peel_subshells,
        num_workers=num_workers,
        normalize=normalize,
    )


# ///////////////////////////////////////////////////////////////////////////////
# Public API
# ///////////////////////////////////////////////////////////////////////////////

__all__ = [
    # Version
    "__version__",
    # CSF file conversion
    "convert_csfs",
    "get_parquet_info",
    # Batch descriptor generation
    "generate_descriptors_from_parquet",
    "read_peel_subshells",
    # Type definitions
    "ConversionStats",
    "DescriptorGenerationStats",
]
