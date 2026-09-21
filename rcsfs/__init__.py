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
from rcsfs import convert_csfs

# Basic conversion
convert_csfs("input.csf", "output.parquet")

# With custom chunk size (default: 3000000 lines = 1M CSFs)
convert_csfs("input.csf", "output.parquet", chunk_size=6000000)

# Limit to 8 workers for shared servers
convert_csfs("input.csf", "output.parquet", num_workers=8)
```

### CSF Descriptor Generation

```python
from rcsfs import read_peel_subshells, generate_descriptors_from_parquet

# Read peel subshells from header file
peel_subshells = read_peel_subshells("data_header.toml")

# Generate descriptors from parquet file
stats = generate_descriptors_from_parquet(
    "csfs_data.parquet",
    "descriptors.parquet",
    peel_subshells=peel_subshells
)
```

For detailed documentation, see function documentation:
- `convert_csfs()`: CSF file to Parquet conversion
- `generate_descriptors_from_parquet()`: Batch descriptor generation
- `read_peel_subshells()`: Extract peel subshells from header file
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from polars import DataFrame

from ._types import (
    ConversionStats,
    CsfBlockInfo,
    CsfDataStats,
    CsfGenerationStats,
    CsfHeaderData,
    CsfHeaderInfo,
    CsfRestoreStats,
    DescriptorGenerationStats,
    InteractionBlockStats,
    InteractionHamiltonian,
    InteractionMethod,
    InteractionStats,
    ParquetInfo,
    PartitionStats,
)

try:
    __version__ = version("rcsfs")
except PackageNotFoundError:
    __version__ = "1.3.1"

from ._rcsfs import (
    convert_csfs as _convert_csfs,
)
from ._rcsfs import (
    generate_csfs_from_transcript as _generate_csfs_from_transcript,
)
from ._rcsfs import (
    get_parquet_info as _get_parquet_info,
)
from ._rcsfs import (
    partition_csfs as _partition_csfs,
)
from ._rcsfs import (
    py_generate_descriptors_from_parquet as _generate_descriptors_from_parquet,
)
from ._rcsfs import (
    py_read_peel_subshells as _read_peel_subshells,
)
from ._rcsfs import (
    py_restore_csfs_from_descriptors as _restore_csfs_from_descriptors,
)
from ._rcsfs import read_csfs_arrow as _read_csfs_arrow
from ._rcsfs import select_interacting_csfs as _select_interacting_csfs

# ///////////////////////////////////////////////////////////////////////////////
# Python Wrapper Functions (with Path support)
# ///////////////////////////////////////////////////////////////////////////////


def read_csfs(
    input_path: str | Path,
    max_line_len: int | None = 256,
    num_workers: int | None = None,
    *,
    include_block_id: bool = False,
    include_coupling_signature: bool = False,
    strict: bool = True,
) -> tuple[CsfHeaderData, DataFrame]:
    """Read CSF header metadata and data rows directly into memory.

    The first five header lines and ``*`` block separators are omitted. Set
    ``include_block_id=True`` to add the zero-based block identifier as a
    ``UInt32`` column. Set ``include_coupling_signature=True`` to append a
    non-null ``List(Int32)`` column containing coupling ``2J`` values for
    occupied peel subshells, in peel order; its final item is total ``2J``.
    Parity is not encoded, so compare signatures within ``block_id``. This
    option adds parsing and memory cost. With ``strict=True`` (the default), an
    incomplete final three-line CSF is rejected instead of silently dropped.
    Arrow buffers produced by Rust are
    transferred to Polars through the Arrow C Stream interface without a
    Parquet round trip. The returned header dictionary has the same schema as
    the TOML sidecar written by :func:`convert_csfs`. For calculation-quality
    input, require ``header["conversion_stats"]["truncated_count"] == 0``.
    """
    header, arrow_stream = _read_csfs_arrow(
        input_path=str(input_path),
        max_line_len=max_line_len,
        num_workers=num_workers,
        include_block_id=include_block_id,
        include_coupling_signature=include_coupling_signature,
        strict=strict,
    )
    return header, DataFrame(arrow_stream)


def convert_csfs(
    input_path: str | Path,
    output_path: str | Path,
    max_line_len: int | None = 256,
    chunk_size: int | None = 3000000,
    num_workers: int | None = None,
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


def get_parquet_info(input_path: str | Path) -> ParquetInfo:
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
        - compression: Compression method used for the first column chunk
        - created_by: Writer identifier stored in the parquet metadata
    """
    return _get_parquet_info(input_path=str(input_path))


# ///////////////////////////////////////////////////////////////////////////////
# Batch Descriptor Generation Functions
# ///////////////////////////////////////////////////////////////////////////////


def read_peel_subshells(header_path: str | Path) -> list[str]:
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
    input_parquet: str | Path,
    output_parquet: str | Path,
    peel_subshells: list[str],
    num_workers: int | None = None,
    normalize: bool = False,
    compression: str | None = None,
    *,
    descriptor_version: int = 2,
    header_path: str | Path | None = None,
) -> DescriptorGenerationStats:
    """
    Generate CSF descriptors from a parquet file using parallel processing.

    This function is optimized for large-scale descriptor generation (tens of millions
    to billions of CSFs). It uses rayon's work-stealing for automatic load balancing
    with streaming batch processing for low memory usage.

    Output Format:
        - Non-normalized: Parquet with multiple Int32 columns `col_0, col_1, ..., col_N`
        - Normalized: Parquet with multiple Float32 columns `col_0, col_1, ..., col_N`
        - Compression defaults to ZSTD level 3; pass ``compression="none"`` to disable
          (much faster write at the cost of ~3-5x larger files).
        Each column corresponds to one position in the descriptor array.
        This multi-column format is much faster than List column format for large datasets.
        Example: For 3 orbitals (descriptor_size=9), columns are: col_0, col_1, ..., col_8

    Args:
        input_parquet: Path to input parquet file (must have line1, line2, line3, idx columns)
        output_parquet: Path to output Parquet file for descriptors
        peel_subshells: List of subshell names (e.g., ['5s', '4d-', '4d', '5p-', '5p', '6s'])
        num_workers: Number of worker threads (default: CPU core count)
        normalize: Whether to normalize descriptors using per-CSF physics-correct
            denominators (default: False). When True, each descriptor triplet
            [n_i, 2Q_i, 2J_cum,i] is normalized by [g_i, n_i*(g_i-n_i),
            min(prefix_i, 2J_target+suffix_i)] respectively, where 2J_target is
            read from the final coupling value of each individual CSF.
            Normalization is V1-only: it raises if the effective
            ``descriptor_version`` is ``2`` (the default), so pass
            ``descriptor_version=1`` explicitly to normalize.
        descriptor_version: ``2`` (default; four-channel per-subshell:
            occupation, printed 2J, seniority, printed coupling 2K; plus
            global ``total_two_j``/``parity`` columns) or ``1`` (legacy
            dense triplet). V2 uses named columns (``sub{i}_n``,
            ``sub{i}_2j``, ``sub{i}_v``, ``sub{i}_2k``, ``total_two_j``,
            ``parity``) rather than positional ``col_{i}`` columns.
        header_path: Path to the source ``{stem}_header.toml``. When given,
            its SHA-256 is recorded in the output Parquet's key-value
            metadata as ``source_header_sha256``, binding the descriptor
            file to the exact header it was generated from. Auto-detected
            from ``input_parquet`` when omitted.
        compression: Parquet compression codec (default: ``zstd-3``). Accepted values:
            ``"none"``/``"uncompressed"``, ``"snappy"``, ``"zstd"``, ``"zstd-N"``
            (N in 1..=22). Pass ``"none"`` to maximize writer throughput when disk
            space is not a concern.

    Returns:
        Dictionary containing generation statistics:
        - success: Whether generation succeeded
        - input_file: Input parquet file path
        - output_file: Output Parquet file path
        - csf_count: Number of CSFs processed
        - descriptor_count: Number of descriptors generated
        - orbital_count: Number of orbitals
        - descriptor_size: Size of each descriptor (3 * orbital_count for V1,
          4 * orbital_count + 2 for V2)
        - descriptor_version: The version tag written (1 or 2)
        - channels_per_subshell: 3 for V1, 4 for V2

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

        >>> # Read with polars
        >>> import polars as pl
        >>> df = pl.read_parquet("descriptors.parquet")
        >>> # Get all descriptor columns (col_0, col_1, ..., col_N)
        >>> descriptor_cols = [col for col in df.columns if col.startswith("col_")]
        >>> descriptors = df[descriptor_cols].to_numpy()  # Shape: (n_csfs, descriptor_size)

        >>> # With normalization (V1-only; explicit descriptor_version=1 required)
        >>> stats = generate_descriptors_from_parquet(
        ...     "csfs_data.parquet",
        ...     "descriptors_normalized.parquet",
        ...     peel_subshells=['5s', '4d-', '4d', '5p-', '5p', '6s'],
        ...     normalize=True,
        ...     descriptor_version=1,
        ... )

        >>> # With custom worker count for large files
        >>> stats = generate_descriptors_from_parquet(
        ...     "csfs_data.parquet",
        ...     "descriptors.parquet",
        ...     peel_subshells=['5s', '4d-', '4d', '5p-', '5p', '6s'],
        ...     num_workers=8
        ... )

        >>> # Disable compression for maximum writer throughput
        >>> stats = generate_descriptors_from_parquet(
        ...     "csfs_data.parquet",
        ...     "descriptors.parquet",
        ...     peel_subshells=['5s', '4d-', '4d', '5p-', '5p', '6s'],
        ...     compression="none",
        ... )

    Performance Considerations:
        - For medium files (1-10M CSFs): num_workers=4-8
        - For large files (>10M CSFs): num_workers=8+
        - More workers = higher CPU usage, faster processing
        - Rayon automatically handles work stealing for optimal load balancing
        - Uses 65536 rows/batch for better I/CPU balance on multi-core systems
        - ``compression="none"`` removes the writer-side ZSTD bottleneck on
          many-core machines; pair with fast local storage (NVMe) for best results.

    Note:
        This implementation uses streaming batch processing to minimize memory usage:
        1. Read parquet in batches (65536 rows per batch)
        2. Parse CSFs to descriptors in parallel (Rayon work-stealing)
        3. Build column arrays directly (Int32 or Float32 builders per column, no ListArray overhead)
        4. Write batch to Parquet file (compression configurable, default ZSTD level 3)
        5. Repeat until all data processed

        Multi-column format is significantly faster than List column format for billion-scale data.
    """
    return _generate_descriptors_from_parquet(
        input_parquet=str(input_parquet),
        output_file=str(output_parquet),
        peel_subshells=peel_subshells,
        num_workers=num_workers,
        normalize=normalize,
        descriptor_version=descriptor_version,
        header_path=str(header_path) if header_path is not None else None,
        compression=compression,
    )


def restore_csfs_from_descriptors(
    descriptor_parquet: str | Path,
    header_path: str | Path,
    output: str | Path,
    indices: list[int] | None = None,
) -> CsfRestoreStats:
    """
    Restore a CSF text file from a V2 descriptor Parquet file.

    The descriptor Parquet must carry ``descriptor_version=2`` key-value
    metadata (written by :func:`generate_descriptors_from_parquet` with
    ``descriptor_version=2``). ``header_path`` must be the exact
    ``{stem}_header.toml`` the descriptors were generated from; if the
    descriptor file recorded a ``source_header_sha256``, it is verified
    against this file's bytes before anything is written.

    Args:
        descriptor_parquet: Path to a V2 descriptor Parquet file.
        header_path: Path to the source ``{stem}_header.toml``.
        output: Destination CSF text file. Must not alias either input.
        indices: Optional 0-based row indices (in the descriptor file's own
            row order) to restore, in the given order. Restores every row
            when omitted.

    Returns:
        Dictionary with ``success``, ``output_file``, ``record_count`` and
        ``output_bytes``.
    """
    return _restore_csfs_from_descriptors(
        descriptor_parquet=str(descriptor_parquet),
        header_path=str(header_path),
        output=str(output),
        indices=indices,
    )


# ///////////////////////////////////////////////////////////////////////////////
# Zero-First Partition Functions
# ///////////////////////////////////////////////////////////////////////////////


def partition_csfs(
    zero_parquet: str | Path,
    zero_header: str | Path,
    full_parquet: str | Path,
    full_header: str | Path,
    output_csf: str | Path,
) -> PartitionStats:
    """
    Partition CSFs into a zero-order + first-order space per symmetry block.

    For each symmetry block, the zero-order reference CSFs are locked to the
    head of the block and the first-order complement (full-block CSFs not
    present in the zero-order block) is appended after them. This mirrors
    GRASP2018's ``rcsfzerofirst`` Fortran utility, but operates on the Parquet
    representation produced by :func:`convert_csfs`.

    Match semantics: a CSF is identified by exact string equality of its
    three-line record ``(line1, line2, line3)``. The 5-line header is taken
    from the full file (the complete space). Blocks are paired by index; both
    files must report the same block count.

    Args:
        zero_parquet: Path to the zero-order reference Parquet file.
        zero_header: Path to the zero-order ``{stem}_header.toml`` sidecar.
        full_parquet: Path to the complete-list Parquet file.
        full_header: Path to the complete-list ``{stem}_header.toml`` sidecar.
        output_csf: Path to the destination CSF text file.

    Returns:
        Dictionary containing partition statistics (success, paths, block and
        CSF counts, first-order complement count).

    Example:
        >>> from rcsfs import convert_csfs, partition_csfs
        >>> convert_csfs("zero.csf", "zero.parquet")
        >>> convert_csfs("full.csf", "full.parquet")
        >>> stats = partition_csfs(
        ...     "zero.parquet", "zero_header.toml",
        ...     "full.parquet", "full_header.toml",
        ...     "reordered.csf",
        ... )
    """
    return _partition_csfs(
        zero_parquet=str(zero_parquet),
        zero_header=str(zero_header),
        full_parquet=str(full_parquet),
        full_header=str(full_header),
        output_csf=str(output_csf),
    )


# ///////////////////////////////////////////////////////////////////////////////
# Structural Interaction Selection
# ///////////////////////////////////////////////////////////////////////////////


def select_interacting_csfs(
    reference_csf: str | Path,
    candidate_csf: str | Path,
    output_csf: str | Path,
    *,
    hamiltonian: InteractionHamiltonian = "dirac_coulomb",
    method: InteractionMethod = "structural_upper_bound",
    num_workers: int | None = None,
    overwrite: bool = False,
) -> InteractionStats:
    """Select a conservative upper bound of CSFs interacting with a reference.

    ``structural_upper_bound`` is the only supported method. It applies cheap
    structural selection rules but does not implement GRASP's complete angular
    and recoupling algebra. Consequently, this function does **not** reproduce
    the exact ``rcsfinteract90`` result: every returned statistics dictionary
    has ``exact=False``, and selected candidates can include false positives.

    Reference CSFs are written first in each symmetry block. Candidate CSFs
    that exactly duplicate a reference record are skipped; structurally
    admissible remaining candidates are appended in candidate-file order.

    **Orbital basis.** The core header lines must be byte-identical, and the
    reference peel subshell list must be a *prefix* of the candidate peel
    subshell list, in the same order. GRASP's own ``rcsfinteract`` requires the
    two peel lists to be identical; this function deliberately relaxes that to
    a prefix so a reference space can be reused against a candidate space that
    appends further correlation orbitals. The candidate header — including the
    appended subshells — becomes the output header, and the appended subshells
    take part in occupation comparisons. Block counts and each block's
    :math:`J/P` must still match.

    **Distinct inputs.** ``reference_csf`` and ``candidate_csf`` must name
    different files; equal paths, symlinks to a common target, and Unix hard
    links are all rejected. ``output_csf`` must likewise not alias either
    input, so a run cannot overwrite the data it is reading.

    Args:
        reference_csf: Reference-space CSF text file.
        candidate_csf: Candidate-space CSF text file whose peel subshell list
            starts with the reference peel subshell list. Must be a different
            file from ``reference_csf``.
        output_csf: Destination CSF text file. Must not alias either input.
        hamiltonian: ``"dirac_coulomb"`` or ``"dirac_coulomb_breit"``.
            The current structural method records this choice but cannot
            distinguish every exact Coulomb/Breit angular zero.
        method: Must be ``"structural_upper_bound"``.
        num_workers: Optional positive worker count; ``None`` uses the runtime
            default.
        overwrite: Replace an existing output file when ``True``.

    Returns:
        Global and per-block selection statistics. ``exact`` is always
        ``False`` for the currently supported method.

    Raises:
        ValueError: An option is unsupported, the inputs are incompatible or
            not distinct, or ``num_workers`` is zero, negative, or too large
            for this platform.
        TypeError: ``num_workers`` is neither an integer nor ``None``.
        FileExistsError: ``output_csf`` exists and ``overwrite`` is ``False``.
        OSError: Input parsing or file I/O fails.
    """
    return _select_interacting_csfs(
        reference_csf=str(reference_csf),
        candidate_csf=str(candidate_csf),
        output_csf=str(output_csf),
        hamiltonian=hamiltonian,
        method=method,
        num_workers=num_workers,
        overwrite=overwrite,
    )


# ///////////////////////////////////////////////////////////////////////////////
# CSF Generation Functions
# ///////////////////////////////////////////////////////////////////////////////


def generate_csfs_from_transcript(
    transcript: str,
    output_path: str | Path,
    normalize: bool = False,
    threads: int | None = None,
) -> CsfGenerationStats:
    """
    Generate CSFs from an in-memory ``rcsfgenerate.log``-format transcript.

    This is the Rust-side engine behind the interactive ``rcsfs csfsgenerate``
    CLI, which assembles the transcript from the user's answers before
    calling this function; the transcript is never written to disk. Parsing,
    occupation enumeration and CSF generation all happen in Rust and are
    written directly to ``output_path``, which must not already exist.

    Args:
        transcript: ``rcsfgenerate.log``-format text: an orbital-order line,
            a core selector (0-6), one reference configuration per line
            terminated by a blank line or ``*``, an active-orbital line, a
            ``jmin,jmax`` line, an excitation-count line, and a final ``n``
            (list continuation is not supported).
        output_path: Destination CSF text file.

        normalize: Retained for API compatibility; descriptor Parquet export
            is performed by the generation CLI pipeline.
        threads: Optional Rayon thread count (default: all cores).

    Returns:
        Dictionary containing generation statistics and status. On success:
        ``output_file``, ``record_count``, ``block_count``,
        ``unique_occupations``. On failure: ``error``.

    Examples:
        >>> transcript = "\\n".join([
        ...     "* ! Orbital order",
        ...     "3",
        ...     "3d(10,i)4s(2,*)4p(6,*)4d(6,*)",
        ...     "3d(10,*)4s(2,i)4p(6,i)4d(6,*)",
        ...     "",
        ...     "5s,5p,5d,4f",
        ...     "0,12",
        ...     "2",
        ...     "n",
        ... ])
        >>> stats = generate_csfs_from_transcript(transcript, "out.c")
    """
    return _generate_csfs_from_transcript(
        transcript=transcript,
        output_path=str(output_path),
        normalize=normalize,
        threads=threads,
    )


def generate_disk_outputs_from_transcript(
    transcript: str,
    csf_output: str | Path,
    csf_parquet_output: str | Path,
    descriptor_output: str | Path,
    header_output: str | Path,
    scratch_dir: str | Path,
    threads: int | None = None,
) -> CsfGenerationStats:
    """Generate staged CSF and reversible V2 descriptors with bounded memory.

    This lower-level API intentionally writes only to caller-owned staging
    paths. The CLI uses it inside its output transaction before it publishes
    the CSF text, CSF Parquet, header and descriptor artifacts together.
    """
    from ._rcsfs import generate_disk_outputs_from_transcript as native_generate

    return native_generate(
        transcript=transcript,
        csf_output=str(csf_output),
        csf_parquet_output=str(csf_parquet_output),
        descriptor_output=str(descriptor_output),
        header_output=str(header_output),
        scratch_dir=str(scratch_dir),
        threads=threads,
    )


# ///////////////////////////////////////////////////////////////////////////////
# Public API
# ///////////////////////////////////////////////////////////////////////////////

__all__ = [  # noqa: RUF022 - grouped by public API area
    # Version
    "__version__",
    # CSF file conversion
    "read_csfs",
    "convert_csfs",
    "get_parquet_info",
    # Batch descriptor generation
    "generate_descriptors_from_parquet",
    "restore_csfs_from_descriptors",
    "read_peel_subshells",
    # Zero-first partition
    "partition_csfs",
    # Structural interaction selection
    "select_interacting_csfs",
    # CSF generation
    "generate_csfs_from_transcript",
    "generate_disk_outputs_from_transcript",
    # Type definitions
    "CsfHeaderInfo",
    "CsfBlockInfo",
    "CsfDataStats",
    "CsfHeaderData",
    "ConversionStats",
    "ParquetInfo",
    "DescriptorGenerationStats",
    "CsfRestoreStats",
    "PartitionStats",
    "CsfGenerationStats",
    "InteractionHamiltonian",
    "InteractionMethod",
    "InteractionBlockStats",
    "InteractionStats",
]
