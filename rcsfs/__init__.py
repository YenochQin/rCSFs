"""Descriptor-native CSF generation and processing.

The primary artifact is an Int8 V3 Parquet plus its byte-bound header TOML.
Use convert_csfs for explicit text import and restore_csfs_from_descriptors
for explicit text export. String-Parquet inputs and implicit conversion are
not supported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from polars import DataFrame

from . import _rcsfs as _native
from ._csfg import generate_csfgs
from ._types import (
    CsfDatasetSplitStats,
    CsfDatasetStats,
    CsfGenerationEstimate,
    CsfGenerationStats,
    CsfgGenerationStats,
    CsfHeaderData,
    CsfRestoreStats,
    InteractionHamiltonian,
    InteractionMethod,
    InteractionStats,
    ParquetInfo,
    PartitionStats,
)

__version__ = _native.__version__
DESCRIPTOR_VERSION = _native.DESCRIPTOR_VERSION


def descriptor_header_path(path: str | Path) -> Path:
    """Return the standard companion path; only descriptor Parquet is accepted."""
    value = Path(path)
    if value.suffix != ".parquet":
        raise ValueError(f"expected descriptor Parquet: {value}")
    stem = value.stem
    if stem.endswith("_desc"):
        stem = stem.removesuffix("_desc")
    else:
        stem = stem.removesuffix("_descriptors")
    return value.with_name(stem + "_header.toml")


def convert_csfs(
    input_path: str | Path,
    output_path: str | Path,
    *,
    header_path: str | Path | None = None,
) -> CsfDatasetStats:
    """Import text CSFs directly to a descriptor/header pair without truncation."""
    return _native.convert_csfs(
        str(input_path),
        str(output_path),
        None if header_path is None else str(header_path),
    )


def read_csfs(
    input_path: str | Path,
    *,
    header_path: str | Path | None = None,
    include_block_id: bool = False,
    include_coupling_signature: bool = False,
) -> tuple[CsfHeaderData, DataFrame]:
    """Read numeric CSFs with a zero-based UInt64 idx column.

    Raw V3 columns store occupation, physical seniority, subshell 2J+1,
    cumulative 2K+1, total 2J+1 and parity. Optional coupling_signature is
    List(Int32), ending in physical total 2J.
    Text input must first be imported with convert_csfs.
    """
    header, stream = _native.read_csfs_arrow(
        str(input_path),
        None if header_path is None else str(header_path),
        include_block_id,
        include_coupling_signature,
    )
    return header, DataFrame(stream)


def select_csfs(
    input_path: str | Path,
    output_path: str | Path,
    indices: Sequence[int],
    *,
    header_path: str | Path | None = None,
    output_header: str | Path | None = None,
) -> CsfDatasetStats:
    """Write a descriptor subset from unique increasing source row indices."""
    return _native.select_csfs(
        str(input_path),
        str(output_path),
        list(indices),
        None if header_path is None else str(header_path),
        None if output_header is None else str(output_header),
    )


def merge_csfs(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    header_paths: Sequence[str | Path] | None = None,
    output_header: str | Path | None = None,
) -> CsfDatasetStats:
    """Append bases with stable orbital mapping, preserving explicit blocks."""
    return _native.merge_csfs(
        [str(p) for p in input_paths],
        str(output_path),
        None if header_paths is None else [str(p) for p in header_paths],
        None if output_header is None else str(output_header),
    )


def match_csfs(
    full_path: str | Path,
    selected_path: str | Path,
    *,
    full_header: str | Path | None = None,
    selected_header: str | Path | None = None,
    require_all: bool = True,
) -> list[int | None]:
    """Return a full-pool index per selected row; reject ambiguous duplicates."""
    return _native.match_csfs(
        str(full_path),
        str(selected_path),
        None if full_header is None else str(full_header),
        None if selected_header is None else str(selected_header),
        require_all,
    )


def restore_csfs_from_descriptors(
    descriptor_parquet: str | Path,
    header_path: str | Path,
    output: str | Path,
    indices: Sequence[int] | None = None,
) -> CsfRestoreStats:
    """Export canonical GRASP text, including explicit closed-core records.

    Selection indices, when supplied, must be unique and increasing. Source
    block boundaries remain distinct even for adjacent equal J/parity blocks.
    """
    return _native.restore_csfs_from_descriptors(
        str(descriptor_parquet),
        str(header_path),
        str(output),
        None if indices is None else list(indices),
    )


def get_parquet_info(input_path: str | Path) -> ParquetInfo:
    return _native.get_parquet_info(str(input_path))


def read_zero_order_counts(
    input_path: str | Path, *, header_path: str | Path | None = None
) -> list[int] | None:
    """Validate the descriptor schema and bound header without loading CSF rows.

    Return one count per explicit block, or None when the header has no partition.
    """
    return _native.read_zero_order_counts(
        str(input_path), None if header_path is None else str(header_path)
    )


def read_peel_subshells(header_path: str | Path) -> list[str]:
    return _native.read_peel_subshells(str(header_path))


def seniority_to_printed_v(
    two_j: int,
    occupation: int,
    subshell_two_j: int,
    seniority: int,
) -> int:
    return _native.seniority_to_printed_v(two_j, occupation, subshell_two_j, seniority)


def partition_csfs(
    zero_parquet: str | Path,
    zero_header: str | Path,
    full_parquet: str | Path,
    full_header: str | Path,
    output_path: str | Path,
) -> PartitionStats:
    """Write reference-first descriptor blocks using exact numeric matching."""
    return _native.partition_csfs(
        str(zero_parquet),
        str(zero_header),
        str(full_parquet),
        str(full_header),
        str(output_path),
    )


def split_csfs_by_active_spaces(
    input_parquet: str | Path,
    header_path: str | Path,
    targets: Mapping[str | Path, str],
) -> CsfDatasetSplitStats:
    """Filter numerical occupations into descriptor datasets for active spaces."""
    return _native.split_csfs_by_active_spaces(
        str(input_parquet),
        str(header_path),
        [(str(path), space) for path, space in targets.items()],
    )


def split_csfs_by_j(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
    *,
    header_path: str | Path | None = None,
) -> CsfDatasetSplitStats:
    """Split each explicit block; filenames include block index, 2J and parity."""
    return _native.split_csfs_by_j(
        str(input_path),
        None if output_dir is None else str(output_dir),
        prefix,
        None if header_path is None else str(header_path),
    )


def generate_csfs_from_transcript(
    transcript: str | Sequence[str],
    output_path: str | Path,
    threads: int | None = None,
    *,
    storage: Literal["disk", "memory"] = "disk",
    memory_budget_mib: int | None = None,
    allow_unchecked_space: bool = False,
) -> CsfGenerationStats:
    """Generate descriptor/header output; no text or string-Parquet intermediate.

    Multiple transcripts form a deduplicated common basis in disk mode.
    A managed disk memory budget is not a process RSS cap.
    """
    return _native.generate_csfs_from_transcript(
        transcript if isinstance(transcript, str) else list(transcript),
        str(output_path),
        threads,
        storage,
        memory_budget_mib,
        allow_unchecked_space,
    )


def estimate_disk_generation(
    transcript: str | Sequence[str],
    threads: int | None = None,
    *,
    memory_budget_mib: int | None = None,
    scratch_dir: str | Path | None = None,
    staging_dir: str | Path | None = None,
    destinations: Mapping[str, str | Path] | None = None,
) -> CsfGenerationEstimate:
    """Estimate the numerical generation pipeline without writing CSFs."""
    return _native.estimate_disk_generation(
        transcript if isinstance(transcript, str) else list(transcript),
        threads,
        memory_budget_mib,
        None if scratch_dir is None else str(scratch_dir),
        None if staging_dir is None else str(staging_dir),
        None
        if destinations is None
        else [(kind, str(path)) for kind, path in destinations.items()],
    )


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
    """Select a conservative interaction upper bound directly from descriptors."""
    return _native.select_interacting_csfs(
        str(reference_csf),
        str(candidate_csf),
        str(output_csf),
        hamiltonian=hamiltonian,
        method=method,
        num_workers=num_workers,
        overwrite=overwrite,
    )


__all__ = [
    "DESCRIPTOR_VERSION",
    "CsfDatasetSplitStats",
    "CsfDatasetStats",
    "CsfGenerationEstimate",
    "CsfGenerationStats",
    "CsfHeaderData",
    "CsfRestoreStats",
    "CsfgGenerationStats",
    "InteractionHamiltonian",
    "InteractionMethod",
    "InteractionStats",
    "ParquetInfo",
    "PartitionStats",
    "__version__",
    "convert_csfs",
    "descriptor_header_path",
    "estimate_disk_generation",
    "generate_csfgs",
    "generate_csfs_from_transcript",
    "get_parquet_info",
    "match_csfs",
    "merge_csfs",
    "partition_csfs",
    "read_csfs",
    "read_peel_subshells",
    "read_zero_order_counts",
    "restore_csfs_from_descriptors",
    "select_csfs",
    "select_interacting_csfs",
    "seniority_to_printed_v",
    "split_csfs_by_active_spaces",
    "split_csfs_by_j",
]
