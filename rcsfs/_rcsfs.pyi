"""Type declarations for the native :mod:`rcsfs._rcsfs` extension."""

from typing import Literal, Protocol

from ._types import (
    CsfDatasetSplitStats,
    CsfDatasetStats,
    CsfGenerationEstimate,
    CsfGenerationStats,
    CsfHeaderData,
    CsfRestoreStats,
    InteractionHamiltonian,
    InteractionMethod,
    InteractionStats,
    ParquetInfo,
    PartitionStats,
)

__version__: str
DESCRIPTOR_VERSION: int
MAX_PRINCIPAL_QUANTUM_NUMBER: int

def seniority_to_printed_v(
    two_j: int,
    occupation: int,
    subshell_two_j: int,
    seniority: int,
) -> int: ...

class ArrowRecordBatchReader(Protocol):
    """Arrow C Stream producer consumed directly by Polars."""

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...

# Header metadata includes electron_count (core + peel) when inferable.
def read_csfs_arrow(
    input_path: str,
    header_path: str | None = None,
    include_block_id: bool = False,
    include_coupling_signature: bool = False,
) -> tuple[CsfHeaderData, ArrowRecordBatchReader]: ...
def convert_csfs(
    input_path: str,
    output_path: str,
    header_path: str | None = None,
) -> CsfDatasetStats: ...
def select_csfs(
    input_path: str,
    output_path: str,
    indices: list[int],
    header_path: str | None = None,
    output_header: str | None = None,
) -> CsfDatasetStats: ...
def merge_csfs(
    input_paths: list[str],
    output_path: str,
    header_paths: list[str] | None = None,
    output_header: str | None = None,
) -> CsfDatasetStats: ...
def match_csfs(
    full_path: str,
    selected_path: str,
    full_header: str | None = None,
    selected_header: str | None = None,
    require_all: bool = True,
) -> list[int | None]: ...
def get_parquet_info(input_path: str) -> ParquetInfo: ...
def read_zero_order_counts(
    input_path: str, header_path: str | None = None
) -> list[int] | None: ...
def partition_csfs(
    zero_parquet: str,
    zero_header: str,
    full_parquet: str,
    full_header: str,
    output_path: str,
) -> PartitionStats: ...
def split_csfs_by_active_spaces(
    input_parquet: str,
    header_path: str,
    targets: list[tuple[str, str]],
) -> CsfDatasetSplitStats: ...
def split_csfs_by_j(
    input_path: str,
    output_dir: str | None = None,
    prefix: str | None = None,
    header_path: str | None = None,
) -> CsfDatasetSplitStats: ...
def restore_csfs_from_descriptors(
    descriptor_parquet: str,
    header_path: str,
    output: str,
    indices: list[int] | None = None,
) -> CsfRestoreStats: ...
def read_peel_subshells(header_path: str) -> list[str]: ...
def generate_csfs_from_transcript(
    transcript: str | list[str],
    output_path: str,
    threads: int | None = None,
    storage: Literal["disk", "memory"] = "disk",
    memory_budget_mib: int | None = None,
    allow_unchecked_space: bool = False,
    *,
    scratch_dir: str | None = None,
) -> CsfGenerationStats: ...
def estimate_disk_generation(
    transcript: str | list[str],
    threads: int | None = None,
    memory_budget_mib: int | None = None,
    scratch_dir: str | None = None,
    staging_dir: str | None = None,
    destinations: list[tuple[str, str]] | None = None,  # one entry per artifact kind
) -> CsfGenerationEstimate: ...
def select_interacting_csfs(
    reference_csf: str,
    candidate_csf: str,
    output_csf: str,
    *,
    hamiltonian: InteractionHamiltonian = "dirac_coulomb",
    method: InteractionMethod = "structural_upper_bound",
    num_workers: int | None = None,
    overwrite: bool = False,
) -> InteractionStats: ...
