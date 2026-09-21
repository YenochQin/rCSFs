"""Shared public result types for the Python and native API boundaries."""

from typing import Literal, NotRequired, TypedDict

type InteractionHamiltonian = Literal[
    "dirac_coulomb",
    "dirac_coulomb_breit",
]
type InteractionMethod = Literal["structural_upper_bound"]


class CsfHeaderInfo(TypedDict):
    """The five source header lines retained from a CSF file."""

    header_lines: list[str]


class CsfBlockInfo(TypedDict):
    """Symmetry-block metadata extracted from ``*`` separators."""

    block_lengths: list[int]
    block_count: int


class CsfDataStats(TypedDict):
    """In-memory CSF parsing statistics matching the TOML sidecar schema."""

    csf_count: int
    total_lines: int
    truncated_count: int


class CsfHeaderData(TypedDict):
    """Header dictionary matching the TOML written by :func:`convert_csfs`."""

    header_info: CsfHeaderInfo
    block_info: CsfBlockInfo
    conversion_stats: CsfDataStats


class ConversionStats(TypedDict):
    """Statistics returned from CSF conversion operations."""

    success: bool
    input_file: NotRequired[str]
    output_file: NotRequired[str]
    parquet_file: NotRequired[str]
    header_file: NotRequired[str]
    max_line_len: NotRequired[int]
    chunk_size: NotRequired[int]
    error: NotRequired[str]
    total_lines: NotRequired[int]
    csf_count: NotRequired[int]
    truncated_count: NotRequired[int]


class ParquetInfo(TypedDict):
    """Metadata returned for a Parquet file."""

    file_path: str
    file_size: int
    num_rows: int
    num_columns: int
    compression: str
    created_by: str
    key_value_metadata: dict[str, str | None]


class DescriptorGenerationStats(TypedDict):
    """Statistics returned from batch descriptor generation."""

    success: bool
    input_file: str
    output_file: str
    csf_count: int
    descriptor_count: int
    orbital_count: int
    descriptor_size: int
    descriptor_version: int
    channels_per_subshell: int


class CsfRestoreStats(TypedDict):
    """Statistics returned from restoring CSFs out of a V2 descriptor file."""

    success: bool
    output_file: str
    record_count: int
    output_bytes: int


class PartitionStats(TypedDict):
    """Statistics returned from a zero-first partition operation."""

    success: bool
    zero_parquet: NotRequired[str]
    full_parquet: NotRequired[str]
    output_file: NotRequired[str]
    block_count: NotRequired[int]
    zero_csf_count: NotRequired[int]
    full_csf_count: NotRequired[int]
    output_csf_count: NotRequired[int]
    first_order_count: NotRequired[int]
    error: NotRequired[str]


class CsfGenerationStats(TypedDict):
    """Statistics returned from transcript-driven CSF generation."""

    success: bool
    output_file: NotRequired[str]
    descriptor_file: NotRequired[str]
    record_count: NotRequired[int]
    block_count: NotRequired[int]
    unique_occupations: NotRequired[int]
    descriptor_count: NotRequired[int]
    generated_count: NotRequired[int]
    duplicate_count: NotRequired[int]
    csf_bytes: NotRequired[int]
    descriptor_bytes: NotRequired[int]
    error: NotRequired[str]


class InteractionBlockStats(TypedDict):
    """Per-symmetry-block structural interaction statistics."""

    block_index: int
    total_two_j: int
    parity: Literal["+", "-"]
    reference_count: int
    candidate_count: int
    exact_reference_skipped: int
    selected_count: int
    rejected_count: int
    output_count: int


class InteractionStats(TypedDict):
    """Statistics from conservative structural interaction selection."""

    exact: Literal[False]
    hamiltonian: InteractionHamiltonian
    method: InteractionMethod
    reference_file: str
    candidate_file: str
    output_file: str
    block_count: int
    reference_count: int
    candidate_count: int
    exact_reference_skipped: int
    selected_count: int
    rejected_count: int
    output_count: int
    output_bytes: int
    blocks: list[InteractionBlockStats]
