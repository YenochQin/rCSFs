"""Shared public result types for the Python and native API boundaries."""

from typing import NotRequired, TypedDict


class ConversionStats(TypedDict):
    """Statistics returned from CSF conversion operations."""

    success: bool
    input_file: NotRequired[str]
    output_file: NotRequired[str]
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


class DescriptorGenerationStats(TypedDict):
    """Statistics returned from batch descriptor generation."""

    success: bool
    input_file: str
    output_file: str
    csf_count: int
    descriptor_count: int
    orbital_count: int
    descriptor_size: int


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
