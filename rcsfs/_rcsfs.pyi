"""Type declarations for the native :mod:`rcsfs._rcsfs` extension."""

from typing import Protocol

from ._types import (
    ConversionStats,
    CsfGenerationStats,
    CsfHeaderData,
    DescriptorGenerationStats,
    ParquetInfo,
    PartitionStats,
)

__version__: str

class ArrowRecordBatchReader(Protocol):
    """Arrow C Stream producer consumed directly by Polars."""

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...

def read_csfs_arrow(
    input_path: str,
    max_line_len: int | None = None,
    num_workers: int | None = None,
    include_block_id: bool = False,
    include_coupling_signature: bool = False,
    strict: bool = True,
) -> tuple[CsfHeaderData, ArrowRecordBatchReader]: ...
def convert_csfs(
    input_path: str,
    output_path: str,
    max_line_len: int | None = None,
    chunk_size: int | None = None,
    num_workers: int | None = None,
) -> ConversionStats: ...
def get_parquet_info(input_path: str) -> ParquetInfo: ...
def partition_csfs(
    zero_parquet: str,
    zero_header: str,
    full_parquet: str,
    full_header: str,
    output_csf: str,
) -> PartitionStats: ...
def py_generate_descriptors_from_parquet(
    input_parquet: str,
    output_file: str,
    peel_subshells: list[str],
    num_workers: int | None = None,
    normalize: bool = False,
    compression: str | None = None,
) -> DescriptorGenerationStats: ...
def py_read_peel_subshells(header_path: str) -> list[str]: ...
def generate_csfs_from_transcript(
    transcript: str,
    output_path: str,
    descriptor_path: str | None = None,
    normalize: bool = False,
    threads: int | None = None,
) -> CsfGenerationStats: ...
