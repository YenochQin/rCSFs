"""
Type stubs for the _rcsfs Rust extension module.

This file provides type hints for the compiled Rust extension module.
"""

from typing import Optional

from typing_extensions import NotRequired, TypedDict

__version__: str

# ///////////////////////////////////////////////////////////////////////////////
# Type Definitions for Return Values
# ///////////////////////////////////////////////////////////////////////////////

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
    num_workers: NotRequired[int]

# ///////////////////////////////////////////////////////////////////////////////
# CSF File Conversion Functions
# ///////////////////////////////////////////////////////////////////////////////

def convert_csfs(
    input_path: str,
    output_path: str,
    max_line_len: Optional[int] = 256,
    chunk_size: Optional[int] = 3000000,
    num_workers: Optional[int] = None,
) -> ConversionStats: ...

def get_parquet_info(input_path: str) -> dict: ...

# ///////////////////////////////////////////////////////////////////////////////
# CSF Descriptor Generation
# ///////////////////////////////////////////////////////////////////////////////

class CSFDescriptorGenerator:
    """
    Generator for CSF descriptor arrays used in ML applications.

    This class parses CSF (Configuration State Function) data and converts
    it into fixed-length numerical descriptor arrays.
    """

    def __init__(self, peel_subshells: list[str]) -> None: ...
    def orbital_count(self) -> int: ...
    def peel_subshells(self) -> list[str]: ...
    def parse_csf(self, line1: str, line2: str, line3: str) -> list[int]: ...
    def parse_csf_from_list(self, csf_lines: list[str]) -> list[int]: ...
    def batch_parse_csfs(self, csf_list: list[list[str]]) -> list[list[int]]: ...
    def get_config(self) -> dict: ...

# ///////////////////////////////////////////////////////////////////////////////
# Batch Descriptor Generation
# ///////////////////////////////////////////////////////////////////////////////

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

def py_generate_descriptors_from_parquet(
    input_parquet: str,
    output_file: str,
    peel_subshells: list[str],
    num_workers: Optional[int] = None,
) -> DescriptorGenerationStats: ...

def py_read_peel_subshells(header_path: str) -> list[str]: ...

# ///////////////////////////////////////////////////////////////////////////////
# CSF Processor Class
# ///////////////////////////////////////////////////////////////////////////////

class CSFProcessor:
    """
    Low-level CSF file processor from the Rust extension.

    This uses parallel processing by default with rayon.
    """

    def __init__(
        self, max_line_len: Optional[int] = 256, chunk_size: Optional[int] = 3000000
    ) -> None: ...
    def set_max_line_len(self, value: int) -> None: ...
    def set_chunk_size(self, value: int) -> None: ...
    def get_config(self) -> dict: ...
    def convert(
        self, input_path: str, output_path: str, num_workers: Optional[int] = None
    ) -> ConversionStats: ...
    def get_metadata(self, input_path: str) -> dict: ...
