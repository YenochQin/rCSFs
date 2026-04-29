"""Type stubs for the compiled ``rcsfs._rcsfs`` PyO3 extension."""

from typing import Optional

from typing_extensions import NotRequired, TypedDict

__version__: str

# ///////////////////////////////////////////////////////////////////////////////
# Type Definitions for Return Values
# ///////////////////////////////////////////////////////////////////////////////

class ConversionStats(TypedDict):
    """Statistics returned by ``convert_csfs``."""

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

# ///////////////////////////////////////////////////////////////////////////////
# CSF File Conversion Functions
# ///////////////////////////////////////////////////////////////////////////////

def convert_csfs(
    input_path: str,
    output_path: str,
    max_line_len: Optional[int] = 256,
    chunk_size: Optional[int] = 3000000,
    num_workers: Optional[int] = None,
) -> ConversionStats:
    """Convert a GRASP CSF text file to Parquet."""
    ...

def get_parquet_info(input_path: str) -> dict:
    """Return basic metadata for a Parquet file."""
    ...

# ///////////////////////////////////////////////////////////////////////////////
# CSF Descriptor Generation
# ///////////////////////////////////////////////////////////////////////////////

# ///////////////////////////////////////////////////////////////////////////////
# Batch Descriptor Generation
# ///////////////////////////////////////////////////////////////////////////////

class DescriptorGenerationStats(TypedDict):
    """Statistics returned by descriptor generation."""

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
    normalize: bool = False,
) -> DescriptorGenerationStats:
    """Generate descriptor columns from a converted CSF Parquet file."""
    ...

def py_read_peel_subshells(header_path: str) -> list[str]:
    """Read peel subshell names from a generated header TOML file."""
    ...
