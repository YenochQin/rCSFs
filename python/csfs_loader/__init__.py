from pathlib import Path
from typing import NotRequired, Optional, TypedDict

from . import _csfs_loader

__all__ = '__version__', 'convert_csfs', 'csfs_header'

# Define the same types as in .pyi file
class ConversionStats(TypedDict):
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

class HeaderInfo(TypedDict):
    header_lines: int
    file_path: str
    line1: NotRequired[str]
    line2: NotRequired[str]
    line3: NotRequired[str]
    line4: NotRequired[str]
    line5: NotRequired[str]

# Try to get version, fallback if not available
try:
    __version__ = _csfs_loader.__version__
    VERSION = _csfs_loader.__version__
except AttributeError:
    __version__ = "0.1.0"
    VERSION = "0.1.0"

def convert_csfs(
    input_path: str | Path,
    output_path: str | Path,
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
    return _csfs_loader.convert_csfs(
        input_path=input_path,
        output_path=output_path,
        max_line_len=max_line_len,
        chunk_size=chunk_size
    )

def csfs_header(input_path: str | Path) -> HeaderInfo:
    """
    Extract header information from CSF file.

    Args:
        input_path: Path to CSF file

    Returns:
        Header information from the CSF file
    """
    return _csfs_loader.csfs_header(input_path=input_path)