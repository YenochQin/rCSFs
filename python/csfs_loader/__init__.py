from pathlib import Path
from typing import NotRequired, Optional, TypedDict

from . import _csfs_loader

__all__ = '__version__', 'convert_csfs', 'convert_csfs_parallel', 'get_parquet_info', 'csfs_header', 'CSFProcessor'

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

class ParallelConversionStats(ConversionStats):
    num_workers: NotRequired[int]

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
        input_path=str(input_path),
        output_path=str(output_path),
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
    return _csfs_loader.csfs_header(input_path=str(input_path))

def convert_csfs_parallel(
    input_path: str | Path,
    output_path: str | Path,
    max_line_len: Optional[int] = 256,
    chunk_size: Optional[int] = 50000,
    num_workers: Optional[int] = None
) -> ParallelConversionStats:
    """
    Convert CSF text file to Parquet format using parallel processing.

    This function is optimized for large-scale data processing and provides
    significant performance improvements over the sequential version.

    Args:
        input_path: Path to input CSF file
        output_path: Path to output Parquet file
        max_line_len: Maximum line length (default: 256)
        chunk_size: Batch processing size (default: 50000, larger than sequential)
        num_workers: Number of worker threads (default: CPU core count)

    Returns:
        Dictionary containing conversion statistics and status

    Features:
        - Multi-threaded parallel processing for large datasets
        - Maintains original CSF order for consistent output
        - Memory-efficient processing with bounded queues
        - Real-time progress monitoring and statistics
        - Automatic worker count optimization
    """
    return _csfs_loader.convert_csfs_parallel(
        input_path=str(input_path),
        output_path=str(output_path),
        max_line_len=max_line_len,
        chunk_size=chunk_size,
        num_workers=num_workers
    )

def get_parquet_info(input_path: str | Path) -> dict:
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
        - compression: Compression method used
    """
    return _csfs_loader.get_parquet_info(input_path=str(input_path))

# CSF Processor class for object-oriented interface
class CSFProcessor:
    """
    CSF file processor class providing an object-oriented interface.

    This class allows for easy configuration and repeated conversion operations
    with consistent settings.
    """

    def __init__(
        self,
        max_line_len: Optional[int] = 256,
        chunk_size: Optional[int] = 30000
    ):
        """
        Create a new CSF processor instance.

        Args:
            max_line_len: Maximum line length (default: 256)
            chunk_size: Batch processing size (default: 30000)
        """
        self._processor = _csfs_loader.CSFProcessor(
            max_line_len=max_line_len,
            chunk_size=chunk_size
        )

    @property
    def max_line_len(self) -> int:
        """Get current maximum line length."""
        return self._processor.get_config()['max_line_len']

    @max_line_len.setter
    def max_line_len(self, value: int) -> None:
        """Set maximum line length."""
        if value <= 0:
            raise ValueError("max_line_len must be greater than 0")
        self._processor.set_max_line_len(value)

    @property
    def chunk_size(self) -> int:
        """Get current chunk size."""
        return self._processor.get_config()['chunk_size']

    @chunk_size.setter
    def chunk_size(self, value: int) -> None:
        """Set chunk size."""
        if value <= 0:
            raise ValueError("chunk_size must be greater than 0")
        self._processor.set_chunk_size(value)

    def get_config(self) -> dict:
        """
        Get current processor configuration.

        Returns:
            Dictionary containing current settings
        """
        return self._processor.get_config()

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path
    ) -> ConversionStats:
        """
        Convert CSF file using sequential processing.

        Args:
            input_path: Path to input CSF file
            output_path: Path to output Parquet file

        Returns:
            Conversion statistics and status
        """
        return self._processor.convert(
            input_path=str(input_path),
            output_path=str(output_path)
        )

    def convert_parallel(
        self,
        input_path: str | Path,
        output_path: str | Path,
        num_workers: Optional[int] = None
    ) -> ParallelConversionStats:
        """
        Convert CSF file using parallel processing.

        Args:
            input_path: Path to input CSF file
            output_path: Path to output Parquet file
            num_workers: Number of worker threads (default: CPU core count)

        Returns:
            Conversion statistics and status
        """
        return self._processor.convert_parallel(
            input_path=str(input_path),
            output_path=str(output_path),
            num_workers=num_workers
        )

    def get_metadata(self, input_path: str | Path) -> dict:
        """
        Get Parquet file information.

        Args:
            input_path: Path to Parquet file

        Returns:
            File information and metadata
        """
        return self._processor.get_metadata(input_path=str(input_path))