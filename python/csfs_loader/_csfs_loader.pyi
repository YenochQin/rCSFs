from typing import Any, Dict, Optional, Union
from pathlib import Path
from typing_extensions import NotRequired, TypedDict

__version__: str

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

def convert_csfs(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    max_line_len: Optional[int] = 256,
    chunk_size: Optional[int] = 100000
) -> ConversionStats: ...

def csfs_header(input_path: Union[str, Path]) -> HeaderInfo: ...
