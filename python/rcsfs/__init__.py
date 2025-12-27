"""
rCSFs - Rust-powered CSF (Configuration State Function) Processing Library

This package provides high-performance Rust implementations for processing
atomic physics CSF data.

Main Components:
- CSF file format conversion to Parquet
- CSF descriptor generation for ML applications
"""

# Import from the Rust extension module
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("rcsfs")
    except PackageNotFoundError:
        __version__ = "0.1.0-dev"
except ImportError:
    __version__ = "0.1.0-dev"

# Re-export the Rust extension
try:
    from rcsfs._rcsfs import (
        # CSF file conversion
        convert_csfs,
        convert_csfs_parallel,
        get_parquet_info,
        csfs_header,
        CSFProcessor,

        # CSF descriptor generation
        CSFDescriptorGenerator,
        py_j_to_double_j,
    )
except ImportError:
    # Fallback for development/testing
    from _rcsfs import (
        # CSF file conversion
        convert_csfs,
        convert_csfs_parallel,
        get_parquet_info,
        csfs_header,
        CSFProcessor,

        # CSF descriptor generation
        CSFDescriptorGenerator,
        py_j_to_double_j,
    )

# Define public API
__all__ = [
    # Version
    "__version__",

    # CSF file conversion
    "convert_csfs",
    "convert_csfs_parallel",
    "get_parquet_info",
    "csfs_header",
    "CSFProcessor",

    # CSF descriptor generation
    "CSFDescriptorGenerator",
    "j_to_double_j",  # Python-friendly name for py_j_to_double_j
]

# Python-friendly wrapper for j_to_double_j
def j_to_double_j(j_str: str) -> int:
    """
    Convert a J-value string to its doubled integer representation (2J).

    This is a Python-friendly wrapper around the Rust implementation.

    Args:
        j_str: J value as string, e.g., "3/2", "2", "5/2", "4-"

    Returns:
        Integer value of 2J

    Examples:
        >>> j_to_double_j("3/2")
        3
        >>> j_to_double_j("2")
        4
        >>> j_to_double_j("4-")
        8
    """
    return py_j_to_double_j(j_str)


# Module docstring additions
__doc__ += """

## Quick Start

### CSF File Conversion

```python
from rcsfs import CSFProcessor

processor = CSFProcessor()
processor.convert_parallel("input.csf", "output.parquet")
```

### CSF Descriptor Generation

```python
from rcsfs import CSFDescriptorGenerator

# Define orbitals
peel_subshells = ['5s', '4d-', '4d', '5p-', '5p', '6s']
gen = CSFDescriptorGenerator(peel_subshells)

# Parse CSF
line1 = '  5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)'
line2 = '                   3/2               2        '
line3 = '                                           4-  '

descriptor = gen.parse_csf(line1, line2, line3)
# Returns: [2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 6.0, 3.0, 3.0, ...]
```

### J-Value Conversion

```python
from rcsfs import j_to_double_j

j_to_double_j("3/2")  # → 3
j_to_double_j("2")    # → 4
j_to_double_j("4-")   # → 8
```

For detailed documentation, see:
- CSF File Conversion: See CSFProcessor class documentation
- CSF Descriptor Generation: See CSFDescriptorGenerator class documentation
"""
