#!/usr/bin/env python3
"""
CSF Descriptor Generator Usage Example

This script demonstrates how to use the rCSFs library to generate
CSF (Configuration State Function) descriptors from CSF data.

Author: YenochQin
Date: 2025-12-26
"""

import numpy as np
from _rcsfs import CSFDescriptorGenerator, py_j_to_double_j


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def example_j_value_conversion():
    """Example 1: Convert J-value strings to 2J integers."""
    print_section("Example 1: J-Value Conversion")

    test_cases = [
        "3/2",   # Fractional: 3/2 -> 3
        "5/2",   # Fractional: 5/2 -> 5
        "2",     # Integer: 2 -> 4
        "4",     # Integer: 4 -> 8
        "4-",    # With parity: 4- -> 8
        "3+",    # With parity: 3+ -> 6
        "",      # Empty: -> 0
    ]

    print("J-value conversions:")
    for j_str in test_cases:
        result = py_j_to_double_j(j_str)
        print(f"  j_to_double_j('{j_str:>4}') = {result}")


def example_single_csf_parsing():
    """Example 2: Parse a single CSF into descriptor."""
    print_section("Example 2: Single CSF Parsing")

    # Define peel subshells (orbital names)
    peel_subshells = ['5s', '4d-', '4d', '5p-', '5p', '6s']

    # Create generator
    generator = CSFDescriptorGenerator(peel_subshells)

    print(f"Orbital count: {generator.orbital_count()}")
    print(f"Peel subshells: {generator.peel_subshells()}")

    # Sample CSF data (3 lines)
    line1 = '  5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)'
    line2 = '                   3/2               2        '
    line3 = '                                           4-  '

    print("\nInput CSF:")
    print(f"  Line 1: {line1}")
    print(f"  Line 2: '{line2}'")
    print(f"  Line 3: '{line3}'")

    # Parse CSF
    descriptor = generator.parse_csf(line1, line2, line3)

    print(f"\nOutput descriptor (length={len(descriptor)}):")
    print(f"  Raw: {descriptor}")

    # Interpret descriptor by orbital
    print("\nInterpreted by orbital:")
    for i, orbital in enumerate(peel_subshells):
        idx = i * 3
        e_count = descriptor[idx]
        middle_j = descriptor[idx + 1]
        coupling_j = descriptor[idx + 2]
        print(f"  {orbital:>4}: e={e_count:.0f}, middle_J={middle_j:.0f}, coupling_J={coupling_j:.0f}")


def example_batch_parsing():
    """Example 3: Batch parse multiple CSFs."""
    print_section("Example 3: Batch CSF Parsing")

    peel_subshells = ['5s', '4d-', '4d', '5p-']
    generator = CSFDescriptorGenerator(peel_subshells)

    # Sample CSF batch
    csf_batch = [
        # CSF 1
        [
            '  5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)',
            '                   3/2      ',
            '                        4-  '
        ],
        # CSF 2
        [
            '  5s ( 1)  4d-( 4)  4d ( 6)  5p-( 2)',
            '                   1/2      ',
            '                        3-  '
        ],
        # CSF 3
        [
            '  5s ( 2)  4d-( 3)  4d ( 6)  5p-( 1)',
            '                   5/2      ',
            '                        5+  '
        ],
    ]

    print(f"Processing {len(csf_batch)} CSFs...")

    # Method 1: Parse individually
    print("\nMethod 1: Individual parsing")
    for i, csf in enumerate(csf_batch, 1):
        desc = generator.parse_csf_from_list(csf)
        print(f"  CSF {i}: {desc[:6]}...")

    # Method 2: Batch parse
    print("\nMethod 2: Batch parsing")
    descriptors = generator.batch_parse_csfs(csf_batch)
    print(f"  Batch result: {len(descriptors)} descriptors")

    # Convert to numpy array for ML applications
    descriptor_matrix = np.array(descriptors, dtype=np.float32)
    print(f"\nNumpy array shape: {descriptor_matrix.shape}")
    print(f"  Expected: ({len(csf_batch)}, 3*{len(peel_subshells)})")


def example_from_parquet():
    """Example 4: Read CSF from Parquet and generate descriptors."""
    print_section("Example 4: Processing Parquet Data")

    # Example: Reading from Parquet file
    # (This assumes you have a Parquet file with line1, line2, line3 columns)

    example_code = '''
import polars as pl
from _rcsfs import CSFDescriptorGenerator

# Read Parquet file
df = pl.read_parquet("csfs_data.parquet")

# Get peel subshells from header or metadata
peel_subshells = ['5s', '4d-', '4d', '5p-', '5p', '6s']
generator = CSFDescriptorGenerator(peel_subshells)

# Method 1: Row-by-row processing
descriptors = []
for row in df.iter_rows(named=True):
    desc = generator.parse_csf(row['line1'], row['line2'], row['line3'])
    descriptors.append(desc)

# Method 2: Using Polars map_elements (faster for large datasets)
def parse_csf_wrapper(struct_val):
    line1, line2, line3 = struct_val
    return generator.parse_csf(line1, line2, line3)

descriptors_df = df.select(
    descriptor = pl.struct(["line1", "line2", "line3"])
        .map_elements(parse_csf_wrapper, return_dtype=pl.List(pl.Float32))
)

# Convert to numpy
import numpy as np
descriptor_matrix = np.array(descriptors_df["descriptor"].to_list())
print(f"Descriptor matrix shape: {descriptor_matrix.shape}")
'''

    print("Example code for Parquet processing:")
    print(example_code)


def example_config():
    """Example 5: Get generator configuration."""
    print_section("Example 5: Generator Configuration")

    peel_subshells = ['5s', '4d-', '4d', '5p-', '5p', '6s', '4f-']
    generator = CSFDescriptorGenerator(peel_subshells)

    config = generator.get_config()
    print("Generator configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("  CSF Descriptor Generator - Usage Examples")
    print("="*60)

    example_j_value_conversion()
    example_single_csf_parsing()
    example_batch_parsing()
    example_from_parquet()
    example_config()

    print("\n" + "="*60)
    print("  All examples completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
