#!/usr/bin/env python3
"""
Test Rust implementation against Python reference implementation
"""

import numpy as np
from rcsfs import CSFDescriptorGenerator, j_to_double_j as py_j_to_double_j

# Load reference descriptors
ref_desc = np.load('/Users/yiqin/Documents/PythonProjects/as3_odd4/cv4odd1as3_odd4_desc.npy')
print(f"Reference descriptors shape: {ref_desc.shape}")
print(f"Expected: 87 values per CSF = {87} = 3 * {87//3} orbitals")

# Parse CSF file header to get peel subshells
csf_file = '/Users/yiqin/Documents/PythonProjects/as3_odd4/cv4odd1as3_odd4.c'

with open(csf_file, 'r') as f:
    lines = f.readlines()

# Find peel subshells line
peel_line = None
for i, line in enumerate(lines):
    if 'Peel subshells:' in line:
        peel_line = i + 1
        break

if peel_line:
    peel_subshells_line = lines[peel_line].strip()
    peel_subshells = peel_subshells_line.split()
    print(f"\nPeel subshells ({len(peel_subshells)}): {peel_subshells}")
else:
    raise ValueError("Could not find peel subshells line")

# Find CSF start
csf_start_idx = None
for i, line in enumerate(lines):
    if 'CSF(s):' in line:
        csf_start_idx = i + 1
        break

if not csf_start_idx:
    raise ValueError("Could not find CSF start line")

print(f"\nFirst CSF starts at line {csf_start_idx}")

# Parse first few CSFs
print("\n" + "="*80)
print("Testing Rust implementation against Python reference")
print("="*80)

# Create Rust generator
rust_gen = CSFDescriptorGenerator(peel_subshells)

# Test first few CSFs
num_test = 5
line_idx = csf_start_idx

for test_num in range(num_test):
    # Find the next CSF (skip empty lines and asterisks)
    while line_idx < len(lines) and lines[line_idx].strip() == '':
        line_idx += 1

    if line_idx >= len(lines):
        break

    # Read CSF (3 lines)
    line1 = lines[line_idx].rstrip('\n')
    line_idx += 1

    # Skip asterisk separators
    while line_idx < len(lines) and lines[line_idx].strip() == '*':
        line_idx += 1

    if line_idx >= len(lines):
        line2 = ''
    else:
        line2 = lines[line_idx].rstrip('\n')
        line_idx += 1

    while line_idx < len(lines) and lines[line_idx].strip() == '*':
        line_idx += 1

    if line_idx >= len(lines):
        line3 = ''
    else:
        line3 = lines[line_idx].rstrip('\n')
        line_idx += 1

    print(f"\n--- CSF {test_num + 1} ---")
    print(f"Line 1: {line1}")
    print(f"Line 2: '{line2}'")
    print(f"Line 3: '{line3}'")

    # Rust implementation
    rust_desc = rust_gen.parse_csf(line1, line2, line3)
    print(f"Rust descriptor length: {len(rust_desc)}")

    # Reference descriptor
    ref_data = ref_desc[test_num]
    print(f"Reference descriptor length: {len(ref_data)}")

    # Compare
    if len(rust_desc) != len(ref_data):
        print(f"ERROR: Length mismatch! Rust={len(rust_desc)}, Ref={len(ref_data)}")
        continue

    # Convert rust_desc to numpy for comparison
    rust_arr = np.array(rust_desc, dtype=np.float32)

    # Check if they match
    if np.allclose(rust_arr, ref_data, rtol=1e-5, atol=1e-6):
        print("✅ MATCH!")
    else:
        print("❌ MISMATCH!")
        max_diff = np.max(np.abs(rust_arr - ref_data))
        num_diff = np.sum(~np.isclose(rust_arr, ref_data, rtol=1e-5, atol=1e-6))
        print(f"   Max difference: {max_diff}")
        print(f"   Number of differing values: {num_diff}/{len(ref_data)}")

        # Show first few values
        print(f"\n   First 24 values comparison:")
        print(f"   Rust:  {rust_arr[:24]}")
        print(f"   Ref:   {ref_data[:24]}")
        print(f"   Diff:  {rust_arr[:24] - ref_data[:24]}")

        # Check if it's just trailing values
        for i in range(min(30, len(ref_data))):
            if not np.isclose(rust_arr[i], ref_data[i], rtol=1e-5, atol=1e-6):
                print(f"\n   First mismatch at index {i}: Rust={rust_arr[i]}, Ref={ref_data[i]}")
                break

print("\n" + "="*80)
print("Test complete!")
print("="*80)
