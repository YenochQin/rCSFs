# V2 CSF descriptor guide

Generate V2 descriptors directly with `rcsfs csfsgenerate`. To import an
existing GRASP text file, write a descriptor and its companion header in one step.

```python
from rcsfs import convert_csfs

stats = convert_csfs("calculation.c", "calculation_descriptors.parquet")
header = stats["header_file"]
```

For M peel subshells, a row contains 4M+2 Int8 values:

| Columns | Meaning |
| --- | --- |
| `sub{i}_n` | Occupation of peel subshell i |
| `sub{i}_2j` | Printed doubled subshell J |
| `sub{i}_v` | Physical seniority, including 0 for empty/closed subshells |
| `sub{i}_2k` | Printed doubled intermediate coupling K |
| `total_two_j` | Total doubled J |
| `parity` | +1 or -1 |

Unoccupied slots are `[0, -1, 0, -1]`. Only unprinted J/coupling fields
use `-1`; physical seniority is always nonnegative. Restoration recovers the
canonical printed seniority from `(single-electron 2j, occupation, subshell 2J, v)`.
Parquet metadata requires `seniority_encoding=physical`, `descriptor_dtype=int8`,
and `max_two_j=126` in addition to version 2. Every J/coupling value must be
0..126 or the permitted -1 missing marker; out-of-range input is rejected before
narrowing. The in-memory row and Arrow columns are signed 8-bit integers.
Regenerate older printed-v or Int32 descriptors from the original text or a generation transcript.

Rust descriptor row APIs use `i8` slices/vectors. Counts, offsets and numerical
work variables retain their wider types. Parquet stores the signed 8-bit logical
type using its INT32 physical type; this does not promise a fourfold reduction
in compressed file size. Arrow/scratch numeric payloads use one byte per value,
while Parquet encoder memory allowances still account for physical INT32 buffers.

```python
from rcsfs import seniority_to_printed_v

assert seniority_to_printed_v(7, 4, 8, 2) == 2   # J=4 needs its v=2 label
assert seniority_to_printed_v(7, 4, 8, 4) == 4   # different state with the same J
assert seniority_to_printed_v(7, 4, 12, 2) == -1 # J=6 uniquely has v=2
```

The Rust `decode_v2_into` interface now also requires the ordered Peel labels
to recover the printed seniority policy.

The shared physical-state table preserves the generator's original state order
and text labels. Descriptor generation infers a missing label only when its
physical seniority is determined; insufficient information is an error.

Both memory and disk generation write the V2 descriptor/header pair.
Regenerate V1 files from their original CSFs to recover physical seniority.
Feature scaling belongs in the consuming ML pipeline.

Use `restore_csfs_from_descriptors(descriptors, header, output)` to restore CSF
text. Keep the exact source header alongside each descriptor file.

## Coupling signatures for grouping

```python
from rcsfs import read_csfs

header, frame = read_csfs(
    "calculation_descriptors.parquet",
    include_block_id=True,
    include_coupling_signature=True,
)
```

`coupling_signature` is a non-null `List(Int32)` derived from the validated V2
row. Each occupied peel subshell contributes its printed coupling, falling back
to printed subshell J or zero; the last entry is total doubled J. Unoccupied
positions are omitted. Compare signatures within `block_id` to keep parity
blocks separate. This deliberately groups coupling families and is not a unique
CSF identifier: it omits occupation and seniority. Descriptor generation and
identity comparison use the complete V2 row.

Signature parsing defaults off. When enabled, malformed quantum fields fail
explicitly. Calculation-quality callers should also require
`header["conversion_stats"]["truncated_count"] == 0`.
Derived datasets retain the source truncation count. Operations combining
multiple input datasets sum their reported counts. These counts describe the
input histories, including when a selection keeps only a subset of the rows.

`CSFDescriptorGenerator` is an internal Rust type, not a Python class. See
[the public API](../README.md#public-python-api) for supported Python functions.
