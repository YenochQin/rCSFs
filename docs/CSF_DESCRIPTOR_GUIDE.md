# V3 CSF descriptor guide

Generate the descriptor and its companion TOML with `rcsfs csfsgenerate`, or
convert an existing GRASP text file with `rcsfs convert`. V3 requires an rCSFs
build with `DESCRIPTOR_VERSION == 3`. Regenerate V1 and V2 data from the original
CSF text or generation inputs.

The [V3 generation plan](CSF_DESCRIPTOR_V3_PLAN.md) records the format design,
validation cases and integration with the other repositories.

```python
from rcsfs import DESCRIPTOR_VERSION, convert_csfs

assert DESCRIPTOR_VERSION == 3
stats = convert_csfs("calculation.c", "calculation_desc.parquet")
```

## Disk layout

Each row stores one ordered jj-CSF as `4M+2` non-null, unscaled Int8 values,
where `M` is the size of the ordered peel table. The four columns for each
zero-based orbital index `i` are followed by the two global columns.

| Column | Meaning |
| --- | --- |
| `sub{i}_occupation` | Subshell occupation, from zero to its capacity |
| `sub{i}_seniority` | Physical seniority, zero in empty and full subshells |
| `sub{i}_two_j_plus_one` | Subshell `2Jᵢ+1`, one in empty and full subshells |
| `sub{i}_cumulative_two_k_plus_one` | `2Kᵢ+1` after coupling through this orbital |
| `total_two_j_plus_one` | Total `2J+1` |
| `parity` | +1 for even, −1 for odd |

The chain begins at `K=0`. Empty and full subshells preserve the preceding K.
The final K equals total J. All angular codes are 1 through 127, corresponding
to physical `2J` or `2K` of 0 through 126. Configuration limits `j_min` and
`j_max` and CLI block summaries still use physical `2J`. A pure closed core with
an empty peel table has one row `[1, 1]`.

Parquet uses the signed 8-bit logical type over its INT32 physical type. The
exact ordered column names are recorded in `feature_columns` and `global_columns`.
Required metadata also includes `descriptor_version=3`,
`channels_per_subshell=4`, `descriptor_dtype=int8`, `seniority_encoding=physical`,
`max_two_j=126`, `normalized=false`, `subshell_count`, and `peel_subshells`.
`source_header_sha256` binds the descriptor to the exact bytes of the TOML. The
TOML holds the original five header lines, the total electron count including
the core, and ordered block lengths. Adjacent blocks with the same symmetry
retain distinct identities.

The supported subshell state tables use occupation, physical seniority and J to
identify a state. This is the ordinary Suite domain with additional state label
`w=0`. Validation checks occupation capacity, allowed subshell states, parity,
electron count, every coupling triangle and the final total J.
In the supported catalog, the seniority-4 terms of a j=9/2 shell at subshell
J=4 and J=6 each have multiplicity two. V3 rejects both terms for four
electrons and four holes because the row has no additional term label.

## Text conversion and restoration

An explicit printed zero constrains the state or coupling. For a blank field,
conversion uses the allowed states, printed couplings and total J to recover a
unique complete chain. Ambiguous input reports the block, row and subshell.
Restoration writes the canonical state and coupling fields from the numeric row.
The same CSF and V3 row survive reimport, including when the source text printed
a normally omitted value. Legacy GRASP text tools with four-character angular
fields cannot read a five-character half-integer such as `105/2`. Keep that
case in the numeric dataset for consumers that support the full V3 range.

```python
from rcsfs import descriptor_header_path, restore_csfs_from_descriptors

basis = "calculation_desc.parquet"
restore_csfs_from_descriptors(basis, descriptor_header_path(basis), "selected.c", indices=[0])
```

## Python grouping and data selection

```python
from rcsfs import read_csfs, select_csfs

header, frame = read_csfs(
    "calculation_desc.parquet",
    include_block_id=True,
    include_coupling_signature=True,
)
select_csfs("calculation_desc.parquet", "selected_desc.parquet", [0])
```

`idx` is the zero-based source ordinal. The optional `coupling_signature`
column is `List(Int32)`. At each occupied peel position it takes the canonical
printed intermediate `2K`, the printed subshell `2J`, or zero, in that order.
The last entry is physical total `2J`. A pure closed core has signature `[0]`.
Use `block_id` when grouping signatures and use complete V3 rows on a common
ordered basis or original ordinals for exact selection.

Selection and splitting preserve source block order and update the bound header.
For partitioned inputs, selection and active-space filtering count the surviving
zero-order prefix rows per block. A retained block can have a zero count.
Whole-block splits preserve the original count. Merge preserves counts when each
input provides them. Zero-first builds counts from the reference for each block.

## Rebuild the acceptance sample

Run this from the checkout's `rCSFs` directory with its V3 CLI on `PATH`.

```fish
set -l rcsfs_source (pwd)
mkdir -p ../temp/tests/descriptor-v3-example
cd ../temp/tests/descriptor-v3-example
rcsfs convert "$rcsfs_source/tests/fixtures/descriptor_v3.c" basis_desc.parquet
rcsfs info basis_desc.parquet --json
```

The sample lists `1s`, `2s`, `3s` and `4s` in the peel table. Each orbital
tuple gives occupation, seniority, `2Jᵢ+1` and `2Kᵢ+1`, in that order.

| Row | 1s | 2s | 3s | 4s | Total `2J+1` | Parity |
| --- | --- | --- | --- | --- | ---: | ---: |
| `1s² 2s¹ 3s¹ 4s¹`, `K=0` at `3s` | (2,0,1,1) | (1,1,2,2) | (1,1,2,1) | (1,1,2,2) | 2 | +1 |
| `1s² 2s¹ 3s¹ 4s¹`, `K=1` at `3s` | (2,0,1,1) | (1,1,2,2) | (1,1,2,3) | (1,1,2,2) | 2 | +1 |
| `1s² 2s² 3s¹ 4s⁰` | (2,0,1,1) | (2,0,1,1) | (1,1,2,2) | (0,0,1,2) | 2 | +1 |

The explicit block lengths are `[2, 1]`.
