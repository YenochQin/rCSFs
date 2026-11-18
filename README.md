# rCSFs

[中文说明](README.zh-CN.md)

High-performance Rust-powered tools for working with atomic-physics CSF (Configuration State Function) data in Python.

rCSFs focuses on two jobs:

1. Convert large CSF text files into Parquet for downstream analysis.
2. Generate fixed-width descriptor tables for machine learning workflows.

The current Python API is function-based and centered around streaming, parallel processing, and Parquet-first data pipelines.

## Descriptor CSF datasets

rCSFs 3.1.0 generates and processes CSFs as V2 descriptor Parquet with a
companion TOML header. The compact Int8 columns preserve occupations, physical
seniority, intermediate coupling, total 2J and parity. The header records the
orbital tables, electron count and explicit block lengths. Parquet metadata binds
the dataset to the exact header bytes by SHA-256.

The Python wheel includes the `rcsfs` package and native CLI. See
[development](docs/development.md) for building from this checkout and
[release preparation](docs/releasing.md) for version and packaging details.

## Generate a dataset

Create an editable TOML template from the calculation directory.

```fish
rcsfs csfsgenerate init-config
```

For example, save this model in `rcsfs.toml`.

```toml
conf = "helium"

[csfsgenerate]
inactive_core = 0
reference_configuration = ["1s(2,*)"]
active_space = "2s"
j_min = 0
j_max = 0
excitations = 2
```

```fish
rcsfs -c rcsfs.toml
rcsfs info helium_descriptors.parquet --json
```

This writes `helium_descriptors.parquet` and `helium_header.toml`. Use
`output` in TOML or `--output` on the CLI to choose the descriptor filename.
An `as` level with `conf` names the pair `{conf}_as{level}raw_descriptors.parquet`
and `{conf}_as{level}raw_header.toml`. Use `--transcript` to read a generation
transcript. Disk generation is the default. Use `--estimate-only` to see the
planned record count and space requirement.

## Import, transform and restore

```fish
rcsfs convert reference.c reference_descriptors.parquet
rcsfs zero-first reference_descriptors.parquet helium_descriptors.parquet ordered_descriptors.parquet --json
rcsfs interacting reference_descriptors.parquet helium_descriptors.parquet -o filtered_descriptors.parquet --json
rcsfs jsplit ordered_descriptors.parquet --prefix block
rcsfs restore-csfs --descriptors ordered_descriptors.parquet --header ordered_header.toml --output ordered.c
```

`convert` imports GRASP text directly. Keep each descriptor file with its
companion header. Text recovery uses an explicit restore command. The
`csfs-split` command filters one dataset into active spaces, each with its own
descriptor and header. If the source filename already represents the largest
active space, list the other spaces as split targets. Output paths must differ
from the input pair. Run `rcsfs COMMAND --help` for flags.

`zero-first` and `interacting` read and write V2 descriptor/header pairs directly.
`zero-first` writes the reference rows first in each explicit symmetry block,
followed by the full-basis rows absent from that reference block.
The output header records the number of leading reference rows in
`[block_info].zero_order_counts`, one integer per block in file order.
`interacting` writes the reference rows followed by the non-reference
candidates that pass the occupation-based structural upper bound. This bound
may retain candidates that full angular selection would discard.

## Python

```python
from pathlib import Path

from rcsfs import (
    convert_csfs,
    descriptor_header_path,
    read_csfs,
    restore_csfs_from_descriptors,
    select_csfs,
)

basis = Path("basis_descriptors.parquet")
convert_csfs(Path("basis.c"), basis)
header, rows = read_csfs(basis, include_block_id=True)
selected = Path("selected_descriptors.parquet")
select_csfs(basis, selected, [0])
restore_csfs_from_descriptors(selected, descriptor_header_path(selected), Path("selected.c"))
```

`read_csfs` returns header data and numerical rows with zero-based indices.
Selection and partial restoration accept unique increasing source indices.
Other operations include `merge_csfs`, `match_csfs`, `partition_csfs`,
`split_csfs_by_active_spaces`, `split_csfs_by_j`, and
`select_interacting_csfs`. For native generation, call
`generate_csfs_from_transcript(transcript, output_path)`.

## Development

See [the V2 descriptor guide](docs/CSF_DESCRIPTOR_GUIDE.md)
for the physical channels and [the changelog](docs/CHANGELOG.md) for changes
since 3.0.1. rCSFs is distributed under the [MIT license](LICENSE).

## Direct CSFG generation

`rcsfs csfggenerate --config csfg.toml --threads 8 --json` generates
GRASPG-compatible `rcsfg.out` and `rlabel.out` directly in Rust. The first
version supports a common continuous S/SD correlation space with an explicit
labeling space. It does not first write a full CSF list. Existing outputs require
`--overwrite`.
See the [configuration, Python interface and limits](docs/CSFG_GENERATION.md)
and the [Gd compatibility/performance report](docs/benchmarks/csfg_gd_20261030.md).

Source organization: [shared kernel and separate CSF/CSFG workflows](docs/GENERATION_SOURCE_LAYOUT.md).
