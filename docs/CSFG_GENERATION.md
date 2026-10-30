# Direct GRASPG-compatible CSFG generation

Status: first implementation, 2026-10-30. This is an unreleased local change.
It generates CSFG files directly in Rust; it does not invoke GRASPG or write a
full CSF list as an intermediate. See the [design record](CSFG_GENERATION_PLAN.md)
for the upstream contract and the deferred features. The
[source layout](GENERATION_SOURCE_LAYOUT.md) groups CSF and CSFG workflows
under `src/configuration_generation/` with one shared generation kernel.

## CLI

```sh
rcsfs csfggenerate --config csfg.toml --threads 8 --json
```

The command reads one `[csfggenerate]` table. With `-c` and no filename, or no
config argument, it reads `rcsfs.toml`. Relative output paths are relative to the
invocation directory. Run it in the desired calculation directory. This first
version does not participate in the top-level multi-command batch runner.

The Gd I model used for validation:

```toml
[csfggenerate]
inactive_core = 4
reference_configuration = ["4d(10,c)5s(2,i)5p(6,i)6s(2,i)4f(7,*)5d(1,*)"]
labeling_space = "6s,5p,5d,4f"
active_space = "10s,9p,9d,8f,8g"
j_min = 2
j_max = 12
excitations = 2
rcsfg_out = "rcsfg.out"
rlabel_out = "rlabel.out"
```

`j_min` and `j_max` mean **2J**, so this example requests J=1..6. The original
`c/i/*/minimum` physical occupations remain intact: in particular 6s² stays
inactive. Do not add the artificial low-n inactive-zero restrictions used to
speed up the older Fortran generator. The Rust planner selects representatives
without changing the physical model.

The output names default to `rcsfg.out` and `rlabel.out`. Optional table keys
are `threads` and `overwrite`. CLI `--threads` overrides the table value;
`--overwrite` explicitly allows replacing both completed outputs. Existing
files are otherwise refused. Unknown keys, `as/conf`, lists, descriptors,
Parquet, storage modes and memory-budget options are not accepted by this
first-version command. The existing `csfsgenerate` command is unchanged.

## Python

```python
from pathlib import Path
from rcsfs import generate_csfgs

stats = generate_csfgs(
    inactive_core=4,
    reference_configuration=["4d(10,c)5s(2,i)5p(6,i)6s(2,i)4f(7,*)5d(1,*)"],
    labeling_space="6s,5p,5d,4f",
    active_space="10s,9p,9d,8f,8g",
    j_min=2, j_max=12, excitations=2,
    rcsfg_out=Path("rcsfg.out"), rlabel_out=Path("rlabel.out"),
    threads=8,
)
assert stats["expanded_csf_count"] == 11_299_739
```

Invalid input raises `ValueError`/`TypeError`; an existing output raises
`FileExistsError` unless overwrite was requested. Both outputs are generated
in a private staging directory before publication. Publication is atomic per
file, **not** across the pair. A `PartialPublicationError` reports already
published and failed destinations; it must not be treated as a successful run.
Config/output aliases and output symlinks are rejected.

## Supported model

- One common labeling space and one common continuous radial range per l;
  the two relativistic κ partners are treated separately.
- One or more references with the same filled, inactive closed core and
  electron count. All occupied reference orbitals and physical occupation
  restrictions must be in the labeling space.
- Nonnegative 0/1/2 excitations; at most two electrons in correlation orbitals.
  Pair-only `d` selectors, different VV/CV/CC radial ranges, radial holes and
  exact preselected jj-CSF reference subsets are outside this first version.
- Existing Rust state-table restrictions, n≤15 and at most 20 occupied peel
  subshells apply. The output compatibility profile permits at most 127 total
  subshells and 50 symmetry blocks, matching suite defaults. Consumers may
  have additional limits (for example the interaction tool's block arrays).
- Generation currently retains the representative CSF chunks in memory.
  There is no bounded-memory/disk mode or process-RSS guarantee for CSFG yet.

The planner enumerates physical occupations, maps them to the five representative
types, and verifies exact, non-overlapping occupation coverage before angular
recursion. The original occupation merge comparator and the CSFG coupling
order are kept distinct. Each Jπ block starts with the ordinary labeling-space
CSFs. The header includes the complete correlation ranges, including radial
orbitals not explicitly used in the representatives. `rlabel.out` includes the
closed-core subshells in its count.

## Statistics and verification

`record_count` counts ordinary labeling CSFs **plus** CSFG representatives;
`generator_count` excludes the labeling CSFs. `type_counts` lists types 1..5.
The result also includes per-block counts, `unique_occupations`,
`representative_occupations`, checked `expanded_csf_count`, output sizes and
planning/generation/writing times in seconds. Expanded counts use the radial
multiplicities; they are not a substitute for independent expansion tests.

For Gd I, 7,154 physical occupation configurations reduce to 767 representative
configurations. The output contains 15,961 labeling CSFs and 1,109,626 generators,
for 1,125,587 total records; the expanded space contains 11,299,739 CSFs.
The two files match the GRASPG reference byte-for-byte, and the unmodified
GRASPG expander reproduces the reference expanded file byte-for-byte.
See the [local benchmark report](benchmarks/csfg_gd_20261030.md).
The [upstream script comparison](benchmarks/graspg_script_comparison_20261030.md)
also passed ten distinct generation inputs and 32 direct active-space layers;
models with differing parity-specific J ranges used an explicit test adapter.

These checks establish generation and file compatibility for the tested model.
They do not establish SCF/RCI convergence, accuracy or downstream speedup.
CSFG order differs from ordinary n-ordered CSFs, so previously generated mixing
coefficients must not be paired with the new files solely because counts agree.

Self-contained Rust/Python tests use small stored fixtures, never external
Fortran programs or the Gd dataset. The large comparisons and timing scripts
remain under ignored `temp/`.

## Upstream attribution

The file contract and five representative types follow
[GRASPG](https://github.com/compas/graspg), source revision e157d9f.
See Ran Si et al., [Graspg — An extension to Grasp2018 based on configuration
state function generators](https://doi.org/10.1016/j.cpc.2025.109604),
*Computer Physics Communications* 312, 109604 (2025), and Yanting Li et al.,
[the CSFG method paper](https://doi.org/10.1016/j.cpc.2022.108562),
*Computer Physics Communications* 283, 108562 (2023). Original GRASP license
notices are retained in [GRASP-MIT.txt](../licenses/GRASP-MIT.txt).
