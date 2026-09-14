# Code Review: Generation Parquet Outputs

**Date:** 2026-09-13
**Commit:** `7805db2af6044ae69c533cdce6e08c14064be097` — `rcsfs: use parquet outputs for generation`
**Base:** `2e1a6f04761bf91618617e862421cddf39083747`
**Comparison:** `git diff 2e1a6f0...7805db2`
**Status:** Four open findings; no implementation fixes made during review.

The change replaces optional descriptor CSV export with a CLI pipeline for CSF
and descriptor Parquet files, and adds TOML generation configuration. The working
tree was clean when reviewed. Source line references below refer to the reviewed
commit.

## Behavior findings

### 1. P1 — Every generation run crashes on an obsolete keyword

**Location:** `rcsfs/cli.py:506–511`; `rcsfs/__init__.py:405`.

The CLI still passes `descriptor_path=None` to the public
`generate_csfs_from_transcript` wrapper, but this commit removes that parameter
from the wrapper. Both interactive and valid TOML generation runs therefore fail
before entering the Rust generator:

```text
TypeError: generate_csfs_from_transcript() got an unexpected keyword argument 'descriptor_path'
```

The existing interactive generation regressions reproduce this failure. The
mocked CLI unit test conceals it because its fake still accepts the old keyword.

**Recommended fix:** Remove the obsolete keyword from the CLI call and update
the affected callers and tests. Cover actual interactive and TOML generation
through the public wrapper, including descriptor-enabled runs.

### 2. P1 — Parquet destinations can overwrite existing files

**Location:** `rcsfs/cli.py:515–530`.

The new pipeline calls `convert_csfs` and
`generate_descriptors_from_parquet` without checking whether destinations already
exist or overlap other paths. Their Rust writers use `File::create`, which
truncates existing files. The generation documentation in
`docs/CSF_GENERATION.md` specifies new output paths and extends that protection
to descriptor sidecars.

A temporary-directory probe confirmed that a preexisting `out.parquet` sentinel
was replaced with Parquet data beginning with `PAR1`, and the command returned
success. Output collisions can also destroy an input needed by a later stage.

**Recommended fix:** Validate all destinations before generation, including
CSF text, both Parquet files, and metadata sidecars. Reject existing destinations
and aliases between outputs or the configuration input; preserve exclusive
creation when opening new files so validation cannot race with another writer.

### 3. P2 — The promised descriptor TOML sidecar is missing

**Location:** `rcsfs/cli.py:526–532`.

The updated README promises “descriptor Parquet plus its metadata sidecar.”
The CLI only invokes `generate_descriptors_from_parquet`, and neither that
wrapper nor its Rust implementation writes descriptor TOML metadata. This
commit also deletes the previous descriptor metadata writer.

A successful one-CSF probe produced `out.c`, `out.parquet`, `out_header.toml`,
and `out_descriptors.parquet`, but no descriptor sidecar. Consumers therefore
do not receive the promised descriptor metadata, including the ordered subshell
mapping and normalization state.

**Recommended fix:** Write a versioned descriptor TOML sidecar and document its
schema. Include the ordered subshells, normalization state, record count, and
Parquet encoding. Add coverage that checks the sidecar against the actual output.

## Standards finding

### 4. P2 — The extension signature and type stub disagree

**Location:** `rcsfs/_rcsfs.pyi:53–59`; `src/lib.rs:299–317`.

The stub removes `descriptor_path` and declares positional arguments as
`(transcript, output_path, normalize, threads)`. The actual PyO3 binding retains
`(transcript, output_path, descriptor_path, normalize, threads)`. A caller
following the stub can therefore pass a normalization boolean into the
descriptor-path parameter and encounter a runtime conversion error.

This violates the repository rule in `AGENTS.md`: “Keep package exports aligned
with `rcsfs/_rcsfs.pyi` when that stub is present.” The Python wrapper also still
passes `descriptor_path=None` to this binding, which now contradicts its stub.

**Recommended fix:** Update the binding, stub, wrapper, and Rust callers together
to expose the intended signature. Remove or explicitly reject obsolete descriptor
arguments instead of silently accepting them.

## Validation and evidence limits

| Check | Result |
| --- | --- |
| `uv run python -m pytest -q` | 29 passed, 4 failed |
| `uv run cargo test` | Passed; 3 external-baseline tests ignored |
| `uv run basedpyright rcsfs/` | 33 errors |

Two Python failures are the interactive generation regressions described in
finding 1. The other two are descriptor compatibility tests that still call the
removed public `descriptor_path` parameter. Type-check errors include signature
mismatches and typing problems in the new TOML configuration branch.

Findings 2 and 3 were verified with real generation in a temporary directory,
using an in-memory shim that drops only the obsolete `descriptor_path` keyword
to reach the downstream pipeline. No repository source was changed for the
probe. Its generation settings were:

```toml
[generate]
order = "*"
core = 0
references = ["1s(2,*)"]
active_orbitals = "1s"
j_min = 0
j_max = 0
excitations = 0
continue_lists = false

[output]
generate_descriptors = true
csf = "out.c"
parquet = "out.parquet"
descriptor_parquet = "out_descriptors.parquet"
normalize = false
```

The Rust test result does not cover the three ignored external differential
checks. The downstream probes isolate findings 2 and 3; unmodified CLI execution
is blocked first by finding 1.
