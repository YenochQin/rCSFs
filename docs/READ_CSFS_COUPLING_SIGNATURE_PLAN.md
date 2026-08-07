# `read_csfs` Coupling Signature — Implementation Plan

- **Date**: 2026-08-07
- **Status**: Proposed
- **Scope**: rCSFs only; no GraspKit implementation changes in this plan

---

## 1. Background

`rcsfs.read_csfs()` already reads a GRASP CSF file directly into a Polars
`DataFrame` through the Arrow C Stream interface:

```text
idx: UInt64
block_id: UInt32       # optional
line1: String
line2: String
line3: String
```

GraspKit's current coupling collector consumes legacy nested Python records and
extracts a pattern with:

```python
tuple(csf[2].lstrip().split())
```

That expression only tokenizes the raw third CSF line. It does not recover the
full fixed-width coupling structure, and it does not produce converted LS/term
labels such as `2P1` or `3D2`.

The complex fixture `tests/fixtures/sample.csf` demonstrates the limitation:

- 28 CSFs in one block;
- 26 raw `line3` values tokenize as `["7/2", "4-"]`;
- 2 raw `line3` values tokenize as `["4-"]`;
- all 28 rows collapse to the same pattern when only the final raw token is
  retained;
- additional intermediate and coupling information is stored by fixed column
  position across `line1`, `line2`, and `line3`.

rCSFs already implements the required fixed-width rules in
`CSFDescriptorGenerator::parse_csf_into()` in `src/csfs_descriptor.rs`:

- each subshell field occupies 9 ASCII characters;
- `line1` supplies the subshell and electron count;
- `line2` supplies the middle J value;
- when a `line2` field contains `;`, the value after the semicolon is used;
- `line3` supplies the coupling J value for the aligned subshell;
- an empty `line3` field falls back to the parsed middle J;
- the final occupied subshell uses the total J from the end of `line3`;
- J values are represented as integer `2J` values.

The new feature should reuse that implementation and expose its coupling result
as an optional Polars column returned by `read_csfs()`.

---

## 2. Goal

Extend the existing `read_csfs()` interface with one keyword-only switch:

```python
header, csfs_df = read_csfs(
    input_path,
    num_workers=8,
    include_block_id=True,
    include_coupling_signature=True,
)
```

When enabled, the returned frame includes:

```text
coupling_signature: List[Int32]
```

The implementation must parse each CSF once in Rust, reuse the established
fixed-width rules, and transfer the derived Arrow column to Polars in the same
Arrow C Stream as the raw CSF columns.

When disabled, behavior, schema, errors, row ordering, and the current fast path
must remain unchanged.

---

## 3. Confirmed Interface Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Python parameter | `include_coupling_signature: bool = False` | Matches `include_block_id`; the name clearly expresses an optional output column. |
| Native binding parameter | Same name and default | Keeps the Python wrapper and PyO3 seam aligned. |
| Output column | `coupling_signature` | Describes a derived equality/grouping key without implying converted LS labels. |
| Arrow/Polars dtype | non-null `List<Int32>` with non-null items | Stable schema across files with different peel-subshell counts; directly groupable and sliceable in Polars. |
| Numeric representation | integer `2J` | Reuses descriptor semantics and avoids equivalent strings such as `4` and `8/2` becoming different groups. |
| Included positions | occupied peel subshells only, in CSF/peel order | Removes absent padding without dropping a physically valid occupied `J=0` value. Occupancy must be determined from the parsed electron-count field, not from `coupling_2J != 0`. |
| Total J | final signature item | Matches the descriptor parser's last-subshell rule. |
| Parity | not encoded in the list | Coupling signatures are compared within a `block_id`; a GRASP block fixes total J and parity. Cross-block comparison must group by `block_id` or add an explicit parity column in a future change. |
| Raw columns | retained unchanged | The feature annotates the frame; it does not replace `line1`/`line2`/`line3`. |
| Default compatibility | exact current schema | Existing callers must not receive an extra column unless they opt in. |

### 3.1 Exact signature invariant

For every returned CSF row, let the shared parser produce triplets:

```text
[n_electrons_i, middle_2J_i, coupling_2J_i]
```

for peel-subshell positions `i = 0..N`. The new column is:

```text
[coupling_2J_i for i in 0..N if n_electrons_i > 0]
```

Important consequences:

- a zero coupling value is retained when its subshell is occupied;
- unoccupied/missing peel positions are omitted;
- the list is ordered and can be used as a Polars grouping key;
- `coupling_level = k` maps to the final `k` entries:

  ```python
  pl.col("coupling_signature").list.slice(-k)
  ```

### 3.2 Column ordering

The schemas must be:

```text
# default
idx, line1, line2, line3

# block IDs only
idx, block_id, line1, line2, line3

# coupling signature only
idx, line1, line2, line3, coupling_signature

# both options
idx, block_id, line1, line2, line3, coupling_signature
```

---

## 4. Non-goals

This change does not:

- generate LS/term-symbol labels;
- parse `line3` with whitespace splitting as a substitute for fixed-width
  parsing;
- change `convert_csfs()` or the canonical CSF Parquet schema;
- add coupling columns to descriptor Parquet output;
- change incomplete-final-CSF, truncation, non-ASCII, or block-separator
  behavior;
- add a parity column or define cross-block coupling equality;
- modify GraspKit's `CSFs_processor` in the same change;
- remove the raw three-line CSF representation.

---

## 5. Architecture and Data Flow

```text
CSF text
  |
  |-- read five header lines
  |      `-- parse peel subshells once
  |
  |-- read/validate/truncate data lines using existing behavior
  |
  |-- group each ordered line triple as one CSF
  |      |
  |      |-- append idx / optional block_id / raw strings
  |      `-- optional shared fixed-width parser
  |             `-- derive occupied coupling_2J list
  |
  `-- one Arrow RecordBatch
          `-- Arrow C Stream -> Polars DataFrame
```

`read_csfs()` remains a deep module: callers learn one optional parameter and
one column contract, while header parsing, fixed-width alignment, J conversion,
fallback rules, Rayon execution, and Arrow construction remain inside rCSFs.

---

## 6. Module Changes

| File | Planned responsibility |
|---|---|
| `src/csfs_descriptor.rs` | Extract reusable peel-header parsing and coupling-signature derivation from the existing descriptor implementation without changing descriptor results. |
| `src/csfs_memory.rs` | Accept the option, initialize the parser only when requested, derive signatures in Rust, and append one Arrow `List<Int32>` column. |
| `src/lib.rs` | Add the keyword to the PyO3 function and pass it to `read_csfs_to_record_batch()`. |
| `rcsfs/__init__.py` | Add the public keyword, document the conditional column, and forward it to the native binding. |
| `rcsfs/_rcsfs.pyi` | Update the native binding declaration. |
| `tests/read_csfs_test.py` | Add Python-facing schema, dtype, fixture, compatibility, and option-combination tests. |
| `src/csfs_descriptor.rs` tests | Lock down shared parser equivalence and signature extraction edge cases. |
| `docs/CSF_DESCRIPTOR_GUIDE.md` | Document the new in-memory annotation after implementation. |
| `docs/CHANGELOG.md` | Record the backward-compatible feature after implementation. |

No new Rust or Python dependencies are required.

---

## 7. Implementation Tasks

### Task 1: Lock down behavior with tests

- [ ] Add a Python test proving the default `read_csfs()` schema remains
  `idx`, `line1`, `line2`, `line3`.
- [ ] Add a test for all four combinations of `include_block_id` and
  `include_coupling_signature`.
- [ ] Add focused Rust tests for coupling signature extraction from descriptor
  triplets, including occupied `J=0` and unoccupied positions.
- [ ] Add parser regression tests for semicolon handling, missing `line3`
  fallback, final-J replacement, and short right-padded lines.
- [ ] Use `tests/fixtures/sample.csf` as the complex integration fixture and
  assert:
  - 28 rows are returned;
  - `coupling_signature` is `List(Int32)`;
  - no signature is null;
  - every signature is non-empty;
  - every signature's final `2J` is `8`, corresponding to total `J=4`;
  - row order and global `idx` remain unchanged.

### Task 2: Deepen the shared fixed-width parser

- [ ] Extract a helper that parses peel subshell names directly from
  `HeaderData.header_info.header_lines` (or a borrowed string slice).
- [ ] Make the existing TOML-based `read_peel_subshells_from_header()` call the
  same helper after loading `header_lines`; do not keep two parsing rules.
- [ ] Add an allocation-conscious internal method that produces a coupling
  signature from the same triplets as `parse_csf_into()`.
- [ ] Preserve all existing descriptor behavior and tests.
- [ ] Do not create a second fixed-width parser in `csfs_memory.rs`.

Suggested internal shape (names are provisional):

```rust
pub(crate) fn parse_peel_subshells_from_header_lines(
    header_lines: &[String],
) -> Result<Vec<String>>;

impl CSFDescriptorGenerator {
    pub(crate) fn parse_coupling_signature_into(
        &self,
        line1: &str,
        line2: &str,
        line3: &str,
        descriptor_buffer: &mut [i32],
        signature_buffer: &mut Vec<i32>,
    ) -> Result<()>;
}
```

The implementation may use `parse_csf_into()` internally and extract every
third value where the preceding electron count is positive. Optimize only
after equivalence tests are green.

### Task 3: Extend the Rust in-memory reader

- [ ] Extend the interface:

  ```rust
  pub fn read_csfs_to_record_batch(
      csfs_path: &Path,
      max_line_len: usize,
      num_workers: Option<usize>,
      include_block_id: bool,
      include_coupling_signature: bool,
  ) -> Result<(HeaderData, RecordBatch), Box<dyn Error + Send + Sync>>;
  ```

- [ ] Keep the current path unchanged when the option is false: do not parse
  peel subshells, initialize a descriptor generator, allocate descriptor
  buffers, or create a list builder.
- [ ] When true, parse peel subshells once from the already-loaded header.
- [ ] Parse signatures from the post-validation/post-truncation strings that
  are actually returned in `line1`/`line2`/`line3`.
- [ ] Preserve indexed Rayon ordering. If signature calculation is parallel,
  use an indexed iterator and caller/thread-owned reusable buffers.
- [ ] Build a non-null Arrow `List<Int32>` column named
  `coupling_signature`.
- [ ] Append the column after `line3` and verify Arrow field/column order always
  matches.
- [ ] Propagate parser failures as read failures; do not silently emit null or
  partial signatures.

### Task 4: Extend the PyO3 and Python interfaces

- [ ] Change `src/lib.rs` to:

  ```rust
  #[pyo3(signature = (
      input_path,
      max_line_len=None,
      num_workers=None,
      include_block_id=false,
      include_coupling_signature=false
  ))]
  ```

- [ ] Forward the new value to `read_csfs_to_record_batch()`.
- [ ] Add the same defaulted argument to `rcsfs/_rcsfs.pyi`.
- [ ] Add the public keyword-only argument to `rcsfs.read_csfs()`:

  ```python
  def read_csfs(
      input_path: str | Path,
      max_line_len: int | None = 256,
      num_workers: int | None = None,
      *,
      include_block_id: bool = False,
      include_coupling_signature: bool = False,
  ) -> tuple[CsfHeaderData, DataFrame]:
  ```

- [ ] Document the exact list invariant, `2J` representation, parity exclusion,
  conditional schema, and added compute/memory cost.

### Task 5: Verify GraspKit-oriented usage

- [ ] Add a Python test or documented example showing native Polars grouping:

  ```python
  coupling_level = 2

  summary = (
      csfs_df
      .with_columns(
          pl.col("coupling_signature")
          .list.slice(-coupling_level)
          .alias("selected_coupling")
      )
      .group_by(
          ["block_id", "selected_coupling"],
          maintain_order=True,
      )
      .agg(
          pl.len().alias("count"),
          pl.col("idx").alias("global_idxs"),
      )
  )
  ```

- [ ] Verify Polars can group the `List(Int32)` column without converting rows
  to Python objects.
- [ ] Document that callers needing block-local rmix indices must derive or
  retain a block-local row index; the returned `idx` remains the existing
  global index.

### Task 6: Measure performance and finish documentation

- [ ] Benchmark `read_csfs()` before and after the change with
  `include_coupling_signature=False`; regression should remain within benchmark
  noise.
- [ ] Benchmark enabled mode with 1, 8, and default worker counts and record
  elapsed time and peak memory.
- [ ] Confirm the file is read only once and no Parquet round trip is added.
- [ ] Update `docs/CSF_DESCRIPTOR_GUIDE.md` and `docs/CHANGELOG.md`.
- [ ] Update public examples if the feature becomes part of the recommended
  GraspKit workflow.

---

## 8. Error and Compatibility Contract

### 8.1 Backward compatibility

The change is backward compatible because the new parameter is keyword-only and
defaults to false. Existing calls such as:

```python
read_csfs(path, num_workers=8)
read_csfs(path, include_block_id=True)
```

must return the same data and schema as before.

### 8.2 Enabled-mode errors

When coupling signatures are requested, the call fails as a whole if:

- peel subshells cannot be extracted from the five header lines;
- the shared fixed-width parser returns an error;
- a signature cannot be built for any otherwise-returned CSF row;
- Arrow list construction violates the non-null schema.

Do not return a frame containing null, shortened, or partially parsed
signatures. Existing field-level descriptor semantics that intentionally map
missing/unparseable optional values to zero remain unchanged unless addressed
by a separate strict-parsing proposal.

### 8.3 Existing input behavior

This feature must not silently change current reader behavior:

- non-ASCII data is rejected;
- zero `max_line_len` or `num_workers` is rejected;
- block separators inside incomplete triples are rejected;
- an incomplete final CSF remains ignored;
- truncated lines remain reported through `truncated_count`.

Because truncation can affect fixed-width physics fields, documentation should
tell coupling-signature callers to require:

```python
header["conversion_stats"]["truncated_count"] == 0
```

for calculation-quality inputs.

---

## 9. Verification Commands

Run from the rCSFs repository using its uv environment:

```bash
uv run cargo fmt --check
uv run cargo test
uv run maturin develop
uv run pytest tests/read_csfs_test.py
uv run pytest
uv run ruff check .
uv run basedpyright rcsfs/
```

Add a focused smoke check:

```python
from pathlib import Path

import polars as pl
from rcsfs import read_csfs

header, frame = read_csfs(
    Path("tests/fixtures/sample.csf"),
    num_workers=2,
    include_block_id=True,
    include_coupling_signature=True,
)

assert frame.height == 28
assert frame.schema["coupling_signature"] == pl.List(pl.Int32)
assert frame["coupling_signature"].null_count() == 0
assert header["conversion_stats"]["truncated_count"] == 0
```

---

## 10. Acceptance Criteria

The work is complete when:

1. `read_csfs(..., include_coupling_signature=True)` returns the documented
   non-null `List(Int32)` column.
2. Every signature is derived through the same fixed-width implementation used
   by descriptor generation.
3. `tests/fixtures/sample.csf` returns 28 ordered rows whose final signature
   value is total `2J = 8`.
4. All option combinations produce the documented column order and dtypes.
5. Default calls remain byte/schema compatible and do not perform signature
   parsing work.
6. Polars can slice and group the signature column without Python-row
   conversion.
7. Rust, Python, lint, type, and formatting checks pass.
8. Enabled and disabled benchmark results are recorded in the implementation
   PR or change log.

---

## 11. Follow-up Work in GraspKit

After the rCSFs interface is released, GraspKit can replace legacy nested-list
coupling collection with a DataFrame flow:

```text
rcsfs.read_csfs(include_block_id=True,
                include_coupling_signature=True)
    -> Polars list slicing by coupling_level
    -> group_by(block_id, selected_coupling)
    -> count / index collection / CI-square aggregation
```

That downstream refactor should be a separate change with its own compatibility
plan. It should not cause rCSFs to depend on GraspKit.
