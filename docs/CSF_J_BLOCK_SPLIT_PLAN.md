# CSF J-Block Split (`rasfsplit`) — Implementation Plan

- **Branch**: `1.3.1-beta2`
- **Version**: no version bump (per owner decision, 2026-01-21)
- **Date**: 2026-01-21
- **Status**: Implemented
- **Fortran reference**: `grasp_2990_NNNP/src/tool/rasfsplit.f90` (Per Jonsson, Nov 2016)

---

## 1. Background & Goal

GRASP's `rasfsplit` tool splits one multi-block CSF list (`name.c`, one
symmetry block per `J^P`) into one single-block CSF file per block, so that
downstream GRASP steps (`rmcdhf`, `rci`, ...) can consume each `J^P`
independently. The ML pipeline in `graspkit-tools` needs the same operation on
its large multi-`J` CSF lists.

**Goal**: reimplement the `.c`-file part of `rasfsplit` inside `rCSFs` as a
streaming Rust core exposed through the Python API and CLI, plus the `.w`
orbital-file copy; the `.m`/`.cm` binary mixing-file split is **not** in scope
(owner implements it elsewhere):

```
name.c  (blocks: 2J=8, 2J=5, 2J=0, ...)
  →  name_8.c    (single block, 5-line header, no separator)
  →  name_8.w    (copy of name.w, when it exists)
  →  name_5.c
  →  name_5.w
  →  ...
```

---

## 2. Fortran `rasfsplit` behavior (what the reference does)

`rasfsplit` handles three input kinds.

### 2.1 `name.c` — CSF list split (in scope)

1. Read the file line by line. A block boundary is any line whose **second
   character is `*`** (i.e. the `" *"` separator written by GRASP tools).
2. **Parity of a block** = last character of the last line before the
   separator: `-` → odd, anything else → even. (That line is the final
   coupling line of the block's last CSF; its trailing token is total `J^P`.)
3. The **first 5 lines** (header) are copied verbatim into every output.
4. Block `k` with parity `p` and per-parity counter `n` is written to
   `name_{even|odd}{n}.c`. **The J value is not used in the filename.**
5. Arbitrary limits: at most **15 odd + 15 even** blocks (fixed-size string
   tables); the program aborts beyond that.
6. The final block (EOF without trailing separator) is closed as a block.

### 2.2 `name.m` / `name.cm` — mixing files (OUT OF SCOPE)

Fortran **unformatted sequential binary** files are split per block by the
Fortran tool. rCSFs never touches this binary format. **Owner implements this
part through other means** (decision 2026-01-21); nothing here constrains that
work except the naming convention chosen in D7, which the owner's tool is free
to reuse (`name_<2J>.m` / `name_<2J>.cm`).

### 2.3 `name.w` — orbital file (in scope)

Plain byte-for-byte copy per block: `name.w` → `name_{even|odd}{n}.w` in the
Fortran tool. We adopt the same behavior with our naming: `name.w` →
`name_<2J>.w` beside each output (see D7).

### 2.4 Reference quirks we will NOT keep

- The 15-blocks-per-parity limit.
- The interactive "same orbital set?" y/n confirmation prompt (a caller
  concern, not a splitter concern).
- Parity-only naming (`_even1`) that discards the J value.
- The `'cp '`-based `.w` copy via `system()` (we copy in Rust).

---

## 3. Scope

| Phase | Content | Status |
|---|---|---|
| **1 (this plan)** | Split one multi-block CSF **text** file into one single-block CSF text file per symmetry block, named `name_<2J>.c`; copy `name.w` to each output stem when present. Rust core + PyO3 binding + Python API + CLI subcommand + tests + docs. | planned |
| ~~2~~ | ~~`name.w` copy~~ | folded into phase 1 (owner decision) |
| — | `.m`/`.cm` Fortran-unformatted mixing-file split | **out of scope** — owner implements separately |

---

## 4. Confirmed design decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | Input representation | **CSF text file directly** (`name.c`) | Matches `rasfsplit` and the user request; avoids a Parquet round trip; the interaction module already establishes text-file inputs as supported. Parquet-based splitting already exists (`split_csfs_by_active_spaces`) and is a different feature. |
| D2 | Core implementation | **New Rust module `src/csf_block_split.rs`**, streaming single pass | Bounded memory (see D3); keeps `lib.rs` thin per repo convention; no new dependencies (`anyhow`, `std` only). |
| D3 | Memory model | **Streaming text router**: one `BufReader` in, one `BufWriter` per output block (staged temp files) | Multi-`J` `.c` files are the *largest* pipeline artifacts (this is why they get split); `CompleteCsfFile::parse_path` materializes everything and is ruled out for the core path. Line buffers + writer buffers only; number of open writers = block count (≪ fd limits in practice; guard documented in D8). |
| D4 | Line fidelity | **Verbatim line copy** | Mirrors `rasfsplit`, which copies lines verbatim; more tolerant than `complete_csf`'s strict canonical round-trip check while still validating everything needed for correctness (D5). |
| D5 | Block `J^P` detection | Parse **line 3's final coupling token + trailing parity byte** of every record, reusing `complete_csf::parse_two_j` and `Parity::parse` (shared as `pub(crate)` helpers if needed); require **all records in a block to agree** | Gives the real `2J` for output naming; detects malformed/mixed blocks that `rasfsplit` would silently mis-split. |
| D6 | Split granularity | **One output per block** (never merge blocks with equal `2J^P`) | Mirrors `rasfsplit` (`_even1`, `_even2`, ...); merging would change record provenance; caller can concatenate trivially. |
| D7 | **Output naming** (owner decision 2026-01-21) | Always `name_<2J>.c` — the integer 2J only, e.g. J=4 → `name_8.c`, J=5/2 → `name_5.c`, J=0 → `name_0.c`. Prefix = input stem by default. **No** parity label, **no** suffixes, **no** alternate naming modes. | Owner-specified simple naming. Upstream workflow tags parity (among other labels) in the file name itself, so a single input file contains one parity and its 2J values are unique; `name_<2J>` can never collide (see D14). |
| D8 | Output transaction | Stage every output as a sibling temp file (`atomic_output::create_temporary_output`), publish with `publish_temporary_output` only after **all** blocks are written; refuse existing outputs unless `overwrite=true`; report already-published files if a later publication fails | Same contract as `split_csfs_by_active_spaces` / `select_interacting_csfs`; never half-write visibly. |
| D9 | Parity source | Parsed parity byte of each record (D5), **not** "last char before separator" | The Fortran heuristic mis-fires on blank-padded or malformed lines; per-record parity is already validated and is the same information. |
| D10 | Parallelism | **Serial, I/O-bound** single pass | The work is a line copy; rayon adds nothing at I/O speed. Keep the code simple; benchmark later if ever needed. |
| D11 | Binding style | Thin `#[pyfunction]` in `lib.rs` delegating to the module; stats returned as a dict; wrapper `rcsfs.split_csfs_by_j(...)` with `pathlib.Path` support; TypedDicts in `_types.py`; `.pyi` stub updated | Follows `partition_csfs` / `split_csfs_by_active_spaces` exactly. |
| D12 | CLI | New subcommand `rcsfs jsplit <input.c>` with aliases `split-j`, `rasfsplit`; options `--prefix`, `--no-copy-w`, `--overwrite`, `--json`. Final outputs are fixed to the invocation directory per the shared CLI output-location policy (2026-09-30 changelog entry) — there is no `--output-dir`; the input may live elsewhere and its sibling `.w` is still discovered next to the **input**. The Python API keeps `output_dir` (default: the input's parent) for library callers. | Note: the alias **`rcsfsplit` is already taken** by the active-space split command (`csfs-split`); `rasfsplit` (the actual Fortran name) is free and is the more accurate alias anyway. No `--naming` option — D7 fixed the single scheme. |
| D13 | TOML batch config | Register a `[jsplit]` section in the CLI config table (`_cli_config.py` `CONFIG_SECTIONS`) | Every other subcommand participates in `rcsfs --config` batch runs; this one should too (graspkit-tools SLURM orchestration uses it). |
| D14 | Duplicate-`2J` semantics (owner decision 2026-01-21) | **Error.** Two blocks in one input file sharing the same 2J — regardless of parity — cannot both map to `name_<2J>.c`, so the input violates the single-parity-per-file contract the naming relies on. Fail with a message naming both block indices and their `2J^P`, before anything is published; staged temporaries are removed. | The owner's workflow tags parity in the input file name, so one file = one parity = unique 2J values. A duplicate therefore means malformed/merged input the caller should split by parity first; silently renaming or overwriting would hide the problem. |
| D15 | `.w` copy (owner decision 2026-01-21) | Source: `<input_dir>/<input_stem>.w`. When present and `copy_w=true` (default), copy it byte-for-byte to each output stem (`name_8.c` → `name_8.w`), staged and published in the same transaction. When absent: skipped silently, reported as `w_file: null` in stats. `copy_w=false` (CLI `--no-copy-w`) disables. | Matches `rasfsplit` behavior; keeps the orbital file beside every single-block list so downstream GRASP steps find it. A missing `.w` is normal for rCSFs-generated lists (no GRASP run behind them), so it must not fail. |
| D16 | Version | **No version bump** (owner decision 2026-01-21); ship on the current version line, note the feature in `docs/CHANGELOG.md` | Owner call. |

---

## 5. Algorithm (streaming split)

```text
open input (BufReader, keep raw lines incl. LF discipline)
├─ read exactly 5 header lines (error if EOF early); validate labels
│    (line 1 "Core subshells:", line 3 "Peel subshells:", line 5 "CSF(s):")
├─ block loop:
│   ├─ collect 3-line records until EOF or a line == " *"
│   ├─ for each record: parse (2J, parity) from line 3 final field
│   │    └─ ensure equal to the block's first record → else error
│   ├─ write the 3 lines verbatim to this block's staged writer
│   └─ on separator / EOF: close block; require non-empty, multiple of 3
├─ after input ends: every block has (2J, parity, csf_count, staged file)
├─ names are prefix_<2J>.c (D7); a duplicate 2J is a hard error (D14)
├─ verify no output aliases the input (atomic_output helpers)
├─ .w handling (D15): if <input stem>.w exists and copy_w → stage one
│    byte-copy per output stem (streamed, not fully buffered)
├─ flush + fsync all staged writers
└─ publish all staged files; on partial failure, name what was published
```

Details:

- **Separator discipline**: exactly `" *"` (like `complete_csf::BLOCK_SEPARATOR`).
  A line whose second char is `*` but which is not exactly `" *"` is an error
  (this is stricter than Fortran, consistent with `complete_csf`).
- **Single-block input**: allowed; produces one output (idempotent behavior,
  still useful for name canonicalization). Not an error.
- **No blocks at all** (header only): error, "CSF file contains no CSF records"
  (matches `complete_csf`).
- **J = 0 blocks** (`0+` / `0-` total coupling): valid; `parse_two_j("0")` → 0;
  output `name_0.c`.
- **LF discipline**: input lines must end with LF, no CR (error message points
  at CRLF as non-canonical), matching `complete_csf::parse_reader`.
- **ASCII check** per line (same as `complete_csf`).
- **Writer count**: number of blocks; typical GRASP files have < 40 `J^P`
  blocks. A hard sanity ceiling (e.g. 10 000 blocks) guards against garbage
  inputs exhausting file descriptors; documented in the error contract.

---

## 6. Rust module specification — `src/csf_block_split.rs`

```rust
/// One output: a single symmetry block written as its own CSF file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JBlockOutputStats {
    pub output_file: String,        // final (published) .c path
    pub w_file: Option<String>,     // final (published) .w path, when copied
    pub block_index: usize,         // 0-based order in the input
    pub total_two_j: u16,           // 2J (integer), e.g. 8 for J=4
    pub parity: String,             // "even" | "odd"
    pub csf_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JBlockSplitStats {
    pub input_csf_count: usize,
    pub block_count: usize,
    pub outputs: Vec<JBlockOutputStats>,
}

pub fn split_csfs_by_j(
    input_csf: &Path,
    output_dir: &Path,          // directory receiving the outputs
    prefix: &str,               // output filename prefix (default: input stem, applied by caller)
    copy_w: bool,               // D15
    overwrite: bool,
) -> Result<JBlockSplitStats>;
```

- Reuses: `atomic_output::{create_temporary_output, publish_temporary_output,
  ensure_output_does_not_alias_input}`, `complete_csf::{Parity, parse_two_j}`
  (export the helpers `pub(crate)` if they are private today).
- Naming is `format!("{prefix}_{two_j}.c")` — inline; the duplicate-`2J` guard
  (D14) lives in the block scan.
- Errors: `anyhow` with line numbers, mirroring `complete_csf` message style
  ("line {n}: ...").
- Public module (`pub mod csf_block_split;` in `lib.rs`) so Rust integration
  tests can reach it, like `csf_partition` / `interaction`.

### PyO3 binding (in `lib.rs`, thin)

```rust
#[pyfunction]
#[pyo3(signature = (input_csf, output_dir=None, prefix=None, *, copy_w=true, overwrite=false))]
fn split_csfs_by_j(...) -> PyResult<Py<PyAny>>   // dict per D11, IO errors → PyIOError
```

---

## 7. Python surface

### 7.1 `rcsfs/__init__.py`

```python
def split_csfs_by_j(
    input_csf: str | Path,
    output_dir: str | Path | None = None,   # default: input's parent
    prefix: str | None = None,              # default: input stem
    *,
    copy_w: bool = True,
    overwrite: bool = False,
) -> JBlockSplitStats: ...
```

### 7.2 `rcsfs/_types.py` — new TypedDicts

`JBlockOutputStats` (`w_file: str | None`), `JBlockSplitStats`
(fields mirror §6; `parity: Literal["even", "odd"]`).

### 7.3 `rcsfs/_rcsfs.pyi` — stub entry for the new `_rcsfs.split_csfs_by_j`.

### 7.4 `__all__` — add the function and both TypedDicts.

### 7.5 CLI (`rcsfs/cli.py`)

```
rcsfs jsplit <input.c>
  [--prefix PREFIX] [--no-copy-w] [--overwrite] [--json]
```

- Outputs are written to the current directory (shared CLI policy); the
  input may live elsewhere and its sibling `.w` is found next to the input.

- `--json` prints the full stats dict; text mode prints one line per block
  (`name_8.c: 12,345 CSFs (2J=8, odd)` plus `  + name_8.w` when copied).
- Exits 1 with `error` JSON on failure, matching the other subcommands.
- Config section `[jsplit]` wired into `_cli_config.py`
  (`CONFIG_SECTIONS`, `init-config` template, key validation).

---

## 8. Validation & error contract

| Condition | Behavior |
|---|---|
| Input missing / unreadable | `OSError` / `PyIOError` |
| Fewer than 5 header lines, or wrong header labels | error naming the line |
| Record group of 1–2 lines at separator/EOF | error ("final CSF record has N of 3 lines") |
| Empty block (separator directly after separator) | error, matching `complete_csf` |
| Records within a block disagree on `2J` or parity | error naming block index and line |
| Line not exactly `" *"` but contains `*` as 2nd char | error (non-canonical separator) |
| CRLF / missing final LF / non-ASCII | error with line number |
| > 10 000 blocks | explicit error before opening writers |
| Duplicate `2J` across blocks (any parity) | **error** naming both block indices and their `2J^P` (D14); nothing published, staged temps removed |
| Output path exists and `overwrite=false` | `FileExistsError` at publish; staged temps removed |
| Output aliases input (`.c` or `.w`) | rejected up-front |
| `copy_w=true` but `<stem>.w` absent | skipped; `w_file: null` in stats; not an error |

---

## 9. Testing plan

### 9.1 New fixture `tests/fixtures/multi_j.c`

Hand-written, 3 blocks with unique 2J: integer J (`4-` → 2J=8 odd),
half-integer J (`5/2+` → 2J=5 even), and **J=0** (`0+` → 2J=0 even).
Small (a few CSFs per block), LF-only, canonical `" *"` separators.
`tests/fixtures/complete.csf` (2 blocks) is reused as-is for a second case.
Optional tiny `tests/fixtures/multi_j.w` to exercise the copy. The
duplicate-`2J` error case writes its own temporary input inside the test.

### 9.2 Rust — `tests/csf_block_split_test.rs`

1. Split `complete.csf` → 2 outputs (`complete_8.c`, `complete_5.c`); per-block
   `2J/parity/csf_count` correct; each output re-parses via
   `CompleteCsfFile::parse_path` with `blocks.len() == 1`.
2. **Byte round-trip**: concatenating the outputs' record sections
   (with `" *"` between, in block order) reproduces the input's data section
   exactly; headers identical to the input's 5 lines.
3. `multi_j.c` naming: `multi_j_8.c`, `multi_j_5.c`, `multi_j_0.c` — bare
   integer 2J, no parity label (D7).
4. `.w` copy: with `multi_j.w` present, each output gains a byte-identical
   `multi_j_<2J>.w`; with `copy_w=false` or the file absent, `w_file: null`
   and no `.w` outputs.
5. Duplicate-`2J` error: a temporary input with two blocks sharing 2J (same or
   different parity) fails, names both block indices and `2J^P`, publishes
   nothing, and leaves no temporary files behind.
6. Errors: truncated final record; empty block; mixed parity in one block;
   CRLF input; non-canonical separator; existing output without `overwrite`;
   output aliasing input (`.c` and `.w`).
7. `overwrite = true` replaces existing outputs.

### 9.3 Python — `tests/csf_block_split_test.py`

1. `rcsfs.split_csfs_by_j` with `Path` and `str` inputs; default output dir =
   input's parent; default prefix = input stem.
2. Stats TypedDict shape (keys exactly as declared in `_types.py`).
3. `.w` behaviors: present → copied; absent → `w_file: null`; `copy_w=False`
   → not copied.
4. CLI: `rcsfs jsplit`, `--json` output parse, `--no-copy-w`,
   exit codes on failure; `[jsplit]` TOML config batch run via `--config` and
   `init-config` adds the section.
5. Type-check/lint pass (`ruff`, `basedpyright rcsfs/`).

### 9.4 Verification commands (per `rCSFs/AGENTS.md`)

```bash
source ../graspkit-tools/.venv/bin/activate
cargo test                        # incl. new integration tests
cargo build --release --features pyo3/extension-module
cp target/release/lib_rcsfs.so rcsfs/_rcsfs.cpython-314-x86_64-linux-gnu.so
pytest                            # incl. new Python tests
ruff check .
basedpyright rcsfs/
# afterwards, for parent-workspace consumers:
cd ../graspkit-tools && uv sync
```

---

## 10. Documentation updates

- `docs/CHANGELOG.md`: new entry describing `split_csfs_by_j` / `rcsfs jsplit`
  with the Fortran lineage (`rasfsplit`) and the `.w` copy; explicitly note
  that `.m`/`.cm` splitting is intentionally not provided.
- `rcsfs/__init__.py` module docstring: add a Quick Start snippet.
- `rCSFs/CLAUDE.md` + `rCSFs/AGENTS.md` public-API tables: add the new symbol
  and CLI command (these files list the canonical API surface).
- This plan flips `Status` to `Implemented` when done.

---

## 11. Implementation steps (ordered)

1. Fixtures `tests/fixtures/multi_j.c` (+ optional `multi_j.w`).
2. `src/csf_block_split.rs`: pure naming function (+ unit tests), stats
   structs, `split_csfs_by_j` core (streaming reader, per-block staged
   writers, `.w` staged copies, publish transaction).
3. Expose `parse_two_j` / `Parity` as `pub(crate)` in `complete_csf.rs` if
   needed (no behavior change).
4. `lib.rs`: `pub mod`, `#[pyfunction]` binding + registration.
5. Rust integration tests (`tests/csf_block_split_test.rs`); `cargo test`.
6. Python: `_types.py` TypedDicts, `__init__.py` wrapper + `__all__`,
   `_rcsfs.pyi` stub; CLI subcommand + `_cli_config.py` section.
7. Python tests (`tests/csf_block_split_test.py`); rebuild in-tree `.so`;
   `pytest`, `ruff`, `basedpyright`.
8. Docs: CHANGELOG, module docstring, CLAUDE/AGENTS API tables, plan status.
   No version bump (D16).
9. Commits (inside `rCSFs`): suggested split —
   `csf_block_split: add streaming J-block split core (Rust)`;
   `python/cli: expose split_csfs_by_j and rcsfs jsplit`;
   `docs: record J-block split plan and changelog`.
   Then parent-workspace gitlink commit per workspace rules.

---

## 12. Resolved questions (owner decisions, 2026-01-21)

| # | Question | Resolution |
|---|---|---|
| 1 | Naming default | `name_<2J>.c` (integer 2J only; no parity label, no extra modes); duplicate 2J errors per D14. **Resolved.** |
| 2 | Duplicate `2J^π` blocks | Cannot occur in the owner's workflow (input file names carry parity tags, so one file = one parity = unique 2J). A duplicate 2J is therefore a hard **error** (D14), not a fallback. **Resolved.** |
| 3 | `.w` / `.m` / `.cm` scope | `.w` copy included (D15); `.m`/`.cm` out of scope — owner implements separately. **Resolved.** |
| 4 | Version bump | None (D16). **Resolved.** |
