# CSF Zero-First Partition — Implementation Plan

- **Branch**: `1.3.1-beta1` (forked from `main` @ `5bce6db`)
- **Target version**: `1.3.1` (minor bump — new backward-compatible feature)
- **Date**: 2026-07-17
- **Status**: Implemented

---

## 1. Background & Goal

GRASP2018 ships a Fortran utility `rcsfzerofirst`
(`src/appl/rcsfzerofirst90/`) that partitions a CSF list into a
**zero-order space + first-order space** per symmetry block (J^P), driven by a
multireference list:

```
output block b = [zero-order CSFs of block b]  ++  [CSFs of full block b NOT in zero-order block b]
                     ^ "locked" to block head       ^ "reordered" complement (hash anti-match)
```

The match key is exact string equality of the three-line CSF record
`(line1, line2, line3)` — identical to `lodcsl_Part.f90:52-54`.

**Goal**: reimplement this flow inside `rCSFs`, but routed through the Parquet
layer (CSF text → Parquet → reorder on Parquet → CSF text) and exposed as a
CLI subcommand:

```
uv run rcsfs zero-first zero.csf full.csf [reordered.csf]
```

---

## 2. Confirmed Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Core implementation lib | **Pure Rust + `arrow` crate** | Zero new deps (arrow/parquet already in `Cargo.toml`); the reorder is a `HashSet` anti-match, polars' DataFrame power is unused; matches rCSFs' Rust-first philosophy. |
| Block alignment | **By block index + count check** | Mirrors grasp original behavior (sequential block pairing); only validate `zero.block_count == full.block_count`. J^P-based alignment is a possible future enhancement. |
| API surface | **Two layers**: low-level `partition_csfs_parquet` (Rust) + high-level orchestration in the CLI | Low-level is reusable for users who already have Parquet; CLI's `zero-first` does the full CSF→CSF pipeline with temp Parquet handling. |
| Match semantics | **`(line1, line2, line3)` full-string equality** | Byte-for-byte identical to grasp `lodcsl_Part.f90`. Trailing-whitespace differences cause miss-matches — preserved as grasp-compatible behavior. |

---

## 3. Current State of rCSFs (gap analysis)

| Capability | Status | Location |
|---|---|---|
| CSF text → Parquet | exists | `convert_csfs` (`src/csfs_conversion.rs`) |
| Block boundary recording | exists | header TOML `block_info.block_lengths` + `block_count` |
| Global row index `idx` | exists | Parquet col 0, continuous across blocks |
| CLI framework | exists | `rcsfs/cli.py` (argparse subparsers, `gen-descriptors` subcmd) |
| `[project.scripts] rcsfs` | exists | `pyproject.toml` |
| **Parquet → CSF text write-back** | **missing** | new |
| **Zero/first-order partition + reorder** | **missing** | new |

Parquet schema (from `convert_csfs`): `idx: UInt64, line1: Utf8, line2: Utf8, line3: Utf8`.
One CSF = one row. Block separators (`*`) are dropped during conversion; block
boundaries are recoverable via `block_lengths`.

---

## 4. Architecture / Data Flow

```
zero.csf ─┐
          ├─ convert_csfs ──→ zero.parquet + zero_header.toml ─┐
          │                                                      │
          │                                                      ├─ partition_csfs ──→ reordered.csf
          │                                                      │  (Rust core: slice by block_lengths,
          │                                                      │   HashSet anti-match, reorder,
          │                                                      │   write CSF text)
full.csf ─┤                                                      │
          └─ convert_csfs ──→ full.parquet + full_header.toml ─┘
```

The CLI `zero-first` subcommand orchestrates: `convert_csfs × 2 → partition →
cleanup temp` (temp isolation via separate subdirs to avoid header filename
collisions when input stems match).

---

## 5. Module Breakdown

| Layer | File | Language | Responsibility |
|---|---|---|---|
| Core | `src/csf_partition.rs` | Rust | Read two Parquets + two header TOMLs, slice by `block_lengths`, HashSet anti-match per block, write reordered CSF text (5-line header + CSF triples + ` *` block separators). |
| Binding | `src/lib.rs` | Rust | Register `partition_csfs` PyO3 function (thin wrapper). |
| Reused | `src/csfs_conversion.rs` | Rust | `HeaderData`/`BlockInfo`/`HeaderInfo` made `pub` so `csf_partition` deserializes the header TOML from a single source of truth. |
| Python API | `rcsfs/__init__.py` | Python | `partition_csfs` wrapper (Path support) + `PartitionStats` TypedDict + `__all__` export. |
| CLI | `rcsfs/cli.py` | Python | New `zero-first` subparser: CSF inputs → orchestrate convert×2 + partition + temp cleanup. |
| Tests (Rust) | `tests/csf_partition_test.rs` | Rust | Multi-block, lock-to-head, complement append, separator placement, block-count mismatch error. |
| Tests (Py) | `tests/cli_test.py` | Python | `zero-first` CLI: monkeypatch `convert_csfs` + `partition_csfs`, assert call args / exit code / output. |

**Zero new Cargo dependencies.** `HashSet`, `BufWriter` are std; `arrow`/`parquet`/`serde`/`toml`/`anyhow` already present.

---

## 6. Core Algorithm

```rust
// Read both header TOMLs
let zero_hdr: HeaderData = toml::from_str(&read(zero_header))?;
let full_hdr: HeaderData = toml::from_str(&read(full_header))?;

// Validate block alignment
if zero_hdr.block_info.block_count != full_hdr.block_info.block_count {
    return Err("block count mismatch: zero={} full={}");
}

let mut zero_reader = ParquetRecordBatchReader::new(zero_parquet)?;
let mut full_reader = ParquetRecordBatchReader::new(full_parquet)?;
let mut zero_pending = Vec::new();   // rows carried across batches
let mut full_pending = Vec::new();
let mut out = BufWriter::new(File::create(output_csf)?);

// Write 5-line header from the FULL file (it is the "complete space")
for line in &full_hdr.header_info.header_lines {
    writeln!(out, "{}", line)?;
}

for b in 0..block_count {
    let zero_len = zero_hdr.block_info.block_lengths[b];
    let full_len = full_hdr.block_info.block_lengths[b];

    // 1. Read entire zero-order block, build HashSet
    let zero_rows = take_rows(&mut zero_reader, &mut zero_pending, zero_len)?;
    let zero_set: HashSet<(&str,&str,&str)> = zero_rows.iter()
        .map(|(l1,l2,l3)| (l1.as_str(), l2.as_str(), l3.as_str())).collect();

    // 2. Write zero-order block first (LOCKED to block head)
    for (l1,l2,l3) in &zero_rows {
        writeln!(out, "{}", l1)?;
        writeln!(out, "{}", l2)?;
        writeln!(out, "{}", l3)?;
    }

    // 3. Stream full block, append COMPLEMENT (not in zero_set)
    let mut taken = 0;
    while taken < full_len {
        let (l1,l2,l3) = take_one_row(&mut full_reader, &mut full_pending)?;
        taken += 1;
        if !zero_set.contains(&(l1.as_str(), l2.as_str(), l3.as_str())) {
            writeln!(out, "{}", l1)?;
            writeln!(out, "{}", l2)?;
            writeln!(out, "{}", l3)?;
        }
    }

    // 4. Block separator between blocks (not after the last)
    if b + 1 < block_count {
        writeln!(out, " *")?;
    }
}
out.flush()?;
```

`take_rows` / `take_one_row` are stream helpers that pull rows from the
`ParquetRecordBatchReader`, buffering carry-over across batches. Memory peak =
zero-block (small, fully loaded for the HashSet) + one batch of the full block.
This mirrors grasp's own `allocate(Found, C_shell, ...)` behavior.

---

## 7. API Surface

### 7.1 Rust (`src/csf_partition.rs`)

```rust
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PartitionStats {
    pub block_count: usize,
    pub zero_csf_count: usize,     // total CSFs in zero-order file
    pub full_csf_count: usize,     // total CSFs in full file
    pub output_csf_count: usize,   // = zero + first-order complement
    pub first_order_count: usize,  // complement count (full not in zero)
}

pub fn partition_csfs(
    zero_parquet: &Path,
    zero_header: &Path,
    full_parquet: &Path,
    full_header: &Path,
    output_csf: &Path,
) -> Result<PartitionStats, Box<dyn std::error::Error + Send + Sync>>;
```

### 7.2 Python (`rcsfs/__init__.py`)

```python
class PartitionStats(TypedDict):
    success: bool
    zero_parquet: str
    full_parquet: str
    output_file: str
    block_count: NotRequired[int]
    zero_csf_count: NotRequired[int]
    full_csf_count: NotRequired[int]
    output_csf_count: NotRequired[int]
    first_order_count: NotRequired[int]
    error: NotRequired[str]

def partition_csfs(
    zero_parquet: Union[str, Path],
    zero_header: Union[str, Path],
    full_parquet: Union[str, Path],
    full_header: Union[str, Path],
    output_csf: Union[str, Path],
) -> PartitionStats: ...
```

### 7.3 CLI (`rcsfs/cli.py`)

```
uv run rcsfs zero-first zero.csf full.csf [reordered.csf] [options]

options:
  --keep-parquet        Keep intermediate Parquet + header TOML (default: clean up)
  --work-dir DIR        Directory for temp files (default: system tempdir)
  --num-workers N       Worker threads for convert_csfs (default: all cores)
  --max-line-len N      Max CSF line length for convert_csfs (default: 256)
  --json                Print PartitionStats as JSON
```

Default output name when `reordered.csf` is omitted: `{full_stem}_zf.csf` in
the full file's directory.

Temp file layout under `--work-dir`:
```
{work_dir}/rcsfs-zero-first-XXXX/
  zero/zero.parquet + zero_header.toml      (stem from zero.csf)
  full/full.parquet + full_header.toml      (stem from full.csf)
```
Separate `zero/` and `full/` subdirs prevent header filename collisions when
the two inputs share a stem.

---

## 8. Implementation Steps

| # | Step | Verify |
|---|---|---|
| 1 | `src/csfs_conversion.rs`: make `HeaderData`/`BlockInfo`/`HeaderInfo` `pub` (+ fields, +`Clone`) | `cargo build` still passes | 
| 2 | `src/csf_partition.rs`: implement `partition_csfs` + stream helpers | unit logic correct |
| 3 | `src/lib.rs`: register `partition_csfs` PyO3 function | `uv run maturin develop` succeeds |
| 4 | `rcsfs/__init__.py`: add `partition_csfs` wrapper + `PartitionStats` + export | `from rcsfs import partition_csfs` works |
| 5 | `rcsfs/cli.py`: add `zero-first` subparser + orchestration | `rcsfs zero-first -h` shows help |
| 6 | `tests/csf_partition_test.rs`: Rust integration tests | `uv run cargo test` green |
| 7 | `tests/cli_test.py`: add `zero-first` CLI tests (monkeypatch) | `uv run pytest` green |
| 8 | Build + full validation | all checks green |

**Step 1 is already done** (commit pending on this branch — see
`git diff src/csfs_conversion.rs`).

---

## 9. Test Plan

### Rust integration (`tests/csf_partition_test.rs`)

Fixture builder: synthesize a CSF text with N blocks, each block a list of
3-line CSF triples. Reuse the multi-block pattern from
`integration_test.rs:216-284` (` *` separators).

Cases:
1. **`test_partition_locks_zero_to_head_and_appends_complement`**
   - zero = first 2 CSFs of each block; full = all 4 CSFs of each block.
   - Assert output block = [zero 2] ++ [other 2], in that exact order.
2. **`test_partition_writes_block_separator_between_blocks_only`**
   - 3 blocks. Assert ` *` appears exactly `block_count - 1` times, never at EOF.
3. **`test_partition_preserves_5_line_header_from_full_file`**
   - Output's first 5 lines == full file's header_lines.
4. **`test_partition_rejects_block_count_mismatch`**
   - zero has 2 blocks, full has 3. Assert `partition_csfs` returns Err.
5. **`test_partition_zero_equals_full_yields_zero_complement`**
   - zero == full. Assert `first_order_count == 0` and output == full verbatim.
6. **`test_partition_disjoint_zero_and_full`**
   - zero and full share no CSFs. Assert output = zero ++ full, `first_order_count == full_len`.

### Python CLI (`tests/cli_test.py`)

Following the existing monkeypatch pattern (`tests/cli_test.py:6-75`):

7. **`test_zero_first_orchestrates_convert_and_partition`**
   - Fake `convert_csfs` (×2) + fake `partition_csfs`; assert call args (input
     paths, isolated temp subdirs, output path) and exit code 0.
8. **`test_zero_first_default_output_name`**
   - Omit `reordered.csf`; assert default = `{full_stem}_zf.csf`.
9. **`test_zero_first_keep_parquet_flag`**
   - `--keep-parquet` writes Parquet under work-dir and does not clean up.
10. **`test_zero_first_propagates_partition_failure`**
    - Fake `partition_csfs` returns `success: False`; assert exit code 1 + stderr.

---

## 10. Acceptance Criteria

- [ ] `uv run cargo test` — all green (existing + new `csf_partition_test`).
- [ ] `uv run pytest` — all green (existing + new CLI tests).
- [ ] `uv run ruff check .` — clean.
- [ ] `uv run mypy rcsfs/` — clean.
- [ ] End-to-end manual: on a real multi-block CSF pair, `uv run rcsfs
      zero-first zero.csf full.csf out.csf` produces a file whose per-block
      content matches grasp `rcsfzerofirst`'s `rcsf.out` (zero-first ordering +
      complement), with correct ` *` separators.
- [ ] `Cargo.toml` version bumped `1.2.2 → 1.3.1`; `docs/CHANGELOG.md` entry
      added under `[1.3.1-beta.1]`.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Full block too large to hold in memory | Stream full block row-by-row against the zero HashSet (zero block is the only one fully loaded — typically small). |
| Trailing-whitespace mismatch causes silent miss | Document as grasp-compatible behavior; the conversion step already `TRIM`s nothing, preserving exact bytes. |
| Two inputs share a file stem → header TOML name collision in temp dir | Isolate under `zero/` and `full/` subdirs. |
| Block ordering assumption breaks on real data | Validate `block_count`; if a future user hits J^P-misalignment, add J^P-based pairing as a `--align-by` option. |

---

## 12. Out of Scope (future work)

- Independent `Parquet → CSF` round-trip write-back as a standalone public API
  (the partition writer is internal for now; extract if reuse is needed).
- J^P-based block alignment (current: by index, matching grasp).
- A `convert` / `info` CLI subcommand mirroring the existing Python API (only
  `zero-first` is added here).
