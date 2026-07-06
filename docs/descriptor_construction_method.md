# CSF Descriptor Construction: Technical Methods Documentation

## 1. Overview

The `rCSFs` library converts Configuration State Functions (CSFs) from the GRASP2018 relativistic atomic structure code into fixed-length numerical descriptor vectors suitable for machine learning. The pipeline has two stages: **parsing** (text → integer vector) and **normalization** (integer vector → float vector in [0, 1]).

Each CSF is represented as a vector of length 3N, where N is the number of *peel subshells* (the active orbital space). For each subshell i, three values are stored:

| Index | Symbol | Meaning |
|-------|--------|---------|
| 3i | n_i | Electron occupation count |
| 3i+1 | 2Q_i | Doubled intermediate seniority/coupling quantum number |
| 3i+2 | 2J_cum,i | Doubled cumulative angular momentum up to subshell i |

---

## 2. CSF Text Format (GRASP Convention)

Each CSF occupies exactly three lines in GRASP's fixed-width format. The lines are divided into 9-character blocks, one per occupied subshell:

```
Line 1: "  5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)"
Line 2: "                   3/2               2        "
Line 3: "                                           4-  "
```

**Line 1** encodes subshell identity and occupation:
- Characters 0–4: subshell label (e.g., `5s`, `4d-`, `4d`)
- Characters 5–8: electron count in parentheses (e.g., `( 2)` means n_i = 2)

**Line 2** encodes intermediate coupling quantum numbers (2Q_i), positionally aligned with Line 1. Values may be fractional (`3/2`) or integer (`2`). Semicolons separate seniority and coupling when both are present; the parser takes the value after the semicolon.

**Line 3** encodes cumulative J-coupling values (2J_cum,i) at each coupling stage. The **last 5 characters** (before a trailing padding character) carry the total angular momentum J and parity of the CSF term (e.g., `4-` means J=4, odd parity).

---

## 3. Parsing Algorithm

**Source**: `src/csfs_descriptor.rs:1095–1214`

**Input**: Three ASCII strings (lines 1, 2, 3) and a list of N peel subshell names.

**Output**: Integer vector **d** of length 3N, initialized to zeros.

**Steps**:

1. **Pad lines** — Lines 2 and 3 are right-padded to match Line 1's length for positional alignment. Line 3 is additionally trimmed: its first 4 characters and last 5 characters are removed before padding (the final J value is extracted separately from those last 5 characters).

2. **Extract final 2J** — The total angular momentum is read from positions [l-5, l-1) of Line 3 (where l is the raw line length). Parity indicators (`+`/`-`) are stripped. The value is converted to 2J via the encoding described in Section 4.

3. **Chunk into 9-character blocks** — All three padded lines are split into 9-character segments, yielding one block per occupied subshell.

4. **Process each block i**:
   - **Subshell name**: characters 0–4, trimmed → looked up in a HashMap mapping subshell names to their position index in the peel list.
   - **Electron count** n_i: characters 6–7, parsed as integer.
   - **Intermediate coupling** 2Q_i: from Line 2's block, converted via `j_to_double_j`. If Line 2 is empty at this position, 2Q_i = 0.
   - **Cumulative coupling** 2J_cum,i: from Line 3's block, converted via `j_to_double_j`. If Line 3 is empty but Line 2 has a value, the coupling inherits the intermediate value.
   - **Last-subshell override**: for the final occupied subshell, 2J_cum,i is always set to the extracted final 2J.

5. **Zero rule**: if n_i = 0, all three descriptor elements for subshell i are forced to zero (unoccupied subshell carries no coupling information).

6. **Unoccupied peel subshells** — Subshells in the peel list but absent from the CSF text remain at their initialized zero values.

---

## 4. J-Value Encoding: Doubled-Integer Convention

**Source**: `src/csfs_descriptor.rs:996–1014`

All angular momentum values are stored as 2J (doubled) to maintain exact integer arithmetic:

| Text | J | Stored as 2J |
|------|---|--------------|
| `3/2` | 3/2 | 3 (numerator) |
| `5/2` | 5/2 | 5 |
| `2` | 2 | 4 |
| `4-` | 4 | 8 (parity stripped) |

Fractional values: the numerator is extracted directly (denominator is always 2). Integer values: multiplied by 2. Trailing parity indicators (`+`/`-`) are stripped before parsing.

---

## 5. Worked Example

Given peel subshells `[5s, 4d-, 4d, 5p-, 5p, 6s]` (N=6, descriptor length = 18):

```
Line 1: "  5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)"
Line 2: "                   3/2               2        "
Line 3: "                                           4-  "
```

| Subshell | n_i | 2Q_i | 2J_cum,i | Notes |
|----------|-----|------|----------|-------|
| 5s | 2 | 0 | 0 | Line 2 empty at this position |
| 4d- | 4 | 0 | 0 | Line 2 empty |
| 4d | 6 | 3 | 3 | `3/2` → 3; Line 3 empty, inherits from Line 2 |
| 5p- | 2 | 0 | 0 | Line 2 empty |
| 5p | 4 | 4 | 4 | `2` → 4 in Line 2; Line 3 empty, inherits |
| 6s | 2 | 0 | **8** | Last subshell → final 2J = `4-` → 8 |

**Descriptor**: `[2, 0, 0, 4, 0, 0, 6, 3, 3, 2, 0, 0, 4, 4, 4, 2, 0, 8]`

---

## 6. Physics-Informed Normalization

**Source**: `src/descriptor_normalization.rs:233–324`

The normalization maps each raw integer descriptor to [0, 1] using per-CSF, position-dependent, physics-derived denominators. This is not a statistical normalization; the bounds are derived from the Dirac-Coulomb theory of relativistic subshells.

### 6.1 Relativistic Subshell Properties

Each subshell type has a fixed capacity g_i = 2j_i + 1 = 2|kappa_i|, where kappa_i is the Dirac quantum number:

| Subshell | l | j | kappa | g_i |
|----------|---|---|-------|-----|
| s, p- | 0, 1 | 1/2 | -1, +1 | 2 |
| p, d- | 1, 2 | 3/2 | -2, +2 | 4 |
| d, f- | 2, 3 | 5/2 | -3, +3 | 6 |
| f, g- | 3, 4 | 7/2 | -4, +4 | 8 |
| g, h- | 4, 5 | 9/2 | -5, +5 | 10 |
| h, i- | 5, 6 | 11/2 | -6, +6 | 12 |
| i | 6 | 13/2 | -7 | 14 |

Full subshell identifiers (e.g., `5s`, `4d-`) are automatically converted to angular notation (e.g., `s `, `d-`) for lookup.

### 6.2 Normalization Formulae

For each subshell i (i = 0, ..., N-1):

**Occupation-dependent upper bound on intermediate coupling:**

```
u_i = n_i * (g_i - n_i)
```

This is tighter than the static kappa_i^2 bound: u_i <= kappa_i^2, with equality only at half-filling (n_i = g_i / 2). At empty (n_i = 0) or full (n_i = g_i) occupation, u_i = 0.

**Prefix and suffix sums:**

```
prefix_i = sum_{k=0}^{i} u_k
suffix_i = sum_{k=i+1}^{N-1} u_k
```

**Position-dependent bound on cumulative coupling:**

```
U_i_occ = min(prefix_i, 2J_target + suffix_i)
```

This dual constraint reflects two physical limits:
- The **prefix bound**: the cumulative coupling cannot exceed the sum of all coupling capacities up to position i.
- The **rear bound**: the cumulative coupling at position i cannot exceed the final target 2J plus the remaining coupling capacity after position i.

**Normalized descriptor:**

```
d_norm[3i]   = n_i / g_i
d_norm[3i+1] = 2Q_i / u_i
d_norm[3i+2] = 2J_cum,i / U_i_occ
```

**Zero-division safety**: when u_i = 0 or U_i_occ = 0, the corresponding element is set to 0.0 (physically, the quantity is forced to zero when the denominator vanishes).

### 6.3 Key Properties

- **All values lie in [0, 1]** for valid CSFs.
- **The last occupied position's coupling normalizes to 1.0** whenever prefix_{N-1} >= 2J_target (which holds for all physically valid CSFs), because the coupling value at the last position equals 2J_target by construction and U_{N-1}_occ = min(prefix_{N-1}, 2J_target) = 2J_target.
- **Per-CSF normalization**: 2J_target is inferred individually for each CSF from the coupling value of the last *occupied* subshell (`infer_two_j_target` at `descriptor_normalization.rs:331–338`), making this correct for mixed-J-pi datasets.

---

## 7. Output Format

Descriptors are written to columnar Parquet files with configurable compression (default ZSTD level 3) and PLAIN encoding (dictionary disabled for throughput):

- **Column names**: `col_0, col_1, ..., col_{3N-1}`
- **Data type**: `Int32` (raw) or `Float32` (normalized)
- **Row order**: preserved from input

This multi-column layout avoids Arrow ListArray overhead and supports efficient bulk reads via Polars or PyArrow:

```python
import polars as pl
df = pl.read_parquet("descriptors.parquet")
X = df.to_numpy()  # shape: (n_csfs, 3N)
```

---

## 8. Pipeline Architecture

The parallel pipeline (`generate_descriptors_from_parquet_parallel`, `src/csfs_descriptor.rs:501–974`) uses three concurrent stages:

1. **Reader thread** — streams input Parquet in 65,536-row batches via Arrow's batch reader.
2. **Worker threads** (W threads, default = CPU count) — compete on a bounded crossbeam channel; each batch is parsed in parallel via Rayon's work-stealing within the worker. When normalization is enabled, `normalize_descriptor_per_csf` is applied per row.
3. **Writer thread** — receives completed batches and maintains a BTreeMap to reassemble output in original order before writing to Parquet.

All three stages overlap I/O and computation. The Python binding releases the GIL during processing (`py.detach()`), enabling concurrent Python threads.

---

## 9. Python API

```python
from rcsfs import read_peel_subshells, generate_descriptors_from_parquet

# Read peel subshells from the header file produced during CSF-to-Parquet conversion
peel_subshells = read_peel_subshells("data_header.toml")
# e.g., ['5s', '4d-', '4d', '5p-', '5p', '6s']

# Generate raw (integer) descriptors
stats = generate_descriptors_from_parquet(
    "csfs_data.parquet",
    "descriptors.parquet",
    peel_subshells=peel_subshells,
)

# Generate normalized (float, [0,1]) descriptors
stats = generate_descriptors_from_parquet(
    "csfs_data.parquet",
    "descriptors_normalized.parquet",
    peel_subshells=peel_subshells,
    normalize=True,
)
```

---

## 10. Summary of Design Decisions

Two key design decisions distinguish this descriptor construction from naive featurization:

1. **Doubled-integer convention** for exact angular momentum arithmetic — avoids floating-point representation of half-integer J values, preserving exact coupling algebra throughout the pipeline.

2. **Per-CSF, position-dependent normalization** that exploits occupation-dependent coupling bounds (u_i = n_i * (g_i - n_i)) rather than fixed subshell-type bounds (kappa_i^2) — produces tighter [0, 1] scaling that reflects the actual physical constraints of each individual CSF, not just the subshell type.
