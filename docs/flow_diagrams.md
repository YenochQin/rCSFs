# rCSFs Code Execution Flow Diagrams

## Diagram 1: Top-Level Architecture

```mermaid
graph TD
    A([User Python Code]) --> B[rcsfs/__init__.py\nPublic API Wrapper]
    B -->|Path conversion| C[[PyO3 Extension\nsrc/lib.rs]]
    C --> D[csfs_conversion.rs\nCSF → Parquet]
    C --> E[csfs_descriptor.rs\nParquet → Descriptors]
    E --> F[descriptor_normalization.rs\nPhysics Normalization]

    D --> G[(CSF Parquet\nUNCOMPRESSED)]
    D --> H[(Header TOML\nhuman-readable)]
    F --> I[(Descriptor Parquet\nZSTD level 3)]

    G --> E

    style C fill:#f9f,stroke:#333
    style F fill:#bbf,stroke:#333
```

---

## Diagram 2: CSF Conversion Pipeline (`convert_csfs`)

```mermaid
graph TD
    A([convert_csfs called]) --> B{Validate params\nmax_line_len > 0\nchunk_size > 0}
    B -->|Invalid| ERR1([PyValueError raised])
    B -->|Valid| C[Open input .csf file\nBufReader]

    C --> D[Extract 5-line header\nfirst 5 lines]
    D --> E[Create Parquet schema\nidx:UInt64, line1-3:Utf8]
    E --> F[Create ArrowWriter\nUNCOMPRESSED]

    F --> G{{Streaming batch loop\nread chunk_size lines}}
    G -->|chunk read| H{Enough for\ncomplete CSFs?\n3 lines each}
    H -->|No / EOF| FIN[Break loop]
    H -->|Yes| I[par_iter over\n3-line CSF chunks]

    I --> J[[Rayon work-stealing\nN CPU threads]]
    J --> K{Line length\n> max_line_len?}
    K -->|Yes| L[Truncate line\ntruncated_count++]
    K -->|No| M[Keep as-is]
    L --> N[Collect ordered results]
    M --> N

    N --> O[Build Arrow columns\nUInt64Builder + StringBuilders]
    O --> P[Write RecordBatch\nto Parquet]
    P --> G

    FIN --> Q[Close ArrowWriter\nflush to disk]
    Q --> R[Write _header.toml\nheader lines + stats]
    R --> S([Return ConversionStats])

    P -.->|error| CLEANUP[ParquetFileGuard\ndelete partial file]
    CLEANUP --> ERR2([Return error])

    style J fill:#f9f,stroke:#333
    style CLEANUP fill:#faa,stroke:#333
```

---

## Diagram 3: Descriptor Generation Pipeline (`generate_descriptors_from_parquet`)

```mermaid
graph TD
    A([generate_descriptors called]) --> B[Parse peel_subshells list\nbuild orbital_index_map]
    B --> C[Open input Parquet\nCreate output writer]
    C --> D[Spawn 3-thread pipeline]

    D --> R[Reader Thread]
    D --> W[N Worker Threads]
    D --> WR[Writer Thread]

    subgraph Reader
        R --> R1{{Read Parquet\n65536 rows/batch}}
        R1 -->|batch ready| R2[Pack WorkItem\nbatch_idx + rows\nArc str zero-copy]
        R2 -->|send| CH1[(work_tx channel\nbounded)]
        R1 -->|EOF| R3[Close work_tx]
    end

    subgraph Workers
        CH1 -->|receive| W1[Get WorkItem]
        W1 --> W2[[rayon par_iter\nover rows in batch]]
        W2 --> W3[parse_csf\nline1, line2, line3]
        W3 --> W4{Parse\nsucceeded?}
        W4 -->|Yes| W5[Vec of i32 descriptors]
        W4 -->|No| W6[Warn + substitute\nall-zeros descriptor]
        W5 --> W7[Collect ResultItem\nbatch_idx + descriptors]
        W6 --> W7
        W7 -->|send| CH2[(result_tx channel\nbounded)]
    end

    subgraph Writer
        CH2 -->|receive| WR1[Buffer in BTreeMap\nfor ordering]
        WR1 -->|next batch ready| WR2{normalize=true?}
        WR2 -->|Yes| WR3[Read 2J_target\nfrom last descriptor col]
        WR3 --> WR4[[normalize_descriptor\n_per_csf]]
        WR4 --> WR5[Build Float32 columns]
        WR2 -->|No| WR6[Build Int32 columns]
        WR5 --> WR7[Write RecordBatch\nZSTD level 3]
        WR6 --> WR7
        WR7 --> WR1
        WR1 -->|channel closed| WR8[Close writer]
    end

    WR8 --> Z([Return DescriptorGenerationStats])

    style W2 fill:#f9f,stroke:#333
    style WR4 fill:#bbf,stroke:#333
    style CH1 fill:#efe,stroke:#333
    style CH2 fill:#efe,stroke:#333
```

---

## Diagram 4: Per-CSF Physics Normalization (`normalize_descriptor_per_csf`)

```mermaid
graph TD
    A([normalize_descriptor_per_csf\ndescriptor, peel_subshells, two_j_target]) --> B{{For each subshell i\ncompute capacities}}

    B --> C[g_i = 2 × abs kappa_i\nmax electrons]
    C --> D[n_i = descriptor 3i\nu_i = n_i × g_i - n_i]

    D --> E[Compute prefix sums\nprefix i = Σ u_k for k=0..i]
    E --> F[Compute suffix sums\nsuffix i = Σ u_k for k=i+1..N]

    F --> G{{For each subshell i\nnormalize triplet}}

    G --> H[OUT 3i = n_i / g_i\nfill fraction → 0..1]

    G --> I{u_i > 0?\nn_i not 0 or g_i}
    I -->|No empty/full subshell| J[OUT 3i+1 = 0.0\nOUT 3i+2 = 0.0]
    I -->|Yes partially occupied| K[OUT 3i+1 = 2Q_i / u_i\nquadrupole normalized]

    K --> L[U_i_occ = min\nprefix i, 2J_target + suffix i]
    L --> M{U_i_occ > 0?}
    M -->|No| N[OUT 3i+2 = 0.0]
    M -->|Yes| O[OUT 3i+2 = 2J_cum,i / U_i_occ\ncoupling normalized]

    H --> P[Collect all triplets]
    J --> P
    N --> P
    O --> P

    P --> Q([Return Vec f32\nall values in 0..1])

    style L fill:#bbf,stroke:#333
```

---

## Diagram 5: `parse_csf` — Single CSF Parsing

```mermaid
graph TD
    A([parse_csf\nline1 line2 line3]) --> B[Init descriptor\n0; 3 × orbital_count]
    B --> C[Extract final J\nfrom end of line3]
    C --> D[Pad all lines\nto equal length]

    D --> E{{Split into\n9-char blocks}}

    E --> F[Extract subshell name\nchars 0-5 trimmed]
    F --> G[Lookup subshell in\norbital_index_map]

    G --> H{Subshell found\nin peel list?}
    H -->|No| SKIP[Skip block]
    H -->|Yes| I[Parse n_electrons\nchars 6-8 of line1]

    I --> J{n_electrons == 0?}
    J -->|Yes| ZERO[Store 0, 0, 0\nat orb_idx × 3]
    J -->|No| K[Parse middle J\nfrom line2 block]

    K --> L{Semicolon\nseparated?}
    L -->|Yes| M[Take last value]
    L -->|No| K2[Use block value]
    M --> N[Parse coupling J\nfrom line3 block]
    K2 --> N

    N --> O{Is last\nsubshell?}
    O -->|Yes| P[Use final J\nfrom line3 end]
    O -->|No| Q[Use block J]

    P --> R[Store n, middle_J, coupling_J\nat descriptor idx × 3]
    Q --> R
    ZERO --> S[Next block]
    R --> S
    SKIP --> S
    S --> E

    E -->|done| T([Return Vec i32\nlength 3 × orbital_count])
```

---

## Logic Breakdown

**Happy Path:**
- `convert_csfs` → reads text, chunks into batches, rayon-parallelizes line truncation, writes ordered Parquet
- `generate_descriptors_from_parquet` → 3-thread pipeline (reader → workers → writer) with rayon inside workers
- `normalize_descriptor_per_csf` → per-CSF u-vector calculation, prefix/suffix sums, three normalized scalars per orbital

**Edge Cases:**
- Lines exceeding `max_line_len` — truncated, counted, file still produced
- Incomplete CSF (not multiple of 3 lines) at EOF — silently dropped
- `parse_csf` failure — all-zeros substituted with a warning, pipeline continues
- Empty or fully-filled subshell (`u_i = 0`) — normalization outputs `0.0` (physically correct, not NaN)
- Out-of-order batch results from workers — BTreeMap in writer enforces batch ordering

---

## Potential Issues

| Location | Issue | Risk |
|---|---|---|
| `csfs_conversion.rs` EOF | Partial CSF at end is **silently dropped** — no warning emitted | Silent data loss if file is malformed |
| `csfs_descriptor.rs` writer | BTreeMap grows unbounded if a worker stalls and results arrive out-of-order | Memory spike on very large files with slow workers |
| `descriptor_normalization.rs` | `two_j_target` is read from `descriptor[descriptor_len - 1]`; if descriptor is empty, this panics | Panic on malformed input with zero orbitals |
| `parse_csf` | 9-char fixed-width chunking assumes exact column alignment — misaligned CSF files produce wrong data silently | Silent corruption on non-standard CSF files |
| Python wrapper | `num_workers=None` passes through to Rust which uses `rayon::current_num_threads()` — no user visibility into actual thread count | Hard to reproduce performance issues |
