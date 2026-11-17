# Native descriptor reader for Fortran

The `native-reader/` crate builds a small static/shared library from the same
`descriptor_transport.rs`, `descriptor_reader.rs`, and `descriptor_ffi.rs` sources
used by rCSFs. It has no Python, CLI, generator, or Arrow C++ dependency. Build it
from a complete rCSFs checkout/source distribution; its shared source paths are
relative to the repository root. Cargo excludes nested packages from the main
crate archive: obtain this standalone build directory from the complete Git
source bundle. The main crate archive does include the shared implementation and
C header, and can alternatively build the ABI with `--no-default-features
--features descriptor-ffi --lib`. The standalone crate is not separately published.

```fish
cargo build --manifest-path native-reader/Cargo.toml --release --locked --offline
cargo test --manifest-path native-reader/Cargo.toml --locked --offline
```

The validated compiler is Rust 1.99.0; `native-reader/Cargo.lock` fixes dependency
versions. An empty Cargo cache must be populated before using `--offline`.
ZSTD uses a C backend at build time. Consumers of the static archive still link
the platform libraries required by Rust; the suite CMake integration handles the
validated Unix link configuration.

ABI version 1 is declared in [rcsfs_descriptor.h](../include/rcsfs_descriptor.h).
Initialize a handle slot to NULL, open a descriptor Parquet and matching header
TOML, query the five metadata integers, allocate tables using those counts, read
rows into caller-owned buffers, then close by address. Table capacities count
orbital/block records; each orbital and symmetry record has two integers.
Row buffers use `channel + row_width * row`, matching Fortran
`values(channel, row)`. Counts are wide; consumers enforce their own capacities.

Handles are independent and require exclusive access. Closing clears the slot
and accepts an already NULL slot. Opening over a live handle is rejected without
destroying it. All fallible calls return a status and bounded, NUL-terminated UTF-8
error text. Failed read output must not be consumed. A failed reader must be
closed and reopened. Ordinary Rust errors and unwinding panics do not cross the
C boundary; this does not make abort/OOM recoverable.

The reader hashes and parses the same header byte snapshot. It validates unique
required footer keys, the ordered non-null Int8 schema, actual nulls, orbital
identity/order, checked block lengths, electron counts, row parity, and block
symmetry. A bounded prepass constructs the block table, then the stream is reset
without restoring text or caching all rows. Zero peel subshells are supported
with an explicit closed-core `[0,+1]` row. Files must remain immutable while open.
Parquet page/footer buffers are managed by the backend in addition to the bounded
row buffers.

Physical state-table checks, omitted J/K resolution, supported `w` restrictions,
and application array installation remain in the suite Fortran adapter. The
transport reader alone is not a certificate of physical correctness. The header
hash does not bind MCP/mixing products or establish payload identity.

The established producer API remains at `descriptor_schema`; it re-exports the
shared transport contract and retains producer-side record validation. The main
rCSFs crate also exposes the C ABI through the optional `descriptor-ffi` feature,
but the standalone native reader is preferred for Fortran to avoid unrelated
dependencies. No Python exports or descriptor file version were changed.
