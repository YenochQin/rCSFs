# rCSFs Guidelines

## Ownership and interfaces

`rCSFs` owns CSF conversion, generation, V2 descriptors, restoration, and splitting. Rust code is in `src/`, the Python API in `rcsfs/`, and maintained tests in `tests/`. The public package is `rcsfs`; the compiled extension is `rcsfs._rcsfs`. The product CLI is the native Rust executable in `src/cli/`; keep Python library APIs independent of command parsing.

- Keep PyO3 bindings in `src/lib.rs` thin and put algorithms in Rust modules.
- Public Python functions and types are defined by `rcsfs/__init__.py` and `rcsfs/_rcsfs.pyi`; update them together. Internal Rust types are not automatically Python exports.
- Preserve `pathlib.Path` support and validate external inputs at the Python boundary.
- Use `src/descriptor_schema.rs` for the V2 contract and `src/descriptor_v2.rs` for encoding, decoding, and restoration. Consult these files when changing formats rather than reproducing their schema elsewhere.

## Environment and build routing

This subrepository uses its own Python 3.15 `.venv`, independently of the paired workspace's Tools environment. Run setup and synchronization here, then activate it before Cargo so PyO3 finds the intended interpreter.

```sh
uv sync --python 3.15 --no-editable --group lint
source .venv/bin/activate
```

Choose the build output the task actually uses:

- Rust algorithms: `cargo test` or a relevant test-name filter needs no wheel build.
- Local installed wheel: `uv sync --no-editable --group lint` rebuilds the package when its source cache keys change.
- Python tests here: pytest's `pythonpath = ["."]` selects the in-tree `rcsfs/` package. Refresh its extension after Rust changes; syncing the environment alone does not update it.
- Distributable wheel: use the PEP 517 backend through `uv build --python .venv/bin/python`; it combines separately audited native-CLI and PyO3 wheels. A single direct Maturin build is only a component. Keep wheel and editable hooks, `.data/scripts` permissions, RECORD, sdist inclusion and release checks in sync.
- Tools consumers: this checkout requires Python 3.15. Keep Tools on its declared released dependency until its environment and dependency constraints have been migrated separately.

For in-tree tests and distributable wheels, read [development.md](docs/development.md). Use wheel installation for this environment; avoid `maturin develop`, which replaces it with an editable installation.

## Format invariants

V2 descriptors store raw `Int8` channels and source-header metadata. The `v` channel is physical seniority, including zero for empty/closed subshells; files require `seniority_encoding=physical`, `descriptor_dtype=int8`, and `max_two_j=126`. Validate 2J/2K in 0..=126 before narrowing; preserve wide indices/counts and arithmetic temporaries. Preserve row order, header association, parity, and the distinction between missing printed J/coupling values (`-1`) and printed zero. Restore seniority labels through the shared `src/seniority.rs` state table; reject old printed-v descriptor files rather than silently interpreting them as physical values. V1 generation and the Python/CLI `normalize` option are removed; ML feature scaling belongs to the consuming pipeline.

J is encoded as integer **2J**: text `3/2` becomes `3`, while `4` or `4-` becomes `8`; parity is stored separately. CSF records have three lines after the five-line header. Test changes to parsing, conversion, restoration, or splitting against realistic fixtures.

## Validation and temporary work

The maintained suite covers this repository's algorithms, APIs, CLI, and formats. It must be self-contained and must not require an external GRASP checkout/executable or private baseline dataset. Stored fixtures may encode expected compatibility behavior.

Put exploratory probes and one-off external comparisons under ignored `temp/`, outside Cargo/pytest discovery. Maintained tests can use ordinary temporary directories for I/O. Keep stable small fixtures in `tests/fixtures/` and add regression coverage for changed public behavior.

Run affected Rust or Python tests; run both when crossing the binding or file-format boundary. Use `python -m pytest tests/rcsfs_test.py` for API checks, `ruff check <changed-paths>` for Python lint, and `basedpyright rcsfs/` for wrapper typing. CI and local test environments share the `test` dependency group; performance measurements use the scripts under `scripts/`. Expand to full suites when the affected contract is shared; documentation-only changes need example/link checks.

## Commit attribution

Use the contributor's configured Git identity for author and committer. Include only human contributors in `Co-authored-by` trailers; omit AI co-author trailers and generated-with or ultraworked-with footers. This applies to every coding agent, including Claude, Sisyphus, and Codex.

Enable the repository's commit-message check in each clone with `git config --local core.hooksPath .githooks`. The hook rejects known AI attribution while preserving human co-authors. Project Claude settings also disable automatic commit attribution.

Use release artifacts for release validation and read build settings from `Cargo.toml`. Keep version metadata consistent and update `docs/CHANGELOG.md` for release-facing changes. Exclude build outputs, local datasets, credentials, and machine paths from commits; follow the workspace's two-layer commit workflow when requested.
