# rCSFs Guidelines

## Ownership and interfaces

`rCSFs` owns CSF conversion, generation, V2 descriptors, restoration, and splitting. Rust code is in `src/`, the Python API in `rcsfs/`, and maintained tests in `tests/`. The public package is `rcsfs`; the compiled extension is `rcsfs._rcsfs`.

- Keep PyO3 bindings in `src/lib.rs` thin and put algorithms in Rust modules.
- Public Python functions and types are defined by `rcsfs/__init__.py` and `rcsfs/_rcsfs.pyi`; update them together. Internal Rust types are not automatically Python exports.
- Preserve `pathlib.Path` support and validate external inputs at the Python boundary.
- Use `src/descriptor_schema.rs` for the V2 contract and `src/descriptor_v2.rs` for encoding, decoding, and restoration. Consult these files when changing formats rather than reproducing their schema elsewhere.

## Environment and build routing

Use only `../graspkit-tools/.venv` for project Python execution and activate it before Cargo so PyO3 finds Python 3.14. Do not run `uv venv`, `uv sync`, or `uv run` in this repository or create a local environment.

```sh
source ../graspkit-tools/.venv/bin/activate
```

Choose the build output the task actually uses:

- Rust algorithms: `cargo test` or a relevant test-name filter needs no wheel build.
- Tools consumers: run `uv sync --extra cpu` from `../graspkit-tools` after Rust changes (`--extra gpu` on CUDA hosts). Source cache keys in this repository's `pyproject.toml` trigger rebuilding the installed wheel.
- Python tests here: pytest's `pythonpath = ["."]` selects the in-tree `rcsfs/` package. Refresh its extension after Rust changes; syncing Tools alone does not update it.
- Distributable wheel: use isolated Maturin with the shared Python as the explicit target interpreter.

For the latter two cases, read [development.md](docs/development.md). Keep Maturin out of the shared runtime and avoid `maturin develop`, which changes the intended wheel installation to an editable one. Isolated build/tool environments are packaging implementation details, not additional project runtimes.

## Format invariants

V2 descriptors store raw `Int32` channels and source-header metadata. Preserve row order, header association, parity, and the distinction between missing printed values (`-1`) and printed zero. V1 generation and the Python/CLI `normalize` option are removed; ML feature scaling belongs to the consuming pipeline.

J is encoded as integer **2J**: text `3/2` becomes `3`, while `4` or `4-` becomes `8`; parity is stored separately. CSF records have three lines after the five-line header. Test changes to parsing, conversion, restoration, or splitting against realistic fixtures.

## Validation and temporary work

The maintained suite covers this repository's algorithms, APIs, CLI, and formats. It must be self-contained and must not require an external GRASP checkout/executable or private baseline dataset. Stored fixtures may encode expected compatibility behavior.

Put exploratory probes and one-off external comparisons under ignored `temp/`, outside Cargo/pytest discovery. Maintained tests can use ordinary temporary directories for I/O. Keep stable small fixtures in `tests/fixtures/` and add regression coverage for changed public behavior.

Run affected Rust or Python tests; run both when crossing the binding or file-format boundary. Use `python -m pytest tests/rcsfs_test.py` for API checks, `ruff check <changed-paths>` for Python lint, and `basedpyright rcsfs/` for wrapper typing. Use `pytest --speed` only for relevant performance work. Expand to full suites when the affected contract is shared; documentation-only changes need example/link checks.

Use release artifacts for release validation and read build settings from `Cargo.toml`. Keep version metadata consistent and update `docs/CHANGELOG.md` for release-facing changes. Exclude build outputs, local datasets, credentials, and machine paths from commits; follow the workspace's two-layer commit workflow when requested.
