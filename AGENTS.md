# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the Rust core: `csfs_conversion.rs`, `csfs_descriptor.rs`, and normalization logic exposed through `src/lib.rs`. `rcsfs/` is the Python package surface, including `__init__.py`, type hints, and the compiled extension module. `tests/` holds Rust integration tests, Python-facing checks, and fixtures such as `tests/fixtures/sample.csf`. Use `docs/` for design notes, review logs, and changelog entries. Treat `dist/` and `target/` as build output unless a release task explicitly needs them.

## Build, Test, and Development Commands
- `uv sync --group dev --group lint` installs Python dev and lint dependencies.
- `maturin develop` builds the Rust extension and installs `rcsfs` into the active environment for local testing.
- `cargo build --release` produces optimized Rust artifacts.
- `cargo test` runs Rust unit and integration tests in `tests/*.rs`.
- `pytest tests/rcsfs_test.py` runs Python API checks against the built extension.
- `ruff check .` and `mypy rcsfs` enforce Python linting and type checks.
- `pixi shell` is an acceptable alternative environment bootstrap when working from the documented Pixi setup.

## Coding Style & Naming Conventions
Follow Rust 2024 idioms: 4-space indentation, `snake_case` for modules/functions, `CamelCase` for types, and small focused modules. Keep PyO3 bindings in `src/lib.rs` thin; push heavy logic into Rust modules. In Python, use PEP 8 naming, type hints for public APIs, and keep package exports aligned with `rcsfs/_rcsfs.pyi`. Prefer clear file names such as `*_test.rs` and avoid committing exploratory notebooks or ad hoc scripts to the root.

## Testing Guidelines
Add Rust coverage for core parsing, conversion, and descriptor behavior in `tests/`. Add Python regression tests when changing the public package API or file I/O behavior. Name Rust tests `*_test.rs`; keep Python tests under `tests/` and start functions with `test_`. Reuse `tests/fixtures/` for stable sample data. Run both `cargo test` and `pytest` before opening a PR.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects such as `update linux build` or `create win artifact`; keep subjects brief and descriptive, and expand in the body when needed. PRs should explain the user-visible change, list validation commands run, and link related issues or docs. Include sample output or screenshots only when CLI/API behavior, generated files, or documentation rendering changes.

## Configuration & Release Notes
This project targets Python 3.14 and builds through Maturin. Keep version changes aligned between `Cargo.toml` and packaging metadata, and document release-facing behavior in `docs/CHANGELOG.md` when appropriate.
