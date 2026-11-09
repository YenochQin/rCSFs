# Building the extension with Python 3.15

Run commands from the `rCSFs/` root unless stated otherwise. This subrepository uses its own Python 3.15 `.venv`. On Windows initialize the compiler/linker environment first and use `.venv/Scripts/activate` and `.venv/Scripts/python.exe` in the commands below.

## Local environment and installed wheel

```sh
uv sync --python 3.15 --no-editable --group lint
source .venv/bin/activate
```

This repository's `pyproject.toml` declares a local PEP 517 backend that combines two Maturin builds and tracks Rust/Python source files in uv cache keys. Sync rebuilds and reinstalls the wheel when those inputs change. `--no-editable` preserves wheel installation. The `dev` dependency group includes Maturin for explicit local builds.

The paired Tools environment remains separate. Follow its own environment and dependency instructions when testing it against this source, which requires Python 3.15.

## In-tree extension used by this repository's Python tests

Pytest adds the repository root to its import path. Its `rcsfs/` package can therefore load a different extension from the installed wheel. For native builds using the default Cargo target directory:

```sh
source .venv/bin/activate
cargo build --release --features pyo3/extension-module,pyo3/generate-import-lib
python - <<'COPY_EXTENSION'
from pathlib import Path
import shutil
import sys
import sysconfig

artifact = {"darwin": "lib_rcsfs.dylib", "win32": "_rcsfs.dll"}.get(sys.platform, "lib_rcsfs.so")
source = Path("target/release") / artifact
suffix = sysconfig.get_config_var("EXT_SUFFIX")
if not source.is_file() or not suffix:
    raise SystemExit("Expected native Cargo artifact or Python extension suffix is missing")
shutil.copy2(source, Path("rcsfs") / f"_rcsfs{suffix}")
COPY_EXTENSION
python -c 'import rcsfs._rcsfs as extension; print(extension.__file__)'
```

The copy derives the ABI suffix from the local interpreter and handles Linux/macOS/Windows library filenames. The shell block uses POSIX syntax; on Windows run the same Python copy logic with the activated local interpreter. Adjust the source path if Cargo uses a custom target directory or target triple; do not copy a library built for another interpreter or architecture. Then run the affected tests, for example `python -m pytest tests/rcsfs_test.py`.

Rust tests use `cargo test --release --locked` with the local environment activated and do not require this extension copy.

## Distributable wheels

For an explicit packaging task, use an isolated tool environment while targeting the local interpreter:

```sh
uv build --python .venv/bin/python --out-dir dist
```

Use `.venv/Scripts/python.exe` on Windows. The combined wheel and source archive are written under `dist/`. This command does not refresh the in-tree extension or install the wheel into the environment. Keep the Maturin version constraint aligned with `pyproject.toml` when updating build dependencies.

Avoid `maturin develop`: its editable extension installation differs from the intended wheel installation. Restore the local installation with `uv sync --no-editable --group lint` when needed.

For source-archive rebuilding, release manifests, license notices and the
build-only platform matrix, follow [the release guide](releasing.md).

## Native Rust CLI

The `cli` feature enables the `rcsfs` binary; `python` remains the default
feature for Python consumers. To build an executable without PyO3 or CPython:

```sh
cargo build --release --locked --no-default-features --features cli --bin rcsfs
cargo test --release --locked --no-default-features --features cli
cargo tree --locked --no-default-features --features cli
```

The dependency tree must not contain `pyo3` or `pyo3-arrow`. The executable is
`target/release/rcsfs` (`rcsfs.exe` on Windows). The native executable tests clear
`PATH` and exercise generation, configuration, file protection and descriptor
round trips. They use only maintained fixtures and small synthetic models.
They do not run GRASP or validate scientific accuracy. The separate
[native workflow](../.github/workflows/native-cli.yml) tests the three release
platforms; executing one platform locally does not validate the others.

To install from source:

```sh
cargo install --path . --locked --no-default-features --features cli --bin rcsfs
```

Put the installation root's `bin` directory on `PATH`. Cargo's default root is
`~/.cargo`. Use `rcsfs --version` to identify the native entry point; it includes
`(native Rust)`. Wheels contain this executable in their scripts installation scheme
and keep the PyO3 library. There is no Python CLI or console-script wrapper.

After changing shared Rust code, also rebuild the Python extension and run the
Python suite using the instructions above. The two builds select different
features; do not copy a Python-free shared library into the Python package.
Run `python scripts/collect_licenses.py` after dependency changes: it collects
all features so notices cover both the Python extension and native executable.


## Combined wheel backend

`build_backend.py` implements PEP 517 and PEP 660. It asks Maturin to build the
CLI with `--bindings bin --no-default-features --features cli --auditwheel check`,
then builds the PyO3 wheel. It accepts only identical package/version and platform
tags, copies the real native executable to `.data/scripts`, and rewrites RECORD.
Both regular and editable installs therefore expose the same native command.

Use `uv build`, `python -m build`, or pip/uv source installation so this backend
runs. A single direct `maturin build` produces only one component. The release
workflow explicitly builds and audits both components before calling the same
merger. A native build requiring a Python repair launcher is rejected; resolve
its native linkage instead of silently reintroducing Python command startup.

The Python CLI integration tests now invoke a real executable through the
private test driver `tests/_native_cli.py`; they do not ship a Python adapter.
It builds the CLI once per test process, or uses `RCSFS_TEST_CLI` when an installed
wheel executable is explicitly supplied. Python implementation-mock tests were
retired; configuration, names, publication, logs and round trips are exercised
through actual native commands and `tests/native_cli_test.rs`.
