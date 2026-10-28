# Building the extension with Python 3.15

Run commands from the `rCSFs/` root unless stated otherwise. This subrepository uses its own Python 3.15 `.venv`. On Windows initialize the compiler/linker environment first and use `.venv/Scripts/activate` and `.venv/Scripts/python.exe` in the commands below.

## Local environment and installed wheel

```sh
uv sync --python 3.15 --no-editable --group lint
source .venv/bin/activate
```

This repository's `pyproject.toml` declares Maturin as the build backend and tracks Rust/Python source files in uv cache keys. Sync rebuilds and reinstalls the wheel when those inputs change. `--no-editable` preserves wheel installation. The `dev` dependency group includes Maturin for explicit local builds.

The paired Tools environment remains separate. Its current Python 3.14 runtime cannot install this checkout, whose metadata requires Python 3.15. Migrate Tools and its dependency constraints separately before testing it against this source.

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
uvx --python .venv/bin/python --from 'maturin[patchelf]>=1.14,<2.0' maturin build --release --locked --compatibility pypi --interpreter .venv/bin/python
```

Use `.venv/Scripts/python.exe` on Windows. Wheels are written under `target/wheels/`. This command does not refresh the in-tree extension or install the wheel into the environment. Keep the Maturin version constraint aligned with `pyproject.toml` when updating build dependencies.

Avoid `maturin develop`: its editable extension installation differs from the intended wheel installation. Restore the local installation with `uv sync --no-editable --group lint` when needed.

For source-archive rebuilding, release manifests, license notices and the
build-only platform matrix, follow [the release guide](releasing.md).
