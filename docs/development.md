# Building the extension in the paired workspace

Run commands from the `rCSFs/` root unless stated otherwise. The only project Python runtime is `../graspkit-tools/.venv`; synchronize it from Tools with the selected CPU/GPU extra when setup requires it. Activate that environment before Cargo so PyO3 links to its Python 3.14 runtime. On Windows initialize the compiler/linker environment first and use the shared venv's Windows activation script.

## Installed wheel used by Tools

From `graspkit-tools/`:

```sh
uv sync --extra cpu
```

Use `--extra gpu` on CUDA hosts. This repository's `pyproject.toml` declares Maturin as the build backend and tracks Rust/Python source files in uv cache keys. Sync rebuilds and reinstalls the wheel when those inputs change. Build isolation provides Maturin; it does not need to be installed into the shared runtime.

## In-tree extension used by this repository's Python tests

Pytest adds the repository root to its import path. Its `rcsfs/` package can therefore load a different extension from the installed wheel. For native builds using the default Cargo target directory:

```sh
source ../graspkit-tools/.venv/bin/activate
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

The copy derives the ABI suffix from the shared interpreter and handles Linux/macOS/Windows library filenames. The shell block uses POSIX syntax; on Windows run the same Python copy logic with the activated shared interpreter. Adjust the source path if Cargo uses a custom target directory or target triple; do not copy a library built for another interpreter or architecture. Then run the affected tests, for example `python -m pytest tests/rcsfs_test.py`.

Rust-only tests use `cargo test` with the shared environment activated and do not require this extension copy.

## Distributable wheels

For an explicit packaging task, use an isolated tool environment while targeting the shared interpreter:

```sh
uvx --python ../graspkit-tools/.venv/bin/python --from 'maturin>=1.14,<2.0' maturin build --release --interpreter ../graspkit-tools/.venv/bin/python
```

Use the shared `Scripts/python.exe` path on Windows. Wheels are written under `target/wheels/`. This command does not refresh the in-tree extension or install the wheel into Tools. Keep the Maturin version constraint aligned with `pyproject.toml` when updating build dependencies.

Avoid `maturin develop` for this workspace: its editable extension installation differs from Tools' declared built path dependency. Restore that installation through a Tools sync with the selected extra when needed.
