"""Test-only subprocess driver; no Python CLI is shipped with rcsfs."""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def executable() -> Path:
    override = os.environ.get("RCSFS_TEST_CLI")
    if override:
        path = Path(override).resolve()
    else:
        result = subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--locked",
                "--no-default-features",
                "--features",
                "cli",
                "--bin",
                "rcsfs",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
        if not target.is_absolute():
            target = ROOT / target
        path = target / "release" / ("rcsfs.exe" if os.name == "nt" else "rcsfs")
    if not path.is_file():
        raise RuntimeError(f"Native CLI is missing: {path}")
    return path


def run(
    args: Sequence[str], *, input_text: str | None = None, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = ""
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [str(executable()), *args],
        input=input_text,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


def main(args: Sequence[str]) -> int:
    result = run(args)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode
