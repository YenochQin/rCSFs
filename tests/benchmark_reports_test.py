"""Registered benchmark reports must stay free of machine-specific data.

Reports under ``docs/benchmarks`` are committed and compared with each other, so
a path from the machine that produced one is both noise and a leak of a local
layout. The filesystem type and capacity are the machine facts a reader needs;
the absolute paths are not.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIRECTORY = REPO_ROOT / "docs" / "benchmarks"
FIXTURE_DIRECTORY = REPO_ROOT / "tests" / "fixtures"


def _load_support() -> object:
    """Import ``scripts/benchmark_support.py`` the way the scripts do."""
    scripts = REPO_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "benchmark_support", scripts / "benchmark_support.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


support = _load_support()


def _absolute_paths(value: object, path: str = "") -> list[str]:
    """Every string in a report that looks like an absolute filesystem path."""
    found: list[str] = []
    if isinstance(value, str):
        if value.startswith("/") or (len(value) > 2 and value[1:3] == ":\\"):
            found.append(f"{path}={value}")
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_absolute_paths(item, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_absolute_paths(item, f"{path}[{index}]"))
    return found


@pytest.mark.parametrize(
    "report", sorted(BENCHMARK_DIRECTORY.glob("*.json")), ids=lambda path: path.name
)
def test_registered_report_has_no_machine_paths(report: Path) -> None:
    document = json.loads(report.read_text(encoding="utf-8"))
    offenders = _absolute_paths(document)
    assert offenders == [], (
        f"{report.name} records machine-specific paths: {offenders[:5]}"
    )


def test_every_registered_fixture_matches_the_manifest() -> None:
    """The scripts' manifest must still describe the fixtures on disk."""
    manifest = support.load_manifest(support.DEFAULT_MANIFEST)
    assert manifest, "the manifest is missing or empty"
    for name, entry in manifest.items():
        fixture = FIXTURE_DIRECTORY / name
        assert fixture.is_file(), f"{name} is registered but missing"
        assert support.sha256_file(fixture) == entry["sha256"], (
            f"{name} drifted from its registered hash"
        )


def test_report_paths_are_normalized_before_being_written() -> None:
    """The normalizer is what keeps the reports above clean."""
    temporary = support.Path(support.tempfile.gettempdir()) / "rcsfs-report-check"
    assert support.normalize_path(str(temporary)).startswith("<system-temp>/")
    assert support.normalize_path(str(support.REPO_ROOT / "docs")) == (
        "<repo-root>/docs"
    )
    # An absolute path outside the known roots keeps only its last component.
    assert support.normalize_path("/mnt/somewhere/scratch") == "<path>/scratch"
    report = {"scratch_root": str(temporary), "counts": [str(temporary / "x")]}
    assert support.normalize_report_paths(report) == {
        "scratch_root": "<system-temp>/rcsfs-report-check",
        "counts": ["<system-temp>/rcsfs-report-check/x"],
    }
