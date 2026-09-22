"""Registered benchmark reports must stay free of machine-specific data.

Reports under ``docs/benchmarks`` are committed and compared with each other, so
a path from the machine that produced one is both noise and a leak of a local
layout. The filesystem type and capacity are the machine facts a reader needs;
the absolute paths are not.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIRECTORY = REPO_ROOT / "docs" / "benchmarks"
FIXTURE_DIRECTORY = REPO_ROOT / "tests" / "fixtures"

#: Reports measured before source identity was captured. Their numbers are the
#: pre-planning baseline and are kept as recorded; the revision, tree and
#: extension of that run cannot be reconstructed now, and the 2026-09-22 matrix
#: supersedes them for every comparison a reader would make today.
LEGACY_REPORTS = {
    "v2_disk_generation_p0b_b1_20260921.json",
    "v2_disk_generation_p0b_b2_20260921.json",
}


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


WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _absolute_paths(value: object, path: str = "") -> list[str]:
    """Every string in a report that looks like an absolute filesystem path."""
    found: list[str] = []
    if isinstance(value, str):
        if (
            value.startswith("/")
            or WINDOWS_ABSOLUTE.match(value)
            or value.startswith("\\\\")
        ):
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


def test_environment_identifies_the_source_and_the_binary() -> None:
    """A report must let a reader tell which source produced the measurement.

    A commit hash alone does not: the tree may have been dirty, and the loaded
    extension may not have been rebuilt from that commit.
    """
    environment = support.environment()
    git = environment["git"]
    assert git["commit"] and git["tree"]
    assert git["dirty"] in (True, False, None)
    if git["dirty"]:
        assert git["dirty_diff_sha256"], "a dirty report must identify its diff"
    else:
        assert "dirty_diff_sha256" not in git
    extension = environment["extension"]
    assert extension["module_sha256"]
    assert extension["module"].startswith("_rcsfs")


@pytest.mark.parametrize(
    "report", sorted(BENCHMARK_DIRECTORY.glob("*.json")), ids=lambda path: path.name
)
def test_registered_report_names_its_source(report: Path) -> None:
    """Every registered report records the revision and the binary it measured."""
    if report.name in LEGACY_REPORTS:
        pytest.skip("measured before source identity was captured")
    document = json.loads(report.read_text(encoding="utf-8"))
    git = document["environment"]["git"]
    assert git["tree"], f"{report.name} does not record the source tree"
    assert document["environment"]["extension"]["module_sha256"], (
        f"{report.name} does not record the measured extension"
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # A drive-absolute path, in either separator spelling.
        ("C:\\Users\\alice\\Temp\\rcsfs-run", "<path>/rcsfs-run"),
        ("C:/Users/alice/Temp/rcsfs-run", "<path>/rcsfs-run"),
        ("D:\\scratch", "<path>/scratch"),
        # A UNC path.
        ("\\\\server\\share\\scratch", "<path>/scratch"),
        # A POSIX path is still handled.
        ("/mnt/data/scratch", "<path>/scratch"),
        # A relative path is left alone: it is not machine-specific.
        ("docs/benchmarks/report.json", "docs/benchmarks/report.json"),
        ("report.json", "report.json"),
    ],
)
def test_paths_of_either_platform_are_normalized(value: str, expected: str) -> None:
    """A report may be written on one platform from measurements on another."""
    assert support.normalize_path(value) == expected


def test_a_windows_root_is_matched_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows filesystems are case-insensitive, and its roots must match too."""
    monkeypatch.setattr(
        support,
        "_PATH_PLACEHOLDERS",
        ((r"C:\Users\Alice\AppData\Local\Temp", "<system-temp>"),),
    )
    assert support.normalize_path(
        r"c:\users\alice\appdata\local\temp\rcsfs-run"
    ) == "<system-temp>/rcsfs-run"
    assert support.normalize_path(r"C:/Users/Alice/AppData/Local/Temp/rcsfs-run") == (
        "<system-temp>/rcsfs-run"
    )
    # A different directory is not swallowed by the root above it.
    assert support.normalize_path(r"C:\Users\Alice\Other") == "<path>/Other"
