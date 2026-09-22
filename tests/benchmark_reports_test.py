"""Registered benchmark reports must stay free of machine-specific data.

Reports under ``docs/benchmarks`` are committed and compared with each other, so
a path from the machine that produced one is both noise and a leak of a local
layout. The filesystem type and capacity are the machine facts a reader needs;
the absolute paths are not.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIRECTORY = REPO_ROOT / "docs" / "benchmarks"
FIXTURE_DIRECTORY = REPO_ROOT / "tests" / "fixtures"

#: Reports measured before source identity was captured. Their numbers are kept
#: as recorded; the revision, tree and extension of those runs cannot be
#: reconstructed now. The two `v2_disk_generation_p0b_*` reports are the
#: pre-planning baseline that the 2026-09-22 matrix supersedes, and the two
#: `rcsfgenerate_*` ones come from a different script that needs a GRASP
#: checkout and is documented as not reproducible from this repository alone.
LEGACY_REPORTS = {
    "v2_disk_generation_p0b_b1_20260921.json",
    "v2_disk_generation_p0b_b2_20260921.json",
    "rcsfgenerate_parallel_20260912.json",
    "rcsfgenerate_serial_20260910.json",
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


def _load_generation_benchmark() -> object:
    """Import the benchmark module without running its command-line entry point."""
    scripts = REPO_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "benchmark_v2_generation_for_test",
            scripts / "benchmark_v2_generation.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


generation_benchmark = _load_generation_benchmark()


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
    assert git["dirty"] in (True, False)
    # `describe` identifies the revision; whether the source was modified is the
    # `dirty` field's job, and it uses a definition that excludes this harness's
    # own reports, so `--dirty` must not be folded into the version string.
    assert not git["describe"].endswith("-dirty")
    if git["dirty"]:
        assert git["dirty_diff_sha256"], "a dirty report must identify its changes"
        assert git["dirty_paths"], "a dirty report must name what changed"
    else:
        assert "dirty_diff_sha256" not in git
        assert "dirty_paths" not in git
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


def test_duplicate_destination_kind_is_rejected(tmp_path: Path) -> None:
    """One artifact has one destination; two would charge its size twice."""
    script = REPO_ROOT / "scripts" / "estimate_v2_generation.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(FIXTURE_DIRECTORY / "b2_cc1_fullas_2exc.rcsfgenerate"),
            "--destination",
            f"header={tmp_path / 'a.toml'}",
            "--destination",
            f"header={tmp_path / 'b.toml'}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--destination header was given more than once" in result.stderr
    assert list(tmp_path.iterdir()) == []


def _identity(**git: object) -> dict[str, object]:
    """A source identity with the given git fields and a fixed extension."""
    return {
        "git": {"commit": "a", "tree": "t", "dirty": False, **git},
        "extension": {"module_sha256": "e"},
    }


def test_a_dirty_source_must_be_accepted_explicitly() -> None:
    """A registered report claims a source state someone else can rebuild."""
    dirty = _identity(dirty=True, dirty_paths=["src/csf_generation/space.rs"])
    with pytest.raises(SystemExit, match="source tree is dirty"):
        support.require_clean_source(dirty, allow_dirty=False)
    support.require_clean_source(dirty, allow_dirty=True)
    support.require_clean_source(_identity(), allow_dirty=False)


def test_a_source_that_moves_during_a_run_is_refused() -> None:
    """The report describes the state at start, so the state must not move.

    A commit or rebuild during a long run ends with a tree that looks pristine
    while the process ran code from the older state; `--allow-dirty-source`
    accepts a stable dirty tree, not a moving one.
    """
    support.verify_source_unchanged(_identity(), _identity())
    for field, before, after in [
        ("commit", "a", "b"),
        ("tree", "t", "u"),
        ("dirty", False, True),
        ("dirty_diff_sha256", None, "f"),
    ]:
        with pytest.raises(SystemExit, match="source changed while it was being measured"):
            support.verify_source_unchanged(
                _identity(**{field: before}), _identity(**{field: after})
            )
    rebuilt = _identity()
    rebuilt["extension"] = {"module_sha256": "different"}
    with pytest.raises(SystemExit, match="extension.module_sha256"):
        support.verify_source_unchanged(_identity(), rebuilt)


def test_finalize_refuses_a_dirty_source_without_the_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is published for a source that cannot be registered."""
    output = tmp_path / "report.json"
    report: dict[str, object] = {"environment": {}, "measurements": []}
    with pytest.raises(SystemExit, match="source tree is dirty"):
        support.finalize_report(
            report, output, _identity(dirty=True, dirty_paths=["a.rs"]), allow_dirty=False
        )
    assert not output.exists()

    # With the override the report still carries the *pre-run* snapshot, not the
    # one taken at the end of the run.
    start = _identity(dirty=True, dirty_diff_sha256="f")
    monkeypatch.setattr(support, "source_identity", lambda: start)
    support.finalize_report(report, output, start, allow_dirty=True)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["environment"]["git"]["dirty"] is True
    assert written["environment"]["git"]["dirty_diff_sha256"] == "f"
    assert written["environment"]["extension"]["module_sha256"] == "e"


def test_the_source_identity_flag_is_registered() -> None:
    """Both scripts share one definition of the flag."""
    parser = argparse.ArgumentParser()
    support.add_source_identity_arguments(parser)
    assert parser.parse_args([]).allow_dirty_source is False
    assert parser.parse_args(["--allow-dirty-source"]).allow_dirty_source is True


def _temporary_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway git repository that the support module believes is its own."""
    repository = tmp_path / "source"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "commit",
            "--allow-empty",
            "-qm",
            "initial",
        ],
        cwd=repository,
        check=True,
    )
    monkeypatch.setattr(support, "REPO_ROOT", repository)
    return repository


def test_a_file_inside_an_untracked_directory_changes_the_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Untracked directories must be listed file by file.

    `git status` collapses an untracked directory into one `?? dir/` row by
    default, so editing a file inside it would leave both the entries and the
    fingerprint unchanged - and a run whose source changed under it would be
    accepted. `--untracked-files=all` is what keeps them visible.
    """
    repository = _temporary_repository(tmp_path, monkeypatch)
    new_directory = repository / "newdir"
    new_directory.mkdir()
    source = new_directory / "a.rs"
    source.write_text("fn a() {}\n")

    entries = support._source_entries()  # noqa: SLF001 - the fingerprint is the unit
    assert [path for _status, path, _origin in entries] == ["newdir/a.rs"]
    before = support._dirty_fingerprint(entries)

    source.write_text("fn a() { /* edited */ }\n")
    entries = support._source_entries()  # noqa: SLF001
    after = support._dirty_fingerprint(entries)
    assert before != after, "editing a file inside an untracked directory must show up"

    # And the same content hashes the same way, so the fingerprint is stable.
    assert support._dirty_fingerprint(support._source_entries()) == after  # noqa: SLF001


def test_a_directory_entry_is_fingerprinted_recursively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-uall` lists files, but a directory row must not become a constant."""
    repository = _temporary_repository(tmp_path, monkeypatch)
    nested = repository / "dir"
    nested.mkdir()
    (nested / "one.rs").write_text("fn one() {}\n")
    entries = [("??", "dir", None)]

    before = support._dirty_fingerprint(entries)  # noqa: SLF001
    (nested / "two.rs").write_text("fn two() {}\n")
    assert support._dirty_fingerprint(entries) != before  # noqa: SLF001

    (nested / "one.rs").write_text("fn one() { /* edited */ }\n")
    assert support._dirty_fingerprint(entries) not in (None, before)  # noqa: SLF001


def test_an_unreadable_untracked_path_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A constant placeholder would make two different trees fingerprint alike."""
    repository = _temporary_repository(tmp_path, monkeypatch)
    (repository / "gone.rs").write_text("fn gone() {}\n")
    entries = support._source_entries()  # noqa: SLF001
    (repository / "gone.rs").unlink()
    with pytest.raises(SystemExit, match="cannot fingerprint"):
        support._dirty_fingerprint(entries)  # noqa: SLF001


def test_an_untracked_symlink_is_fingerprinted_as_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source fingerprint must neither follow nor confuse symbolic links."""
    repository = _temporary_repository(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    first = outside / "first.rs"
    second = outside / "second.rs"
    first.write_text("same\n", encoding="utf-8")
    second.write_text("same\n", encoding="utf-8")
    link = repository / "linked.rs"
    link.symlink_to(first)

    entries = support._source_entries()  # noqa: SLF001
    before = support._dirty_fingerprint(entries)  # noqa: SLF001
    first.write_text("external edit\n", encoding="utf-8")
    assert support._dirty_fingerprint(entries) == before  # noqa: SLF001

    link.unlink()
    link.symlink_to(second)
    assert support._dirty_fingerprint(entries) != before  # noqa: SLF001


def test_recursive_directory_fingerprinting_never_follows_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fallback directory row may contain a link cycle without recursing."""
    repository = _temporary_repository(tmp_path, monkeypatch)
    directory = repository / "newdir"
    directory.mkdir()
    (directory / "source.rs").write_text("fn source() {}\n", encoding="utf-8")
    (directory / "cycle").symlink_to(".")
    entries = [("??", "newdir", None)]

    first = support._dirty_fingerprint(entries)  # noqa: SLF001
    assert support._dirty_fingerprint(entries) == first  # noqa: SLF001
    (directory / "cycle").unlink()
    (directory / "cycle").symlink_to("..")
    assert support._dirty_fingerprint(entries) != first  # noqa: SLF001


def test_an_unreadable_descendant_of_a_directory_entry_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fallback recursion must propagate a nested directory scan failure."""
    repository = _temporary_repository(tmp_path, monkeypatch)
    directory = repository / "newdir"
    private = directory / "private"
    private.mkdir(parents=True)
    real_scandir = support.os.scandir

    def refused_scandir(path: object) -> object:
        if Path(path) == private:
            raise PermissionError("permission denied")
        return real_scandir(path)

    monkeypatch.setattr(support.os, "scandir", refused_scandir)
    with pytest.raises(SystemExit, match="cannot fingerprint newdir/private"):
        support._dirty_fingerprint([("??", "newdir", None)])  # noqa: SLF001


def test_git_status_warning_refuses_the_source_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable directory reported only on stderr must not look clean."""
    completed = subprocess.CompletedProcess(
        ["git", "status"],
        0,
        stdout="",
        stderr="warning: could not open directory 'private/': Permission denied\n",
    )
    monkeypatch.setattr(support.subprocess, "run", lambda *_args, **_kwargs: completed)
    with pytest.raises(SystemExit, match="cannot inspect the source tree"):
        support._source_entries()  # noqa: SLF001


def test_git_status_failure_does_not_masquerade_as_a_clean_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable Git answer must fail closed before benchmarking."""
    monkeypatch.setattr(support, "_run_git", lambda *_args, **_kwargs: None)
    with pytest.raises(SystemExit, match="git status failed"):
        support._source_entries()  # noqa: SLF001


def test_strategy_agreement_compares_content_not_only_file_lengths() -> None:
    """Equal-sized but different outputs are not a successful P6b differential."""
    common = {
        "segment_codec": "none",
        "threads": 8,
        "memory_budget_mib": None,
        "generated_count": 10,
        "record_count": 10,
        "duplicate_count": 0,
        "block_count": 1,
        "csf_bytes": 100,
        "descriptor_bytes": 200,
        "csf_text_sha256": "text",
        "csf_parquet_sha256": "parquet",
        "descriptor_sha256": "descriptor",
        "header_sha256": "header-a",
    }
    measurements = [
        {**common, "deduplication": "verified_unique"},
        {**common, "deduplication": "exact", "header_sha256": "header-b"},
    ]
    with pytest.raises(SystemExit, match="de-duplication strategies disagree"):
        generation_benchmark._check_strategy_agreement(measurements)  # noqa: SLF001


def test_each_benchmark_measurement_has_an_isolated_rss_scope(tmp_path: Path) -> None:
    """Per-run RSS must come from a process whose high-water mark starts fresh."""
    transcript = tmp_path / "tiny.rcsfgenerate"
    transcript.write_text(
        "* ! Orbital order\n0\n2s(2,*)2p(1,*)\n\n3s,3p,3d\n1,3\n1\nn\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "benchmark_v2_generation.py"),
            str(transcript),
            "--threads",
            "1",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--allow-dirty-source",
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    document = json.loads(report.read_text(encoding="utf-8"))
    assert document["settings"]["measurement_process_model"] == "spawned_process_per_run"
    measurement = document["measurements"][0]
    assert measurement["rss_scope"] == "single_measurement_process"
    for artifact in ("csf_text", "csf_parquet", "descriptor", "header"):
        assert re.fullmatch(r"[0-9a-f]{64}", measurement[f"{artifact}_sha256"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # An ordinary modification.
        (" M src/lib.rs\0", [(" M", "src/lib.rs", None)]),
        # A name with a space and one that merely looks quoted: `-z` never quotes.
        ('?? a file.rs\0?? "quoted".rs\0', [("??", "a file.rs", None), ("??", '"quoted".rs', None)]),
        # A rename carries its origin in the next field.
        ("R  dst.rs\0src/src.rs\0", [("R ", "dst.rs", "src/src.rs")]),
        # A newline inside a name survives because fields are NUL-separated.
        ("?? line\nbreak.rs\0", [("??", "line\nbreak.rs", None)]),
        # Mixed rows, and an empty field is not an entry.
        (" M a\0\0?? b\0", [(" M", "a", None), ("??", "b", None)]),
    ],
)
def test_status_parsing_handles_awkward_paths(
    raw: str, expected: list[tuple[str, str, str | None]]
) -> None:
    """`-z` is what keeps quoted, renamed and newline-bearing names intact."""
    assert support._parse_status_z(raw) == expected  # noqa: SLF001 - the parser is the unit


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


def _digests(report: Path, *, codec: str, deduplication: str) -> dict[str, str]:
    """The published-artifact digests one report recorded for one combination."""
    document = json.loads(report.read_text(encoding="utf-8"))
    for measurement in document["measurements"]:
        if (
            measurement.get("segment_codec") == codec
            and measurement.get("deduplication") == deduplication
        ):
            return {
                key: measurement[key]
                for key in ("csf_text_sha256", "descriptor_sha256", "header_sha256")
            }
    raise AssertionError(f"{report.name} has no {deduplication}/{codec} measurement")


@pytest.mark.parametrize("codec", ["none", "zstd"])
def test_the_one_pass_tail_published_what_the_two_pass_tail_published(
    codec: str,
) -> None:
    """P4 replaced the tail; the content it publishes must not have moved.

    The two campaigns were measured at different revisions (`47e04ea` and
    `de97127`), so this compares two committed reports rather than two runs of
    the current code. The CSF text, descriptor and header must be identical
    byte for byte; the CSF Parquet digest is allowed to differ because its
    row-group boundaries follow whichever path batched it. If the tail ever
    changes the first three, this test fails on the registered evidence instead
    of on a claim in a report.
    """
    reference = _digests(
        BENCHMARK_DIRECTORY / "v2_disk_generation_dedup_b2_20260922.json",
        codec=codec,
        deduplication="verified_unique",
    )
    measured = _digests(
        BENCHMARK_DIRECTORY / "v2_disk_generation_final_encoding_b2_20260922.json",
        codec=codec,
        deduplication="verified_unique",
    )
    for key in ("csf_text_sha256", "descriptor_sha256", "header_sha256"):
        assert measured[key] == reference[key], (
            f"the one-pass tail changed {key} for the {codec} codec"
        )
    # The CSF Parquet is deliberately *not* compared here. Its row-group
    # boundaries follow whichever path batched it, which the format contract
    # does not fix, so equal or unequal bytes are both legitimate; asserting
    # either direction would turn a layout detail into a contract. Its logical
    # rows are compared where they can be: the live differentials in
    # `streaming.rs` (both tails, and across thread counts).
