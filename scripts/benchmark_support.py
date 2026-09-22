"""Shared plumbing for the registered V2 generation benchmarks.

Both :mod:`benchmark_v2_generation` and :mod:`estimate_v2_generation` publish
reports that are compared with registered baselines, so the parts that decide
what a report may claim live here once: the manifest contract, the environment
and filesystem metadata, and the path normalization that keeps
machine-specific absolute paths out of a committed report.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "transcripts.toml"

def _placeholder_prefixes() -> tuple[tuple[str, str], ...]:
    """Absolute prefixes replaced by a placeholder when a report is written.

    Both the resolved and unresolved spelling of each root is registered:
    macOS reports the system temp directory as ``/var/folders/...`` while
    resolving it yields ``/private/var/folders/...``, and a report has to be
    normalized whichever form the measured path used. The longest prefix is
    tried first so a nested root cannot be shadowed by its parent.
    """
    entries: list[tuple[str, str]] = []
    for path, placeholder in (
        (Path(tempfile.gettempdir()), "<system-temp>"),
        (REPO_ROOT, "<repo-root>"),
        (Path.home(), "<home>"),
    ):
        for candidate in {str(path), str(Path(path).resolve())}:
            entries.append((candidate, placeholder))
    return tuple(sorted(entries, key=lambda item: len(item[0]), reverse=True))


_PATH_PLACEHOLDERS = _placeholder_prefixes()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Return the registered transcripts keyed by file name."""
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    entries = document.get("transcript", [])
    if not isinstance(entries, list):
        raise ValueError(f"{path} has a non-list transcript table")
    registered: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{path} has a non-table transcript entry")
        file_name = entry.get("file")
        if not isinstance(file_name, str):
            raise ValueError(f"{path} has a transcript entry without a file name")
        registered[file_name] = entry
    return registered


def verify_registered_transcript(
    path: Path,
    digest: str,
    registered: dict[str, dict[str, Any]],
    *,
    unique_occupations: int | None = None,
    records: int | None = None,
) -> dict[str, Any]:
    """Bind an observation to the registered baseline, or report it as new.

    One implementation serves both scripts: a drifted hash, a configuration
    count that no longer matches, or a record total that disagrees all mean the
    comparison the report is about to make would be meaningless, so they stop
    the report instead of being written beside a baseline they contradict.
    """
    entry = registered.get(path.name)
    if entry is None:
        return {"registered": False, "sha256": digest}
    expected_digest = entry.get("sha256")
    if expected_digest != digest:
        raise SystemExit(
            f"transcript {path.name} is registered with sha256 {expected_digest} "
            f"but hashes to {digest}; re-register the fixture and its baseline together"
        )
    record: dict[str, Any] = {
        "registered": True,
        "name": entry.get("name"),
        "sha256": digest,
        "description": entry.get("description"),
        "expected_unique_occupations": entry.get("unique_occupations"),
        "expected_records": entry.get("records"),
    }
    expected_occupations = record["expected_unique_occupations"]
    if (
        unique_occupations is not None
        and expected_occupations is not None
        and unique_occupations != expected_occupations
    ):
        raise SystemExit(
            f"{path.name} enumerated {unique_occupations} configurations but the "
            f"registered baseline is {expected_occupations}"
        )
    expected_records = record["expected_records"]
    if records is not None and expected_records is not None and records != expected_records:
        raise SystemExit(
            f"{path.name} counted {records} records but the registered baseline is "
            f"{expected_records}"
        )
    return record


#: Paths excluded from the source-identity computation. The registered reports
#: are this harness's own output, so writing one must not make the next report
#: claim a dirty source.
_SOURCE_EXCLUSIONS = (":!docs/benchmarks",)


def _run_git(*arguments: str, keep_indent: bool = False) -> str | None:
    """Run a git command in the repository, or `None` when git cannot answer.

    `keep_indent` matters for porcelain output, whose two-character status
    column starts with a space that `strip()` would remove from the first line.
    """
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.rstrip("\n") if keep_indent else completed.stdout.strip()


def _source_status() -> str | None:
    """Tracked modifications and untracked files, ignoring the report outputs."""
    return _run_git("status", "--porcelain", "--", *_SOURCE_EXCLUSIONS, keep_indent=True)


def _dirty_fingerprint(status: str) -> str | None:
    """Identify everything that makes the source dirty, tracked or not.

    `git diff HEAD` alone would miss an untracked source file and would include
    the report outputs this harness writes, so the fingerprint is built from the
    status entries with the same exclusions: the diff of the tracked ones plus
    the name and content hash of each new file.
    """
    digest = hashlib.sha256()
    digest.update(status.encode())
    tracked_diff = _run_git("diff", "HEAD", "--", *_SOURCE_EXCLUSIONS)
    if tracked_diff is None:
        return None
    digest.update(tracked_diff.encode())
    untracked = _run_git(
        "ls-files", "--others", "--exclude-standard", "--", *_SOURCE_EXCLUSIONS
    )
    if untracked is None:
        return None
    for name in sorted(line for line in untracked.splitlines() if line):
        digest.update(name.encode())
        path = REPO_ROOT / name
        try:
            digest.update(sha256_file(path).encode())
        except OSError:
            # A file that vanished between listing and reading still has to be
            # accounted for, so its absence is recorded rather than skipped.
            digest.update(b"<unreadable>")
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    status = _source_status()
    dirty = None if status is None else bool(status)
    metadata: dict[str, Any] = {
        "commit": _run_git("rev-parse", "HEAD"),
        # The tree hash identifies the source state even if the commit is later
        # rewritten or the branch moves, so a reader can check what was built.
        "tree": _run_git("rev-parse", "HEAD^{tree}"),
        # Without `--dirty`: the `dirty` field below carries that, and its
        # definition excludes the report outputs this harness writes, so the two
        # would otherwise disagree whenever only a report changed.
        "describe": _run_git("describe", "--tags", "--always"),
        "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree is reported rather than rejected here; the scripts refuse
        # to register a report from one unless the caller insists.
        "dirty": dirty,
        "dirty_ignores": [exclusion.lstrip(":!") for exclusion in _SOURCE_EXCLUSIONS],
    }
    if dirty:
        metadata["dirty_diff_sha256"] = _dirty_fingerprint(status)
        # Porcelain rows are "<XY> <path>"; the status column is two characters
        # wide and may itself start with a space.
        metadata["dirty_paths"] = sorted(
            line[3:] for line in status.splitlines() if len(line) > 3
        )
    return metadata


def extension_metadata() -> dict[str, Any]:
    """Identify the extension this process actually loaded.

    A commit hash says which source was checked out, not which binary was
    measured. Recording the loaded module's hash lets a reader rebuild the
    reported tree and compare.
    """
    import rcsfs._rcsfs as native

    path = Path(str(native.__file__))
    return {
        "version": str(native.__version__),
        "module": path.name,
        "module_sha256": sha256_file(path),
        "path": str(path),
    }


def filesystem_type(path: Path) -> str | None:
    """Filesystem name from the mount table, or None when it is unreadable.

    The mount table is read instead of platform-specific helpers because
    macOS `stat -f %T` reports the mount point rather than the filesystem on
    current releases, and a wrong name is worse than an absent one.
    """
    try:
        completed = subprocess.run(
            ["mount"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    resolved = str(path.resolve())
    best: tuple[int, str] | None = None
    for line in completed.stdout.splitlines():
        # "<device> on <mount point> (<type>, <options>)"
        on_index = line.find(" on ")
        open_index = line.rfind(" (")
        if on_index < 0 or open_index < on_index:
            continue
        mount_point = line[on_index + 4 : open_index]
        fields = line[open_index + 2 :].rstrip(")").split(",")
        if not fields or not fields[0].strip():
            continue
        if resolved != mount_point and not resolved.startswith(
            mount_point.rstrip("/") + "/"
        ):
            continue
        if best is None or len(mount_point) > best[0]:
            best = (len(mount_point), fields[0].strip())
    return None if best is None else best[1]


def filesystem_metadata(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "type": filesystem_type(path),
        "total_bytes": usage.total,
        "free_bytes": usage.free,
    }


def _total_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None
    if not isinstance(page_size, int) or not isinstance(pages, int):
        return None
    return page_size * pages


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "total_memory_bytes": _total_memory_bytes(),
        "git": _git_metadata(),
        "extension": extension_metadata(),
    }


#: A Windows drive-absolute path, with either separator.
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
#: A Windows UNC path (\\server\share\...).
_WINDOWS_UNC = re.compile(r"^\\\\[^\\/]+[\\/][^\\/]+")


def _looks_windows(value: str) -> bool:
    """Whether a path is written in Windows form rather than POSIX form."""
    return bool(_WINDOWS_DRIVE.match(value)) or "\\" in value


def _is_absolute(value: str) -> bool:
    return (
        value.startswith("/")
        or bool(_WINDOWS_DRIVE.match(value))
        or bool(_WINDOWS_UNC.match(value))
    )


def _last_component(value: str) -> str:
    return re.split(r"[\\/]", value.rstrip("\\/"))[-1]


def _matches_prefix(value: str, prefix: str) -> bool:
    """Whether `value` is inside `prefix`, tolerating the platform's spelling.

    Windows paths are compared case-insensitively because the filesystems are;
    POSIX paths are compared exactly, where `/Home` and `/home` are different
    directories.
    """
    separator = "\\" if _looks_windows(value) else "/"
    trimmed = prefix.rstrip("/\\")
    if _looks_windows(value) or _looks_windows(prefix):
        return value.casefold().replace("\\", "/").startswith(
            trimmed.casefold().replace("\\", "/") + "/"
        )
    return value.startswith(trimmed + separator)


def normalize_path(value: str) -> str:
    """Replace the machine-specific part of an absolute path.

    Both spellings are handled: a report may be written on one platform from
    measurements taken on another, and `/var/folders/...` and
    `C:\\Users\\...\\Temp\\...` are equally machine-specific.
    """
    for prefix, placeholder in _PATH_PLACEHOLDERS:
        if value == prefix or value.casefold() == prefix.casefold():
            return placeholder
        if _matches_prefix(value, prefix):
            remainder = value[len(prefix) :].lstrip("/\\")
            return f"{placeholder}/{remainder}"
    if _is_absolute(value):
        # An absolute path outside the known roots is machine-specific too. Keep
        # its last component so the shape of the layout stays legible.
        return "<path>/" + _last_component(value)
    return value


def normalize_report_paths(value: Any) -> Any:
    """Recursively replace absolute paths in a report before it is written.

    Reports are committed, so a path from the machine that produced them is
    noise that also leaks a local layout. Filesystem type and capacity, which
    are the machine facts a reader needs, are kept as they are.
    """
    if isinstance(value, str):
        return normalize_path(value)
    if isinstance(value, dict):
        return {key: normalize_report_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_report_paths(item) for item in value]
    return value


def require_clean_source(git: dict[str, Any], allow_dirty: bool) -> None:
    """Refuse to register a measurement whose source is not identified.

    A report is a claim about a specific source state. When the tree is dirty
    the claim becomes "this revision plus these changes", which is auditable but
    weaker, and a measurement taken while the source is being edited is not a
    baseline anyone should compare against - one of these runs was disturbed
    exactly that way. Registering such a report therefore takes an explicit
    decision.
    """
    if not git.get("dirty"):
        return
    if allow_dirty:
        return
    paths = ", ".join(git.get("dirty_paths") or ["<unknown>"])
    raise SystemExit(
        f"the source tree is dirty ({paths}); a registered report must name a "
        f"source state someone else can rebuild. Commit the changes, or pass "
        f"--allow-dirty-source to record this measurement as an identified but "
        f"uncommitted one."
    )


def write_report(report: dict[str, Any], output: Path | None) -> None:
    """Write a normalized report to stdout or to ``output``."""
    import json

    encoded = json.dumps(normalize_report_paths(report), indent=2, sort_keys=True) + "\n"
    if output is None:
        print(encoded, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - manual check
    print(sys.modules[__name__].__doc__)
