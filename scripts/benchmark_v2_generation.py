"""Measure the V2 disk-generation stages for a supplied transcript.

This is an explicit benchmark, not a pytest test. It only uses the public
Python API and a private temporary output set, so it does not require a GRASP
checkout, an executable or a private dataset.

Measurement contract:

* The timed region covers the generation call itself, including publishing the
  staged artifacts. Creating and deleting the temporary output set is *not*
  part of it; deletion is reported as its own ``cleanup_seconds`` value.
* Every thread count gets one discarded warm-up run before the measured runs,
  and runs are recorded in execution order so a page-cache bias is visible.
* Transcripts registered in ``tests/fixtures/transcripts.toml`` are verified
  against their recorded SHA-256 and expected record counts. An unregistered
  transcript is measured and reported as such rather than silently accepted as
  a baseline.
* Physical device I/O is only reported when the platform exposes it. It is
  ``None`` elsewhere instead of being estimated.

Rust progress and diagnostics remain on stderr; the JSON report is written to
stdout or to ``--output``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

# Running a script places ``scripts/`` first on sys.path. Prefer the checkout's
# editable package so the benchmark measures the source under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcsfs import generate_disk_outputs_from_transcript

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "transcripts.toml"
REPORT_SCHEMA = "rcsfs-v2-generation-benchmark/1"

#: Files sampled by the scratch monitor. Small enough to stay cheap next to a
#: multi-gigabyte run, large enough to observe a stage boundary.
SCRATCH_SAMPLE_INTERVAL_SECONDS = 0.2

_STAGE_KEYS = (
    "unique_occupations",
    "generated_count",
    "record_count",
    "descriptor_count",
    "duplicate_count",
    "block_count",
    "csf_bytes",
    "descriptor_bytes",
    "stage_stats",
    "resource_stats",
    "plan_stats",
)


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


def _git_metadata() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
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
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "describe": run("describe", "--tags", "--always", "--dirty"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree is reported rather than rejected: a local measurement of
        # uncommitted work is still useful, but it must not be mistaken for a
        # reproducible baseline.
        "dirty": None if status is None else bool(status),
    }


def _filesystem_type(path: Path) -> str | None:
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


def _filesystem_metadata(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "type": _filesystem_type(path),
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


def _peak_rss_bytes() -> int | None:
    """Peak RSS since process start; Linux reports KiB, macOS bytes."""
    try:
        import resource
    except ImportError:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):
        return None
    return int(usage) if sys.platform == "darwin" else int(usage) * 1024


def _process_io_bytes() -> dict[str, int] | None:
    """Process-wide read/write byte counters where the kernel exposes them."""
    path = Path("/proc/self/io")
    if not path.is_file():
        return None
    counters: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"read_bytes", "write_bytes"}:
                counters[key] = int(value.strip())
    except (OSError, ValueError):
        return None
    return counters or None


def _environment() -> dict[str, Any]:
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
    }


class ScratchMonitor(threading.Thread):
    """Sample temporary scratch usage so a peak can be reported.

    Polling is inherently lossy and adds a small amount of I/O of its own, so
    the interval is recorded with the result instead of being presented as an
    exact peak.
    """

    def __init__(self, root: Path, interval: float = SCRATCH_SAMPLE_INTERVAL_SECONDS):
        super().__init__(daemon=True)
        self.root = root
        self.interval = interval
        self.peak_bytes = 0
        self.peak_files = 0
        self.final_bytes = 0
        self.final_files = 0
        self.samples = 0
        self._stop = threading.Event()

    def _sample(self) -> tuple[int, int]:
        total = 0
        files = 0
        for directory, _subdirectories, names in os.walk(self.root, onerror=lambda _error: None):
            for name in names:
                try:
                    total += os.stat(os.path.join(directory, name)).st_size
                except OSError:
                    continue
                files += 1
        return total, files

    def run(self) -> None:
        while not self._stop.is_set():
            total, files = self._sample()
            self.samples += 1
            self.peak_bytes = max(self.peak_bytes, total)
            self.peak_files = max(self.peak_files, files)
            self.final_bytes = total
            self.final_files = files
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=10.0)


def _measure_outputs(root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    sizes: dict[str, Any] = {}
    for name, path in paths.items():
        sizes[f"{name}_bytes"] = path.stat().st_size if path.is_file() else None
    sizes["output_file_count"] = sum(1 for path in paths.values() if path.is_file())
    return sizes


def _run_once(
    transcript: str,
    threads: int | None,
    memory_budget_mib: int | None,
    scratch_root: Path,
) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix="rcsfs-v2-benchmark-", dir=scratch_root))
    paths = {
        "csf_text": root / "output.c",
        "csf_parquet": root / "output.parquet",
        "descriptor": root / "descriptors.parquet",
        "header": root / "header.toml",
    }
    scratch = root / "scratch"
    monitor = ScratchMonitor(scratch)
    io_before = _process_io_bytes()
    result: dict[str, Any] = {
        "threads": threads,
        "memory_budget_mib": memory_budget_mib,
    }
    try:
        monitor.start()
        # The timed region excludes temporary-set creation and deletion, so it
        # is a generation measurement rather than a filesystem measurement.
        started = time.perf_counter()
        stats = generate_disk_outputs_from_transcript(
            transcript,
            paths["csf_text"],
            paths["csf_parquet"],
            paths["descriptor"],
            paths["header"],
            scratch,
            threads=threads,
            memory_budget_mib=memory_budget_mib,
        )
        wall_seconds = time.perf_counter() - started
        io_after = _process_io_bytes()
        monitor.stop()
        result["wall_seconds"] = wall_seconds
        result["success"] = stats.get("success", False)
        for key in _STAGE_KEYS:
            if key in stats:
                result[key] = stats[key]
        result.update(_measure_outputs(root, paths))
        result["scratch_peak_bytes"] = monitor.peak_bytes
        result["scratch_peak_file_count"] = monitor.peak_files
        result["scratch_final_bytes"] = monitor.final_bytes
        result["scratch_final_file_count"] = monitor.final_files
        result["scratch_samples"] = monitor.samples
        result["rss_peak_bytes"] = _peak_rss_bytes()
        result["process_io"] = None
        if io_before is not None and io_after is not None:
            result["process_io"] = {
                key: io_after[key] - io_before.get(key, 0) for key in io_after
            }
    finally:
        monitor.stop()
        cleanup_started = time.perf_counter()
        shutil.rmtree(root, ignore_errors=True)
        # Reported separately: deleting tens of gigabytes of scratch is real
        # work, but it is not part of generating the CSFs.
        result["cleanup_seconds"] = time.perf_counter() - cleanup_started
    return result


def _summarize(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int | None, int | None], list[dict[str, Any]]] = {}
    for measurement in measurements:
        key = (measurement["threads"], measurement.get("memory_budget_mib"))
        groups.setdefault(key, []).append(measurement)
    summary: list[dict[str, Any]] = []
    for (threads, budget), group in sorted(
        groups.items(), key=lambda item: (item[0][0] is not None, item[0][0] or 0, item[0][1] or 0)
    ):
        walls = [entry["wall_seconds"] for entry in group]
        entry: dict[str, Any] = {
            "threads": threads,
            "memory_budget_mib": budget,
            "runs": len(group),
            "wall_seconds_median": statistics.median(walls),
            "wall_seconds_min": min(walls),
            "wall_seconds_max": max(walls),
            "wall_seconds_range": max(walls) - min(walls),
            "cleanup_seconds_median": statistics.median(
                [item["cleanup_seconds"] for item in group]
            ),
            "scratch_peak_bytes_max": max(item["scratch_peak_bytes"] for item in group),
            "stages": {},
        }
        names = [stage["name"] for stage in group[0].get("stage_stats", [])]
        for name in names:
            elapsed = [
                stage["elapsed_millis"]
                for item in group
                for stage in item.get("stage_stats", [])
                if stage["name"] == name
            ]
            cpu = [
                stage["cpu_millis"]
                for item in group
                for stage in item.get("stage_stats", [])
                if stage["name"] == name and stage.get("cpu_millis") is not None
            ]
            if not elapsed:
                continue
            entry["stages"][name] = {
                "wall_seconds_median": statistics.median(elapsed) / 1000.0,
                "wall_seconds_min": min(elapsed) / 1000.0,
                "wall_seconds_max": max(elapsed) / 1000.0,
                "cpu_seconds_median": None if not cpu else statistics.median(cpu) / 1000.0,
            }
        summary.append(entry)
    return summary


def _check_registered(
    path: Path,
    digest: str,
    registered: dict[str, dict[str, Any]],
    measurements: list[dict[str, Any]],
) -> dict[str, Any]:
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
    for measurement in measurements:
        if not measurement.get("success"):
            continue
        occupations = measurement.get("unique_occupations")
        if occupations is not None and occupations != record["expected_unique_occupations"]:
            raise SystemExit(
                f"{path.name} enumerated {occupations} configurations but the registered "
                f"baseline is {record['expected_unique_occupations']}"
            )
        records = measurement.get("generated_count")
        if records is not None and records != record["expected_records"]:
            raise SystemExit(
                f"{path.name} generated {records} CSFs but the registered baseline is "
                f"{record['expected_records']}"
            )
        # The planner's count is what scheduling and the capacity model trust.
        # A drifting counter has to fail here rather than silently mis-plan.
        plan_stats = measurement.get("plan_stats")
        if isinstance(plan_stats, dict):
            estimated = plan_stats.get("estimated_total_records")
            if estimated is not None and records is not None and estimated != records:
                raise SystemExit(
                    f"{path.name} planned {estimated} records but generated {records}; "
                    f"the workload counter and the generator disagree"
                )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("transcript", type=Path)
    _ = parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[None],
        metavar="N",
        help="Thread counts to measure; omit to use the Rayon default.",
    )
    _ = parser.add_argument(
        "--memory-budget-mib",
        type=int,
        nargs="+",
        default=[None],
        metavar="N",
        help="Managed-memory budgets in MiB to measure; omit for the default.",
    )
    _ = parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured runs per combination (default: 3).",
    )
    _ = parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Discarded warm-up runs per combination before the measured runs.",
    )
    _ = parser.add_argument(
        "--scratch-root",
        type=Path,
        default=None,
        help="Directory that holds the temporary output set (default: the system temp dir).",
    )
    _ = parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    _ = parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be greater than 0")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    if any(thread is not None and thread <= 0 for thread in args.threads):
        parser.error("--threads values must be greater than 0")
    if any(budget is not None and budget <= 0 for budget in args.memory_budget_mib):
        parser.error("--memory-budget-mib values must be greater than 0")
    if not args.transcript.is_file():
        parser.error(f"transcript does not exist: {args.transcript}")
    scratch_root = args.scratch_root
    if scratch_root is None:
        scratch_root = Path(tempfile.gettempdir())
    if not scratch_root.is_dir():
        parser.error(f"scratch root is not a directory: {scratch_root}")

    transcript = args.transcript.read_text(encoding="utf-8")
    digest = sha256_file(args.transcript)
    registered = load_manifest(args.manifest)
    combinations = [
        (threads, budget)
        for threads in args.threads
        for budget in args.memory_budget_mib
    ]

    warmups: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    order = 0
    for threads, budget in combinations:
        for _ in range(args.warmup):
            warmup = _run_once(transcript, threads, budget, scratch_root)
            warmup["order"] = order
            warmup["kind"] = "warmup"
            order += 1
            warmups.append(warmup)
        for _ in range(args.repeats):
            measurement = _run_once(transcript, threads, budget, scratch_root)
            measurement["order"] = order
            measurement["kind"] = "measured"
            order += 1
            measurements.append(measurement)

    transcript_record = _check_registered(
        args.transcript,
        digest,
        registered,
        [*warmups, *measurements],
    )
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at_unix_seconds": time.time(),
        "environment": _environment(),
        "transcript": {
            **transcript_record,
            "path": args.transcript.name,
            "size_bytes": args.transcript.stat().st_size,
        },
        "settings": {
            "repeats": args.repeats,
            "warmup_runs": args.warmup,
            "threads": [None if value is None else value for value in args.threads],
            "memory_budget_mib": [
                None if value is None else value for value in args.memory_budget_mib
            ],
            "scratch_root": str(scratch_root),
            "scratch_sampling_interval_seconds": SCRATCH_SAMPLE_INTERVAL_SECONDS,
            "timed_region": "generation call including artifact publication; excludes set-up and deletion",
        },
        "filesystem": _filesystem_metadata(scratch_root),
        "measurements": measurements,
        "warmups": warmups,
        "summary": _summarize(measurements),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
