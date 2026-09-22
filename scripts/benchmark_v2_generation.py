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
* Every warm-up and measured run executes in a fresh spawned process. This
  makes ``ru_maxrss`` a per-run high-water mark instead of a value inherited
  from earlier combinations in the same benchmark invocation.

Rust progress and diagnostics remain on stderr; the JSON report is written to
stdout or to ``--output``.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

# Running a script places ``scripts/`` first on sys.path. Prefer the checkout's
# editable package so the benchmark measures the source under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_support import (
    DEFAULT_MANIFEST,
    add_source_identity_arguments,
    environment,
    finalize_report,
    filesystem_metadata,
    load_manifest,
    require_clean_source,
    sha256_file,
    source_identity,
    verify_registered_transcript,
)
from rcsfs import generate_disk_outputs_from_transcript

REPORT_SCHEMA = "rcsfs-v2-generation-benchmark/1"

#: Failures the benchmark expects and records as measurements rather than
#: crashing on: a rejected configuration is a result, not a broken run.
_EXPECTED_REJECTIONS = (
    "memory budget exceeded",
    "not enough free space",
    "cannot check free space",
)

#: Files sampled by the scratch monitor. Small enough to stay cheap next to a
#: multi-gigabyte run, large enough to observe a stage boundary.
SCRATCH_SAMPLE_INTERVAL_SECONDS = 0.2

#: The extension reads the temporary-segment codec from the environment; the
#: benchmark sets it per run so one invocation can compare codecs.
SEGMENT_CODEC_ENV = "RCSFS_SEGMENT_CODEC"

#: What `RCSFS_SEGMENT_CODEC` accepts.
SEGMENT_CODECS = ("none", "lz4", "zstd")

#: The extension reads the de-duplication strategy from the environment too, so
#: one invocation can compare the proof-backed path against the exact one.
DEDUPLICATION_ENV = "RCSFS_DEDUPLICATION"

#: What `RCSFS_DEDUPLICATION` accepts.
DEDUPLICATION_STRATEGIES = ("verified_unique", "exact")

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

_PUBLISHED_DIGEST_KEYS = (
    "csf_text_sha256",
    "csf_parquet_sha256",
    "descriptor_sha256",
    "header_sha256",
)


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
        if path.is_file():
            sizes[f"{name}_bytes"] = path.stat().st_size
            # Hash after the timed region so the P6b differential compares the
            # bytes that were published without charging verification to the
            # generation time. Both strategies use the same writers, so a raw
            # hash is the strongest useful check for this campaign.
            sizes[f"{name}_sha256"] = sha256_file(path)
        else:
            sizes[f"{name}_bytes"] = None
            sizes[f"{name}_sha256"] = None
    sizes["output_file_count"] = sum(1 for path in paths.values() if path.is_file())
    return sizes


def _run_once_in_process(
    transcript: str,
    threads: int | None,
    memory_budget_mib: int | None,
    segment_codec: str,
    deduplication: str,
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
        "segment_codec": segment_codec,
        "deduplication": deduplication,
    }
    previous_codec = os.environ.get(SEGMENT_CODEC_ENV)
    previous_deduplication = os.environ.get(DEDUPLICATION_ENV)
    os.environ[SEGMENT_CODEC_ENV] = segment_codec
    os.environ[DEDUPLICATION_ENV] = deduplication
    try:
        monitor.start()
        # The timed region excludes temporary-set creation and deletion, so it
        # is a generation measurement rather than a filesystem measurement.
        started = time.perf_counter()
        try:
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
        except (OSError, RuntimeError) as error:
            # A budget or space rejection is a result the matrix is meant to
            # record, not a broken run. Anything else is a real failure.
            message = str(error)
            if not any(expected in message for expected in _EXPECTED_REJECTIONS):
                raise
            monitor.stop()
            result["wall_seconds"] = time.perf_counter() - started
            result["success"] = False
            result["outcome"] = "rejected"
            result["error"] = message
            result["rss_peak_bytes"] = _peak_rss_bytes()
            result["rss_scope"] = "single_measurement_process"
            return result
        wall_seconds = time.perf_counter() - started
        io_after = _process_io_bytes()
        monitor.stop()
        reported_codec = stats.get("segment_codec")
        if reported_codec != segment_codec:
            # A report that names a codec the run did not use would turn P2a's
            # measured ratios into fiction. Refuse instead of recording it.
            raise RuntimeError(
                f"asked for segment codec {segment_codec!r} but the run reported "
                f"{reported_codec!r}"
            )
        reported_deduplication = stats.get("deduplication")
        if reported_deduplication != deduplication:
            # The same for the strategy: it decides whether duplicate_count is a
            # measurement or a consequence of the uniqueness proof.
            raise RuntimeError(
                f"asked for de-duplication {deduplication!r} but the run reported "
                f"{reported_deduplication!r}"
            )
        result["wall_seconds"] = wall_seconds
        result["success"] = stats.get("success", False)
        for key in _STAGE_KEYS:
            if key in stats:
                result[key] = stats[key]
        # The stages must account for the call they describe. A phase that is
        # left out of the timers is how a "stage is 2x faster" claim becomes
        # unverifiable, so the residual is recorded and checked rather than
        # being discovered later by arithmetic on the end-to-end time.
        stage_seconds = sum(
            stage.get("elapsed_millis", 0) for stage in stats.get("stage_stats", [])
        ) / 1000.0
        result["stage_seconds"] = stage_seconds
        result["unattributed_seconds"] = wall_seconds - stage_seconds
        # This process performs exactly one run, so its lifetime high-water
        # mark is an independent measurement. Capture it before output hashing,
        # which is verification work outside the generation call.
        result["rss_peak_bytes"] = _peak_rss_bytes()
        result["rss_scope"] = "single_measurement_process"
        result.update(_measure_outputs(root, paths))
        result["scratch_peak_bytes"] = monitor.peak_bytes
        result["scratch_peak_file_count"] = monitor.peak_files
        result["scratch_final_bytes"] = monitor.final_bytes
        result["scratch_final_file_count"] = monitor.final_files
        result["scratch_samples"] = monitor.samples
        result["process_io"] = None
        if io_before is not None and io_after is not None:
            result["process_io"] = {
                key: io_after[key] - io_before.get(key, 0) for key in io_after
            }
    finally:
        if previous_codec is None:
            os.environ.pop(SEGMENT_CODEC_ENV, None)
        else:
            os.environ[SEGMENT_CODEC_ENV] = previous_codec
        if previous_deduplication is None:
            os.environ.pop(DEDUPLICATION_ENV, None)
        else:
            os.environ[DEDUPLICATION_ENV] = previous_deduplication
        monitor.stop()
        cleanup_started = time.perf_counter()
        shutil.rmtree(root, ignore_errors=True)
        # Reported separately: deleting tens of gigabytes of scratch is real
        # work, but it is not part of generating the CSFs.
        result["cleanup_seconds"] = time.perf_counter() - cleanup_started
    return result


def _run_once(
    transcript: str,
    threads: int | None,
    memory_budget_mib: int | None,
    segment_codec: str,
    deduplication: str,
    scratch_root: Path,
) -> dict[str, Any]:
    """Run one combination in a fresh process with an independent RSS peak."""
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        future = executor.submit(
            _run_once_in_process,
            transcript,
            threads,
            memory_budget_mib,
            segment_codec,
            deduplication,
            scratch_root,
        )
        return future.result()


def _summarize(
    measurements: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int | None, int | None], list[dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    for measurement in measurements:
        key = (
            measurement.get("deduplication", "verified_unique"),
            measurement.get("segment_codec", "none"),
            measurement["threads"],
            measurement.get("memory_budget_mib"),
        )
        if measurement.get("outcome") == "rejected":
            # A refused run has no stage timings to average; it is reported on
            # its own so a low budget cannot masquerade as a slow one.
            rejected.append(
                {
                    "threads": measurement["threads"],
                    "memory_budget_mib": measurement.get("memory_budget_mib"),
                    "segment_codec": measurement.get("segment_codec", "none"),
                    "deduplication": measurement.get("deduplication", "verified_unique"),
                    "error": measurement.get("error"),
                }
            )
            continue
        groups.setdefault(key, []).append(measurement)
    summary: list[dict[str, Any]] = []
    for (deduplication, codec, threads, budget), group in sorted(
        groups.items(),
        key=lambda item: (
            DEDUPLICATION_STRATEGIES.index(item[0][0]),
            SEGMENT_CODECS.index(item[0][1]),
            item[0][2] is not None,
            item[0][2] or 0,
            item[0][3] or 0,
        ),
    ):
        walls = [entry["wall_seconds"] for entry in group]
        entry: dict[str, Any] = {
            "deduplication": deduplication,
            "segment_codec": codec,
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
            "dedup_temp_bytes_median": statistics.median(
                [
                    stage["output_bytes"]
                    for item in group
                    for stage in item.get("stage_stats", [])
                    if stage["name"] == "deduplication"
                ]
                or [0]
            ),
            "segment_bytes_median": statistics.median(
                [
                    stage["output_bytes"]
                    for item in group
                    for stage in item.get("stage_stats", [])
                    if stage["name"] == "csf_generation"
                ]
                or [0]
            ),
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
    return summary, rejected


def _check_strategy_agreement(measurements: list[dict[str, Any]]) -> None:
    """The proof-backed path must publish exactly what the exact path publishes.

    Both strategies are asked for in one invocation precisely so this can be
    checked on the registered inputs rather than only in a unit test. Counts and
    every published artifact have to match; a difference means the skipped
    comparison was not skipping nothing.
    """
    same = {}
    for measurement in measurements:
        if measurement.get("outcome") == "rejected":
            continue
        missing = [
            name for name in _PUBLISHED_DIGEST_KEYS if not measurement.get(name)
        ]
        if missing:
            raise SystemExit(
                "cannot verify de-duplication strategy agreement: successful run "
                f"is missing published artifact digests {missing}"
            )
        key = (
            measurement.get("segment_codec", "none"),
            measurement["threads"],
            measurement.get("memory_budget_mib"),
        )
        figures = (
            measurement.get("generated_count"),
            measurement.get("record_count"),
            measurement.get("duplicate_count"),
            measurement.get("block_count"),
            measurement.get("csf_bytes"),
            measurement.get("descriptor_bytes"),
            *(measurement.get(name) for name in _PUBLISHED_DIGEST_KEYS),
        )
        seen = same.setdefault(key, (measurement.get("deduplication"), figures))
        if seen[1] != figures:
            raise SystemExit(
                f"de-duplication strategies disagree for {key}: "
                f"{seen[0]} reported {seen[1]}, {measurement.get('deduplication')} "
                f"reported {figures}"
            )


#: Largest share of the call that may be outside every stage timer before the
#: report is refused. A small residual is unavoidable (transcript parsing, the
#: pre-flight's statvfs calls, the binding's own marshalling); anything larger
#: means a phase was added without a timer, and no stage claim can be checked.
MAX_UNATTRIBUTED_SHARE = 0.10

#: An absolute floor, so a sub-second run is not refused over a millisecond.
MIN_UNATTRIBUTED_SECONDS = 0.05


def _check_stage_coverage(measurements: list[dict[str, Any]]) -> None:
    """Refuse a run whose stages do not account for the call they time."""
    for measurement in measurements:
        if measurement.get("outcome") == "rejected":
            continue
        wall = measurement["wall_seconds"]
        unattributed = measurement.get("unattributed_seconds")
        if unattributed is None:
            raise SystemExit("a successful run recorded no stage accounting")
        allowed = max(MIN_UNATTRIBUTED_SECONDS, wall * MAX_UNATTRIBUTED_SHARE)
        if unattributed > allowed:
            raise SystemExit(
                f"the stages account for {wall - unattributed:.3f}s of a "
                f"{wall:.3f}s call; {unattributed:.3f}s is outside every stage "
                f"timer, which is more than the {allowed:.3f}s allowed. Add a "
                f"timer for the phase that is missing, or the report's stage "
                f"numbers cannot be checked against the end-to-end time."
            )


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
        "--segment-codec",
        choices=SEGMENT_CODECS,
        nargs="+",
        default=["none"],
        help=(
            "Temporary Arrow IPC segment codecs to measure; each codec runs every "
            "requested thread/budget combination. The run reports the codec it "
            "actually used, so a mismatch fails instead of being recorded."
        ),
    )
    _ = parser.add_argument(
        "--deduplication",
        choices=DEDUPLICATION_STRATEGIES,
        nargs="+",
        default=["verified_unique"],
        help=(
            "De-duplication strategies to measure. `verified_unique` is the "
            "default the uniqueness proof licenses and reports zero duplicates "
            "by construction; `exact` compares every row and measures the count."
        ),
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
    add_source_identity_arguments(parser)
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

    # Snapshot and gate first: a source that cannot be registered must not cost
    # a full benchmark, and the snapshot is what the process will actually run.
    start_identity = source_identity()
    require_clean_source(start_identity, args.allow_dirty_source)

    transcript = args.transcript.read_text(encoding="utf-8")
    digest = sha256_file(args.transcript)
    registered = load_manifest(args.manifest)
    combinations = [
        (deduplication, codec, threads, budget)
        for deduplication in args.deduplication
        for codec in args.segment_codec
        for threads in args.threads
        for budget in args.memory_budget_mib
    ]

    warmups: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    order = 0
    for deduplication, codec, threads, budget in combinations:
        for _ in range(args.warmup):
            warmup = _run_once(
                transcript, threads, budget, codec, deduplication, scratch_root
            )
            warmup["order"] = order
            warmup["kind"] = "warmup"
            order += 1
            warmups.append(warmup)
        for _ in range(args.repeats):
            measurement = _run_once(
                transcript, threads, budget, codec, deduplication, scratch_root
            )
            measurement["order"] = order
            measurement["kind"] = "measured"
            order += 1
            measurements.append(measurement)

    successful = [
        measurement
        for measurement in [*warmups, *measurements]
        if measurement.get("outcome") != "rejected"
    ]
    transcript_record = verify_registered_transcript(
        args.transcript,
        digest,
        registered,
        unique_occupations=next(
            (item.get("unique_occupations") for item in successful), None
        ),
        records=next((item.get("generated_count") for item in successful), None),
    )
    _check_stage_coverage(measurements)
    _check_strategy_agreement(measurements)
    summary, rejected = _summarize(measurements)
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at_unix_seconds": time.time(),
        "environment": environment(),
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
            "segment_codecs": list(args.segment_codec),
            "deduplication": list(args.deduplication),
            "scratch_root": str(scratch_root),
            "scratch_sampling_interval_seconds": SCRATCH_SAMPLE_INTERVAL_SECONDS,
            "measurement_process_model": "spawned_process_per_run",
            "timed_region": "generation call including artifact publication; excludes set-up and deletion",
        },
        "filesystem": filesystem_metadata(scratch_root),
        "measurements": measurements,
        "warmups": warmups,
        "summary": summary,
        "rejected": rejected,
    }
    finalize_report(report, args.output, start_identity, args.allow_dirty_source)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
