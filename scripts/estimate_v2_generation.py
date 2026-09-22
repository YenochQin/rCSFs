"""Count a transcript's workload and report its capacity, without generating.

This is the scriptable pre-flight for the V2 disk path. It runs the same
enumeration, counting and scheduling as a real run, so the counts it reports are
the counts that run would use, but it creates no scratch directory and publishes
no file. It is what makes a full-scale input (B3) reportable without producing
the CSFs.

Transcripts registered in ``tests/fixtures/transcripts.toml`` are verified
against their recorded SHA-256 and, when the entry registers them, their
configuration and record counts. An unregistered transcript is reported as
such rather than silently accepted as a baseline.

The report is JSON on stdout, or at ``--output``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Running a script places ``scripts/`` first on sys.path. Prefer the checkout's
# editable package so the estimate describes the source under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcsfs import estimate_disk_generation

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

REPORT_SCHEMA = "rcsfs-v2-generation-capacity/1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("transcript", type=Path)
    _ = parser.add_argument("--threads", type=int, default=None)
    _ = parser.add_argument("--memory-budget-mib", type=int, default=None)
    _ = parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    _ = parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Directory that will hold the run's scratch data.",
    )
    _ = parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Directory that will hold the staged output set before publication.",
    )
    _ = parser.add_argument(
        "--destination",
        action="append",
        default=[],
        metavar="KIND=PATH",
        help=(
            "Where a published artifact will go, as kind=path. Kinds: csf_text, "
            "csf_parquet, descriptor, header, descriptor_metadata. Repeat once per "
            "artifact; each volume is checked against the sizes that coexist on it."
        ),
    )
    _ = parser.add_argument(
        "--allow-unchecked-space",
        action="store_true",
        help=(
            "Accepted for symmetry with the generation CLI. An estimate only reports "
            "its checks, so an unmeasurable volume is recorded as unchecked either way."
        ),
    )
    add_source_identity_arguments(parser)
    _ = parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be greater than 0")
    if args.memory_budget_mib is not None and args.memory_budget_mib <= 0:
        parser.error("--memory-budget-mib must be greater than 0")
    if not args.transcript.is_file():
        parser.error(f"transcript does not exist: {args.transcript}")
    destinations: dict[str, str] = {}
    for value in args.destination:
        kind, separator, path = value.partition("=")
        if not separator or not kind or not path:
            parser.error(f"--destination expects KIND=PATH, not {value!r}")
        if kind in destinations:
            # Two entries for one artifact would charge its size twice and model
            # a requirement the run never reaches.
            parser.error(f"--destination {kind} was given more than once")
        destinations[kind] = path

    # Snapshot and gate first: a source that cannot be registered must not cost
    # a full estimate, and the snapshot is what the process will actually run.
    start_identity = source_identity()
    require_clean_source(start_identity, args.allow_dirty_source)

    transcript = args.transcript.read_text(encoding="utf-8")
    started = time.perf_counter()
    estimate = dict(
        estimate_disk_generation(
            transcript,
            args.threads,
            memory_budget_mib=args.memory_budget_mib,
            scratch_dir=args.scratch_dir,
            staging_dir=args.staging_dir,
            destinations=destinations or None,
        )
    )
    wall_seconds = time.perf_counter() - started
    digest = sha256_file(args.transcript)
    registered = load_manifest(args.manifest)
    transcript_record = verify_registered_transcript(
        args.transcript,
        digest,
        registered,
        unique_occupations=estimate.get("unique_occupations"),
        records=estimate.get("pre_deduplication_records"),
    )

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
            "threads": args.threads,
            "memory_budget_mib": args.memory_budget_mib,
            "counted_records_are": "pre-de-duplication",
            "space_checked": bool(
                args.scratch_dir or args.staging_dir or destinations
            ),
        },
        "wall_seconds": wall_seconds,
        "filesystem": filesystem_metadata(Path.cwd()),
        "estimate": estimate,
    }
    finalize_report(report, args.output, start_identity, args.allow_dirty_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
