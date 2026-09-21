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
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Running a script places ``scripts/`` first on sys.path. Prefer the checkout's
# editable package so the estimate describes the source under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcsfs import estimate_disk_generation

from benchmark_v2_generation import (
    DEFAULT_MANIFEST,
    _environment,
    _filesystem_metadata,
    load_manifest,
    sha256_file,
)

REPORT_SCHEMA = "rcsfs-v2-generation-capacity/1"


def _verify_registered(
    path: Path, digest: str, registered: dict[str, dict[str, Any]], report: dict[str, Any]
) -> dict[str, Any]:
    """Check the estimate against the registered baseline, if there is one."""
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
    occupations = report.get("unique_occupations")
    if occupations is not None and occupations != record["expected_unique_occupations"]:
        raise SystemExit(
            f"{path.name} enumerated {occupations} configurations but the registered "
            f"baseline is {record['expected_unique_occupations']}"
        )
    records = report.get("pre_deduplication_records")
    if records is not None and records != record["expected_records"]:
        raise SystemExit(
            f"{path.name} counted {records} records but the registered baseline is "
            f"{record['expected_records']}"
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("transcript", type=Path)
    _ = parser.add_argument("--threads", type=int, default=None)
    _ = parser.add_argument("--memory-budget-mib", type=int, default=None)
    _ = parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    _ = parser.add_argument(
        "--destination",
        type=Path,
        action="append",
        default=[],
        help="Directory that will receive a published artifact; its free space is reported.",
    )
    _ = parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be greater than 0")
    if args.memory_budget_mib is not None and args.memory_budget_mib <= 0:
        parser.error("--memory-budget-mib must be greater than 0")
    if not args.transcript.is_file():
        parser.error(f"transcript does not exist: {args.transcript}")

    transcript = args.transcript.read_text(encoding="utf-8")
    started = time.perf_counter()
    estimate = dict(
        estimate_disk_generation(
            transcript, args.threads, memory_budget_mib=args.memory_budget_mib
        )
    )
    wall_seconds = time.perf_counter() - started
    digest = sha256_file(args.transcript)
    registered = load_manifest(args.manifest)
    transcript_record = _verify_registered(args.transcript, digest, registered, estimate)

    # The destinations are known here, not inside the counting call, and
    # publication copies into them, so each is checked for its own share.
    destination_checks: list[dict[str, Any]] = []
    for directory in args.destination:
        if not directory.is_dir():
            raise SystemExit(f"destination is not a directory: {directory}")
        required = estimate["bytes"]["required_output"]
        free = shutil.disk_usage(directory).free
        destination_checks.append(
            {
                "path": str(directory),
                "required_bytes": required,
                "free_bytes": free,
                "sufficient": free >= required,
            }
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
            "threads": args.threads,
            "memory_budget_mib": args.memory_budget_mib,
            "counted_records_are": "pre-de-duplication",
        },
        "wall_seconds": wall_seconds,
        "filesystem": _filesystem_metadata(Path.cwd()),
        "estimate": estimate,
        "destination_checks": destination_checks,
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
