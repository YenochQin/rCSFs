"""Measure the V2 disk-generation stages for a supplied transcript.

This is an explicit benchmark, not a pytest test. It only uses the public
Python API and a temporary output set, so it does not require a GRASP checkout.
Rust progress and diagnostics remain on stderr; the JSON report is written to
stdout or to ``--output``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Running a script places ``scripts/`` first on sys.path. Prefer the checkout's
# editable package so the benchmark measures the source under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcsfs import generate_disk_outputs_from_transcript


def _run_once(transcript: str, threads: int | None) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rcsfs-v2-benchmark-") as directory:
        root = Path(directory)
        stats = generate_disk_outputs_from_transcript(
            transcript,
            root / "output.c",
            root / "output.parquet",
            root / "descriptors.parquet",
            root / "header.toml",
            root / "scratch",
            threads=threads,
        )
    result: dict[str, Any] = {
        "threads": threads,
        "wall_seconds": time.perf_counter() - started,
        "success": stats.get("success", False),
    }
    for key in (
        "unique_occupations",
        "generated_count",
        "record_count",
        "descriptor_count",
        "duplicate_count",
        "block_count",
        "csf_bytes",
        "descriptor_bytes",
        "stage_stats",
    ):
        if key in stats:
            result[key] = stats[key]
    return result


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
    _ = parser.add_argument("--repeats", type=int, default=1)
    _ = parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be greater than 0")
    if any(thread is not None and thread <= 0 for thread in args.threads):
        parser.error("--threads values must be greater than 0")

    transcript = args.transcript.read_text(encoding="utf-8")
    measurements = [
        _run_once(transcript, threads)
        for threads in args.threads
        for _ in range(args.repeats)
    ]
    report = {
        "transcript_name": args.transcript.name,
        "repeats": args.repeats,
        "measurements": measurements,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
