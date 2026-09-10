"""Repeat and verify serial rcsfgenerate baselines (macOS/Linux, no cache purge)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time
import tomllib


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def time_metrics(text: str) -> dict[str, float | int]:
    if platform.system() == "Darwin":
        fields = {
            "peak_rss_bytes": "maximum resident set size",
            "block_input_operations": "block input operations",
            "block_output_operations": "block output operations",
        }
        values = {}
        for key, label in fields.items():
            match = re.search(rf"^\s*(\d+)\s+{label}$", text, re.M)
            if not match:
                raise ValueError(f"missing {label} in system time output")
            values[key] = int(match[1])
        return values
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    if not match:
        raise ValueError("missing peak RSS in GNU time output")
    return {"peak_rss_bytes": int(match[1]) * 1024}


def run_once(
    executable: Path,
    kind: str,
    transcript: Path,
    work: Path,
    expected: str,
    timeout: int,
) -> dict:
    work.mkdir()
    # The fixture is a recorded here-document, not an executable shell script.
    lines = transcript.read_text().splitlines()
    if lines and lines[0].startswith("rcsfgenerate<<"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "EOF":
        lines.pop()
    raw = "\n".join(line.split("!")[0].rstrip() for line in lines) + "\n"
    (work / "input.txt").write_text(raw)
    command = [str(executable)]
    if kind == "rust":
        command += [str(transcript), "rcsf.out", "2000000"]
    time_flags = ["-l"] if platform.system() == "Darwin" else ["-v"]
    # communicate() drains pipes with an event-driven wait. Waiting on a child
    # with redirected files and a timeout otherwise adds polling latency to
    # short Rust runs (up to 50 ms on CPython).
    with (work / "input.txt").open("rb") as stdin:
        started = time.perf_counter()
        completed = subprocess.run(
            ["/usr/bin/time", *time_flags, *command],
            cwd=work,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=timeout,
        )
        wall_seconds = time.perf_counter() - started
    (work / "stdout.txt").write_bytes(completed.stdout)
    (work / "resources.txt").write_bytes(completed.stderr)
    result = {
        "wall_seconds": wall_seconds,
        **time_metrics((work / "resources.txt").read_text()),
    }
    output = work / "rcsf.out"
    actual = digest(output)
    if actual != expected:
        raise ValueError(
            f"output SHA-256 mismatch in {work}: expected {expected}, got {actual}"
        )
    result.update(output_sha256=actual, output_bytes=output.stat().st_size)
    if kind == "rust":
        result["stages"] = tomllib.loads((work / "stdout.txt").read_text())
    if (work / "profile.toml").exists():
        result["profile"] = tomllib.loads((work / "profile.toml").read_text())
    # Snapshot only; deleted/scratch files and bytes rewritten are not represented.
    result["surviving_files"] = {
        p.name: p.stat().st_size for p in sorted(work.iterdir()) if p.is_file()
    }
    # Avoid retaining gigabytes of reproducible output; checksums and logs remain.
    for name in ["rcsf.out", "clist.new", "fil1.dat", "clist.out", "rcsf.inp"]:
        (work / name).unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        nargs=3,
        metavar=("NAME", "KIND", "EXECUTABLE"),
        required=True,
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="new results directory"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    variants = [
        (name, kind, Path(exe).resolve(strict=True)) for name, kind, exe in args.variant
    ]
    if any(kind not in {"fortran", "rust"} for _, kind, _ in variants):
        parser.error("kind must be fortran or rust")
    if len({name for name, _, _ in variants}) != len(variants) or any(
        not re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name, _, _ in variants
    ):
        parser.error("variant names must be unique simple directory names")
    args.output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    results = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cache_policy": "one untabulated warmup per variant/case; no OS cache purge; output verification outside timed interval; no fsync",
        "variants": {
            name: {"kind": kind, "executable": str(exe), "sha256": digest(exe)}
            for name, kind, exe in variants
        },
        "cases": {},
    }
    for case in ["e1_cc1as1", "o1_cc1as1"]:
        transcript = repository / "tests/fixtures" / f"{case}.rcsfgenerate"
        expected = digest(args.baseline_dir / f"{case}.c")
        results["cases"][case] = {
            "input_sha256": digest(transcript),
            "expected_output_sha256": expected,
            "variants": {},
        }
        for name, kind, exe in variants:
            warmup = run_once(
                exe,
                kind,
                transcript,
                args.output / f"{case}-{name}-warmup",
                expected,
                args.timeout,
            )
            runs = []
            for repeat in range(args.repeats):
                row = run_once(
                    exe,
                    kind,
                    transcript,
                    args.output / f"{case}-{name}-{repeat}",
                    expected,
                    args.timeout,
                )
                runs.append(row)
                print(
                    f"{case} {name} {repeat + 1}: {row['wall_seconds']:.6f}s, RSS {row['peak_rss_bytes']} bytes",
                    flush=True,
                )
            summary = {
                "warmup": warmup,
                "runs": runs,
                "median_wall_seconds": statistics.median(
                    row["wall_seconds"] for row in runs
                ),
                "median_peak_rss_bytes": statistics.median(
                    row["peak_rss_bytes"] for row in runs
                ),
            }
            results["cases"][case]["variants"][name] = summary
            (args.output / "results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
    print(args.output / "results.json")


if __name__ == "__main__":
    main()
