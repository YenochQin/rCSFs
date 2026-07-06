"""Command line interface for rcsfs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import generate_descriptors_from_parquet, read_peel_subshells


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rcsfs",
        description="Command line tools for rCSFs data processing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_descriptors = subparsers.add_parser(
        "gen-descriptors",
        help="Generate descriptor Parquet data from a CSF Parquet file.",
    )
    gen_descriptors.add_argument("input_parquet", type=Path)
    gen_descriptors.add_argument("output_parquet", type=Path)
    gen_descriptors.add_argument(
        "--header",
        required=True,
        type=Path,
        help="Header TOML file generated during CSF conversion.",
    )
    gen_descriptors.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker threads to use.",
    )
    gen_descriptors.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize descriptor values.",
    )
    gen_descriptors.add_argument(
        "--compression",
        default=None,
        help=(
            "Parquet compression codec. Accepted: none (uncompressed), snappy, "
            'zstd, zstd-N (N in 1..=22). Default: zstd-3. Pass "none" to remove '
            "the writer-side compression bottleneck on many-core machines."
        ),
        choices=["none", "uncompressed", "snappy", "zstd"]
        + [f"zstd-{i}" for i in range(1, 23)],
        metavar="{none,snappy,zstd,zstd-N}",
    )
    gen_descriptors.add_argument(
        "--json",
        action="store_true",
        help="Print descriptor generation statistics as JSON.",
    )

    return parser


def _print_gen_descriptors_summary(stats: dict[str, object], normalize: bool) -> None:
    descriptor_kind = "normalized descriptors" if normalize else "descriptors"
    output_file = stats.get("output_file", "")
    print(f"Generated {descriptor_kind}: {output_file}")

    descriptor_count = stats.get("descriptor_count")
    if descriptor_count is not None:
        print(f"descriptor_count: {descriptor_count}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "gen-descriptors":
        peel_subshells = read_peel_subshells(args.header)
        stats = generate_descriptors_from_parquet(
            args.input_parquet,
            args.output_parquet,
            peel_subshells=peel_subshells,
            num_workers=args.num_workers,
            normalize=args.normalize,
            compression=args.compression,
        )
        if args.json:
            json.dump(stats, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        elif stats.get("success") is True:
            _print_gen_descriptors_summary(stats, normalize=args.normalize)
        else:
            error = stats.get("error", "unknown error")
            print(f"Descriptor generation failed: {error}", file=sys.stderr)

        return 0 if stats.get("success") is True else 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
