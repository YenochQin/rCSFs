"""Command line interface for rcsfs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from . import convert_csfs, generate_descriptors_from_parquet, partition_csfs, read_peel_subshells


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

    zero_first = subparsers.add_parser(
        "zero-first",
        help="Reorder CSFs into zero-order + first-order space per symmetry block.",
        description=(
            "Partition a CSF list so that, within each symmetry block, the "
            "zero-order reference CSFs are locked to the head of the block and "
            "the first-order complement is appended after them. Mirrors "
            "GRASP2018's rcsfzerofirst utility but routes through the Parquet "
            "layer produced by convert_csfs."
        ),
    )
    zero_first.add_argument("zero_csf", type=Path, help="Zero-order reference CSF file.")
    zero_first.add_argument("full_csf", type=Path, help="Complete CSF list to be partitioned.")
    zero_first.add_argument(
        "output_csf",
        nargs="?",
        type=Path,
        help="Destination CSF file. Default: {full_stem}_zf.csf beside the full input.",
    )
    zero_first.add_argument(
        "--keep-parquet",
        action="store_true",
        help="Keep intermediate Parquet + header TOML files (default: clean up).",
    )
    zero_first.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for intermediate Parquet files. Default: system temp dir.",
    )
    zero_first.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker threads for CSF-to-Parquet conversion.",
    )
    zero_first.add_argument(
        "--max-line-len",
        type=int,
        default=256,
        help="Maximum CSF line length for conversion (default: 256).",
    )
    zero_first.add_argument(
        "--json",
        action="store_true",
        help="Print partition statistics as JSON.",
    )

    return parser


def _print_gen_descriptors_summary(stats: Mapping[str, object], normalize: bool) -> None:
    descriptor_kind = "normalized descriptors" if normalize else "descriptors"
    output_file = stats.get("output_file", "")
    print(f"Generated {descriptor_kind}: {output_file}")

    descriptor_count = stats.get("descriptor_count")
    if descriptor_count is not None:
        print(f"descriptor_count: {descriptor_count}")


def _print_zero_first_summary(stats: Mapping[str, object]) -> None:
    output_file = stats.get("output_file", "")
    print(f"Partitioned CSFs: {output_file}")
    first_order = stats.get("first_order_count")
    if first_order is not None:
        print(f"first_order_count: {first_order}")
    block_count = stats.get("block_count")
    if block_count is not None:
        print(f"block_count: {block_count}")


def _convert_to_parquet(
    src: Path, dest_parquet: Path, args: argparse.Namespace, label: str
) -> str:
    stats = convert_csfs(
        src,
        dest_parquet,
        max_line_len=args.max_line_len,
        num_workers=args.num_workers,
    )
    if stats.get("success") is not True:
        error = stats.get("error", "unknown error")
        raise RuntimeError(f"{label} conversion failed: {error}")
    header = stats.get("header_file")
    if not isinstance(header, str):
        header = str(dest_parquet.parent / f"{src.stem}_header.toml")
    return header


def _run_zero_first(args: argparse.Namespace) -> int:
    zero_path: Path = args.zero_csf
    full_path: Path = args.full_csf
    output_path = (
        args.output_csf
        if args.output_csf is not None
        else full_path.with_name(f"{full_path.stem}_zf.csf")
    )

    base_dir = args.work_dir if args.work_dir is not None else Path(tempfile.gettempdir())
    base_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="rcsfs-zero-first-", dir=str(base_dir)))

    try:
        zero_dir = root / "zero"
        full_dir = root / "full"
        zero_dir.mkdir()
        full_dir.mkdir()

        zero_pq = zero_dir / f"{zero_path.stem}.parquet"
        full_pq = full_dir / f"{full_path.stem}.parquet"

        zero_header = _convert_to_parquet(zero_path, zero_pq, args, "Zero-order")
        full_header = _convert_to_parquet(full_path, full_pq, args, "Full-list")

        stats = partition_csfs(zero_pq, zero_header, full_pq, full_header, output_path)

        if args.keep_parquet:
            print(f"Intermediate Parquet kept under: {root}")

        if args.json:
            json.dump(stats, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        elif stats.get("success") is True:
            _print_zero_first_summary(stats)
        else:
            error = stats.get("error", "unknown error")
            print(f"Partition failed: {error}", file=sys.stderr)

        return 0 if stats.get("success") is True else 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if not args.keep_parquet:
            shutil.rmtree(root, ignore_errors=True)


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

    if args.command == "zero-first":
        return _run_zero_first(args)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
