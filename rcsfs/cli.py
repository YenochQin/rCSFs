"""Command line interface for rcsfs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, cast

from . import (
    convert_csfs,
    generate_csfs_from_transcript,
    generate_descriptors_from_parquet,
    partition_csfs,
    read_peel_subshells,
)

#: Maximum reference configurations accepted, matching GRASP's `rcsfgenerate`.
_MAX_REFERENCE_CONFIGURATIONS = 100


class GenDescriptorsArgs(Protocol):
    """Parsed arguments for the ``gen-descriptors`` subcommand."""

    command: Literal["gen-descriptors"]
    input_parquet: Path
    output_parquet: Path
    header: Path
    num_workers: int | None
    normalize: bool
    compression: str | None
    json: bool


class ZeroFirstArgs(Protocol):
    """Parsed arguments for the ``zero-first`` subcommand."""

    command: Literal["zero-first"]
    zero_csf: Path
    full_csf: Path
    output_csf: Path | None
    keep_parquet: bool
    work_dir: Path | None
    num_workers: int | None
    max_line_len: int
    json: bool


class CsfsGenerateArgs(Protocol):
    """Parsed arguments for the ``csfsgenerate`` subcommand."""

    command: Literal["csfsgenerate"]
    output: Path

    descriptors: Path | None
    normalize: bool
    threads: int | None
    json: bool


type CliArgs = GenDescriptorsArgs | ZeroFirstArgs | CsfsGenerateArgs


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
    _ = gen_descriptors.add_argument("input_parquet", type=Path)
    _ = gen_descriptors.add_argument("output_parquet", type=Path)
    _ = gen_descriptors.add_argument(
        "--header",
        required=True,
        type=Path,
        help="Header TOML file generated during CSF conversion.",
    )
    _ = gen_descriptors.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker threads to use.",
    )
    _ = gen_descriptors.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize descriptor values.",
    )
    _ = gen_descriptors.add_argument(
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
    _ = gen_descriptors.add_argument(
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
    _ = zero_first.add_argument(
        "zero_csf", type=Path, help="Zero-order reference CSF file."
    )
    _ = zero_first.add_argument(
        "full_csf", type=Path, help="Complete CSF list to be partitioned."
    )
    _ = zero_first.add_argument(
        "output_csf",
        nargs="?",
        type=Path,
        help="Destination CSF file. Default: {full_stem}_zf.csf beside the full input.",
    )
    _ = zero_first.add_argument(
        "--keep-parquet",
        action="store_true",
        help="Keep intermediate Parquet + header TOML files (default: clean up).",
    )
    _ = zero_first.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for intermediate Parquet files. Default: system temp dir.",
    )
    _ = zero_first.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker threads for CSF-to-Parquet conversion.",
    )
    _ = zero_first.add_argument(
        "--max-line-len",
        type=int,
        default=256,
        help="Maximum CSF line length for conversion (default: 256).",
    )
    _ = zero_first.add_argument(
        "--json",
        action="store_true",
        help="Print partition statistics as JSON.",
    )

    csfsgenerate = subparsers.add_parser(
        "csfsgenerate",
        help="Interactively generate a new CSF list, replicating GRASP's rcsfgenerate dialog.",
        description=(
            "Ask the same sequence of questions as GRASP2018's rcsfgenerate "
            "(orbital order, core, reference configurations, active orbitals, "
            "2J range, excitation count) and generate the resulting CSF list "
            "with the Rust generator. Options with no equivalent question in "
            "the original dialog (output path, descriptor export, "
            "thread count) are plain CLI flags."
        ),
    )
    _ = csfsgenerate.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("rcsf.out"),
        help="Destination CSF text file (default: rcsf.out; must not already exist).",
    )
    _ = csfsgenerate.add_argument(
        "--descriptors",
        type=Path,
        default=None,
        help="Optional descriptor CSV output path.",
    )
    _ = csfsgenerate.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize descriptor values (only used with --descriptors).",
    )
    _ = csfsgenerate.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Rayon thread count for generation (default: all cores).",
    )
    _ = csfsgenerate.add_argument(
        "--json",
        action="store_true",
        help="Print generation statistics as JSON.",
    )

    return parser


def _parse_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
) -> CliArgs:
    """Parse the CLI's discriminated argument union at one dynamic boundary."""
    namespace = cast(object, parser.parse_args(argv))
    return cast(CliArgs, namespace)


def _print_gen_descriptors_summary(
    stats: Mapping[str, object], normalize: bool
) -> None:
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
    src: Path, dest_parquet: Path, args: ZeroFirstArgs, label: str
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


def _run_zero_first(args: ZeroFirstArgs) -> int:
    zero_path = args.zero_csf
    full_path = args.full_csf
    output_path = (
        args.output_csf
        if args.output_csf is not None
        else full_path.with_name(f"{full_path.stem}_zf.csf")
    )

    base_dir = (
        args.work_dir if args.work_dir is not None else Path(tempfile.gettempdir())
    )
    _ = base_dir.mkdir(parents=True, exist_ok=True)
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
            _ = sys.stdout.write("\n")
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


def _prompt(message: str) -> str:
    """Sole indirection point over ``input()`` so tests can script answers."""
    return input(message)


def _read_order() -> str:
    while True:
        answer = _prompt(
            "Default, reverse, symmetry or user specified ordering? (*/r/s/u) "
        ).strip()
        if answer == "*":
            return answer
        print(
            f"Orbital order {answer!r} is not supported yet; only * (default) "
            "is implemented. Answer *.",
            file=sys.stderr,
        )


def _read_core() -> int:
    print("Select core")
    print(" 0  No core")
    print(" 1  He (2)")
    print(" 2  Ne (10)")
    print(" 3  Ar (18)")
    print(" 4  Kr (36)")
    print(" 5  Xe (54)")
    print(" 6  Rn (86)")
    while True:
        answer = _prompt("Core? (0-6) ").strip()
        try:
            value = int(answer)
        except ValueError:
            print("Enter an integer 0..=6.", file=sys.stderr)
            continue
        if 0 <= value <= 6:
            return value
        print("Core selector must be 0..=6.", file=sys.stderr)


def _read_references() -> list[str]:
    print(
        f"Enter list of (maximum {_MAX_REFERENCE_CONFIGURATIONS}) configurations. "
        "End list with a blank line or an asterisk (*)"
    )
    references: list[str] = []
    while len(references) < _MAX_REFERENCE_CONFIGURATIONS:
        line = _prompt(f"Give configuration {len(references) + 1}: ")
        stripped = line.strip()
        if stripped in ("", "*"):
            break
        if stripped.count("(") != stripped.count(","):
            print(
                "Each orbital must be closed (c), inactive (i), active (*) "
                "or have a minimal occupation; redo!",
                file=sys.stderr,
            )
            continue
        references.append(stripped)
    if not references:
        raise SystemExit("at least one reference configuration is required")
    return references


def _read_active_orbitals() -> str:
    while True:
        line = _prompt(
            "Give set of active orbitals, as defined by the highest principal "
            "quantum number per l-symmetry, in a comma delimited list in "
            "s,p,d etc order, e.g. 5s,4p,3d: "
        ).strip()
        tokens = [token.strip() for token in line.split(",")]
        if tokens and all(2 <= len(token) <= 3 for token in tokens):
            return line
        print(
            "Orbitals should be given in comma delimited list, redo!",
            file=sys.stderr,
        )


def _read_j_range() -> tuple[int, int]:
    while True:
        line = _prompt("Resulting 2*J-number? lower, higher (J=1 -> 2*J=2 etc.): ")
        parts = [part for part in line.replace(",", " ").split() if part]
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass
        print("Enter two integers: lower,higher", file=sys.stderr)


def _read_excitations() -> int:
    while True:
        line = _prompt(
            "Number of excitations (if negative number e.g. -2, correlation "
            "orbitals will always be doubly occupied): "
        ).strip()
        try:
            return int(line)
        except ValueError:
            print("Enter an integer.", file=sys.stderr)


def _read_continue() -> bool:
    answer = _prompt("Generate more lists ? (y/n) ").strip().lower()
    return answer == "y"


def _print_csfsgenerate_summary(stats: Mapping[str, object]) -> None:
    output_file = stats.get("output_file", "")
    print(f"Generated CSFs: {output_file}")
    record_count = stats.get("record_count")
    if record_count is not None:
        print(f"record_count: {record_count}")
    block_count = stats.get("block_count")
    if block_count is not None:
        print(f"block_count: {block_count}")
    descriptor_file = stats.get("descriptor_file")
    if descriptor_file is not None:
        print(f"descriptor_file: {descriptor_file}")


def _run_csfsgenerate(args: CsfsGenerateArgs) -> int:
    order = _read_order()
    core = _read_core()
    references = _read_references()
    active_orbitals = _read_active_orbitals()
    j_min, j_max = _read_j_range()
    excitations = _read_excitations()
    if _read_continue():
        print(
            "Multiple lists are not supported yet; only the first list "
            "would be generated. Aborting.",
            file=sys.stderr,
        )
        return 1

    transcript = "\n".join(
        [
            f"{order} ! Orbital order",
            str(core),
            *references,
            "",
            active_orbitals,
            f"{j_min},{j_max}",
            str(excitations),
            "n",
        ]
    )

    stats = generate_csfs_from_transcript(
        transcript,
        args.output,
        descriptor_path=args.descriptors,
        normalize=args.normalize,
        threads=args.threads,
    )

    if args.json:
        json.dump(stats, sys.stdout, indent=2, sort_keys=True)
        _ = sys.stdout.write("\n")
    elif stats.get("success") is True:
        _print_csfsgenerate_summary(stats)
    else:
        error = stats.get("error", "unknown error")
        print(f"CSF generation failed: {error}", file=sys.stderr)

    return 0 if stats.get("success") is True else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = _parse_args(parser, argv)

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
            _ = sys.stdout.write("\n")
        elif stats.get("success") is True:
            _print_gen_descriptors_summary(stats, normalize=args.normalize)
        else:
            error = stats.get("error", "unknown error")
            print(f"Descriptor generation failed: {error}", file=sys.stderr)

        return 0 if stats.get("success") is True else 1

    if args.command == "csfsgenerate":
        return _run_csfsgenerate(args)

    return _run_zero_first(args)


if __name__ == "__main__":
    raise SystemExit(main())
