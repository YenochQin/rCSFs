"""Command line interface for rcsfs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Literal, Protocol, cast

from . import (
    convert_csfs,
    generate_disk_outputs_from_transcript,
    generate_csfs_from_transcript,
    generate_descriptors_from_parquet,
    partition_csfs,
    read_peel_subshells,
    restore_csfs_from_descriptors,
    select_interacting_csfs,
)
from ._types import InteractionHamiltonian, InteractionMethod

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
    descriptor_version: int
    compression: str | None
    json: bool


class RestoreCsfsArgs(Protocol):
    """Parsed arguments for the ``restore-csfs`` subcommand."""

    command: Literal["restore-csfs"]
    descriptors: Path
    header: Path
    output: Path
    indices: list[int] | None
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
    config: Path | None
    generate_descriptors: bool
    parquet: Path | None
    descriptor_parquet: Path | None

    normalize: bool
    threads: int | None
    generation_storage: Literal["memory", "disk"] | None
    scratch_dir: Path | None
    json: bool


class InteractingArgs(Protocol):
    """Parsed arguments for the ``interacting`` subcommand."""

    command: Literal["interacting"]
    reference: Path
    candidates: Path
    output: Path
    hamiltonian: InteractionHamiltonian
    method: InteractionMethod
    num_workers: int
    overwrite: bool
    json: bool


type CliArgs = (
    GenDescriptorsArgs
    | ZeroFirstArgs
    | CsfsGenerateArgs
    | InteractingArgs
    | RestoreCsfsArgs
)


#: CLI spellings accepted for each domain Hamiltonian value.
_HAMILTONIAN_ALIASES: dict[str, InteractionHamiltonian] = {
    "dc": "dirac_coulomb",
    "dirac-coulomb": "dirac_coulomb",
    "dirac_coulomb": "dirac_coulomb",
    "dcb": "dirac_coulomb_breit",
    "dirac-coulomb-breit": "dirac_coulomb_breit",
    "dirac_coulomb_breit": "dirac_coulomb_breit",
}

#: CLI spellings accepted for each domain selection method.
_METHOD_ALIASES: dict[str, InteractionMethod] = {
    "structural-upper-bound": "structural_upper_bound",
    "structural_upper_bound": "structural_upper_bound",
}


def _parse_hamiltonian(value: str) -> InteractionHamiltonian:
    try:
        return _HAMILTONIAN_ALIASES[value.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "expected dc, dcb, dirac-coulomb, or dirac-coulomb-breit"
        ) from exc


def _parse_interaction_method(value: str) -> InteractionMethod:
    try:
        return _METHOD_ALIASES[value.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "only structural-upper-bound is implemented in this release"
        ) from exc


def _parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


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
        help="Normalize descriptor values. Not supported with --descriptor-version 2.",
    )
    _ = gen_descriptors.add_argument(
        "--descriptor-version",
        type=int,
        choices=[1, 2],
        default=2,
        help=(
            "Descriptor format version: 2 (default; four-channel per-subshell "
            "plus total_two_j/parity globals) or 1 (legacy dense triplet)."
        ),
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
        "--config",
        type=Path,
        default=None,
        help="TOML generation configuration; replaces the interactive dialog.",
    )
    _ = csfsgenerate.add_argument(
        "--generate-descriptors",
        action="store_true",
        help="Also write CSF Parquet and descriptor Parquet outputs.",
    )
    _ = csfsgenerate.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="CSF Parquet output (default: same stem as the CSF file).",
    )
    _ = csfsgenerate.add_argument(
        "--descriptor-parquet",
        type=Path,
        default=None,
        help="Descriptor Parquet output (default: <stem>_descriptors.parquet).",
    )
    _ = csfsgenerate.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize descriptor values when --generate-descriptors is enabled.",
    )
    _ = csfsgenerate.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Rayon thread count for generation (default: all cores).",
    )
    _ = csfsgenerate.add_argument(
        "--generation-storage",
        choices=["memory", "disk"],
        default=None,
        help="Generation backend. disk streams reversible V2 records through scratch storage.",
    )
    _ = csfsgenerate.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Existing directory for disk-generation scratch data.",
    )
    _ = csfsgenerate.add_argument(
        "--json",
        action="store_true",
        help="Print generation statistics as JSON.",
    )

    interacting = subparsers.add_parser(
        "interacting",
        help="Select a structural upper bound of CSFs interacting with references.",
        description=(
            "Select candidate CSFs that pass inexpensive structural conditions "
            "for interaction with at least one reference CSF. This is a "
            "STRUCTURAL UPPER BOUND, NOT an exact reproduction of GRASP's "
            "rcsfinteract90 angular-algebra calculation."
        ),
    )
    _ = interacting.add_argument(
        "reference", type=Path, help="Reference (MR) CSF file."
    )
    _ = interacting.add_argument(
        "candidates", type=Path, help="Candidate CSF file to filter."
    )
    _ = interacting.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("rcsf.out"),
        help="Destination CSF file (default: rcsf.out).",
    )
    _ = interacting.add_argument(
        "--hamiltonian",
        type=_parse_hamiltonian,
        default="dirac_coulomb",
        metavar="{dc,dcb,dirac-coulomb,dirac-coulomb-breit}",
        help=(
            "Hamiltonian whose structural selection rules are applied (default: dc)."
        ),
    )
    _ = interacting.add_argument(
        "--method",
        type=_parse_interaction_method,
        default="structural_upper_bound",
        metavar="structural-upper-bound",
        help=(
            "Selection method. Only structural-upper-bound is currently "
            "implemented; it can include false positives."
        ),
    )
    _ = interacting.add_argument(
        "--threads",
        "--num-workers",
        dest="num_workers",
        type=_parse_positive_int,
        default=8,
        help="Worker thread count (default: 8).",
    )
    _ = interacting.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    _ = interacting.add_argument(
        "--json",
        action="store_true",
        help="Print selection statistics as JSON.",
    )

    restore_csfs = subparsers.add_parser(
        "restore-csfs",
        help="Restore a CSF text file from a V2 descriptor Parquet file.",
        description=(
            "Rebuild a CSF text file from a V2 descriptor Parquet file and its "
            "source {stem}_header.toml. The header path must be explicit and "
            "exact; if the descriptor file recorded source_header_sha256, it is "
            "verified against this file before anything is written."
        ),
    )
    _ = restore_csfs.add_argument(
        "--descriptors", required=True, type=Path, help="V2 descriptor Parquet file."
    )
    _ = restore_csfs.add_argument(
        "--header", required=True, type=Path, help="Source {stem}_header.toml file."
    )
    _ = restore_csfs.add_argument(
        "--output", required=True, type=Path, help="Destination CSF text file."
    )
    _ = restore_csfs.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional 0-based row indices to restore, in the given order.",
    )
    _ = restore_csfs.add_argument(
        "--json",
        action="store_true",
        help="Print restoration statistics as JSON.",
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


def _write_gen_descriptors_sidecar(
    output_parquet: Path,
    stats: Mapping[str, object],
    peel_subshells: list[str],
    normalize: bool,
) -> None:
    """Write the `{descriptor_stem}.toml` mirror of the Parquet KV metadata.

    This is the plan D5 layer-3 sidecar for tools that read TOML without
    opening the Parquet file. `gen-descriptors` previously wrote none of
    this; `csfsgenerate --generate-descriptors` already writes an equivalent
    file via `_generate_outputs`, and reuses the same `format_version` key
    for the descriptor version tag rather than introducing a second key.
    """
    sidecar = output_parquet.with_suffix(".toml")
    descriptor_version = stats.get("descriptor_version", 1)
    _ = sidecar.write_text(
        f'format_version = {descriptor_version}\nencoding = "parquet"\n'
        f"normalized = {str(normalize).lower()}\n"
        f"record_count = {stats.get('descriptor_count', 0)}\n"
        f"subshells = {json.dumps(peel_subshells)}\n",
        encoding="utf-8",
    )


def _validate_gen_descriptors_sidecar_path(
    input_parquet: Path, output_parquet: Path, header_path: Path
) -> None:
    """Reject a sidecar path that aliases an input or the Parquet output."""
    sidecar = output_parquet.with_suffix(".toml")
    for label, path in (
        ("input Parquet", input_parquet),
        ("output Parquet", output_parquet),
        ("header", header_path),
    ):
        aliases = sidecar.resolve() == path.resolve()
        if not aliases and sidecar.exists() and path.exists():
            aliases = sidecar.samefile(path)
        if aliases:
            raise ValueError(f"descriptor sidecar aliases {label}: {sidecar}")


def _print_restore_csfs_summary(stats: Mapping[str, object]) -> None:
    output_file = stats.get("output_file", "")
    print(f"Restored CSFs: {output_file}")
    record_count = stats.get("record_count")
    if record_count is not None:
        print(f"record_count: {record_count}")


def _print_zero_first_summary(stats: Mapping[str, object]) -> None:
    output_file = stats.get("output_file", "")
    print(f"Partitioned CSFs: {output_file}")
    first_order = stats.get("first_order_count")
    if first_order is not None:
        print(f"first_order_count: {first_order}")
    block_count = stats.get("block_count")
    if block_count is not None:
        print(f"block_count: {block_count}")


def _print_interacting_summary(stats: Mapping[str, object]) -> None:
    output_file = stats.get("output_file", "")
    print(f"Selected interacting CSFs (structural upper bound): {output_file}")
    for key in ("reference_count", "candidate_count", "selected_count"):
        value = stats.get(key)
        if value is not None:
            print(f"{key}: {value}")


def _run_interacting(args: InteractingArgs) -> int:
    print(
        "WARNING: STRUCTURAL UPPER BOUND ONLY; NOT an exact rcsfinteract90 "
        "angular-algebra calculation. False positives may be retained.",
        file=sys.stderr,
    )
    try:
        stats = select_interacting_csfs(
            args.reference,
            args.candidates,
            args.output,
            hamiltonian=args.hamiltonian,
            method=args.method,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        if args.json:
            json.dump(
                {"success": False, "error": str(exc)},
                sys.stdout,
                indent=2,
                sort_keys=True,
            )
            _ = sys.stdout.write("\n")
        else:
            print(f"Interaction selection failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        json.dump(stats, sys.stdout, indent=2, sort_keys=True)
        _ = sys.stdout.write("\n")
    else:
        _print_interacting_summary(stats)

    return 0


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


def _generate_outputs(transcript: str, args: CsfsGenerateArgs) -> dict[str, object]:
    csf_parquet = args.parquet or args.output.with_suffix(".parquet")
    descriptor_parquet = args.descriptor_parquet or args.output.with_name(
        f"{args.output.stem}_descriptors.parquet"
    )
    header = csf_parquet.parent / f"{args.output.stem}_header.toml"
    metadata = descriptor_parquet.with_suffix(".toml")
    destinations = [args.output]
    if args.generate_descriptors:
        destinations.extend([csf_parquet, header, descriptor_parquet, metadata])
    resolved = [path.resolve() for path in destinations]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Generation output paths must be distinct")
    if args.config is not None and args.config.resolve() in resolved:
        raise ValueError("Generation output must not overwrite the configuration input")
    for path in destinations:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
    if not args.generate_descriptors:
        return dict(
            generate_csfs_from_transcript(
                transcript, args.output, normalize=args.normalize, threads=args.threads
            )
        )
    if args.generation_storage == "disk" and args.normalize:
        raise ValueError("normalize is not supported by reversible V2 descriptors")

    # The existing converters truncate their destinations. Run them only in a
    # private staging directory, then hold exclusive handles for publication.
    # Staged under the current working directory rather than the system temp
    # dir: disk-mode generation can write far more Arrow/Parquet data than a
    # tmpfs-backed /tmp has room for, so the caller's own filesystem is the
    # safer default. Removed automatically on exit either way.
    with tempfile.TemporaryDirectory(
        prefix="rcsfs-generation-", dir=str(Path.cwd())
    ) as directory:
        root = Path(directory)
        csf_dir = root / "text"
        csf_dir.mkdir()
        csf = csf_dir / args.output.name
        parquet_dir = root / "parquet"
        parquet_dir.mkdir()
        parquet = parquet_dir / "csfs.parquet"
        descriptors = root / "features.parquet"
        staged_header = parquet_dir / f"{csf.stem}_header.toml"
        if args.generation_storage == "disk":
            scratch_base = args.scratch_dir if args.scratch_dir is not None else root
            if not scratch_base.is_dir():
                raise FileNotFoundError(
                    f"Scratch directory does not exist: {scratch_base}"
                )
            scratch = scratch_base / f"rcsfs-disk-{csf.stem}"
            if scratch.exists():
                raise FileExistsError(
                    f"Disk-generation scratch already exists: {scratch}"
                )
            stats = dict(
                generate_disk_outputs_from_transcript(
                    transcript,
                    csf,
                    parquet,
                    descriptors,
                    staged_header,
                    scratch,
                    threads=args.threads,
                )
            )
            shutil.rmtree(scratch, ignore_errors=True)
        else:
            stats = dict(
                generate_csfs_from_transcript(
                    transcript, csf, normalize=args.normalize, threads=args.threads
                )
            )
        if stats.get("success") is not True:
            return stats
        if args.generation_storage != "disk":
            conversion = convert_csfs(csf, parquet)
            if conversion.get("success") is not True:
                return {
                    "success": False,
                    "error": conversion.get("error", "CSF conversion failed"),
                }
        shells = read_peel_subshells(staged_header)
        if args.generation_storage == "disk":
            result: Mapping[str, object] = stats
        else:
            result = generate_descriptors_from_parquet(
                parquet,
                descriptors,
                peel_subshells=shells,
                normalize=args.normalize,
                descriptor_version=1,
                header_path=staged_header,
                compression="zstd",
            )
        if result.get("success") is not True:
            return {
                "success": False,
                "error": result.get("error", "Descriptor generation failed"),
            }
        sidecar = root / "features.toml"
        _ = sidecar.write_text(
            f'format_version = {result.get("descriptor_version", 2)}\nencoding = "parquet"\n'
            f"normalized = {str(args.normalize).lower()}\n"
            f"record_count = {result['descriptor_count']}\n"
            f"subshells = {json.dumps(shells)}\n",
            encoding="utf-8",
        )
        sources = [csf, parquet, staged_header, descriptors, sidecar]
        with ExitStack() as stack:
            handles = [stack.enter_context(path.open("xb")) for path in destinations]
            for source, handle in zip(sources, handles, strict=True):
                with source.open("rb") as reader:
                    shutil.copyfileobj(reader, handle)
        stats.update(
            output_file=str(args.output),
            parquet_file=str(csf_parquet),
            descriptor_parquet_file=str(descriptor_parquet),
            descriptor_metadata_file=str(metadata),
        )
        return stats


def _config_table(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Expected a TOML table")
    return cast(dict[str, object], value)


def _config_int(value: object) -> int:
    if type(value) is not int:
        raise ValueError("Expected an integer")
    return value


def _config_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("Expected a boolean")
    return value


def _config_string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Expected a string")
    return value


def _config_references(value: object) -> list[str]:
    if not isinstance(value, list):
        raise TypeError("references must be an array of strings")
    return [_config_string(item) for item in cast(list[object], value)]


def _run_csfsgenerate(args: CsfsGenerateArgs) -> int:
    if args.config is not None:
        try:
            config = _config_table(
                tomllib.loads(args.config.read_text(encoding="utf-8"))
            )
            generate = _config_table(config["generate"])
            order = _config_string(generate.get("order", "*"))
            core = _config_int(generate["core"])
            references = _config_references(generate["references"])
            active_orbitals = _config_string(generate["active_orbitals"])
            j_min = _config_int(generate["j_min"])
            j_max = _config_int(generate["j_max"])
            excitations = _config_int(generate["excitations"])
            output = _config_table(config.get("output", {}))
            if args.output == Path("rcsf.out") and output.get("csf") is not None:
                args.output = Path(_config_string(output["csf"]))
            if not args.generate_descriptors:
                args.generate_descriptors = _config_bool(
                    output.get("generate_descriptors", False)
                )
            if args.parquet is None and output.get("parquet") is not None:
                args.parquet = Path(_config_string(output["parquet"]))
            if (
                args.descriptor_parquet is None
                and output.get("descriptor_parquet") is not None
            ):
                args.descriptor_parquet = Path(
                    _config_string(output["descriptor_parquet"])
                )
            if not args.normalize:
                args.normalize = _config_bool(output.get("normalize", False))
            if args.generation_storage is None and generate.get("storage") is not None:
                args.generation_storage = cast(
                    Literal["memory", "disk"],
                    _config_string(generate["storage"]),
                )
            if args.scratch_dir is None and generate.get("scratch_dir") is not None:
                args.scratch_dir = Path(_config_string(generate["scratch_dir"]))
            if _config_bool(generate.get("continue_lists", False)):
                raise ValueError("continue_lists is not supported yet; use false")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            print(f"Invalid generation config: {exc}", file=sys.stderr)
            return 2
    else:
        order = _read_order()
        core = _read_core()
        references = _read_references()
        active_orbitals = _read_active_orbitals()
        j_min, j_max = _read_j_range()
        excitations = _read_excitations()
    if args.config is None and _read_continue():
        print(
            "Multiple lists are not supported yet; only the first list "
            "would be generated. Aborting.",
            file=sys.stderr,
        )
        return 1

    if args.generation_storage is None:
        args.generation_storage = (
            "disk"
            if args.generate_descriptors and args.config is not None
            else "memory"
        )
    if args.generation_storage not in ("memory", "disk"):
        print("Invalid generation storage: expected memory or disk", file=sys.stderr)
        return 2

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

    try:
        stats = _generate_outputs(transcript, args)
    except (OSError, ValueError, RuntimeError) as exc:
        stats = {"success": False, "error": str(exc)}

    if args.json:
        json.dump(stats, sys.stdout, indent=2, sort_keys=True)
        _ = sys.stdout.write("\n")
    elif stats.get("success") is True:
        _print_csfsgenerate_summary(stats)
    else:
        error = stats.get("error", "unknown error")
        print(f"CSF generation failed: {error}", file=sys.stderr)

    return 0 if stats.get("success") is True else 1


def _run_restore_csfs(args: RestoreCsfsArgs) -> int:
    try:
        stats = restore_csfs_from_descriptors(
            args.descriptors,
            args.header,
            args.output,
            indices=args.indices,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        if args.json:
            json.dump(
                {"success": False, "error": str(exc)},
                sys.stdout,
                indent=2,
                sort_keys=True,
            )
            _ = sys.stdout.write("\n")
        else:
            print(f"CSF restoration failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        json.dump(stats, sys.stdout, indent=2, sort_keys=True)
        _ = sys.stdout.write("\n")
    else:
        _print_restore_csfs_summary(stats)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = _parse_args(parser, argv)

    if args.command == "gen-descriptors":
        try:
            _validate_gen_descriptors_sidecar_path(
                args.input_parquet, args.output_parquet, args.header
            )
            peel_subshells = read_peel_subshells(args.header)
            stats = generate_descriptors_from_parquet(
                args.input_parquet,
                args.output_parquet,
                peel_subshells=peel_subshells,
                num_workers=args.num_workers,
                normalize=args.normalize,
                descriptor_version=args.descriptor_version,
                header_path=args.header,
                compression=args.compression,
            )
        except (OSError, ValueError) as exc:
            if args.json:
                json.dump(
                    {"success": False, "error": str(exc)},
                    sys.stdout,
                    indent=2,
                    sort_keys=True,
                )
                _ = sys.stdout.write("\n")
            else:
                print(f"Descriptor generation failed: {exc}", file=sys.stderr)
            return 1
        if stats.get("success") is True:
            _write_gen_descriptors_sidecar(
                args.output_parquet, stats, peel_subshells, normalize=args.normalize
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

    if args.command == "interacting":
        return _run_interacting(args)

    if args.command == "restore-csfs":
        return _run_restore_csfs(args)

    return _run_zero_first(args)


if __name__ == "__main__":
    raise SystemExit(main())
