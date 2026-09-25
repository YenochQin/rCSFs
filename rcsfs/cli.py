"""Command line interface for rcsfs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, TextIO, cast

from . import (
    convert_csfs,
    estimate_disk_generation,
    generate_csfs_from_transcript,
    generate_descriptors_from_parquet,
    generate_disk_outputs_from_transcript,
    partition_csfs,
    read_peel_subshells,
    restore_csfs_from_descriptors,
    select_interacting_csfs,
    split_csfs_by_active_spaces,
)
from ._types import InteractionHamiltonian, InteractionMethod
from ._publication import PartialPublicationError, publish_outputs
from ._cli_config import parse_cli_args

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


class CsfsSplitArgs(Protocol):
    """Parsed arguments for the active-space CSF split command."""

    command: Literal["csfs-split", "split-active", "rcsfsplit"]
    split_csfs_parquet: Path
    csfs_header: Path
    active_spaces: list[str]
    output_dir: Path
    prefix: str | None
    json: bool


class CsfsGenerateArgs(Protocol):
    """Parsed arguments for the ``csfsgenerate`` subcommand."""

    command: Literal["csfsgenerate"]
    rcsfs_out: Path
    config: Path | None
    generation: Mapping[str, object] | None
    generate_descriptors: bool
    rcsfs_parquet: Path | None
    descriptor: Path | None

    normalize: bool
    threads: int | None
    memory_budget_mib: int | None
    estimate_only: bool
    allow_unchecked_space: bool
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
    | CsfsSplitArgs
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

    def add_config_argument(command_parser: argparse.ArgumentParser) -> None:
        _ = command_parser.add_argument(
            "-c",
            "--config",
            type=Path,
            help="CLI TOML file (default: ./rcsfs.toml when present).",
        )

    gen_descriptors = subparsers.add_parser(
        "gen-descriptors",
        help="Generate descriptor Parquet data from a CSF Parquet file.",
    )
    add_config_argument(gen_descriptors)
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
    add_config_argument(zero_first)
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

    split_active = subparsers.add_parser(
        "csfs-split",
        aliases=["split-active", "rcsfsplit"],
        help="Split one CSF Parquet file into GRASP-style active-space CSF lists.",
        description=(
            "Read CSF Parquet once and independently select each active space. "
            "Outputs may overlap; no output is overwritten."
        ),
    )
    add_config_argument(split_active)
    _ = split_active.add_argument("split_csfs_parquet", type=Path)
    _ = split_active.add_argument(
        "--header",
        dest="csfs_header",
        required=True,
        type=Path,
        help="Matching CSF *_header.toml file.",
    )
    _ = split_active.add_argument(
        "--space",
        "--active-space",
        dest="active_spaces",
        required=True,
        action="append",
        metavar="LABEL=5s,4p,3d",
        help="Output label and maximum orbitals; repeat for each active space.",
    )
    _ = split_active.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Existing directory for output CSF text files.",
    )
    _ = split_active.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix (default: input Parquet stem).",
    )
    _ = split_active.add_argument("--json", action="store_true")

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
    add_config_argument(csfsgenerate)
    _ = csfsgenerate.add_argument(
        "rcsfs_out",
        nargs="?",
        type=Path,
        default=Path("rcsf.out"),
        help="Destination CSF text file (default: rcsf.out; must not already exist).",
    )
    _ = csfsgenerate.add_argument(
        "--generate-descriptors",
        action="store_true",
        help="Also write CSF Parquet and descriptor Parquet outputs.",
    )
    _ = csfsgenerate.add_argument(
        "--parquet",
        "--rcsfs-parquet",
        dest="rcsfs_parquet",
        type=Path,
        default=None,
        help="CSF Parquet output (default: same stem as the CSF file).",
    )
    _ = csfsgenerate.add_argument(
        "--descriptor-parquet",
        "--descriptor",
        dest="descriptor",
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
        type=_parse_positive_int,
        default=None,
        help="Rayon thread count for generation (default: all cores).",
    )
    _ = csfsgenerate.add_argument(
        "--memory-budget-mib",
        type=_parse_positive_int,
        default=None,
        help="Managed generation-memory budget in MiB (disk storage only).",
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
        "--allow-unchecked-space",
        action="store_true",
        help=(
            "Proceed when a volume's free space cannot be measured on this platform. "
            "By default the run is refused instead of skipping the pre-flight."
        ),
    )
    _ = csfsgenerate.add_argument(
        "--estimate-only",
        action="store_true",
        help=(
            "Count the workload and report the capacity estimate without generating "
            "anything (requires --generate-descriptors)."
        ),
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
    add_config_argument(interacting)
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
    add_config_argument(restore_csfs)
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


def _run_csfs_split(args: CsfsSplitArgs) -> int:
    try:
        if not args.output_dir.is_dir():
            raise ValueError(f"output directory does not exist: {args.output_dir}")
        prefix = (
            args.prefix if args.prefix is not None else args.split_csfs_parquet.stem
        )
        if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
            raise ValueError("output prefix must contain only letters, digits, _ or -")
        targets: dict[str | Path, str] = {}
        labels: set[str] = set()
        for item in args.active_spaces:
            label, separator, orbitals = item.partition("=")
            if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", label):
                raise ValueError(f"invalid --space {item!r}; expected LABEL=5s,4p,3d")
            if label in labels:
                raise ValueError(f"duplicate active-space label: {label}")
            labels.add(label)
            targets[args.output_dir / f"{prefix}{label}.c"] = orbitals
        if len(targets) != len(args.active_spaces):
            raise ValueError("active-space labels produce duplicate output paths")
        stats = split_csfs_by_active_spaces(
            args.split_csfs_parquet, args.csfs_header, targets
        )
    except (OSError, ValueError, RuntimeError) as exc:
        if args.json:
            json.dump({"success": False, "error": str(exc)}, sys.stdout, indent=2)
            _ = sys.stdout.write("\n")
        else:
            print(f"Active-space split failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        json.dump(stats, sys.stdout, indent=2)
        _ = sys.stdout.write("\n")
    else:
        for output in stats["outputs"]:
            print(f"{output['output_file']}: {output['csf_count']} CSFs")
    return 0


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
    if stats.get("estimate_only") is True:
        _print_estimate_summary(stats)
        return
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
    # The default is uncompressed, so only a non-default choice is worth a line.
    segment_codec = stats.get("segment_codec")
    if isinstance(segment_codec, str) and segment_codec != "none":
        print(f"segment_codec: {segment_codec}")
    # Say how duplicate_count was obtained: "verified_unique" reports zero by
    # construction, "exact" by comparing every row.
    deduplication = stats.get("deduplication")
    if isinstance(deduplication, str):
        print(f"deduplication: {deduplication}")
    stage_stats = stats.get("stage_stats")
    if isinstance(stage_stats, list):
        for value in cast(list[object], stage_stats):
            if not isinstance(value, Mapping):
                continue
            stage = cast(Mapping[str, object], value)
            name = stage.get("name")
            elapsed = stage.get("elapsed_millis")
            cpu = stage.get("cpu_millis")
            if isinstance(name, str) and isinstance(elapsed, int):
                print(f"stage_{name}_seconds: {elapsed / 1000:.3f}")
                if isinstance(cpu, int):
                    print(f"stage_{name}_cpu_seconds: {cpu / 1000:.3f}")
    resource_stats = stats.get("resource_stats")
    if isinstance(resource_stats, Mapping):
        resource = cast(Mapping[str, object], resource_stats)
        for key in (
            "memory_budget_mib",
            "budget_bytes",
            "peak_managed_bytes",
            "current_managed_bytes",
            "occupation_bytes",
        ):
            value = resource.get(key)
            if value is not None:
                print(f"{key}: {value}")
    plan_stats = stats.get("plan_stats")
    if isinstance(plan_stats, Mapping):
        plan = cast(Mapping[str, object], plan_stats)
        for key in (
            "task_count",
            "target_records_per_task",
            "estimated_total_records",
            "zero_record_configurations",
            "unsplittable_tasks",
            "unsplittable_records",
        ):
            value = plan.get(key)
            if value is not None:
                print(f"plan_{key}: {value}")
        per_task = plan.get("estimated_records_per_task")
        if isinstance(per_task, Mapping):
            distribution = cast(Mapping[str, object], per_task)
            for key in ("minimum", "p50", "p95", "maximum"):
                value = distribution.get(key)
                if value is not None:
                    print(f"plan_estimated_records_per_task_{key}: {value}")


def _require_usable_space(
    estimate: Mapping[str, object], *, allow_unchecked: bool
) -> None:
    """Refuse to start when the reported space checks are not a clearance.

    Insufficient space always stops the run. A volume the platform cannot
    measure stops it too unless the caller accepted an unchecked pre-flight, so
    an unknown value is never read as sufficient by default.
    """
    checks = estimate.get("space_checks")
    if not isinstance(checks, list):
        return
    for value in cast(list[object], checks):
        if not isinstance(value, Mapping):
            continue
        check = cast(Mapping[str, object], value)
        path = check.get("path")
        required = check.get("required_bytes")
        if check.get("sufficient") is False:
            raise ValueError(
                f"Not enough free space at {path}: the run needs {required} bytes "
                f"(including the safety margin) and {check.get('free_bytes')} are available"
            )
        if check.get("sufficient") is None and not allow_unchecked:
            raise ValueError(
                f"Cannot check free space at {path}: this platform does not report it. "
                f"Pass --allow-unchecked-space to run without the pre-flight."
            )


def _print_estimate_summary(
    stats: Mapping[str, object], *, file: TextIO | None = None
) -> None:
    """Print the counted workload and the capacity estimate."""
    stream: TextIO = sys.stdout if file is None else file
    plan_stats = stats.get("plan_stats")
    plan: Mapping[str, object] = (
        cast(Mapping[str, object], plan_stats)
        if isinstance(plan_stats, Mapping)
        else {}
    )
    print(f"unique_occupations: {stats.get('unique_occupations')}", file=stream)
    print(
        f"pre_dedup_records: {stats.get('pre_deduplication_records')}",
        file=stream,
    )
    print(f"peel_subshells: {stats.get('peel_subshells')}", file=stream)
    print(f"v2_columns: {stats.get('v2_columns')}", file=stream)
    print(f"segment_codec: {stats.get('segment_codec', 'none')}", file=stream)
    print(
        f"deduplication: {stats.get('deduplication', 'verified_unique')}",
        file=stream,
    )
    for key in (
        "task_count",
        "target_records_per_task",
        "zero_record_configurations",
        "unsplittable_tasks",
        "unsplittable_records",
    ):
        value = plan.get(key)
        if value is not None:
            print(f"plan_{key}: {value}", file=stream)
    bytes_table = stats.get("bytes")
    if isinstance(bytes_table, Mapping):
        for key, value in cast(Mapping[str, object], bytes_table).items():
            print(f"bytes_{key}: {value}", file=stream)
    checks = stats.get("space_checks")
    if isinstance(checks, list):
        for value in cast(list[object], checks):
            if not isinstance(value, Mapping):
                continue
            check = cast(Mapping[str, object], value)
            print(
                f"space_check {check.get('path')}: required {check.get('required_bytes')} "
                f"free {check.get('free_bytes')} sufficient {check.get('sufficient')}",
                file=stream,
            )
    assumptions = stats.get("assumptions")
    if isinstance(assumptions, list):
        for value in cast(list[object], assumptions):
            print(f"assumption: {value}", file=stream)
    print(f"failure_recovery: {stats.get('failure_recovery')}", file=stream)


def _generate_outputs(transcript: str, args: CsfsGenerateArgs) -> dict[str, object]:
    rcsfs_parquet = args.rcsfs_parquet or args.rcsfs_out.with_suffix(".parquet")
    descriptor_output = args.descriptor or args.rcsfs_out.with_name(
        f"{args.rcsfs_out.stem}_descriptors.parquet"
    )
    csfs_header = rcsfs_parquet.parent / f"{args.rcsfs_out.stem}_header.toml"
    metadata = descriptor_output.with_suffix(".toml")
    destinations = [args.rcsfs_out]
    if args.generate_descriptors:
        destinations.extend([rcsfs_parquet, csfs_header, descriptor_output, metadata])
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
    if args.memory_budget_mib is not None and not args.generate_descriptors:
        raise ValueError(
            "memory_budget_mib requires descriptor-producing disk generation"
        )
    if not args.generate_descriptors:
        return dict(
            generate_csfs_from_transcript(
                transcript,
                args.rcsfs_out,
                normalize=args.normalize,
                threads=args.threads,
            )
        )
    if args.generation_storage == "disk" and args.normalize:
        raise ValueError("normalize is not supported by reversible V2 descriptors")
    if args.memory_budget_mib is not None and args.generation_storage != "disk":
        raise ValueError("memory_budget_mib requires disk generation storage")
    if args.estimate_only and args.generation_storage != "disk":
        raise ValueError(
            "estimate_only covers the disk descriptor path; add --generate-descriptors"
        )
    if args.generation_storage == "disk":
        scratch_base = args.scratch_dir if args.scratch_dir is not None else Path.cwd()
        # The estimate run every check itself, from the same model the
        # generation path uses: requirements that share a volume are added and
        # each phase is compared by its maximum. Only the paths are the CLI's
        # business -- scratch, the staging directory and each destination.
        estimate = dict(
            estimate_disk_generation(
                transcript,
                args.threads,
                memory_budget_mib=args.memory_budget_mib,
                scratch_dir=scratch_base,
                staging_dir=Path.cwd(),
                destinations={
                    "csf_text": args.rcsfs_out,
                    "csf_parquet": rcsfs_parquet,
                    "descriptor": descriptor_output,
                    # The header and the descriptor sidecar can land on different
                    # volumes, so each is charged to its own destination.
                    "header": csfs_header,
                    "descriptor_metadata": metadata,
                },
            )
        )
        if args.estimate_only:
            estimate["estimate_only"] = True
            return estimate
        # The estimate only reports; a run has to decide. This is that decision,
        # made before the staging directory or the scratch directory exist.
        _require_usable_space(estimate, allow_unchecked=args.allow_unchecked_space)
        _print_estimate_summary(estimate, file=sys.stderr)

    # The existing converters truncate their destinations. Run them only in a
    # private staging directory, then publish each complete file under a new
    # final name without replacing another process's file.
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
        csf = csf_dir / args.rcsfs_out.name
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
            # The scratch directory is this operation's own; remove it whether
            # generation succeeded or failed, so a failed run can be retried
            # instead of colliding with its own leftovers.
            try:
                stats = dict(
                    generate_disk_outputs_from_transcript(
                        transcript,
                        csf,
                        parquet,
                        descriptors,
                        staged_header,
                        scratch,
                        threads=args.threads,
                        memory_budget_mib=args.memory_budget_mib,
                        allow_unchecked_space=args.allow_unchecked_space,
                    )
                )
            finally:
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
        publish_outputs(sources, destinations)
        stats.update(
            output_file=str(args.rcsfs_out),
            parquet_file=str(rcsfs_parquet),
            descriptor_parquet_file=str(descriptor_output),
            descriptor_metadata_file=str(metadata),
        )
        return stats


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


def _config_reference_configuration(value: object) -> list[str]:
    if not isinstance(value, list):
        raise TypeError("reference_configuration must be an array of strings")
    return [_config_string(item) for item in cast(list[object], value)]


def _run_csfsgenerate(args: CsfsGenerateArgs) -> int:
    if args.generation is not None:
        try:
            generate = args.generation
            orbital_order = _config_string(generate.get("orbital_order", "*"))
            inactive_core = _config_int(generate["inactive_core"])
            reference_configuration = _config_reference_configuration(
                generate["reference_configuration"]
            )
            active_space = _config_string(generate["active_space"])
            j_min = _config_int(generate["j_min"])
            j_max = _config_int(generate["j_max"])
            excitations = _config_int(generate["excitations"])
            if _config_bool(generate.get("continue_lists", False)):
                raise ValueError("continue_lists is not supported yet; use false")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            print(f"Invalid generation config: {exc}", file=sys.stderr)
            return 2
    else:
        orbital_order = _read_order()
        inactive_core = _read_core()
        reference_configuration = _read_references()
        active_space = _read_active_orbitals()
        j_min, j_max = _read_j_range()
        excitations = _read_excitations()
    if args.generation is None and _read_continue():
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
            f"{orbital_order} ! Orbital order",
            str(inactive_core),
            *reference_configuration,
            "",
            active_space,
            f"{j_min},{j_max}",
            str(excitations),
            "n",
        ]
    )

    try:
        stats = _generate_outputs(transcript, args)
    except PartialPublicationError as exc:
        stats = {
            "success": False,
            "error": str(exc),
            "published_outputs": [str(path) for path in exc.published],
            "failed_destination": str(exc.destination),
        }
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
    args = cast(CliArgs, cast(object, parse_cli_args(parser, argv)))

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

    if args.command == "csfs-split":
        return _run_csfs_split(args)
    if args.command == "split-active":
        return _run_csfs_split(args)
    if args.command == "rcsfsplit":
        return _run_csfs_split(args)

    if args.command == "zero-first":
        return _run_zero_first(args)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
