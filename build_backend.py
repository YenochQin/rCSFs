"""PEP 517/660 backend for one wheel containing the library and native CLI.

Maturin audits the Python extension and Python-free executable separately.
Only a matching-platform, real executable is added to the library wheel;
Maturin's Python launcher for repaired bin wheels is deliberately rejected.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import tempfile
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

Settings = Mapping[str, Any] | None
_NATIVE_MAGIC = (
    b"\x7fELF",
    b"MZ",
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
)


def _arguments(config_settings: Settings) -> list[str]:
    import maturin

    args = list(maturin.get_maturin_pep517_args(config_settings))
    # Maturin's PEP 517 wrapper otherwise defaults to unaudited linux/off.
    config = tomllib.loads(Path("pyproject.toml").read_text())
    tool = config.get("tool", {}).get("maturin", {})
    normalized: list[str] = []
    for arg in args:
        if arg.startswith(("--compatibility=", "--manylinux=", "--auditwheel=")):
            normalized.extend(arg.split("=", 1))
        else:
            normalized.append(arg)
    args = normalized
    if "--compatibility" not in args and "--manylinux" not in args:
        args.extend(["--compatibility", tool.get("compatibility", "pypi")])
    for index, arg in enumerate(args):
        if arg in {"--compatibility", "--manylinux"} and (
            index + 1 == len(args) or args[index + 1] in {"off", "linux"}
        ):
            raise ValueError(
                "Mixed wheels require audited pypi/manylinux/musllinux compatibility"
            )
        if arg == "--skip-auditwheel":
            raise ValueError(
                "Use --auditwheel check or repair; the native CLI is always audited"
            )
        if arg == "--auditwheel" and (
            index + 1 == len(args) or args[index + 1] not in {"check", "repair"}
        ):
            raise ValueError("Wheel auditing cannot be skipped")
    if "--locked" not in args and "--frozen" not in args:
        args.append("--locked")
    return args


def _native_arguments(args: Sequence[str]) -> list[str]:
    result: list[str] = []
    skip = False
    value_options = {"--features", "-F", "--bindings", "-b", "--auditwheel"}
    for arg in args:
        if skip:
            skip = False
            continue
        if arg in value_options:
            skip = True
        elif arg in {"--all-features", "--no-default-features"} or arg.startswith(
            ("--features=", "--bindings=", "--auditwheel=", "-F")
        ):
            continue
        else:
            result.append(arg)
    if skip:
        raise ValueError("Missing value in native build options")
    return [
        *result,
        "--bindings",
        "bin",
        "--no-default-features",
        "--features",
        "cli",
        "--auditwheel",
        "check",
    ]


def _wheel_parts(path: Path) -> tuple[str, str]:
    if path.suffix != ".whl":
        raise ValueError(f"Not a wheel: {path.name}")
    prefix, _, _, platform = path.stem.rsplit("-", 3)
    return prefix, platform


def merge_wheels(library: Path, native: Path, output_directory: Path) -> Path:
    """Merge audited wheels without changing the library's Python ABI tag."""
    prefix, platform = _wheel_parts(library)
    native_prefix, native_platform = _wheel_parts(native)
    if prefix != native_prefix or set(platform.split(".")) != set(
        native_platform.split(".")
    ):
        raise ValueError(
            "Library and native CLI must have the same package/version and platform tags; build both with the same explicit compatibility policy"
        )
    executable = "rcsfs.exe" if platform.startswith("win") else "rcsfs"
    destination = f"{prefix}.data/scripts/{executable}"
    with zipfile.ZipFile(native) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.endswith(f".data/scripts/{executable}")
        ]
        if len(matches) != 1:
            raise ValueError("Native wheel must contain exactly one rcsfs executable")
        binary = archive.read(matches[0])
        if not binary.startswith(_NATIVE_MAGIC):
            raise ValueError(
                "The rcsfs command must be a native executable, not a Python/shell launcher"
            )
        info = archive.getinfo(matches[0])
        binary_info = zipfile.ZipInfo(destination, date_time=info.date_time)
        binary_info.create_system = 3
        binary_info.external_attr = 0o100755 << 16
        binary_info.compress_type = zipfile.ZIP_DEFLATED

    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / library.name
    with zipfile.ZipFile(library) as source:
        names = source.namelist()
        if destination in names:
            raise ValueError("Library wheel already contains an rcsfs executable")
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(record_names) != 1:
            raise ValueError("Library wheel must contain exactly one RECORD")
        record_name = record_names[0]
        for name in names:
            if name.endswith(".dist-info/entry_points.txt") and b"rcsfs" in source.read(
                name
            ):
                raise ValueError(
                    "Remove the Python rcsfs console entry point before bundling the binary"
                )
        with tempfile.NamedTemporaryFile(
            dir=output_directory, suffix=".whl", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            rows: list[tuple[str, str, str]] = []
            with zipfile.ZipFile(
                temporary_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as target:
                for item in source.infolist():
                    if item.filename == record_name or item.is_dir():
                        continue
                    payload = source.read(item.filename)
                    target.writestr(item, payload)
                    rows.append(_record(item.filename, payload))
                target.writestr(binary_info, binary)
                rows.append(_record(destination, binary))
                buffer = io.StringIO(newline="")
                writer = csv.writer(buffer, lineterminator="\n")
                writer.writerows([*rows, (record_name, "", "")])
                target.writestr(record_name, buffer.getvalue())
            # Close the source first: Windows cannot replace an open archive.
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary_path, output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output


def _record(name: str, payload: bytes) -> tuple[str, str, str]:
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
    )
    return name, f"sha256={digest}", str(len(payload))


def _build(
    wheel_directory: str,
    config_settings: Settings,
    metadata_directory: str | None,
    *,
    editable: bool,
) -> str:
    import maturin

    output = Path(wheel_directory).resolve()
    args = _arguments(config_settings)
    with tempfile.TemporaryDirectory(prefix="rcsfs-wheel-") as temporary:
        root = Path(temporary)
        native_dir, library_dir = root / "native", root / "library"
        native_dir.mkdir()
        library_dir.mkdir()
        # Build native first so the final local library artifact remains PyO3.
        native_name = maturin.build_wheel(
            str(native_dir), {"maturin.build-args": _native_arguments(args)}
        )
        build = maturin.build_editable if editable else maturin.build_wheel
        library_name = build(
            str(library_dir), {"maturin.build-args": args}, metadata_directory
        )
        return merge_wheels(
            library_dir / library_name, native_dir / native_name, output
        ).name


def build_wheel(
    wheel_directory: str,
    config_settings: Settings = None,
    metadata_directory: str | None = None,
) -> str:
    return _build(wheel_directory, config_settings, metadata_directory, editable=False)


def build_editable(
    wheel_directory: str,
    config_settings: Settings = None,
    metadata_directory: str | None = None,
) -> str:
    return _build(wheel_directory, config_settings, metadata_directory, editable=True)


def build_sdist(sdist_directory: str, config_settings: Settings = None) -> str:
    import maturin

    return maturin.build_sdist(sdist_directory, config_settings)


def get_requires_for_build_wheel(config_settings: Settings = None) -> list[str]:
    import maturin

    return maturin.get_requires_for_build_wheel(config_settings)


get_requires_for_build_editable = get_requires_for_build_wheel
get_requires_for_build_sdist = get_requires_for_build_wheel


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: Settings = None
) -> str:
    import maturin

    return maturin.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


prepare_metadata_for_build_editable = prepare_metadata_for_build_wheel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine the audited extension and CLI wheels"
    )
    parser.add_argument("library", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    print(merge_wheels(options.library, options.native, options.out))
