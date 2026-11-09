"""Check release archives and exercise the compatible wheel without installing it.

Dependencies: packaging plus the project's runtime dependencies for --smoke.
Run with this repository's Python 3.15 interpreter.
"""

from __future__ import annotations

import argparse
import base64
import csv
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version

NAME = "rcsfs"
LICENSE = "MIT"
WHEEL_ROOTS = {"rcsfs", "rcsfs.libs"}
WHEEL_REQUIRED = {
    "rcsfs/__init__.py",
    "rcsfs/_rcsfs.pyi",
    "rcsfs/py.typed",
}
SDIST_PATTERNS = {
    "src/*.rs",
    "rcsfs/*.py",
    "rcsfs/*.pyi",
    "rcsfs/py.typed",
    "Cargo.toml",
    "Cargo.toml.orig",
    "Cargo.lock",
    "build.rs",
    "build_backend.py",
    "pyproject.toml",
    "PKG-INFO",
    "LICENSE",
    "licenses/*.txt",
    "README.md",
    "README.zh-CN.md",
    "docs/releasing.md",
    "scripts/check_release.py",
    "tests/fixtures/sample.csf",
}
SDIST_REQUIRED = {
    "pyproject.toml",
    "Cargo.toml",
    "Cargo.lock",
    "build.rs",
    "build_backend.py",
    "src/lib.rs",
    "rcsfs/__init__.py",
    "rcsfs/py.typed",
    "LICENSE",
    "licenses/GRASP-MIT.txt",
    "licenses/RUST-DEPENDENCIES.txt",
    "tests/fixtures/sample.csf",
    "scripts/check_release.py",
}


def check_metadata(raw: bytes) -> str:
    metadata = BytesParser(policy=policy.default).parsebytes(raw)
    assert metadata["Name"] == NAME, metadata["Name"]
    version = Version(str(metadata["Version"]))
    assert not version.is_prerelease and not version.is_devrelease, version
    assert str(metadata["Requires-Python"]).replace(" ", "") == ">=3.15"
    assert metadata["License-Expression"] == LICENSE
    assert metadata["Author-email"] and metadata["Description-Content-Type"]
    assert len(str(metadata.get_payload()).strip()) > 100, "Missing project description"
    assert "Development Status :: 5 - Production/Stable" in metadata.get_all(
        "Classifier", []
    )
    assert metadata.get_all("License-File"), "Missing license files"
    for requirement in metadata.get_all("Requires-Dist", []):
        assert " @ " not in requirement, f"Non-index dependency: {requirement}"
    return str(version)


def check_archive(path: Path) -> tuple[str, bool]:
    if path.suffix == ".whl":
        _, filename_version, _, tags = parse_wheel_filename(path.name)
        assert all(not tag.platform.startswith("linux_") for tag in tags), (
            "Plain linux wheels are not PyPI release artifacts; build manylinux/musllinux."
        )
        with zipfile.ZipFile(path) as archive:
            names = {name for name in archive.namelist() if not name.endswith("/")}
            assert WHEEL_REQUIRED <= names, WHEEL_REQUIRED - names
            command_name = (
                "rcsfs.exe"
                if all(tag.platform.startswith("win") for tag in tags)
                else "rcsfs"
            )
            native_path = f"{NAME}-{filename_version}.data/scripts/{command_name}"
            assert native_path in names, "Missing native CLI in wheel scripts"
            assert all(
                PurePosixPath(name).parts[0] in WHEEL_ROOTS
                or PurePosixPath(name).parts[0].endswith(".dist-info")
                or name == native_path
                for name in names
            ), f"Unexpected wheel contents: {names}"
            assert not any(
                name in names
                for name in [
                    "rcsfs/cli.py",
                    "rcsfs/_cli_config.py",
                    "rcsfs/_cli_log.py",
                    "rcsfs/_csfg_cli.py",
                ]
            ), "Retired Python CLI modules are packaged"
            assert archive.read(native_path).startswith(
                (
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
            ), "Native CLI was replaced with a launcher"
            assert archive.getinfo(native_path).external_attr >> 16 & 0o111, (
                "CLI is not executable"
            )
            assert not any(
                "__pycache__" in name or name.endswith((".pyc", ".pdb"))
                for name in names
            )
            (metadata_name,) = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            version = check_metadata(archive.read(metadata_name))
            assert Version(version) == filename_version, (
                "Filename/metadata version mismatch"
            )
            metadata = BytesParser(policy=policy.default).parsebytes(
                archive.read(metadata_name)
            )
            dist_info = metadata_name.rsplit("/", 1)[0]
            for license_file in metadata.get_all("License-File", []):
                assert f"{dist_info}/licenses/{license_file}" in names, license_file
            for required in (
                "LICENSE",
                "licenses/GRASP-MIT.txt",
                "licenses/RUST-DEPENDENCIES.txt",
            ):
                assert required in metadata.get_all("License-File", []), required
            expected_lock = hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "Cargo.lock").read_bytes()
            ).hexdigest()
            notices = archive.read(
                f"{dist_info}/licenses/licenses/RUST-DEPENDENCIES.txt"
            )
            assert f"Cargo.lock SHA-256: {expected_lock}".encode() in notices, (
                "Stale dependency notices"
            )
            if NAME == "rcsfs":
                binaries = [
                    name
                    for name in names
                    if name.startswith("rcsfs/_rcsfs")
                    and name.endswith((".so", ".pyd"))
                ]
                assert len(binaries) == 1, (
                    f"Expected only the built native extension: {binaries}"
                )
                expected_suffix = (
                    ".pyd"
                    if all(tag.platform.startswith("win") for tag in tags)
                    else ".so"
                )
                assert binaries[0].endswith(expected_suffix), binaries
                entry_points = f"{dist_info}/entry_points.txt"
                assert entry_points not in names or b"rcsfs" not in archive.read(
                    entry_points
                )
                record = {
                    row[0]: row[1:]
                    for row in csv.reader(
                        archive.read(f"{dist_info}/RECORD").decode().splitlines()
                    )
                }
                for name in names - {f"{dist_info}/RECORD"}:
                    data = archive.read(name)
                    digest = (
                        base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                        .rstrip(b"=")
                        .decode()
                    )
                    assert record[name] == [f"sha256={digest}", str(len(data))], (
                        f"Bad RECORD entry: {name}"
                    )
        return version, bool(tags & set(sys_tags()))

    with tarfile.open(path, "r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        roots = {PurePosixPath(member.name).parts[0] for member in files}
        assert len(roots) == 1, roots
        names = {
            str(PurePosixPath(member.name).relative_to(next(iter(roots))))
            for member in files
        }
        unexpected = {
            name
            for name in names
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in SDIST_PATTERNS)
        }
        assert not unexpected, (
            f"Unexpected source archive contents: {sorted(unexpected)}"
        )
        assert SDIST_REQUIRED <= names, SDIST_REQUIRED - names
        assert not any(
            "__pycache__" in name or name.endswith((".pyc", ".so", ".pyd", ".pdb"))
            for name in names
        )
        (metadata_member,) = [
            member for member in files if member.name == f"{next(iter(roots))}/PKG-INFO"
        ]
        stream = archive.extractfile(metadata_member)
        assert stream is not None
        version = check_metadata(stream.read())
    return version, False


def smoke(unpacked: Path) -> None:
    sys.path.insert(0, str(unpacked))
    from importlib.metadata import distribution

    import rcsfs
    from rcsfs import _rcsfs

    assert Path(rcsfs.__file__).is_relative_to(unpacked)
    assert Path(_rcsfs.__file__).is_relative_to(unpacked)
    metadata = distribution(NAME)
    assert rcsfs.__version__ == _rcsfs.__version__ == metadata.version
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/sample.csf"
    stats = rcsfs.convert_csfs(
        fixture, unpacked / "sample.parquet", chunk_size=90, num_workers=2
    )
    assert stats["success"] and stats["csf_count"] > 0
    peel = rcsfs.read_peel_subshells(stats["header_file"])
    descriptor = rcsfs.generate_descriptors_from_parquet(
        unpacked / "sample.parquet",
        unpacked / "descriptor.parquet",
        peel,
        num_workers=2,
        header_path=stats["header_file"],
    )
    assert descriptor["descriptor_version"] == 2
    assert descriptor["descriptor_count"] == stats["csf_count"]
    assert not any(
        entry.group == "console_scripts" and entry.name == "rcsfs"
        for entry in metadata.entry_points
    )
    binary_name = "rcsfs.exe" if sys.platform == "win32" else "rcsfs"
    binary = unpacked / f"{NAME}-{metadata.version}.data" / "scripts" / binary_name
    binary.chmod(binary.stat().st_mode | 0o111)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    env["PATH"] = ""
    version_result = subprocess.run(
        [str(binary), "--version"], env=env, capture_output=True, text=True, check=True
    )
    assert (
        metadata.version in version_result.stdout
        and "native Rust" in version_result.stdout
    )
    result = subprocess.run(
        [str(binary), "info", str(unpacked / "descriptor.parquet"), "--json"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    assert info["num_rows"] == stats["csf_count"]
    assert info["key_value_metadata"]["descriptor_dtype"] == "int8"
    assert rcsfs.seniority_to_printed_v(7, 4, 12, 2) == -1
    print(
        f"Wheel library, Int8 descriptors and Python-free CLI passed: {metadata.version}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--unpacked", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.unpacked:
        smoke(args.directory.resolve())
        return
    archives = sorted(args.directory.glob("*.whl")) + sorted(
        args.directory.glob("*.tar.gz")
    )
    assert archives, f"No release archives in {args.directory}"
    versions: set[str] = set()
    compatible: list[Path] = []
    for path in archives:
        version, can_import = check_archive(path)
        versions.add(version)
        if can_import:
            compatible.append(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        print(f"{digest}  {path.name}")
    assert len(versions) == 1, f"Mixed release versions: {versions}"
    if args.smoke:
        assert compatible, "No wheel compatible with this interpreter/platform"
        for wheel in compatible:
            with tempfile.TemporaryDirectory(prefix="release-smoke-") as temporary:
                unpacked = Path(temporary)
                with zipfile.ZipFile(wheel) as archive:
                    archive.extractall(unpacked)
                subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        str(Path(__file__).resolve()),
                        str(unpacked),
                        "--unpacked",
                    ],
                    cwd=unpacked,
                    check=True,
                )
    print(f"Validated {len(archives)} archive(s) for {NAME} {next(iter(versions))}.")


if __name__ == "__main__":
    main()
