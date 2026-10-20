"""Check release archives and exercise the compatible wheel without installing it.

Dependencies: packaging plus the project's runtime dependencies for --smoke.
Run with the shared Tools interpreter in the workspace.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
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
    "rcsfs/cli.py",
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
    assert str(metadata["Requires-Python"]).replace(" ", "") == ">=3.14"
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
            assert all(
                PurePosixPath(name).parts[0] in WHEEL_ROOTS
                or PurePosixPath(name).parts[0].endswith(".dist-info")
                for name in names
            ), f"Unexpected wheel contents: {names}"
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
                assert f"{dist_info}/entry_points.txt" in names
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
    (entry,) = [
        entry
        for entry in metadata.entry_points
        if entry.group == "console_scripts" and entry.name == "rcsfs"
    ]
    assert entry.value == "rcsfs.cli:main"
    try:
        entry.load()(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    print(
        f"Wheel extension, Parquet, V2 descriptors and CLI passed: {metadata.version}"
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
