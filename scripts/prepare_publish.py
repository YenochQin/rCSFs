"""Assemble one version from the complete, successful build matrix; never upload."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path

from check_release import NAME, check_archive
from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import Version

ARTIFACTS = {
    "rcsfs-x86_64-unknown-linux-gnu": r"manylinux_2_28_x86_64",
    "rcsfs-x86_64-pc-windows-msvc": r"win_amd64",
    "rcsfs-aarch64-apple-darwin": r"macosx_\d+_\d+_arm64",
}
CANONICAL_ARTIFACT = "rcsfs-x86_64-unknown-linux-gnu"


def prepare(artifacts: Path, outdir: Path, version: str) -> list[Path]:
    expected = Version(version)
    if expected.is_prerelease or expected.is_devrelease or str(expected) != version:
        raise ValueError("Use an exact, stable package version.")
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError("Output directory must be empty.")
    if {path.name for path in artifacts.iterdir()} != set(ARTIFACTS):
        raise ValueError("Expected the complete build artifact matrix.")

    selected: list[Path] = []
    for artifact, platform in ARTIFACTS.items():
        directory = artifacts / artifact
        wheels = list(directory.glob("*.whl"))
        sources = list(directory.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sources) != 1 or len(list(directory.iterdir())) != 2:
            raise ValueError(
                f"Expected one wheel and one source archive in {artifact}."
            )
        wheel, source = wheels[0], sources[0]
        for path in (wheel, source):
            package, parsed_version = (
                parse_wheel_filename(path.name)[:2]
                if path.suffix == ".whl"
                else parse_sdist_filename(path.name)
            )
            if package != NAME or parsed_version != expected:
                raise ValueError(f"Unexpected package or version: {path.name}")
            metadata_version, _ = check_archive(path)
            if metadata_version != version:
                raise ValueError(
                    f"Metadata version differs from requested version: {path.name}"
                )
        tags = parse_wheel_filename(wheel.name)[3]
        if platform == "any":
            valid = all(str(tag) == "py3-none-any" for tag in tags)
        else:
            valid = all(
                tag.interpreter == "cp315"
                and tag.abi == "cp315"
                and re.fullmatch(platform, tag.platform)
                for tag in tags
            )
        if not valid:
            raise ValueError(f"Unexpected platform or Python ABI: {wheel.name}")
        if artifact == CANONICAL_ARTIFACT:
            selected.append(source)
        if platform != "any" or artifact == CANONICAL_ARTIFACT:
            selected.append(wheel)

    if len({path.name for path in selected}) != len(selected):
        raise ValueError("Selected release files have duplicate names.")
    outdir.mkdir(parents=True, exist_ok=True)
    for path in selected:
        shutil.copy2(path, outdir / path.name)
    return sorted(outdir.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    paths = prepare(args.artifacts, args.outdir, args.version)
    lines = [
        f"Prepared {NAME} {args.version}",
        "",
        f"Source commit: {os.environ.get('GITHUB_SHA', 'local')}",
        "",
        "| File | SHA-256 |",
        "| --- | --- |",
    ]
    for path in paths:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        lines.append(f"| {path.name} | `{digest}` |")
    summary = "\n".join(lines) + "\n"
    print(summary)
    if location := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(location).open("a", encoding="utf-8") as stream:
            stream.write(summary)


if __name__ == "__main__":
    main()
