"""Promote a successful PyPI dry run to a GitHub Release draft; never publish."""

from __future__ import annotations

import argparse
import ast
import base64
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import tomllib
from typing import Any
from urllib.parse import quote
import zipfile

PACKAGE = "rcsfs"


def gh(*args: str) -> bytes:
    return subprocess.run(
        ["gh", *args], check=True, stdout=subprocess.PIPE
    ).stdout


def api(repository: str, path: str) -> Any:
    return json.loads(gh("api", f"repos/{repository}/{path}"))


def pages(repository: str, path: str) -> list[Any]:
    separator = "&" if "?" in path else "?"
    return json.loads(gh(
        "api", "--paginate", "--slurp",
        f"repos/{repository}/{path}{separator}per_page=100",
    ))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_run(
    run: dict[str, Any], jobs: list[dict[str, Any]], repository: str
) -> str:
    require(run["repository"]["full_name"].lower() == repository.lower(), "Wrong repository.")
    require(run["head_repository"]["full_name"].lower() == repository.lower(), "Fork runs are not accepted.")
    require(run["path"] == ".github/workflows/publish.yml", "Use a publish.yml run.")
    require(run["event"] == "workflow_dispatch" and run["head_branch"] == "main", "Use a manual main run.")
    require(run["status"] == "completed" and run["conclusion"] == "success", "The source run must have succeeded.")
    for name, conclusion in (("preflight", "success"), ("prepare", "success"), ("publish", "skipped")):
        matches = [job for job in jobs if job["name"] == name]
        require(len(matches) == 1 and matches[0]["conclusion"] == conclusion, f"Expected {name}: {conclusion}; use a successful dry run.")
    builds = [job for job in jobs if job["name"].startswith("build /")]
    require(len(builds) == 3 and all(job["conclusion"] == "success" for job in builds), "All three platform builds must succeed.")
    sha = run["head_sha"]
    require(re.fullmatch(r"[0-9a-f]{40}", sha) is not None, "Invalid source commit.")
    return sha


def source_version(repository: str, sha: str) -> str:
    filename = "src/graspkit/version.py" if PACKAGE == "graspkit-atomic" else "Cargo.toml"
    blob = api(repository, f"contents/{filename}?ref={sha}")
    text = base64.b64decode(blob["content"]).decode()
    if filename.endswith(".toml"):
        version = tomllib.loads(text)["package"]["version"]
    else:
        values = [ast.literal_eval(node.value) for node in ast.parse(text).body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)]
        require(len(values) == 1, "Missing source version.")
        version = values[0]
    require(isinstance(version, str) and re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version) is not None, "Use a stable three-part version.")
    return version


def validate_tag(repository: str, tag: str, sha: str) -> None:
    refs = api(repository, f"git/matching-refs/tags/{quote(tag, safe='')}")
    refs = [ref for ref in refs if ref["ref"] == f"refs/tags/{tag}"]
    if not refs:
        return
    target = refs[0]["object"]
    for _ in range(10):
        if target["type"] != "tag":
            break
        target = api(repository, f"git/tags/{target['sha']}")["object"]
    require(target["type"] == "commit" and target["sha"] == sha, "Existing tag points to a different commit.")


def unpack_artifact(data: bytes, directory: Path) -> list[Path]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate artifact filenames.")
        require(all(re.fullmatch(r"[A-Za-z0-9_.+-]+\.(whl|tar\.gz)", name) for name in names), "Unexpected artifact path or file.")
        for name in names:
            (directory / name).write_bytes(archive.read(name))
    return sorted(directory.iterdir())


def validate_files(files: list[Path], version: str) -> None:
    prefix = f"{PACKAGE.replace('-', '_')}-{version}"
    sources = [path for path in files if path.name == f"{prefix}.tar.gz"]
    wheels = [path for path in files if path.suffix == ".whl"]
    count = 1 if PACKAGE == "graspkit-atomic" else 3
    require(len(sources) == 1 and len(wheels) == count and len(files) == count + 1, "Incomplete distribution set or mixed versions.")
    platforms = set()
    for path in files:
        if path.suffix == ".whl":
            match = re.fullmatch(re.escape(prefix) + r"-(py3-none-any|cp314-cp314-(manylinux_2_28_x86_64|win_amd64|macosx_\d+_\d+_arm64))\.whl", path.name)
            require(match is not None, f"Unexpected wheel name or ABI: {path.name}")
            platforms.add(match[2] or "any")
            with zipfile.ZipFile(path) as wheel:
                metadata = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
                require(len(metadata) == 1, "Expected one wheel metadata file.")
                raw = wheel.read(metadata[0])
        else:
            with tarfile.open(path, "r:gz") as archive:
                metadata = [member for member in archive.getmembers() if member.isfile() and len(member.name.split('/')) == 2 and member.name.endswith("/PKG-INFO")]
                require(len(metadata) == 1, "Expected one source metadata file.")
                raw = archive.extractfile(metadata[0]).read()
        info = BytesParser().parsebytes(raw)
        require(re.sub(r"[-_.]+", "-", info["Name"] or "").lower() == PACKAGE and info["Version"] == version, "Package metadata does not match source/tag.")
    if PACKAGE == "graspkit-atomic":
        require(platforms == {"any"}, "Expected a universal wheel.")
    else:
        require(len(platforms) == 3 and "manylinux_2_28_x86_64" in platforms and "win_amd64" in platforms and any(p.startswith("macosx_") for p in platforms), "Missing native platform wheel.")


def create_draft(repository: str, run_id: str, tag: str) -> str:
    require(re.fullmatch(r"[1-9]\d*", run_id) is not None, "Run ID must be a positive integer.")
    require(re.fullmatch(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", tag) is not None, "Use a tag such as v1.2.3.")
    run = api(repository, f"actions/runs/{run_id}")
    jobs = [job for page in pages(repository, f"actions/runs/{run_id}/jobs?filter=latest") for job in page["jobs"]]
    sha = validate_run(run, jobs, repository)
    version = source_version(repository, sha)
    require(tag == f"v{version}", "Tag must match the source version of the selected run.")
    releases = [release for page in pages(repository, "releases") for release in page]
    require(not any(release["tag_name"] == tag for release in releases), "A release already exists for this tag; nothing was changed.")
    validate_tag(repository, tag, sha)
    artifacts = [artifact for page in pages(repository, f"actions/runs/{run_id}/artifacts") for artifact in page["artifacts"] if artifact["name"] == "pypi-distributions"]
    require(len(artifacts) == 1 and not artifacts[0]["expired"], "Expected one unexpired pypi-distributions artifact.")
    artifact = artifacts[0]
    require(artifact["workflow_run"]["head_sha"] == sha, "Artifact source commit mismatch.")
    data = gh("api", f"repos/{repository}/actions/artifacts/{artifact['id']}/zip")
    require(artifact.get("digest") == f"sha256:{hashlib.sha256(data).hexdigest()}", "Missing or mismatched artifact SHA-256 digest.")
    with tempfile.TemporaryDirectory(prefix="github-release-") as temporary:
        directory = Path(temporary)
        files = unpack_artifact(data, directory)
        validate_files(files, version)
        lines = []
        for path in files:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            lines.append(f"{digest}  {path.name}\n")
        hashes = "".join(lines)
        checksum = directory / "SHA256SUMS"
        checksum.write_text(hashes)
        notes = directory / "release-notes.md"
        notes.write_text(f"{PACKAGE} {version}\n\nSource commit: `{sha}`\n\nValidated dry run: https://github.com/{repository}/actions/runs/{run_id}\n\nArtifact ID: `{artifact['id']}`\n\nThese files are reused without rebuilding. This workflow does not upload to PyPI.\n\n```text\n{hashes}```\n")
        url = gh("release", "create", tag, *map(str, files), str(checksum), "--repo", repository, "--target", sha, "--draft", "--title", f"{PACKAGE} {version}", "--notes-file", str(notes)).decode().strip()
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    require(os.environ.get("GITHUB_REF") == "refs/heads/main", "Run this workflow from main.")
    url = create_draft(os.environ["GITHUB_REPOSITORY"], args.run_id, args.tag)
    print(f"Created draft: {url}")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as stream:
            stream.write(f"Created GitHub Release **draft**: {url}\n\nPyPI upload and formal Release publication were not performed.\n")


if __name__ == "__main__":
    main()
