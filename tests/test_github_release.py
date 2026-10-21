"""Offline tests for promoting verified artifacts to draft Releases."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "github_release", Path(__file__).resolve().parents[1] / "scripts/github_release.py"
)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)

SHA = "a" * 40
REPO = "owner/project"
VERSION = "1.2.3"
TAG = f"v{VERSION}"


def distributions() -> dict[str, bytes]:
    prefix = f"{release.PACKAGE.replace('-', '_')}-{VERSION}"
    metadata = f"Name: {release.PACKAGE}\nVersion: {VERSION}\n".encode()
    tags = (["py3-none-any"] if release.PACKAGE == "graspkit-atomic" else [
        "cp314-cp314-manylinux_2_28_x86_64", "cp314-cp314-win_amd64",
        "cp314-cp314-macosx_11_0_arm64",
    ])
    files = {}
    for tag in tags:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as wheel:
            wheel.writestr(f"{prefix}.dist-info/METADATA", metadata)
        files[f"{prefix}-{tag}.whl"] = stream.getvalue()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as source:
        info = tarfile.TarInfo(f"{prefix}/PKG-INFO")
        info.size = len(metadata)
        source.addfile(info, io.BytesIO(metadata))
    files[f"{prefix}.tar.gz"] = stream.getvalue()
    return files


class GitHubReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run = {
            "repository": {"full_name": REPO}, "head_repository": {"full_name": REPO},
            "path": ".github/workflows/publish.yml", "event": "workflow_dispatch",
            "head_branch": "main", "status": "completed", "conclusion": "success", "head_sha": SHA,
        }
        self.jobs = [
            {"name": name, "conclusion": result} for name, result in [
                ("preflight", "success"), ("prepare", "success"), ("publish", "skipped"),
                ("build / build (linux)", "success"), ("build / build (windows)", "success"),
                ("build / build (macos)", "success"),
            ]
        ]
        self.files = distributions()
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as artifact:
            for name, data in self.files.items():
                artifact.writestr(name, data)
        self.archive = stream.getvalue()
        self.artifact = {
            "id": 99, "name": "pypi-distributions", "expired": False,
            "workflow_run": {"head_sha": SHA},
            "digest": f"sha256:{hashlib.sha256(self.archive).hexdigest()}",
        }
        self.refs = []
        self.releases = []
        self.created = []
        self.bad_source_version = False

    def fake_gh(self, *args: str) -> bytes:
        if args[:2] == ("release", "create"):
            self.created.append(args)
            self.assertIn("--draft", args)
            self.assertEqual(args[args.index("--target") + 1], SHA)
            self.assertEqual(args[2], TAG)
            assets = args[3:args.index("--repo")]
            self.assertEqual({Path(p).name for p in assets}, set(self.files) | {"SHA256SUMS"})
            for path in assets:
                if Path(path).name in self.files:
                    self.assertEqual(Path(path).read_bytes(), self.files[Path(path).name])
            self.assertIn(SHA, Path(args[args.index("--notes-file") + 1]).read_text())
            return b"https://github.com/owner/project/releases/draft"
        endpoint = args[-1].removeprefix(f"repos/{REPO}/")
        if endpoint == "actions/artifacts/99/zip":
            return self.archive
        if endpoint == "actions/runs/123":
            result = self.run
        elif endpoint.startswith("actions/runs/123/jobs?"):
            result = [{"jobs": self.jobs}]
        elif endpoint.startswith("actions/runs/123/artifacts?"):
            result = [{"artifacts": [self.artifact]}]
        elif endpoint.startswith("releases?"):
            result = [self.releases]
        elif endpoint.startswith("git/matching-refs/tags/"):
            result = self.refs
        elif endpoint.startswith("git/tags/"):
            result = {"object": {"type": "commit", "sha": SHA}}
        elif endpoint.startswith("contents/"):
            version = "2.0.0" if self.bad_source_version else VERSION
            source = (f'__version__ = "{version}"' if release.PACKAGE == "graspkit-atomic" else f'[package]\nversion = "{version}"')
            result = {"content": base64.b64encode(source.encode()).decode()}
        else:
            raise AssertionError(f"Unexpected endpoint: {endpoint}")
        return json.dumps(result).encode()

    def create(self, tag: str = TAG) -> str:
        with patch.object(release, "gh", side_effect=self.fake_gh):
            return release.create_draft(REPO, "123", tag)

    def test_complete_flow_preserves_bytes_and_creates_only_a_draft(self) -> None:
        self.assertIn("/draft", self.create())
        self.assertEqual(len(self.created), 1)

    def test_failed_or_untrusted_runs_never_create_release(self) -> None:
        for key, value in [("head_branch", "dev"), ("path", ".github/workflows/other.yml"),
                           ("conclusion", "failure"), ("event", "pull_request"),
                           ("head_repository", {"full_name": "fork/project"})]:
            with self.subTest(key=key):
                original = copy.deepcopy(self.run)
                self.run[key] = value
                with self.assertRaises(ValueError):
                    self.create()
                self.run = original
        self.assertFalse(self.created)

    def test_upload_enabled_or_incomplete_jobs_are_rejected(self) -> None:
        for name in ("preflight", "prepare", "publish", "build / build (windows)"):
            with self.subTest(name=name):
                original = copy.deepcopy(self.jobs)
                next(job for job in self.jobs if job["name"] == name)["conclusion"] = "success" if name == "publish" else "failure"
                with self.assertRaises(ValueError):
                    self.create()
                self.jobs = original
        self.assertFalse(self.created)

    def test_expired_tampered_or_wrong_commit_artifacts_are_rejected(self) -> None:
        for key, value in [("expired", True), ("digest", "sha256:wrong"), ("workflow_run", {"head_sha": "b" * 40})]:
            with self.subTest(key=key):
                original = copy.deepcopy(self.artifact)
                self.artifact[key] = value
                with self.assertRaises(ValueError):
                    self.create()
                self.artifact = original
        self.assertFalse(self.created)

    def test_existing_release_is_never_overwritten(self) -> None:
        self.releases = [{"tag_name": TAG, "draft": True}]
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.create()
        self.assertFalse(self.created)

    def test_existing_wrong_tag_is_rejected(self) -> None:
        self.refs = [{"ref": f"refs/tags/{TAG}", "object": {"type": "commit", "sha": "b" * 40}}]
        with self.assertRaisesRegex(ValueError, "different commit"):
            self.create()
        self.assertFalse(self.created)

    def test_matching_lightweight_and_annotated_tags_are_accepted(self) -> None:
        for kind in ("commit", "tag"):
            with self.subTest(kind=kind):
                self.refs = [{"ref": f"refs/tags/{TAG}", "object": {"type": kind, "sha": SHA}}]
                self.create()

    def test_tag_must_match_historical_source_version(self) -> None:
        self.bad_source_version = True
        with self.assertRaisesRegex(ValueError, "source version"):
            self.create()
        self.assertFalse(self.created)

    def test_shell_like_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.create("v1.2.3; echo bad")
        self.assertFalse(self.created)

    def test_incomplete_matrix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            files = release.unpack_artifact(self.archive, Path(temporary))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                release.validate_files(files[:-1], VERSION)

    def test_metadata_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            files = release.unpack_artifact(self.archive, Path(temporary))
            wheel = next(p for p in files if p.suffix == ".whl")
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("package.dist-info/METADATA", "Name: wrong-package\nVersion: 1.2.3\n")
            with self.assertRaisesRegex(ValueError, "metadata"):
                release.validate_files(files, VERSION)

    def test_unexpected_artifact_paths_are_rejected(self) -> None:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("../escape.whl", b"bad")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Unexpected artifact"):
                release.unpack_artifact(stream.getvalue(), Path(temporary))


if __name__ == "__main__":
    unittest.main()
