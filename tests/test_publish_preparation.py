"""Release assembly must reject incomplete or mixed platform artifacts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from packaging.utils import parse_sdist_filename, parse_wheel_filename


@pytest.fixture
def publisher(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.delitem(sys.modules, "check_release", raising=False)
    spec = importlib.util.spec_from_file_location(
        "release_assembler", scripts / "prepare_publish.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Archive internals are validated by check_release; these tests isolate the
    # selection policy, including three conflicting copies of the source archive.
    def archive_version(path: Path) -> tuple[str, bool]:
        parsed = (
            parse_wheel_filename(path.name)
            if path.suffix == ".whl"
            else parse_sdist_filename(path.name)
        )
        return str(parsed[1]), False

    monkeypatch.setattr(module, "check_archive", archive_version)
    return module


def matrix(root: Path, publisher: ModuleType) -> Path:
    artifacts = root / "artifacts"
    package = publisher.NAME.replace("-", "_")
    for artifact, platform in publisher.ARTIFACTS.items():
        directory = artifacts / artifact
        directory.mkdir(parents=True)
        if platform == "any":
            tag = "py3-none-any"
        elif "manylinux" in platform:
            tag = "cp314-cp314-manylinux_2_28_x86_64"
        elif "win_amd64" in platform:
            tag = "cp314-cp314-win_amd64"
        else:
            tag = "cp314-cp314-macosx_11_0_arm64"
        (directory / f"{package}-1.0.0-{tag}.whl").write_text(f"wheel from {artifact}")
        (directory / f"{package}-1.0.0.tar.gz").write_text(f"source from {artifact}")
    return artifacts


def test_complete_matrix_selects_canonical_source_without_overwriting(
    tmp_path: Path, publisher: ModuleType
) -> None:
    artifacts = matrix(tmp_path, publisher)
    paths = publisher.prepare(artifacts, tmp_path / "dist", "1.0.0")
    expected_wheels = 1 if publisher.NAME == "grasp-kit" else 3
    assert len(paths) == expected_wheels + 1
    (source,) = [path for path in paths if path.name.endswith(".tar.gz")]
    assert source.read_text() == f"source from {publisher.CANONICAL_ARTIFACT}"
    wheels = [path for path in paths if path.suffix == ".whl"]
    assert len(wheels) == expected_wheels
    if expected_wheels == 1:
        assert wheels[0].read_text() == f"wheel from {publisher.CANONICAL_ARTIFACT}"


def test_missing_platform_cannot_produce_a_partial_release(
    tmp_path: Path, publisher: ModuleType
) -> None:
    artifacts = matrix(tmp_path, publisher)
    directory = artifacts / next(iter(publisher.ARTIFACTS))
    for path in directory.iterdir():
        path.unlink()
    directory.rmdir()
    with pytest.raises(ValueError, match="complete build artifact matrix"):
        publisher.prepare(artifacts, tmp_path / "dist", "1.0.0")
    assert not (tmp_path / "dist").exists()


def test_mixed_versions_are_rejected_before_copying(
    tmp_path: Path, publisher: ModuleType
) -> None:
    artifacts = matrix(tmp_path, publisher)
    wheel = next(artifacts.rglob("*.whl"))
    wheel.rename(wheel.with_name(wheel.name.replace("1.0.0", "1.0.1")))
    with pytest.raises(ValueError, match="Unexpected package or version"):
        publisher.prepare(artifacts, tmp_path / "dist", "1.0.0")
    assert not (tmp_path / "dist").exists()


def test_matching_filename_cannot_hide_a_metadata_version_mismatch(
    tmp_path: Path, publisher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = matrix(tmp_path, publisher)
    monkeypatch.setattr(publisher, "check_archive", lambda path: ("2.0.0", False))
    with pytest.raises(ValueError, match="Metadata version differs"):
        publisher.prepare(artifacts, tmp_path / "dist", "1.0.0")
    assert not (tmp_path / "dist").exists()


def test_wrong_python_abi_is_rejected(tmp_path: Path, publisher: ModuleType) -> None:
    artifacts = matrix(tmp_path, publisher)
    wheel = next(artifacts.rglob("*.whl"))
    prefix = publisher.NAME.replace("-", "_")
    wheel.rename(wheel.with_name(f"{prefix}-1.0.0-cp313-cp313-win_amd64.whl"))
    with pytest.raises(ValueError, match="Unexpected platform or Python ABI"):
        publisher.prepare(artifacts, tmp_path / "dist", "1.0.0")
    assert not (tmp_path / "dist").exists()


def test_existing_output_is_not_silently_overwritten(
    tmp_path: Path, publisher: ModuleType
) -> None:
    artifacts = matrix(tmp_path, publisher)
    output = tmp_path / "dist"
    output.mkdir()
    existing = output / "keep.txt"
    existing.write_text("existing output")
    with pytest.raises(ValueError, match="Output directory must be empty"):
        publisher.prepare(artifacts, output, "1.0.0")
    assert existing.read_text() == "existing output"


def test_unexpected_artifact_file_is_rejected(
    tmp_path: Path, publisher: ModuleType
) -> None:
    artifacts = matrix(tmp_path, publisher)
    (artifacts / publisher.CANONICAL_ARTIFACT / "unexpected.txt").write_text(
        "unexpected"
    )
    with pytest.raises(ValueError, match="one wheel and one source archive"):
        publisher.prepare(artifacts, tmp_path / "dist", "1.0.0")
    assert not (tmp_path / "dist").exists()
