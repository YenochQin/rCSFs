"""Wheel composition and PEP 517/660 routing without compiling test binaries."""

import base64
import csv
import hashlib
import zipfile
from pathlib import Path

import maturin
import pytest

import build_backend as backend


def _wheels(
    tmp_path: Path,
    platform: str = "manylinux_2_34_x86_64",
    *,
    payload: bytes = b"\x7fELFnative",
) -> tuple[Path, Path]:
    prefix = "rcsfs-3.0.0"
    library = tmp_path / f"{prefix}-cp315-cp315-{platform}.whl"
    native = tmp_path / f"{prefix}-py3-none-{platform}.whl"
    executable = "rcsfs.exe" if platform.startswith("win") else "rcsfs"
    with zipfile.ZipFile(library, "w") as z:
        z.writestr("rcsfs/__init__.py", "# library API\n")
        z.writestr("rcsfs/_rcsfs.so", b"extension")
        z.writestr(f"{prefix}.dist-info/METADATA", "Name: rcsfs\nVersion: 3.0.0\n")
        z.writestr(f"{prefix}.dist-info/RECORD", "old record\n")
    with zipfile.ZipFile(native, "w") as z:
        z.writestr(f"{prefix}.data/scripts/{executable}", payload)
    return library, native


@pytest.mark.parametrize(
    "platform,payload",
    [
        ("manylinux_2_34_x86_64", b"\x7fELFcli"),
        ("win_amd64", b"MZcli"),
        ("macosx_11_0_arm64", b"\xcf\xfa\xed\xfecli"),
    ],
)
def test_merge_preserves_library_adds_executable_and_rehashes_record(
    tmp_path: Path, platform: str, payload: bytes
) -> None:
    library, native = _wheels(tmp_path, platform, payload=payload)
    before = library.read_bytes()
    output = backend.merge_wheels(library, native, tmp_path / "out")
    assert library.read_bytes() == before
    with zipfile.ZipFile(output) as z:
        record = next(name for name in z.namelist() if name.endswith("/RECORD"))
        rows = {
            row[0]: row[1:] for row in csv.reader(z.read(record).decode().splitlines())
        }
        assert rows[record] == ["", ""]
        for name in z.namelist():
            if name == record:
                continue
            data = z.read(name)
            sha = (
                base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .rstrip(b"=")
                .decode()
            )
            assert rows[name] == [f"sha256={sha}", str(len(data))]
        binary = next(name for name in z.namelist() if ".data/scripts/" in name)
        assert z.read(binary) == payload
        assert z.getinfo(binary).external_attr >> 16 & 0o111
        assert z.read("rcsfs/_rcsfs.so") == b"extension"
        assert not any(name.endswith("entry_points.txt") for name in z.namelist())


def test_merge_can_replace_library_archive_after_closing_it(tmp_path: Path) -> None:
    library, native = _wheels(tmp_path)
    assert backend.merge_wheels(library, native, tmp_path) == library
    with zipfile.ZipFile(library) as z:
        assert any(".data/scripts/rcsfs" in name for name in z.namelist())


@pytest.mark.parametrize("payload", [b"#!/usr/bin/python\n", b"#!/bin/sh\n"])
def test_python_or_shell_launchers_are_rejected(tmp_path: Path, payload: bytes) -> None:
    library, native = _wheels(tmp_path, payload=payload)
    with pytest.raises(ValueError, match="native executable"):
        backend.merge_wheels(library, native, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_different_platform_or_version_is_rejected(tmp_path: Path) -> None:
    library, native = _wheels(tmp_path)
    for filename in [
        native.name.replace("2_34", "2_28"),
        native.name.replace("3.0.0", "3.0.1"),
    ]:
        with pytest.raises(ValueError, match="same package/version and platform"):
            backend.merge_wheels(library, native.with_name(filename), tmp_path / "out")


def test_native_build_forces_no_python_and_auditing() -> None:
    args = backend._native_arguments(
        [
            "--features",
            "pyo3/extension-module,python",
            "--all-features",
            "--bindings=pyo3",
            "--target",
            "aarch64-apple-darwin",
            "--compatibility",
            "pypi",
            "--auditwheel",
            "repair",
        ]
    )
    assert "pyo3/extension-module,python" not in args and "--all-features" not in args
    assert args[-7:] == [
        "--bindings",
        "bin",
        "--no-default-features",
        "--features",
        "cli",
        "--auditwheel",
        "check",
    ]
    assert "aarch64-apple-darwin" in args


@pytest.mark.parametrize(
    "arguments",
    [
        "--compatibility off",
        "--compatibility=linux",
        "--auditwheel skip",
        "--skip-auditwheel",
    ],
)
def test_unaudited_platform_options_are_rejected(arguments: str) -> None:
    with pytest.raises(ValueError):
        backend._arguments({"maturin.build-args": arguments})


@pytest.mark.parametrize("editable", [False, True])
def test_both_build_hooks_merge_native_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, editable: bool
) -> None:
    calls = []
    library, native = _wheels(tmp_path)

    def build(directory, settings, metadata_directory=None):
        args = settings["maturin.build-args"]
        source = native if "bin" in args else library
        calls.append(("wheel", args))
        target = Path(directory) / source.name
        target.write_bytes(source.read_bytes())
        return source.name

    def build_editable(directory, settings, metadata_directory=None):
        calls.append(("editable", settings["maturin.build-args"]))
        target = Path(directory) / library.name
        target.write_bytes(library.read_bytes())
        return library.name

    monkeypatch.setattr(maturin, "build_wheel", build)
    monkeypatch.setattr(maturin, "build_editable", build_editable)
    hook = backend.build_editable if editable else backend.build_wheel
    name = hook(str(tmp_path / "output"))
    assert calls[0][0] == "wheel" and "bin" in calls[0][1]
    assert calls[1][0] == ("editable" if editable else "wheel")
    assert "--compatibility" in calls[0][1] and "pypi" in calls[0][1]
    with zipfile.ZipFile(tmp_path / "output" / name) as z:
        assert "rcsfs-3.0.0.data/scripts/rcsfs" in z.namelist()
