"""Python API and CLI tests for ``rcsfs.split_csfs_by_j`` / ``rcsfs jsplit``."""

import json
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MULTI_J_CSF = (FIXTURES / "multi_j.c").read_text()
MULTI_J_W = (FIXTURES / "multi_j.w").read_bytes()


@pytest.mark.parametrize("string_input", [False, True])
def test_split_csfs_by_j_names_outputs_by_two_j(
    tmp_path: Path, string_input: bool
) -> None:
    from rcsfs import JBlockOutputStats, JBlockSplitStats, split_csfs_by_j

    input_csf = tmp_path / "multi_j.c"
    input_csf.write_text(MULTI_J_CSF)

    stats = split_csfs_by_j(str(input_csf) if string_input else input_csf)

    assert set(stats) == set(JBlockSplitStats.__annotations__)
    assert stats["success"] is True
    assert stats["input_file"] == str(input_csf)
    assert stats["block_count"] == 3
    assert stats["input_csf_count"] == 4
    outputs = stats["outputs"]
    assert all(
        set(output) == set(JBlockOutputStats.__annotations__) for output in outputs
    )
    assert [Path(output["output_file"]).name for output in outputs] == [
        "multi_j_8.c",
        "multi_j_5.c",
        "multi_j_0.c",
    ]
    assert [
        (output["total_two_j"], output["parity"], output["csf_count"])
        for output in outputs
    ] == [
        (8, "odd", 2),
        (5, "even", 1),
        (0, "even", 1),
    ]
    assert [output["block_index"] for output in outputs] == [0, 1, 2]
    # No sibling .w exists, so no copies are reported or written.
    assert all(output["w_file"] is None for output in outputs)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "multi_j.c",
        "multi_j_0.c",
        "multi_j_5.c",
        "multi_j_8.c",
    ]
    # Every output repeats the five header lines and one block only.
    for name, two_j in (("multi_j_8.c", 8), ("multi_j_5.c", 5), ("multi_j_0.c", 0)):
        content = (tmp_path / name).read_text()
        assert content.startswith("Core subshells:\n")
        assert content.count("\n *\n") == 0, "single-block outputs carry no separator"
        assert name == f"multi_j_{two_j}.c"


def test_split_csfs_by_j_copies_w_and_honors_output_dir_and_prefix(
    tmp_path: Path,
) -> None:
    from rcsfs import split_csfs_by_j

    source = tmp_path / "inputs"
    source.mkdir()
    (source / "multi_j.c").write_text(MULTI_J_CSF)
    (source / "multi_j.w").write_bytes(MULTI_J_W)
    destination = tmp_path / "outputs"
    destination.mkdir()

    stats = split_csfs_by_j(
        source / "multi_j.c", output_dir=destination, prefix="tagged", copy_w=True
    )

    assert sorted(path.name for path in destination.iterdir()) == [
        "tagged_0.c",
        "tagged_0.w",
        "tagged_5.c",
        "tagged_5.w",
        "tagged_8.c",
        "tagged_8.w",
    ]
    for output in stats["outputs"]:
        assert output["w_file"] is not None
        assert Path(output["w_file"]).read_bytes() == MULTI_J_W
    # copy_w=False stages no copies even when the .w exists.
    stats = split_csfs_by_j(
        source / "multi_j.c", output_dir=destination, prefix="bare", copy_w=False
    )
    assert all(output["w_file"] is None for output in stats["outputs"])
    assert not (destination / "bare_8.w").exists()


def test_split_csfs_by_j_errors(tmp_path: Path) -> None:
    from rcsfs import split_csfs_by_j

    input_csf = tmp_path / "multi_j.c"
    input_csf.write_text(MULTI_J_CSF)
    split_csfs_by_j(input_csf)

    # Callers can explicitly refuse replacement; the default replaces outputs.
    with pytest.raises(FileExistsError):
        split_csfs_by_j(input_csf, overwrite=False)
    stats = split_csfs_by_j(input_csf)
    assert stats["block_count"] == 3

    # Bad prefix is a usage error.
    with pytest.raises(ValueError, match="prefix"):
        split_csfs_by_j(input_csf, prefix="a/b")

    # Missing input and missing output directory are OSError.
    with pytest.raises(OSError):
        split_csfs_by_j(tmp_path / "absent.c")
    with pytest.raises(OSError, match="output directory"):
        split_csfs_by_j(input_csf, output_dir=tmp_path / "absent_dir")


@pytest.mark.parametrize("native_binding", [False, True])
def test_split_csfs_by_j_replaces_existing_c_and_w_by_default(
    tmp_path: Path, native_binding: bool
) -> None:
    from rcsfs import _rcsfs, split_csfs_by_j

    input_csf = tmp_path / "multi_j.c"
    input_csf.write_text(MULTI_J_CSF)
    orbital = tmp_path / "multi_j.w"
    orbital.write_bytes(MULTI_J_W)
    original = split_csfs_by_j(input_csf)
    expected_csfs = {
        output["output_file"]: Path(output["output_file"]).read_bytes()
        for output in original["outputs"]
    }
    for output in original["outputs"]:
        Path(output["output_file"]).write_text("stale CSF output\n")
        assert output["w_file"] is not None
        Path(output["w_file"]).write_bytes(b"stale orbital output")
    updated_orbital = MULTI_J_W + b"\x00updated"
    orbital.write_bytes(updated_orbital)

    stats = (
        _rcsfs.split_csfs_by_j(str(input_csf))
        if native_binding
        else split_csfs_by_j(input_csf)
    )

    for output in stats["outputs"]:
        assert (
            Path(output["output_file"]).read_bytes()
            == expected_csfs[output["output_file"]]
        )
        assert output["w_file"] is not None
        assert Path(output["w_file"]).read_bytes() == updated_orbital
    assert input_csf.read_text() == MULTI_J_CSF
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("marker", [" *garbage", " * ", "x*garbage"])
def test_split_csfs_by_j_rejects_noncanonical_separator_without_outputs(
    tmp_path: Path, marker: str
) -> None:
    from rcsfs import split_csfs_by_j

    lines = MULTI_J_CSF.splitlines()
    lines[6] = marker
    input_csf = tmp_path / "malformed.c"
    input_csf.write_text("\n".join(lines) + "\n")
    with pytest.raises(OSError, match="line 7: block separator must be exactly"):
        split_csfs_by_j(input_csf)
    assert list(tmp_path.iterdir()) == [input_csf]


def test_split_csfs_by_j_failed_rerun_preserves_existing_outputs(
    tmp_path: Path,
) -> None:
    from rcsfs import split_csfs_by_j

    input_csf = tmp_path / "multi_j.c"
    input_csf.write_text(MULTI_J_CSF)
    (tmp_path / "multi_j.w").write_bytes(MULTI_J_W)
    stats = split_csfs_by_j(input_csf)
    existing: dict[Path, bytes] = {}
    for output in stats["outputs"]:
        for name in (output["output_file"], output["w_file"]):
            if name is not None:
                path = Path(name)
                existing[path] = path.read_bytes()
    lines = MULTI_J_CSF.splitlines()
    lines[13] = " *garbage"  # A later block, after an output has been staged.
    input_csf.write_text("\n".join(lines) + "\n")

    with pytest.raises(OSError, match="line 14: block separator must be exactly"):
        split_csfs_by_j(input_csf)
    assert all(path.read_bytes() == content for path, content in existing.items())
    assert not list(tmp_path.glob("*.tmp"))


def test_split_csfs_by_j_error_classification_does_not_depend_on_path_text(
    tmp_path: Path,
) -> None:
    from rcsfs import split_csfs_by_j

    input_csf = tmp_path / "input.c"
    input_csf.write_text(MULTI_J_CSF)
    with pytest.raises(OSError, match="output directory does not exist") as caught:
        split_csfs_by_j(input_csf, output_dir=tmp_path / "already exists but missing")
    assert type(caught.value) is OSError
    assert list(tmp_path.iterdir()) == [input_csf]


@pytest.mark.skipif(os.name != "posix", reason="requires Unix symlink support")
@pytest.mark.parametrize("extension", ["c", "w"])
def test_split_csfs_by_j_publication_conflict_raises_file_exists_error(
    tmp_path: Path, extension: str
) -> None:
    from rcsfs import split_csfs_by_j

    input_csf = tmp_path / "multi_j.c"
    input_csf.write_text(MULTI_J_CSF)
    if extension == "w":
        (tmp_path / "multi_j.w").write_bytes(MULTI_J_W)
    conflict = tmp_path / f"multi_j_5.{extension}"
    conflict.symlink_to(tmp_path / "missing")

    with pytest.raises(FileExistsError, match="already published:") as caught:
        split_csfs_by_j(input_csf, overwrite=False)
    assert str(tmp_path / "multi_j_8.c") in str(caught.value)
    assert (tmp_path / "multi_j_8.c").is_file()
    assert conflict.is_symlink()
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_jsplit_runs_and_prints_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rcsfs import cli

    monkeypatch.chdir(tmp_path)
    input_csf = tmp_path / "multi_j.c"
    input_csf.write_text(MULTI_J_CSF)
    (tmp_path / "multi_j.w").write_bytes(MULTI_J_W)

    exit_code = cli.main(["jsplit", str(input_csf)])
    captured = capsys.readouterr()
    assert exit_code == 0
    lines = captured.out.rstrip("\n").splitlines()
    assert len(lines) == 3
    assert "multi_j_8.c: 2 CSFs (2J=8, odd) (+ " in lines[0]
    assert "multi_j_5.c: 1 CSFs (2J=5, even) (+ " in lines[1]
    assert "multi_j_0.c: 1 CSFs (2J=0, even) (+ " in lines[2]
    assert all(line.rstrip(")").endswith(".w") for line in lines)

    # Rerunning the command replaces existing output files by default.
    (tmp_path / "multi_j_8.c").write_text("stale CSF output\n")
    (tmp_path / "multi_j_8.w").write_bytes(b"stale orbital output")
    exit_code = cli.main(["jsplit", str(input_csf)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert (tmp_path / "multi_j_8.c").read_text().startswith("Core subshells:\n")
    assert (tmp_path / "multi_j_8.w").read_bytes() == MULTI_J_W

    exit_code = cli.main(["jsplit", str(input_csf), "--overwrite", "--no-copy-w"])
    assert exit_code == 0


def test_cli_jsplit_json_and_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rcsfs import cli

    monkeypatch.chdir(tmp_path)
    input_csf = tmp_path / "multi_j.c"
    input_csf.write_text(MULTI_J_CSF)

    exit_code = cli.main(["rasfsplit", str(input_csf), "--json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    stats = json.loads(captured.out)
    assert stats["success"] is True
    assert stats["block_count"] == 3
    assert [Path(output["output_file"]).name for output in stats["outputs"]] == [
        "multi_j_8.c",
        "multi_j_5.c",
        "multi_j_0.c",
    ]

    # JSON reruns also replace outputs by default.
    exit_code = cli.main(["jsplit", str(input_csf), "--json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["success"] is True

    exit_code = cli.main(["jsplit", str(tmp_path / "missing.c"), "--json"])
    captured = capsys.readouterr()
    assert exit_code == 1
    error = json.loads(captured.out)
    assert error["success"] is False
    assert "CSF input does not exist" in error["error"]


def test_cli_jsplit_reads_toml_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rcsfs import cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "multi_j.c").write_text(MULTI_J_CSF)
    (tmp_path / "rcsfs.toml").write_text(
        '[jsplit]\ninput_csf = "multi_j.c"\nprefix = "cfg"\ncopy_w = false\n'
    )

    exit_code = cli.main(["jsplit"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "cfg_8.c: 2 CSFs" in captured.out
    assert sorted(
        path.name for path in tmp_path.iterdir() if path.name.startswith("cfg")
    ) == [
        "cfg_0.c",
        "cfg_5.c",
        "cfg_8.c",
    ]


def test_cli_jsplit_batch_config_writes_to_cwd_and_copies_input_sibling_w(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rcsfs import cli

    source = tmp_path / "inputs"
    source.mkdir()
    (source / "multi_j.c").write_text(MULTI_J_CSF)
    (source / "multi_j.w").write_bytes(MULTI_J_W)
    destination = tmp_path / "outputs"
    destination.mkdir()
    monkeypatch.chdir(destination)
    config = destination / "batch.toml"
    config.write_text(
        '[jsplit]\ninput_csf = "../inputs/multi_j.c"\nprefix = "cfg"\njson = true\n'
    )

    assert cli.main(["--config", str(config)]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["block_count"] == 3
    assert sorted(path.name for path in source.iterdir()) == ["multi_j.c", "multi_j.w"]
    assert sorted(path.name for path in destination.iterdir()) == [
        "batch.toml",
        "cfg_0.c",
        "cfg_0.w",
        "cfg_5.c",
        "cfg_5.w",
        "cfg_8.c",
        "cfg_8.w",
    ]
    for output in stats["outputs"]:
        assert Path(output["output_file"]).parent == destination
        assert output["w_file"] is not None
        assert Path(output["w_file"]).read_bytes() == MULTI_J_W

    (destination / "cfg_8.c").write_text("stale CSF output\n")
    (destination / "cfg_8.w").write_bytes(b"stale orbital output")
    assert cli.main(["--config", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["success"] is True
    assert (destination / "cfg_8.c").read_text().startswith("Core subshells:\n")
    assert (destination / "cfg_8.w").read_bytes() == MULTI_J_W


def test_cli_jsplit_init_config_adds_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rcsfs import cli

    monkeypatch.chdir(tmp_path)
    exit_code = cli.main(["jsplit", "init-config"])
    assert exit_code == 0
    config = (tmp_path / "rcsfs.toml").read_text()
    assert "[jsplit]" in config
    assert 'input_csf = "generated.c"' in config
    assert "# overwrite = true" in config
