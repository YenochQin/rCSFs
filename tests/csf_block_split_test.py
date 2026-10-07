"""Python API and CLI tests for ``rcsfs.split_csfs_by_j`` / ``rcsfs jsplit``."""

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MULTI_J_CSF = (FIXTURES / "multi_j.c").read_text()
MULTI_J_W = (FIXTURES / "multi_j.w").read_bytes()


def test_split_csfs_by_j_names_outputs_by_two_j(tmp_path: Path) -> None:
    from rcsfs import split_csfs_by_j

    input_csf = tmp_path / "multi_j.c"
    input_csf.write_text(MULTI_J_CSF)

    stats = split_csfs_by_j(input_csf)

    assert stats["success"] is True
    assert stats["input_file"] == str(input_csf)
    assert stats["block_count"] == 3
    assert stats["input_csf_count"] == 4
    outputs = stats["outputs"]
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

    # Existing outputs are refused unless overwrite is set.
    with pytest.raises(FileExistsError):
        split_csfs_by_j(input_csf)
    stats = split_csfs_by_j(input_csf, overwrite=True)
    assert stats["block_count"] == 3

    # Bad prefix is a usage error.
    with pytest.raises(ValueError, match="prefix"):
        split_csfs_by_j(input_csf, prefix="a/b")

    # Missing input and missing output directory are OSError.
    with pytest.raises(OSError):
        split_csfs_by_j(tmp_path / "absent.c")
    with pytest.raises(OSError, match="output directory"):
        split_csfs_by_j(input_csf, output_dir=tmp_path / "absent_dir")


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

    # Failure path: existing outputs, refused without --overwrite.
    exit_code = cli.main(["jsplit", str(input_csf)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "J-block split failed" in captured.err

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

    # JSON error path: existing outputs are refused with the error dict.
    exit_code = cli.main(["jsplit", str(input_csf), "--json"])
    captured = capsys.readouterr()
    assert exit_code == 1
    error = json.loads(captured.out)
    assert error["success"] is False
    assert "already exists" in error["error"]


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
