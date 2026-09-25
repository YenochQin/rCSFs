"""Public API and CLI coverage for GRASP-style active-space splitting."""

from pathlib import Path

import pytest

from rcsfs import convert_csfs, split_csfs_by_active_spaces
from rcsfs.cli import main


def _source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.c"
    source.write_text(
        "Core subshells:\n\nPeel subshells:\n  2s  5g  15g\nCSF(s):\n"
        "  2s ( 2)  5g ( 1)\nL2-a\nL3-a\n"
        " *\n"
        "  2s ( 2)  15g ( 1)\nL2-b\nL3-b\n",
        encoding="ascii",
    )
    parquet = tmp_path / "source.parquet"
    result = convert_csfs(source, parquet, chunk_size=90, num_workers=1)
    assert result["success"] is True
    return parquet, Path(result["header_file"])


def test_public_api_selects_independent_overlapping_spaces(tmp_path: Path) -> None:
    parquet, header = _source(tmp_path)
    small = tmp_path / "small.c"
    large = tmp_path / "large.c"
    result = split_csfs_by_active_spaces(
        parquet, header, {small: "5s,5g", large: "5s,15g"}
    )

    assert result["input_csf_count"] == 2
    assert [output["csf_count"] for output in result["outputs"]] == [1, 2]
    assert [output["block_lengths"] for output in result["outputs"]] == [
        [1, 0],
        [1, 1],
    ]
    assert "15g ( 1)" not in small.read_text(encoding="ascii")
    assert "15g ( 1)" in large.read_text(encoding="ascii")


@pytest.mark.parametrize("command", ["csfs-split", "split-active", "rcsfsplit"])
def test_split_active_cli_uses_grasp_style_labels(tmp_path: Path, command: str) -> None:
    parquet, header = _source(tmp_path)
    active_space_flag = "--active-space" if command == "csfs-split" else "--space"
    assert (
        main(
            [
                command,
                str(parquet),
                "--header",
                str(header),
                "--output-dir",
                str(tmp_path),
                active_space_flag,
                "_small=5s,5g",
                active_space_flag,
                "_large=5s,15g",
            ]
        )
        == 0
    )
    assert (tmp_path / "source_small.c").exists()
    assert (tmp_path / "source_large.c").exists()


def test_split_active_cli_rejects_bad_labels_before_writing(tmp_path: Path) -> None:
    parquet, header = _source(tmp_path)
    assert (
        main(
            [
                "split-active",
                str(parquet),
                "--header",
                str(header),
                "--output-dir",
                str(tmp_path),
                "--space",
                "../outside=5s,5g",
            ]
        )
        == 1
    )
    assert not (tmp_path / "sourceoutside.c").exists()


def test_csfs_split_reads_new_default_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet, header = _source(tmp_path)
    (tmp_path / "rcsfs.toml").write_text(
        "[csfs-split]\n"
        f'split_csfs_parquet = "{parquet.name}"\n'
        f'csfs_header = "{header.name}"\n'
        'active_spaces = ["_small=5s,5g", "_large=5s,15g"]\n'
        'output_dir = "."\n'
    )
    monkeypatch.chdir(tmp_path)
    assert main(["csfs-split"]) == 0
    assert (tmp_path / "source_small.c").exists()
    assert (tmp_path / "source_large.c").exists()
