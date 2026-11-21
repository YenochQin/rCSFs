"""Public API and CLI coverage for GRASP-style active-space splitting."""

from pathlib import Path

import pytest
from _native_cli import main

from rcsfs import convert_csfs, read_csfs, split_csfs_by_active_spaces


def _source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.c"
    source.write_text(
        "Core subshells:\n  1s\nPeel subshells:\n  5s  15s\nCSF(s):\n"
        "  5s ( 2)\n\n        0+\n *\n 15s ( 2)\n\n        0+\n",
        encoding="ascii",
    )
    parquet = tmp_path / "source_descriptors.parquet"
    result = convert_csfs(source, parquet)
    assert result["success"] is True
    return parquet, Path(result["header_file"])


def test_public_api_selects_independent_overlapping_spaces(tmp_path: Path) -> None:
    parquet, header = _source(tmp_path)
    small = tmp_path / "small_descriptors.parquet"
    large = tmp_path / "large_descriptors.parquet"
    result = split_csfs_by_active_spaces(parquet, header, {small: "5s", large: "15s"})

    assert [output["record_count"] for output in result["outputs"]] == [1, 2]
    assert [output["block_lengths"] for output in result["outputs"]] == [
        [1],
        [1, 1],
    ]
    assert read_csfs(small)[1]["sub0_occupation"].to_list() == [2]
    assert read_csfs(large)[1]["sub1_occupation"].to_list() == [0, 2]


def test_split_active_cli_uses_grasp_style_labels(tmp_path: Path) -> None:
    parquet, header = _source(tmp_path)
    assert (
        main(
            [
                "csfs-split",
                str(parquet),
                "--header",
                str(header),
                "--space",
                "_small=5s",
                "--space",
                "_large=15s",
            ]
        )
        == 0
    )
    assert (tmp_path / "source_small_desc.parquet").exists()
    assert (tmp_path / "source_large_desc.parquet").exists()


def test_split_active_cli_rejects_bad_labels_before_writing(tmp_path: Path) -> None:
    parquet, header = _source(tmp_path)
    assert (
        main(
            [
                "csfs-split",
                str(parquet),
                "--header",
                str(header),
                "--space",
                "../outside=5s",
            ]
        )
        == 2
    )
    assert not (tmp_path / "sourceoutside_desc.parquet").exists()


def test_csfs_split_reads_new_default_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet, header = _source(tmp_path)
    (tmp_path / "rcsfs.toml").write_text(
        "[csfs-split]\n"
        f'split_csfs_parquet = "{parquet.name}"\n'
        f'csfs_header = "{header.name}"\n'
        'active_spaces = ["_small=5s", "_large=15s"]\n'
    )
    monkeypatch.chdir(tmp_path)
    assert main(["csfs-split"]) == 0
    assert (tmp_path / "source_small_desc.parquet").exists()
    assert (tmp_path / "source_large_desc.parquet").exists()


def test_csfs_split_rejects_removed_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet, header = _source(tmp_path)
    (tmp_path / "rcsfs.toml").write_text(
        "[csfs-split]\n"
        f'split_csfs_parquet = "{parquet.name}"\n'
        f'csfs_header = "{header.name}"\n'
        'active_spaces = ["AS1=5s"]\n'
        'output_dir = "split/nested"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["csfs-split"]) == 2
    assert not (tmp_path / "split").exists()


pytestmark = pytest.mark.usefixtures("cli_cwd")
