"""Descriptor comparison helpers validate the header before interpreting columns."""

from pathlib import Path

import pytest

from scripts.compare_descriptor_outputs import read_peel_subshells


def test_comparison_reads_peel_subshells_from_header(tmp_path: Path) -> None:
    header = tmp_path / "header.toml"
    header.write_text(
        '[header_info]\nheader_lines = ["h1", "h2", "h3", "1s 2p- 2p"]\n',
        encoding="utf-8",
    )

    assert read_peel_subshells(header) == ["1s", "2p-", "2p"]


def test_comparison_rejects_non_string_peel_line(tmp_path: Path) -> None:
    header = tmp_path / "header.toml"
    header.write_text(
        '[header_info]\nheader_lines = ["h1", "h2", "h3", 42]\n',
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="is not a string"):
        read_peel_subshells(header)


def test_comparison_rejects_missing_peel_line(tmp_path: Path) -> None:
    header = tmp_path / "header.toml"
    header.write_text(
        '[header_info]\nheader_lines = ["h1", "h2", "h3"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Could not find"):
        read_peel_subshells(header)
