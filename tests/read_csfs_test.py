from pathlib import Path
import tomllib
from typing import get_type_hints

import polars as pl
import pytest

from rcsfs import CsfHeaderData, convert_csfs, read_csfs


def _write_multiblock_csf(path: Path) -> None:
    path.write_text(
        "Header 1\n"
        "Header 2\n"
        "Header 3\n"
        "Header 4\n"
        "Header 5\n"
        "config-a\n"
        "middle-a\n"
        "final-a\n"
        " *\n"
        "config-b\n"
        "middle-b\n"
        "final-b\n"
        "config-c\n"
        "middle-c\n"
        "final-c\n",
        encoding="ascii",
    )


def test_read_csfs_runtime_return_annotation_resolves() -> None:
    assert get_type_hints(read_csfs)["return"] == tuple[CsfHeaderData, pl.DataFrame]


def test_read_csfs_returns_polars_dataframe_and_skips_block_separators(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "multiblock.csf"
    _write_multiblock_csf(input_path)

    header, frame = read_csfs(input_path, num_workers=2)

    assert header == {
        "header_info": {
            "header_lines": ["Header 1", "Header 2", "Header 3", "Header 4", "Header 5"]
        },
        "block_info": {"block_lengths": [1, 2], "block_count": 2},
        "conversion_stats": {
            "csf_count": 3,
            "total_lines": 10,
            "truncated_count": 0,
        },
    }
    assert isinstance(frame, pl.DataFrame)
    assert frame.schema == {
        "idx": pl.UInt64,
        "line1": pl.String,
        "line2": pl.String,
        "line3": pl.String,
    }
    assert frame.to_dict(as_series=False) == {
        "idx": [0, 1, 2],
        "line1": ["config-a", "config-b", "config-c"],
        "line2": ["middle-a", "middle-b", "middle-c"],
        "line3": ["final-a", "final-b", "final-c"],
    }


def test_read_csfs_can_include_zero_based_block_id(tmp_path: Path) -> None:
    input_path = tmp_path / "multiblock.csf"
    _write_multiblock_csf(input_path)

    _, frame = read_csfs(input_path, include_block_id=True, num_workers=2)

    assert frame.columns == ["idx", "block_id", "line1", "line2", "line3"]
    assert frame["block_id"].dtype == pl.UInt32
    assert frame["block_id"].to_list() == [0, 1, 1]


@pytest.mark.parametrize(
    ("include_block_id", "include_coupling_signature", "expected_columns"),
    [
        (False, False, ["idx", "line1", "line2", "line3"]),
        (True, False, ["idx", "block_id", "line1", "line2", "line3"]),
        (
            False,
            True,
            ["idx", "line1", "line2", "line3", "coupling_signature"],
        ),
        (
            True,
            True,
            [
                "idx",
                "block_id",
                "line1",
                "line2",
                "line3",
                "coupling_signature",
            ],
        ),
    ],
)
def test_read_csfs_option_combinations_preserve_column_order(
    include_block_id: bool,
    include_coupling_signature: bool,
    expected_columns: list[str],
) -> None:
    input_path = Path(__file__).parent / "fixtures" / "sample.csf"

    _, frame = read_csfs(
        input_path,
        num_workers=2,
        include_block_id=include_block_id,
        include_coupling_signature=include_coupling_signature,
    )

    assert frame.columns == expected_columns
    if include_coupling_signature:
        assert frame.schema["coupling_signature"] == pl.List(pl.Int32)


def test_read_csfs_coupling_signatures_match_complex_fixture() -> None:
    input_path = Path(__file__).parent / "fixtures" / "sample.csf"

    header, frame = read_csfs(
        input_path,
        num_workers=2,
        include_block_id=True,
        include_coupling_signature=True,
    )

    signatures = frame["coupling_signature"]
    assert frame.height == 28
    assert frame["idx"].to_list() == list(range(28))
    assert signatures.dtype == pl.List(pl.Int32)
    assert signatures.null_count() == 0
    assert all(signatures.list.len() > 0)
    assert signatures.list.last().to_list() == [8] * 28
    assert header["conversion_stats"]["truncated_count"] == 0

    coupling_level = 2
    summary = (
        frame.with_columns(
            pl.col("coupling_signature")
            .list.slice(-coupling_level)
            .alias("selected_coupling")
        )
        .group_by(["block_id", "selected_coupling"], maintain_order=True)
        .agg(
            pl.len().alias("count"),
            pl.col("idx").alias("global_idxs"),
        )
    )
    assert summary["count"].sum() == 28


def test_read_csfs_header_matches_convert_csfs_toml(tmp_path: Path) -> None:
    input_path = tmp_path / "multiblock.csf"
    output_path = tmp_path / "multiblock.parquet"
    _write_multiblock_csf(input_path)

    header, _ = read_csfs(input_path, max_line_len=6, num_workers=2)
    stats = convert_csfs(
        input_path,
        output_path,
        max_line_len=6,
        chunk_size=4,
        num_workers=2,
    )

    assert stats["success"] is True
    sidecar = tomllib.loads(Path(stats["header_file"]).read_text(encoding="utf-8"))
    assert header == sidecar


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("max_line_len", 0), ("num_workers", 0)],
)
def test_read_csfs_rejects_zero_parameters(
    tmp_path: Path, parameter: str, value: int
) -> None:
    input_path = tmp_path / "multiblock.csf"
    _write_multiblock_csf(input_path)

    with pytest.raises(ValueError, match=f"{parameter} must be greater than 0"):
        read_csfs(input_path, **{parameter: value})


def test_read_csfs_rejects_separator_inside_incomplete_csf(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid-block.csf"
    input_path.write_text("h1\nh2\nh3\nh4\nh5\nconfig\nmiddle\n*\n", encoding="ascii")

    with pytest.raises(OSError, match="not a multiple of 3"):
        read_csfs(input_path)


def test_read_csfs_rejects_incomplete_final_csf_by_default(tmp_path: Path) -> None:
    input_path = tmp_path / "incomplete-final.csf"
    input_path.write_text(
        "h1\nh2\nh3\nh4\nh5\n"
        "config-a\nmiddle-a\nfinal-a\n"
        "config-incomplete\nmiddle-incomplete\n",
        encoding="ascii",
    )

    with pytest.raises(OSError, match="final CSF block.*not a multiple of 3"):
        read_csfs(input_path)


def test_read_csfs_can_ignore_incomplete_final_csf_when_not_strict(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "incomplete-final.csf"
    input_path.write_text(
        "h1\nh2\nh3\nh4\nh5\n"
        "config-a\nmiddle-a\nfinal-a\n"
        "config-incomplete\nmiddle-incomplete\n",
        encoding="ascii",
    )

    _, frame = read_csfs(input_path, strict=False)

    assert frame.height == 1
    assert frame["line1"].to_list() == ["config-a"]


def test_read_csfs_rejects_non_ascii_data(tmp_path: Path) -> None:
    input_path = tmp_path / "non-ascii.csf"
    input_path.write_text("h1\nh2\nh3\nh4\nh5\n配置\nmiddle\nfinal\n", encoding="utf-8")

    with pytest.raises(OSError, match="non-ASCII"):
        read_csfs(input_path)
