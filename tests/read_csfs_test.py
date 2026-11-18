import tomllib
from pathlib import Path
from typing import get_type_hints

import polars as pl
import pytest

from rcsfs import CsfHeaderData, convert_csfs, read_csfs

FIXTURE = Path(__file__).parent / "fixtures" / "descriptor_multiblock.c"


def test_read_csfs_runtime_return_annotation_resolves() -> None:
    assert get_type_hints(read_csfs)["return"] == tuple[CsfHeaderData, pl.DataFrame]


@pytest.mark.parametrize("blocks", [False, True])
@pytest.mark.parametrize("signatures", [False, True])
def test_read_numeric_rows_preserves_schema_blocks_and_signatures(
    tmp_path: Path, blocks: bool, signatures: bool
) -> None:
    output = tmp_path / "basis_descriptors.parquet"
    header_path = tmp_path / "custom.toml"
    convert_csfs(FIXTURE, output, header_path=header_path)
    header, rows = read_csfs(
        output,
        header_path=header_path,
        include_block_id=blocks,
        include_coupling_signature=signatures,
    )
    assert header == tomllib.loads(header_path.read_text())
    assert header["header_info"]["electron_count"] == 4
    assert header["block_info"]["block_lengths"] == [1, 1]
    assert rows["idx"].dtype == pl.UInt64
    assert rows["idx"].to_list() == [0, 1]
    assert rows["total_two_j"].to_list() == [0, 2]
    assert rows["parity"].to_list() == [1, -1]
    assert rows["sub0_n"].to_list() == [2, 1]
    assert rows["sub1_2j"].to_list() == [-1, 3]
    assert all(
        dtype == pl.Int8
        for column, dtype in rows.schema.items()
        if column not in {"idx", "block_id", "coupling_signature"}
    )
    if blocks:
        assert rows["block_id"].dtype == pl.UInt32
        assert rows["block_id"].to_list() == [0, 1]
    else:
        assert "block_id" not in rows
    if signatures:
        assert rows["coupling_signature"].dtype == pl.List(pl.Int32)
        assert rows["coupling_signature"].to_list() == [[0], [1, 2]]
    else:
        assert "coupling_signature" not in rows


def test_read_requires_descriptor_and_exact_companion_header(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="descriptor Parquet"):
        read_csfs(FIXTURE)
    output = tmp_path / "basis_descriptors.parquet"
    header = tmp_path / "basis_header.toml"
    convert_csfs(FIXTURE, output)
    header.write_text(header.read_text() + "\n# stale\n")
    with pytest.raises(OSError, match="source_header_sha256"):
        read_csfs(output)
