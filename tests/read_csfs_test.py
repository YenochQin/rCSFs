import hashlib
import tomllib
from pathlib import Path
from typing import get_type_hints

import polars as pl
import pyarrow.parquet as pq
import pytest

from rcsfs import (
    CsfHeaderData,
    convert_csfs,
    descriptor_header_path,
    merge_csfs,
    partition_csfs,
    read_csfs,
    read_zero_order_counts,
    select_csfs,
    select_interacting_csfs,
    split_csfs_by_j,
)

FIXTURE = Path(__file__).parent / "fixtures" / "descriptor_multiblock.c"


@pytest.mark.parametrize("entry", ['"bad"', "[1]", "[true, 1]", "[-1, 1]", "[1, 2]"])
def test_read_rejects_invalid_zero_order_counts(tmp_path: Path, entry: str) -> None:
    path = tmp_path / "basis_descriptors.parquet"
    convert_csfs(FIXTURE, path)
    header = descriptor_header_path(path)
    raw = header.read_bytes().replace(
        b"[block_info]\n", f"[block_info]\nzero_order_counts = {entry}\n".encode()
    )
    table = pq.read_table(path)
    metadata = dict(pq.read_metadata(path).metadata or {})
    metadata.pop(b"ARROW:schema", None)
    metadata[b"source_header_sha256"] = hashlib.sha256(raw).hexdigest().encode()
    header.write_bytes(raw)
    pq.write_table(table.replace_schema_metadata(metadata), path)
    with pytest.raises(OSError, match="zero_order_counts"):
        read_csfs(path)
    with pytest.raises(OSError, match="zero_order_counts"):
        read_zero_order_counts(path)


def test_whole_block_split_keeps_zero_order_count(tmp_path: Path) -> None:
    path = tmp_path / "basis_descriptors.parquet"
    reference = tmp_path / "ref_descriptors.parquet"
    ordered = tmp_path / "ordered_descriptors.parquet"
    text = tmp_path / "basis.c"
    text.write_bytes(
        b"Core subshells:\n  1s\nPeel subshells:\n  2s   3s   4s\nCSF(s):\n"
        b"  2s ( 2)\n\n        0+\n  3s ( 2)\n\n        0+\n *\n"
        b"  3s ( 2)\n\n        0+\n  4s ( 2)\n\n        0+\n  2s ( 2)\n\n        0+\n"
    )
    convert_csfs(text, path)
    assert read_zero_order_counts(path) is None
    select_csfs(path, reference, [1, 2, 4])
    partition_csfs(
        reference, descriptor_header_path(reference), path, descriptor_header_path(path), ordered
    )
    stats = split_csfs_by_j(ordered, output_dir=tmp_path)
    assert read_zero_order_counts(ordered) == [1, 2]
    assert len(stats["outputs"]) == 2
    for block, count, length in zip(stats["outputs"], [1, 2], [2, 3], strict=True):
        header, rows = read_csfs(block["output_file"])
        assert header["block_info"]["zero_order_counts"] == [count]
        assert rows.height == length


def test_truncation_history_survives_read_selection_and_combination(
    tmp_path: Path,
) -> None:
    pairs = [tmp_path / name for name in ("one.parquet", "two.parquet")]
    for count, path in enumerate(pairs, start=1):
        convert_csfs(FIXTURE, path)
        header = descriptor_header_path(path)
        raw = header.read_bytes().replace(
            b"truncated_count = 0", f"truncated_count = {count}".encode()
        )
        raw = raw.replace(b"total_lines = 6", b"total_lines = 7")
        header.write_bytes(raw)
        table = pq.read_table(path)
        metadata = pq.read_metadata(path).metadata.copy()
        metadata.pop(b"ARROW:schema", None)
        metadata[b"source_header_sha256"] = hashlib.sha256(raw).hexdigest().encode()
        pq.write_table(table.replace_schema_metadata(metadata), path)
        assert read_csfs(path)[0] == tomllib.loads(raw.decode())

    selected = tmp_path / "selected.parquet"
    select_csfs(pairs[0], selected, [0])
    merged = tmp_path / "merged.parquet"
    merge_csfs(pairs, merged)
    interacting = tmp_path / "interacting.parquet"
    select_interacting_csfs(*pairs, interacting)
    for path, count in [(selected, 1), (merged, 3), (interacting, 3)]:
        stored = tomllib.loads(descriptor_header_path(path).read_text(encoding="utf-8"))
        assert stored["conversion_stats"]["truncated_count"] == count
        assert read_csfs(path)[0] == stored


@pytest.mark.parametrize("fraction", ["3/4", "3/0", "3/x"])
def test_convert_rejects_invalid_subshell_j_denominator(
    tmp_path: Path, fraction: str
) -> None:
    source = tmp_path / "invalid.c"
    source.write_bytes(FIXTURE.read_bytes().replace(b"3/2", fraction.encode()))
    output = tmp_path / "invalid.parquet"
    with pytest.raises(OSError, match="denominator"):
        convert_csfs(source, output)
    assert list(tmp_path.iterdir()) == [source]


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
    assert rows["total_two_j_plus_one"].to_list() == [1, 3]
    assert rows["parity"].to_list() == [1, -1]
    assert rows["sub0_occupation"].to_list() == [2, 1]
    assert rows["sub1_two_j_plus_one"].to_list() == [1, 4]
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
