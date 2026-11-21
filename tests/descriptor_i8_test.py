from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rcsfs import (
    convert_csfs,
    restore_csfs_from_descriptors,
)


def _inputs(tmp_path: Path, *, large_j: bool = False) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.c"
    text = (
        "Core subshells:\n  1s\nPeel subshells:\n  2s  2p-  2p\nCSF(s):\n"
        "  2s ( 1)  2p-( 1)\n      1/2      1/2\n                  0-\n *\n"
        "  2s ( 1)  2p-( 1)\n      1/2      1/2\n                  1-\n"
    )
    if large_j:
        text = text.replace("                  0-", "                 64-")
    source.write_text(text, encoding="utf-8", newline="\n")
    descriptors = tmp_path / "source_descriptors.parquet"
    header = tmp_path / "source_header.toml"
    if not large_j:
        convert_csfs(source, descriptors)
    return source, header, descriptors


def test_int8_parquet_and_one_byte_buffers_round_trip(tmp_path: Path) -> None:
    source, header, descriptors = _inputs(tmp_path)
    table = pq.read_table(descriptors)
    assert all(field.type == pa.int8() and not field.nullable for field in table.schema)
    for column in table.columns:
        assert column.combine_chunks().buffers()[1].size == table.num_rows
    metadata = pq.read_metadata(descriptors).metadata
    assert metadata[b"descriptor_dtype"] == b"int8"
    assert metadata[b"max_two_j"] == b"126"
    restored = tmp_path / "restored.c"
    restore_csfs_from_descriptors(descriptors, header, restored)
    assert restored.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("indices", [None, [0]])
def test_restore_rejects_int32_columns_even_with_int8_metadata(
    tmp_path: Path, indices: list[int] | None
) -> None:
    _, header, descriptors = _inputs(tmp_path)
    table = pq.read_table(descriptors)
    metadata = dict(pq.read_metadata(descriptors).metadata)
    metadata.pop(b"ARROW:schema", None)
    old_schema = pa.schema(
        [pa.field(field.name, pa.int32(), nullable=False) for field in table.schema],
        metadata=metadata,
    )
    legacy = tmp_path / "int32.parquet"
    pq.write_table(table.cast(old_schema), legacy)
    output = tmp_path / "rejected.c"
    with pytest.raises((ValueError, RuntimeError, OSError), match="Int8"):
        restore_csfs_from_descriptors(legacy, header, output, indices=indices)
    assert not output.exists()


@pytest.mark.parametrize("indices", [None, [0]])
def test_restore_rejects_zero_as_angular_code(
    tmp_path: Path, indices: list[int] | None
) -> None:
    _, header, descriptors = _inputs(tmp_path)
    table = pq.read_table(descriptors)
    column = table.schema.get_field_index("total_two_j_plus_one")
    table = table.set_column(
        column, table.schema.field(column), pa.array([0, 3], type=pa.int8())
    )
    metadata = dict(pq.read_metadata(descriptors).metadata)
    metadata.pop(b"ARROW:schema", None)
    invalid = tmp_path / "invalid.parquet"
    pq.write_table(table.replace_schema_metadata(metadata), invalid)
    output = tmp_path / "rejected.c"
    with pytest.raises((ValueError, RuntimeError, OSError), match="code must be in 1..=127"):
        restore_csfs_from_descriptors(invalid, header, output, indices=indices)
    assert not output.exists()


def test_out_of_range_generation_does_not_publish_partial_descriptors(
    tmp_path: Path,
) -> None:
    source, header, output = _inputs(tmp_path, large_j=True)
    with pytest.raises((ValueError, RuntimeError, OSError), match="angular bound"):
        convert_csfs(source, output)
    assert not output.exists() and not header.exists()
