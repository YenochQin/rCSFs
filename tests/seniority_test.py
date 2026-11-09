from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rcsfs import (
    convert_csfs,
    generate_descriptors_from_parquet,
    read_peel_subshells,
    restore_csfs_from_descriptors,
    seniority_to_printed_v,
)


@pytest.mark.parametrize(
    ("two_j", "occupation", "subshell_two_j", "seniority", "expected"),
    [
        (7, 4, 8, 2, 2),
        (7, 4, 8, 4, 4),
        (7, 4, 12, 2, -1),
        (7, 4, 10, 4, -1),
        (9, 4, 0, 0, 0),
        (3, 1, 3, 1, -1),
        (3, 3, 3, 1, -1),
        (3, 0, 0, 0, -1),
    ],
)
def test_seniority_to_printed_v(
    two_j: int,
    occupation: int,
    subshell_two_j: int,
    seniority: int,
    expected: int,
) -> None:
    assert (
        seniority_to_printed_v(two_j, occupation, subshell_two_j, seniority) == expected
    )


def test_seniority_to_printed_v_rejects_illegal_state() -> None:
    with pytest.raises(ValueError, match="illegal physical seniority"):
        seniority_to_printed_v(7, 4, 12, 4)


def _descriptors(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.c"
    source.write_text(
        "Core subshells:\n  1s\nPeel subshells:\n  2s  2p  3s  4f\nCSF(s):\n"
        "  2s ( 1)  2p ( 2)  3s ( 1)\n"
        "      1/2        0      1/2\n"
        "                           0+\n *\n"
        "  4f ( 4)\n   2;   4\n         4+\n"
        "  4f ( 4)\n   4;   4\n         4+\n *\n"
        "  4f ( 4)\n        6\n         6+\n *\n"
        "  4f ( 4)\n        5\n         5+\n"
    )
    csf_parquet = tmp_path / "source.parquet"
    stats = convert_csfs(source, csf_parquet, num_workers=1)
    header = Path(stats["header_file"])
    descriptors = tmp_path / "descriptor.parquet"
    generate_descriptors_from_parquet(
        csf_parquet,
        descriptors,
        read_peel_subshells(header),
        num_workers=1,
        header_path=header,
    )
    return source, header, descriptors


def test_physical_seniority_values_and_exact_text_restoration(tmp_path: Path) -> None:
    source, header, descriptors = _descriptors(tmp_path)
    table = pq.read_table(descriptors)
    assert pq.read_metadata(descriptors).metadata[b"seniority_encoding"] == b"physical"
    assert [table[f"sub{i}_v"].to_pylist() for i in range(4)] == [
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 2, 4, 2, 4],
    ]
    output = tmp_path / "restored.c"
    restore_csfs_from_descriptors(descriptors, header, output)
    assert output.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("encoding", [None, b"printed"])
@pytest.mark.parametrize("indices", [None, [0]])
def test_restore_rejects_legacy_encoding_before_output(
    tmp_path: Path, encoding: bytes | None, indices: list[int] | None
) -> None:
    _, header, descriptors = _descriptors(tmp_path)
    table = pq.read_table(descriptors)
    metadata = dict(pq.read_metadata(descriptors).metadata)
    metadata.pop(b"ARROW:schema", None)
    if encoding is None:
        del metadata[b"seniority_encoding"]
    else:
        metadata[b"seniority_encoding"] = encoding
    legacy = tmp_path / "legacy.parquet"
    pq.write_table(table.replace_schema_metadata(metadata), legacy)
    output = tmp_path / "rejected.c"
    with pytest.raises(
        (ValueError, RuntimeError, OSError), match="seniority_encoding=physical"
    ):
        restore_csfs_from_descriptors(legacy, header, output, indices=indices)
    assert not output.exists()


def test_restore_rejects_negative_physical_seniority(tmp_path: Path) -> None:
    _, header, descriptors = _descriptors(tmp_path)
    table = pq.read_table(descriptors)
    column = table.schema.get_field_index("sub0_v")
    table = table.set_column(
        column, table.schema.field(column), pa.array([-1, 0, 0, 0, 0], type=pa.int8())
    )
    metadata = dict(pq.read_metadata(descriptors).metadata)
    metadata.pop(b"ARROW:schema", None)
    invalid = tmp_path / "invalid.parquet"
    pq.write_table(table.replace_schema_metadata(metadata), invalid)
    output = tmp_path / "rejected.c"
    with pytest.raises((ValueError, RuntimeError, OSError), match="physical seniority"):
        restore_csfs_from_descriptors(invalid, header, output)
    assert not output.exists()
