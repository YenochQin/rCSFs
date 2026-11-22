"""Cross-entry-point acceptance rows from the V3 format specification."""

import hashlib
import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest
from _native_cli import main

from rcsfs import (
    DESCRIPTOR_VERSION,
    convert_csfs,
    descriptor_header_path,
    match_csfs,
    merge_csfs,
    read_csfs,
    restore_csfs_from_descriptors,
    select_csfs,
)

FIXTURE = Path(__file__).parent / "fixtures" / "descriptor_v3.c"
A_ZERO = (2, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1)
A_ONE = (2, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 3, 1, 1, 2, 2, 2, 1)
B = (2, 0, 1, 1, 2, 0, 1, 1, 1, 1, 2, 2, 0, 0, 1, 2, 2, 1)


@pytest.mark.parametrize(("last_shell", "sign"), [("15m-", "-"), ("15l", "+")])
@pytest.mark.parametrize("total_two_j", [99, 101, 125])
def test_five_character_total_j_roundtrip(
    tmp_path: Path, last_shell: str, sign: str, total_two_j: int
) -> None:
    # Six j=19/2 subshells couple to K=57. The final j=17/2 admits all three targets.
    shells = ["10m", "11m", "12m", "13m", "14m", "15m", last_shell]
    source = tmp_path / "high.c"
    source.write_text(
        "Core subshells:\n\nPeel subshells:\n  "
        + "  ".join(shells)
        + "\nCSF(s):\n"
        + "".join(f"{shell:>4} ( 1)" for shell in shells)
        + "\n"
        + "     19/2" * 6
        + "     17/2\n"
        + "                   19     57/2       38     95/2       57"
        + f"{str(total_two_j) + '/2':>7}{sign}\n",
        encoding="ascii",
        newline="\n",
    )
    first = tmp_path / "high_desc.parquet"
    convert_csfs(source, first)
    text = tmp_path / "restored.c"
    restore_csfs_from_descriptors(first, descriptor_header_path(first), text)
    second = tmp_path / "again_desc.parquet"
    convert_csfs(text, second)
    frame = read_csfs(first)[1]
    assert frame.select("total_two_j_plus_one", "parity").row(0) == (
        total_two_j + 1,
        1 if sign == "+" else -1,
    )
    assert frame.equals(read_csfs(second)[1])


@pytest.mark.parametrize("occupation", [4, 6])
@pytest.mark.parametrize("angular", [4, 6])
def test_ambiguous_jj_term_requires_w(
    tmp_path: Path, occupation: int, angular: int
) -> None:
    source = tmp_path / "ambiguous.c"
    source.write_text(
        f"Core subshells:\n\nPeel subshells:\n  5g\nCSF(s):\n  5g ({occupation:2})\n   4;{angular:4}\n{angular:9}+\n",
        encoding="ascii",
        newline="\n",
    )
    output = tmp_path / "ambiguous_desc.parquet"
    with pytest.raises((ValueError, OSError), match="additional w label"):
        convert_csfs(source, output)
    assert not output.exists()


def test_spec_rows_and_canonical_round_trip(tmp_path: Path) -> None:
    descriptor = tmp_path / "basis_desc.parquet"
    convert_csfs(FIXTURE, descriptor)
    header, frame = read_csfs(
        descriptor, include_block_id=True, include_coupling_signature=True
    )
    assert DESCRIPTOR_VERSION == 3
    assert header["header_info"]["electron_count"] == 5
    assert header["block_info"]["block_lengths"] == [2, 1]
    assert frame["block_id"].to_list() == [0, 0, 1]
    assert frame["coupling_signature"].dtype == pl.List(pl.Int32)
    assert frame["coupling_signature"].to_list() == [
        [0, 1, 0, 1],
        [0, 1, 2, 1],
        [0, 0, 1],
    ]
    assert frame.drop("idx", "block_id", "coupling_signature").rows() == [
        A_ZERO,
        A_ONE,
        B,
    ]

    selected = tmp_path / "selected_desc.parquet"
    select_csfs(descriptor, selected, [1, 2])
    restored = tmp_path / "selected.c"
    restore_csfs_from_descriptors(selected, descriptor_header_path(selected), restored)
    imported = tmp_path / "imported_desc.parquet"
    convert_csfs(restored, imported)
    assert read_csfs(imported, include_block_id=True)[1].drop(
        "idx", "block_id"
    ).rows() == [A_ONE, B]
    assert read_csfs(imported)[0]["block_info"]["block_lengths"] == [1, 1]


def test_generation_backends_match_each_spec_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = []
    for backend in ["memory", "disk"]:
        descriptor = tmp_path / f"{backend}_desc.parquet"
        config = tmp_path / f"{backend}.toml"
        config.write_text(
            "[csfsgenerate]\ninactive_core = 0\n"
            'reference_configuration = ["1s(2,i)2s(1,i)3s(1,i)4s(1,i)", "1s(2,i)2s(2,i)3s(1,i)"]\n'
            'active_space = "4s"\nj_min = 1\nj_max = 1\nexcitations = 0\ndescriptor_version = 3\n'
            f'generation_storage = "{backend}"\noutput = {json.dumps(str(descriptor))}\n',
            encoding="utf-8",
        )
        assert main(["csfsgenerate", "--config", str(config), "--threads", "2"]) == 0
        paths.append(descriptor)
    memory = read_csfs(paths[0])[1].drop("idx").rows()
    disk = read_csfs(paths[1])[1].drop("idx").rows()
    assert disk == memory
    assert sorted(disk) == sorted([A_ZERO, A_ONE, B])
    assert disk.index(A_ZERO) < disk.index(A_ONE)


def test_ambiguous_text_reports_block_row_and_subshell(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous.c"
    data = FIXTURE.read_bytes()
    first_coupling = data.splitlines()[7]
    source.write_bytes(
        data.replace(first_coupling, first_coupling[:29] + b" " + first_coupling[30:])
    )
    descriptor = tmp_path / "ambiguous_desc.parquet"
    with pytest.raises(OSError, match=r"block 1, row 1: .*multiple chains at 3s"):
        convert_csfs(source, descriptor)
    assert not descriptor.exists()
    assert not descriptor_header_path(descriptor).exists()


def test_same_width_v2_metadata_is_rejected(tmp_path: Path) -> None:
    descriptor = tmp_path / "basis_desc.parquet"
    convert_csfs(FIXTURE, descriptor)
    table = pq.read_table(descriptor)
    metadata = dict(table.schema.metadata or {})
    metadata[b"descriptor_version"] = b"2"
    legacy = tmp_path / "old_desc.parquet"
    pq.write_table(table.replace_schema_metadata(metadata), legacy)
    with pytest.raises(OSError, match=r"descriptor_version=2.*regenerate V3"):
        read_csfs(legacy, header_path=descriptor_header_path(descriptor))


def test_remap_inherits_cumulative_code_at_head_middle_and_tail(tmp_path: Path) -> None:
    records = b"  2s ( 1)  4s ( 1)\n      1/2      1/2\n                  1+\n"
    paths = []
    for name, peel in [("narrow", "2s 4s"), ("wide", "1s 2s 3s 4s 5s")]:
        source = tmp_path / f"{name}.c"
        source.write_bytes(
            f"Core subshells:\n\nPeel subshells:\n{peel}\nCSF(s):\n".encode() + records
        )
        descriptor = tmp_path / f"{name}_desc.parquet"
        convert_csfs(source, descriptor)
        paths.append(descriptor)
    narrow, wide = paths
    matched = match_csfs(wide, narrow)
    assert matched == [0]
    merged = tmp_path / "merged_desc.parquet"
    merge_csfs([narrow, wide], merged)
    expected = (0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 1, 2, 1, 1, 2, 3, 0, 0, 1, 3, 3, 1)
    assert read_csfs(merged)[1].drop("idx").rows() == [expected, expected]


@pytest.mark.parametrize("backend", ["memory", "disk"])
def test_pure_closed_core_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "core.toml"
    config.write_text(
        '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,c)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        f'generation_storage = "{backend}"\noutput = "core_desc.parquet"\n',
        encoding="utf-8",
    )
    assert (
        main(["csfsgenerate", "--config", str(config), "--descriptor-version", "3"])
        == 0
    )
    header, rows = read_csfs(
        tmp_path / "core_desc.parquet", include_coupling_signature=True
    )
    assert header["header_info"]["electron_count"] == 2
    assert rows.drop("idx", "coupling_signature").rows() == [(1, 1)]
    assert rows["coupling_signature"].to_list() == [[0]]


def test_cli_restoration_keeps_odd_parity_and_original_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "odd.c"
    source.write_bytes(
        b"Core subshells:\n\nPeel subshells:\n  1s  2p-\nCSF(s):\n"
        b"  1s ( 1)\n      1/2\n      1/2+\n *\n"
        b"  2p-( 1)\n      1/2\n      1/2-\n"
    )
    descriptor = tmp_path / "odd_desc.parquet"
    assert main(["convert", source.name, descriptor.name]) == 0
    assert (
        main(
            [
                "restore-csfs",
                "--descriptors",
                descriptor.name,
                "--header",
                "odd_header.toml",
                "--output",
                "selected.c",
                "--indices",
                "1",
            ]
        )
        == 0
    )
    restored = tmp_path / "selected_desc.parquet"
    convert_csfs(tmp_path / "selected.c", restored)
    assert read_csfs(restored)[1].drop("idx").rows() == [
        (0, 0, 1, 1, 1, 1, 2, 2, 2, -1)
    ]


def test_open_subshell_j_zero_inherits_the_preceding_k(tmp_path: Path) -> None:
    paths = []
    for name, middle in [("printed", "0"), ("inferred", "")]:
        source = tmp_path / f"{name}.c"
        source.write_bytes(
            (
                "Core subshells:\n\nPeel subshells:\n  1s   2p   3s\nCSF(s):\n"
                "  1s ( 1)  2p ( 2)  3s ( 1)\n"
                f"      1/2{middle:>9}      1/2\n                          0+\n"
            ).encode()
        )
        descriptor = tmp_path / f"{name}_desc.parquet"
        convert_csfs(source, descriptor)
        paths.append(descriptor)
    for path in paths:
        _, frame = read_csfs(path, include_coupling_signature=True)
        assert frame.drop("idx", "coupling_signature").rows() == [
            (1, 1, 2, 2, 2, 0, 1, 2, 1, 1, 2, 1, 1, 1)
        ]
        assert frame["coupling_signature"].to_list() == [[1, 0, 0]]


def test_backends_keep_orbitals_without_target_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for backend in ("memory", "disk"):
        config = tmp_path / f"{backend}.toml"
        config.write_text(
            '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,i)", "2p(2,i)"]\n'
            'active_space = "1s,2p"\nj_min = 2\nj_max = 2\nexcitations = 0\n'
            f'generation_storage = "{backend}"\noutput = "{backend}_desc.parquet"\n',
            encoding="utf-8",
        )
        assert main(["csfsgenerate", "--config", str(config)]) == 0
    memory_header, memory = read_csfs(tmp_path / "memory_desc.parquet")
    disk_header, disk = read_csfs(tmp_path / "disk_desc.parquet")
    for backend, header in [("memory", memory_header), ("disk", disk_header)]:
        descriptor = tmp_path / f"{backend}_desc.parquet"
        assert (
            header["descriptor_sha256"]
            == hashlib.sha256(descriptor.read_bytes()).hexdigest()
        )
    assert memory_header["block_info"] == disk_header["block_info"]
    assert memory_header["header_info"] == disk_header["header_info"]
    assert memory_header["conversion_stats"] == disk_header["conversion_stats"]
    assert memory.equals(disk)
    assert memory["sub0_occupation"].to_list() == [0]
