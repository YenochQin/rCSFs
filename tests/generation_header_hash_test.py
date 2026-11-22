"""Generated descriptor bindings survive CLI publication and reject stale headers."""

import hashlib
import tomllib
from pathlib import Path

import _native_cli as cli
import pytest
import pyarrow.parquet as pq

from rcsfs import (
    descriptor_header_path,
    generate_csfs_from_transcript,
    get_parquet_info,
    restore_csfs_from_descriptors,
)

pytestmark = pytest.mark.usefixtures("cli_cwd")


def _assert_binding(descriptors: Path, header: Path) -> str:
    metadata = get_parquet_info(descriptors)["key_value_metadata"]
    raw = header.read_bytes()
    expected = hashlib.sha256(raw.split(b"\n", 1)[1]).hexdigest()
    document = tomllib.loads(raw.decode())
    assert (
        document["descriptor_sha256"]
        == hashlib.sha256(descriptors.read_bytes()).hexdigest()
    )
    symmetries = pq.read_table(
        descriptors, columns=["total_two_j_plus_one", "parity"]
    ).to_pylist()
    start = 0
    for block, length in zip(
        document["block_info"]["blocks"],
        document["block_info"]["block_lengths"],
        strict=True,
    ):
        assert (block["start_row"], block["length"]) == (start, length)
        assert all(
            (row["total_two_j_plus_one"] - 1, row["parity"])
            == (block["two_j"], block["parity"])
            for row in symmetries[start : start + length]
        )
        start += length
    assert start == len(symmetries)
    assert metadata["source_header_sha256"] == expected
    assert metadata["source_header_filename"] == header.name
    return expected


@pytest.mark.parametrize("storage", ["disk", "memory"])
@pytest.mark.parametrize("core", [0, 1])
@pytest.mark.parametrize(
    "descriptor_setting, descriptor_args, descriptor_name",
    [
        ("", [], "calc_desc.parquet"),
        ('output = "custom.parquet"\n', [], "custom.parquet"),
        (
            'output = "ignored.parquet"\n',
            ["--output", "override.parquet"],
            "override.parquet",
        ),
    ],
    ids=["default", "toml", "cli"],
)
def test_generated_descriptor_binding_survives_cli_publication(
    tmp_path: Path,
    storage: str,
    core: int,
    descriptor_setting: str,
    descriptor_args: list[str],
    descriptor_name: str,
) -> None:
    config = tmp_path / "rcsfs.toml"
    config.write_text(
        'conf = "calc"\n[csfsgenerate]\n'
        f'inactive_core = {core}\nreference_configuration = ["2s(2,*)"]\n'
        'active_space = "3s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        f'generation_storage = "{storage}"\nthreads = 2\n' + descriptor_setting,
        encoding="utf-8",
    )
    arguments = (
        ["csfsgenerate", "-c", str(config), *descriptor_args]
        if descriptor_args
        else ["-c", str(config)]
    )
    assert cli.main(arguments) == 0
    descriptors = tmp_path / descriptor_name
    header = descriptor_header_path(descriptors)
    original_hash = _assert_binding(descriptors, header)
    assert tomllib.loads(header.read_text())["header_info"]["electron_count"] == (
        2 + 2 * core
    )
    assert not list(tmp_path.glob("*.c"))
    restored = tmp_path / "restored.c"
    restore_csfs_from_descriptors(descriptors, header, restored)
    assert restored.is_file()

    # Regenerating the same filenames with a different source must refresh the binding.
    config.write_text(
        config.read_text().replace("2s(2,*)", "3s(2,*)"), encoding="utf-8"
    )
    assert cli.main(arguments) == 0
    assert _assert_binding(descriptors, header) != original_hash
    original_csf = restored.read_bytes()
    restored = tmp_path / "regenerated_restored.c"
    restore_csfs_from_descriptors(descriptors, header, restored)
    assert restored.read_bytes() != original_csf


@pytest.mark.parametrize("multiple_lists", [False, True])
@pytest.mark.parametrize("core", [0, 1])
def test_disk_api_binds_completed_header_and_rejects_tampering(
    tmp_path: Path, multiple_lists: bool, core: int
) -> None:
    first = f"* ! Orbital order\n{core}\n2s(2,*)\n\n3s\n0,0\n0\nn\n"
    second = f"* ! Orbital order\n{core}\n3s(2,*)\n\n3s\n0,0\n0\nn\n"
    descriptors = tmp_path / "desc.parquet"
    header = descriptor_header_path(descriptors)
    stats = generate_csfs_from_transcript(
        [first, second] if multiple_lists else first, descriptors, threads=2
    )
    assert stats["success"] is True
    _assert_binding(descriptors, header)
    electrons = 2 + 2 * core
    assert (
        tomllib.loads(header.read_text())["header_info"]["electron_count"] == electrons
    )
    restored = tmp_path / "restored.c"
    restore_csfs_from_descriptors(descriptors, header, restored)
    assert "2s ( 2)" in restored.read_text()

    # The electron count is part of the exact header bytes bound to the descriptors.
    header.write_bytes(
        header.read_bytes().replace(
            f"electron_count = {electrons}".encode(),
            f"electron_count = {electrons + 1}".encode(),
        )
    )
    for label, indices in (("all", None), ("subset", [0])):
        output = tmp_path / f"stale_{label}.c"
        with pytest.raises(OSError, match="does not match the hash"):
            restore_csfs_from_descriptors(descriptors, header, output, indices=indices)
        assert not output.exists()
