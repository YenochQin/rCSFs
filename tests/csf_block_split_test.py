"""Split and independently restore explicit descriptor blocks."""

import json
from pathlib import Path

import _native_cli as cli
import pytest

from rcsfs import convert_csfs, read_csfs, split_csfs_by_j

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    output = tmp_path / "basis_descriptors.parquet"
    convert_csfs(FIXTURES / "descriptor_multiblock.c", output)
    return output


def test_split_api_keeps_the_explicit_block_rows(dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "split"
    output.mkdir()
    stats = split_csfs_by_j(dataset, output_dir=output, prefix="slice")
    names = [
        "slice_b000_0_even_descriptors.parquet",
        "slice_b001_2_odd_descriptors.parquet",
    ]
    assert [Path(item["output_file"]).name for item in stats["outputs"]] == names
    _, input_rows = read_csfs(dataset)
    for row, name in enumerate(names):
        header, frame = read_csfs(output / name)
        assert header["block_info"]["block_lengths"] == [1]
        assert frame.drop("idx").equals(input_rows.slice(row, 1).drop("idx"))
    assert not list(output.glob("*.c"))


def test_existing_companion_preserved_without_partial_publication(
    dataset: Path, tmp_path: Path
) -> None:
    sentinel = tmp_path / "slice_b001_2_odd_header.toml"
    sentinel.write_bytes(b"sentinel")
    with pytest.raises(OSError, match="already exists"):
        split_csfs_by_j(dataset, output_dir=tmp_path, prefix="slice")
    assert list(tmp_path.glob("slice*")) == [sentinel]
    assert sentinel.read_bytes() == b"sentinel"


def test_cli_later_header_conflict_preserves_all_destinations(
    dataset: Path, tmp_path: Path
) -> None:
    (tmp_path / "basis.w").write_bytes(b"orbitals")
    sentinel = tmp_path / "slice_b001_2_odd_header.toml"
    sentinel.write_bytes(b"previous header")
    args = ["jsplit", str(dataset), "--prefix", "slice", "--json"]
    failed = cli.run(args, cwd=tmp_path)
    assert failed.returncode != 0
    assert "already exists" in failed.stderr
    assert list(tmp_path.glob("slice*")) == [sentinel]
    assert sentinel.read_bytes() == b"previous header"
    sentinel.unlink()
    result = cli.run(args, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    for item in json.loads(result.stdout)["outputs"]:
        assert read_csfs(tmp_path / item["output_file"])[1].height == 1
        assert (tmp_path / item["w_file"]).read_bytes() == b"orbitals"


@pytest.mark.parametrize("copy", [False, True])
def test_cli_jsplit_config_copies_wavefunction_only_when_requested(
    dataset: Path, tmp_path: Path, copy: bool
) -> None:
    wave = tmp_path / "basis.w"
    wave.write_bytes((FIXTURES / "multi_j.w").read_bytes())
    config = tmp_path / "rcsfs.toml"
    config.write_text(
        '[jsplit]\ninput_csf = "basis_descriptors.parquet"\nprefix = "slice"\n'
        f"copy_w = {str(copy).lower()}\n"
    )
    result = cli.run(["jsplit", "-c", str(config), "--json"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    outputs = json.loads(result.stdout)["outputs"]
    assert len(outputs) == 2
    for block in outputs:
        assert read_csfs(tmp_path / block["output_file"])[1].height == 1
        if copy:
            assert (tmp_path / block["w_file"]).read_bytes() == wave.read_bytes()
        else:
            assert "w_file" not in block
