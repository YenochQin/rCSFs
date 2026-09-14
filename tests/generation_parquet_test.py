"""Generation CLI output safety and metadata regressions."""

import json
import tomllib
from pathlib import Path

import polars as pl
import pytest

from rcsfs import cli, read_peel_subshells
from rcsfs._rcsfs import generate_csfs_from_transcript


def config_file(tmp_path: Path, **outputs: object) -> Path:
    config = tmp_path / "generation.toml"
    values = {
        "csf": str(tmp_path / "out.c"),
        "generate_descriptors": True,
        "normalize": False,
        **outputs,
    }
    config.write_text(
        '[generate]\norder = "*"\ncore = 0\nreferences = ["1s(2,*)"]\n'
        'active_orbitals = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        "continue_lists = false\n[output]\n"
        + "\n".join(f"{key} = {json.dumps(value)}" for key, value in values.items())
    )
    return config


@pytest.mark.parametrize("normalize", [False, True])
def test_config_generation_metadata(tmp_path: Path, normalize: bool) -> None:
    config = config_file(tmp_path, normalize=normalize)
    assert cli.main(["csfsgenerate", "--config", str(config)]) == 0
    frame = pl.read_parquet(tmp_path / "out_descriptors.parquet")
    metadata = tomllib.loads((tmp_path / "out_descriptors.toml").read_text())
    assert metadata == {
        "format_version": 1,
        "encoding": "parquet",
        "normalized": normalize,
        "record_count": frame.height,
        "subshells": read_peel_subshells(tmp_path / "out_header.toml"),
    }
    assert frame.height == 1
    assert frame.width == 3 * len(metadata["subshells"])
    assert frame.dtypes == [pl.Float32 if normalize else pl.Int32] * frame.width


@pytest.mark.parametrize(
    "name",
    [
        "out.c",
        "out.parquet",
        "out_header.toml",
        "out_descriptors.parquet",
        "out_descriptors.toml",
    ],
)
def test_existing_output_preserved(tmp_path: Path, name: str) -> None:
    config = config_file(tmp_path)
    sentinel = tmp_path / name
    sentinel.write_bytes(b"keep me")
    assert cli.main(["csfsgenerate", "--config", str(config)]) == 1
    assert sentinel.read_bytes() == b"keep me"
    assert set(tmp_path.iterdir()) == {config, sentinel}


@pytest.mark.parametrize("output", ["out.c", "out_header.toml", "generation.toml"])
def test_output_aliases_rejected(tmp_path: Path, output: str) -> None:
    config = config_file(tmp_path, descriptor_parquet=str(tmp_path / output))
    before = config.read_bytes()
    assert cli.main(["csfsgenerate", "--config", str(config)]) == 1
    assert config.read_bytes() == before
    assert set(tmp_path.iterdir()) == {config}


def test_symlink_output_rejected(tmp_path: Path) -> None:
    config = config_file(tmp_path)
    target = tmp_path / "missing"
    (tmp_path / "out.parquet").symlink_to(target)
    assert cli.main(["csfsgenerate", "--config", str(config)]) == 1
    assert not target.exists()
    assert not (tmp_path / "out.c").exists()


def test_publication_race_preserves_other_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_file(tmp_path)
    original = cli.generate_descriptors_from_parquet
    sentinel = tmp_path / "out.parquet"

    def racing_writer(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        sentinel.write_bytes(b"another writer")
        return result

    monkeypatch.setattr(cli, "generate_descriptors_from_parquet", racing_writer)
    assert cli.main(["csfsgenerate", "--config", str(config)]) == 1
    assert sentinel.read_bytes() == b"another writer"


def test_extension_positional_signature(tmp_path: Path) -> None:
    transcript = "* ! Orbital order\n0\n1s(2,*)\n\n1s\n0,0\n0\nn\n"
    assert generate_csfs_from_transcript(transcript, str(tmp_path / "out.c"), False, 1)[
        "success"
    ]
    with pytest.raises(TypeError):
        generate_csfs_from_transcript(
            transcript, str(tmp_path / "other.c"), descriptor_path=None
        )
