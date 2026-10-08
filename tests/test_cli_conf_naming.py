"""Shared raw filenames stay aligned across generation and active-space splitting."""

from pathlib import Path

import pytest

from rcsfs import cli
from rcsfs._cli_config import parse_cli_args


@pytest.mark.parametrize("conf", ["e1_cv1", "e1_cv1_"])
@pytest.mark.parametrize("explicit_split_sources", [False, True])
def test_batch_generation_and_split_share_conf_as_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    conf: str,
    explicit_split_sources: bool,
) -> None:
    split_sources = (
        'split_csfs_parquet = "e1_cv1_as6raw.parquet"\n'
        'csfs_header = "e1_cv1_as6raw_header.toml"\n'
        if explicit_split_sources
        else ""
    )
    contents = (
        f'conf = "{conf}"\n'
        "[[csfsgenerate]]\nas = 0\ninactive_core = 0\n"
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 0\n"
        'generation_storage = "memory"\ngenerate_descriptors = false\n'
        "[[csfsgenerate]]\nas = 6\ninactive_core = 0\n"
        'reference_configuration = ["1s(2,*)"]\nactive_space = "2s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 2\n"
        'generation_storage = "memory"\ngenerate_descriptors = false\n'
        "[csfs-split]\n"
        + split_sources
        + 'active_spaces = ["AS1=1s", "AS2=2s", "AS3=2s", '
        '"AS4=2s", "AS5=2s", "AS6=2s"]\n'
    )
    config = tmp_path / "rcsfs.toml"
    config.write_text(contents, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    generated = parse_cli_args(
        cli.build_parser(), ["csfsgenerate", "-c"], config_index=0
    )
    assert generated.rcsfs_out == Path("e1_cv1_as0raw.c")
    split = parse_cli_args(cli.build_parser(), ["csfs-split", "-c", "rcsfs.toml"])
    assert split.split_csfs_parquet == Path("e1_cv1_as6raw.parquet")
    assert split.csfs_header == Path("e1_cv1_as6raw_header.toml")

    assert cli.main(["-c", "rcsfs.toml"]) == 0
    for level in range(7):
        assert (tmp_path / f"e1_cv1_as{level}raw.c").is_file()
    assert (tmp_path / "e1_cv1_as6raw.parquet").is_file()
    assert (tmp_path / "e1_cv1_as6raw_header.toml").is_file()
    assert not list(tmp_path.glob("e1_cv1as*"))
    assert not list(tmp_path.glob("e1_cv1__as*"))
    assert not list(tmp_path.glob("*desc*.parquet"))
    assert config.read_text(encoding="utf-8") == contents
