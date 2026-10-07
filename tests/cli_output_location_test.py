"""Every CLI command publishes locally even when its inputs live elsewhere."""

import shutil
from pathlib import Path

import pytest

from rcsfs import cli

pytestmark = pytest.mark.usefixtures("cli_cwd")


def test_all_processing_outputs_and_scratch_stay_in_invocation_directory(
    tmp_path: Path,
) -> None:
    config = tmp_path / "rcsfs.toml"
    config.write_text(
        'conf = "calc_"\n[csfsgenerate]\nas = 0\ninactive_core = 0\n'
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 0\ngenerate_descriptors = true\n"
        '[csfs-split]\nactive_spaces = ["AS0=1s"]\n',
        encoding="utf-8",
    )
    assert cli.main(["csfsgenerate", "-c"]) == 0
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for path in tmp_path.glob("calc_*"):
        shutil.copy2(path, inputs / path.name)
    shutil.copy2(inputs / "calc_as0raw.c", inputs / "reference.c")
    original = {path.name: path.read_bytes() for path in inputs.iterdir()}

    parquet = inputs / "calc_as0raw.parquet"
    header = inputs / "calc_as0raw_header.toml"
    descriptors = inputs / "calc__desc.parquet"
    source = inputs / "calc_as0raw.c"
    assert (
        cli.main(
            [
                "gen-descriptors",
                str(parquet),
                "rebuilt.parquet",
                "--header",
                str(header),
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "restore-csfs",
                "--descriptors",
                str(descriptors),
                "--header",
                str(header),
                "--output",
                "restored.c",
            ]
        )
        == 0
    )
    assert cli.main(["zero-first", str(source), str(source)]) == 0
    assert (
        cli.main(
            [
                "interacting",
                str(inputs / "reference.c"),
                str(source),
                "--output",
                "selected.c",
            ]
        )
        == 0
    )
    assert cli.main(["csfs-split", str(parquet), "--header", str(header)]) == 0

    for name in (
        "rebuilt.parquet",
        "rebuilt.toml",
        "restored.c",
        "calc_as0raw_zf.csf",
        "selected.c",
        "calc_as0raw.c",
    ):
        assert (tmp_path / name).is_file()
    assert {path.name: path.read_bytes() for path in inputs.iterdir()} == original
    assert not (tmp_path / "split").exists()
    assert not list(tmp_path.glob("rcsfs-*/"))


@pytest.mark.parametrize(
    "command, table",
    [
        (
            "csfsgenerate",
            '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\nactive_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\nrcsfs_out = "elsewhere/out.c"\n',
        ),
        (
            "gen-descriptors",
            '[gen-descriptors]\ninput_parquet = "input.parquet"\nheader = "header.toml"\noutput_parquet = "elsewhere/out.parquet"\n',
        ),
        (
            "restore-csfs",
            '[restore-csfs]\ndescriptors = "desc.parquet"\nheader = "header.toml"\noutput = "elsewhere/out.c"\n',
        ),
        (
            "zero-first",
            '[zero-first]\nzero_csf = "zero.c"\nfull_csf = "full.c"\noutput_csf = "elsewhere/out.c"\n',
        ),
        (
            "interacting",
            '[interacting]\nreference = "ref.c"\ncandidates = "all.c"\noutput = "elsewhere/out.c"\n',
        ),
    ],
)
def test_external_output_paths_fail_before_processing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, table: str
) -> None:
    (tmp_path / "rcsfs.toml").write_text(table, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.main([command, "-c", "rcsfs.toml"])
    assert exc.value.code == 2
    assert "current directory" in capsys.readouterr().err
    assert not (tmp_path / "elsewhere").exists()


@pytest.mark.parametrize(
    "tokens",
    [
        ["csfs-split", "--output-dir", "split"],
        ["zero-first", "--work-dir", "scratch"],
        ["zero-first", "--keep-parquet"],
    ],
)
def test_removed_directory_and_retention_flags_are_rejected(tokens: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(tokens)
