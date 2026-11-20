"""Every CLI command publishes locally even when its inputs live elsewhere."""

import shutil
from pathlib import Path

import _native_cli as cli
import pytest

pytestmark = pytest.mark.usefixtures("cli_cwd")


def test_all_processing_outputs_and_scratch_stay_in_invocation_directory(
    tmp_path: Path,
) -> None:
    config = tmp_path / "rcsfs.toml"
    config.write_text(
        'conf = "calc_"\n[csfsgenerate]\nas = 0\ninactive_core = 0\n'
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 0\n"
        '[csfs-split]\nactive_spaces = ["AS1=1s"]\n',
        encoding="utf-8",
    )
    assert cli.main(["csfsgenerate", "-c"]) == 0
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for path in tmp_path.glob("calc_*"):
        shutil.copy2(path, inputs / path.name)
    descriptors = inputs / "calc_as0raw_desc.parquet"
    header = inputs / "calc_as0raw_header.toml"
    shutil.copy2(descriptors, inputs / "reference_descriptors.parquet")
    shutil.copy2(header, inputs / "reference_header.toml")
    original = {path.name: path.read_bytes() for path in inputs.iterdir()}
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
    assert cli.main(["zero-first", str(descriptors), str(descriptors)]) == 0
    assert (
        cli.main(
            [
                "interacting",
                str(inputs / "reference_descriptors.parquet"),
                str(descriptors),
                "--output",
                "selected_descriptors.parquet",
            ]
        )
        == 0
    )
    assert cli.main(["csfs-split", str(descriptors), "--header", str(header)]) == 0

    for name in (
        "restored.c",
        "zero_first_desc.parquet",
        "selected_descriptors.parquet",
        "calc_as1raw_desc.parquet",
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
            '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\nactive_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\noutput = "elsewhere/out_descriptors.parquet"\n',
        ),
        (
            "convert",
            '[convert]\ninput_csf = "input.c"\noutput_parquet = "elsewhere/out.parquet"\n',
        ),
        (
            "restore-csfs",
            '[restore-csfs]\ndescriptors = "desc.parquet"\nheader = "header.toml"\noutput = "elsewhere/out.c"\n',
        ),
        (
            "zero-first",
            '[zero-first]\nzero_csf = "zero_descriptors.parquet"\nfull_csf = "full_descriptors.parquet"\noutput_csf = "elsewhere/out_descriptors.parquet"\n',
        ),
        (
            "interacting",
            '[interacting]\nreference = "ref_descriptors.parquet"\ncandidates = "all_descriptors.parquet"\noutput = "elsewhere/out_descriptors.parquet"\n',
        ),
    ],
)
def test_external_output_paths_fail_before_processing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, table: str
) -> None:
    (tmp_path / "rcsfs.toml").write_text(table, encoding="utf-8")
    assert cli.main([command, "-c", "rcsfs.toml"]) == 1
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
    assert cli.main(tokens) == 2
