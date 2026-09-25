"""The single CLI configuration boundary, including all command surfaces."""

from pathlib import Path

import pytest

from rcsfs import cli
from rcsfs._cli_config import parse_cli_args


@pytest.mark.parametrize(
    ("command", "table", "expected"),
    [
        (
            "gen-descriptors",
            '[gen-descriptors]\ninput_parquet = "input.parquet"\noutput_parquet = "out.parquet"\nheader = "input_header.toml"\nnum_workers = 4\n',
            {"input_parquet": Path("input.parquet"), "num_workers": 4},
        ),
        (
            "zero-first",
            '[zero-first]\nzero_csf = "zero.c"\nfull_csf = "full.c"\nkeep_parquet = true\n',
            {"zero_csf": Path("zero.c"), "keep_parquet": True},
        ),
        (
            "csfs-split",
            '[csfs-split]\nsplit_csfs_parquet = "all.parquet"\ncsfs_header = "all_header.toml"\nactive_spaces = ["as1=5s,4p", "as2=6s,5p"]\noutput_dir = "split"\n',
            {
                "split_csfs_parquet": Path("all.parquet"),
                "csfs_header": Path("all_header.toml"),
                "active_spaces": ["as1=5s,4p", "as2=6s,5p"],
                "output_dir": Path("split"),
            },
        ),
        (
            "interacting",
            '[interacting]\nreference = "mr.c"\ncandidates = "all.c"\nnum_workers = 16\n',
            {"reference": Path("mr.c"), "num_workers": 16},
        ),
        (
            "restore-csfs",
            '[restore-csfs]\ndescriptors = "desc.parquet"\nheader = "all_header.toml"\noutput = "restored.c"\nindices = [1, 4]\n',
            {"output": Path("restored.c"), "indices": [1, 4]},
        ),
        (
            "csfsgenerate",
            '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\nactive_space = "2s"\nj_min = 0\nj_max = 0\nexcitations = 0\nrcsfs_out = "generated.c"\n',
            {"rcsfs_out": Path("generated.c"), "generation_storage": None},
        ),
    ],
)
def test_default_config_supplies_each_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    table: str,
    expected: dict[str, object],
) -> None:
    (tmp_path / "rcsfs.toml").write_text(table)
    monkeypatch.chdir(tmp_path)
    args = parse_cli_args(cli.build_parser(), [command])
    for key, value in expected.items():
        assert getattr(args, key) == value
    assert args.config == Path("rcsfs.toml")
    if command == "csfsgenerate":
        assert args.generation["reference_configuration"] == ["1s(2,*)"]


def test_alias_and_explicit_flags_override_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        '[csfs-split]\nsplit_csfs_parquet = "all.parquet"\ncsfs_header = "head.toml"\n'
        'active_spaces = ["as1=5s"]\noutput_dir = "from_toml"\njson = true\n'
    )
    monkeypatch.chdir(tmp_path)
    args = parse_cli_args(
        cli.build_parser(),
        ["rcsfsplit", "--output-dir", "from_cli", "--space", "as2=6s"],
    )
    assert args.output_dir == Path("from_cli")
    assert args.active_spaces == ["as2=6s"]
    assert args.json is True


def test_explicit_config_replaces_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        '[interacting]\nreference = "default.c"\ncandidates = "all.c"\n'
    )
    other = tmp_path / "other.toml"
    other.write_text('[interacting]\nreference = "other.c"\ncandidates = "all.c"\n')
    monkeypatch.chdir(tmp_path)
    args = parse_cli_args(cli.build_parser(), ["interacting", "-c", str(other)])
    assert args.reference == Path("other.c")


def test_invalid_config_fails_before_command_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        '[restore-csfs]\noutput = "out.c"\nunknown = true\n'
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["restore-csfs"])
    assert exc.value.code == 2
    assert not (tmp_path / "out.c").exists()


def test_legacy_generation_config_remains_supported(tmp_path: Path) -> None:
    legacy = tmp_path / "generation.toml"
    legacy.write_text(
        '[generate]\ncore = 0\nreferences = ["1s(2,*)"]\n'
        'active_orbitals = "2s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'memory_budget_mib = 64\n[output]\ncsf = "legacy.c"\n'
    )
    args = parse_cli_args(cli.build_parser(), ["csfsgenerate", "-c", str(legacy)])
    assert args.rcsfs_out == Path("legacy.c")
    assert args.memory_budget_mib == 64
    assert args.generation["inactive_core"] == 0


def test_default_config_runs_generator_without_config_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'rcsfs_out = "generated.c"\n'
    )
    monkeypatch.chdir(tmp_path)
    assert cli.main(["csfsgenerate"]) == 0
    assert (tmp_path / "generated.c").is_file()
    assert (tmp_path / "rcsfs.toml").is_file()


def test_user_named_generation_and_split_sections_share_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        '[csfsgenerate]\norbital_order = "*"\ninactive_core = 1\n'
        'reference_configuration = ["2s(2,i)2p(6,i)3s(2,1)3p(6,i)3d(8,*)4s(2,*)"]\n'
        'active_space = "9s,9p,9d,9f,7g"\nj_min = 8\nj_max = 8\nexcitations = 3\n'
        'rcsfs_out = "rcsfs_3exc.c"\ngenerate_descriptors = true\n'
        'rcsfs_parquet = "rcsfs_3exc.parquet"\n'
        'descriptor = "rcsfs_3exc_descriptors.parquet"\n'
        'generation_storage = "disk"\nscratch_dir = "."\nthreads = 8\n'
        '[csfs-split]\nsplit_csfs_parquet = "rcsfs_3exc.parquet"\n'
        'csfs_header = "rcsfs_3exc_header.toml"\n'
        'active_spaces = ["AS5=5s,5p,5d,5f,5g", "AS6=6s,6p,6d,6f,6g"]\n'
        'output_dir = "split"\n'
    )
    monkeypatch.chdir(tmp_path)
    generation = parse_cli_args(cli.build_parser(), ["csfsgenerate"])
    assert generation.generation == {
        "orbital_order": "*",
        "inactive_core": 1,
        "reference_configuration": ["2s(2,i)2p(6,i)3s(2,1)3p(6,i)3d(8,*)4s(2,*)"],
        "active_space": "9s,9p,9d,9f,7g",
        "j_min": 8,
        "j_max": 8,
        "excitations": 3,
    }
    assert generation.rcsfs_out == Path("rcsfs_3exc.c")
    assert generation.rcsfs_parquet == Path("rcsfs_3exc.parquet")
    assert generation.descriptor == Path("rcsfs_3exc_descriptors.parquet")
    assert generation.threads == 8
    for command in ("csfs-split", "split-active", "rcsfsplit"):
        split = parse_cli_args(cli.build_parser(), [command])
        assert split.split_csfs_parquet == Path("rcsfs_3exc.parquet")
        assert split.csfs_header == Path("rcsfs_3exc_header.toml")
        assert split.active_spaces == [
            "AS5=5s,5p,5d,5f,5g",
            "AS6=6s,6p,6d,6f,6g",
        ]


def test_multiple_generation_lists_keep_their_own_parameters(tmp_path: Path) -> None:
    config = tmp_path / "multiple.toml"
    config.write_text(
        '[csfsgenerate]\ninactive_core = 0\nrcsfs_out = "out.c"\n'
        '[[csfsgenerate.lists]]\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "2s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        '[[csfsgenerate.lists]]\nreference_configuration = ["2s(2,*)"]\n'
        'active_space = "3s"\nj_min = 2\nj_max = 2\nexcitations = -2\n'
    )
    args = parse_cli_args(cli.build_parser(), ["csfsgenerate", "-c", str(config)])
    assert args.generation == {
        "lists": [
            {
                "inactive_core": 0,
                "orbital_order": "*",
                "reference_configuration": ["1s(2,*)"],
                "active_space": "2s",
                "j_min": 0,
                "j_max": 0,
                "excitations": 0,
            },
            {
                "inactive_core": 0,
                "orbital_order": "*",
                "reference_configuration": ["2s(2,*)"],
                "active_space": "3s",
                "j_min": 2,
                "j_max": 2,
                "excitations": -2,
            },
        ]
    }


def test_multiple_generation_lists_reject_root_level_list_parameters(
    tmp_path: Path,
) -> None:
    config = tmp_path / "ambiguous.toml"
    config.write_text(
        "[csfsgenerate]\ninactive_core = 0\nexcitations = 2\n"
        '[[csfsgenerate.lists]]\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "2s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        '[[csfsgenerate.lists]]\nreference_configuration = ["2s(2,*)"]\n'
        'active_space = "2s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
    )
    with pytest.raises(SystemExit) as exc:
        parse_cli_args(cli.build_parser(), ["csfsgenerate", "-c", str(config)])
    assert exc.value.code == 2


def test_previous_flat_and_split_tables_still_parse(tmp_path: Path) -> None:
    previous = tmp_path / "previous.toml"
    previous.write_text(
        '[csfsgenerate]\norder = "*"\ncore = 0\nreferences = ["1s(2,*)"]\n'
        'active_orbitals = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'output = "old.c"\nparquet = "old.parquet"\n'
        'descriptor_parquet = "old_descriptors.parquet"\n'
        '[split-active]\ninput_parquet = "old.parquet"\nheader = "old_header.toml"\n'
        'space = ["AS1=1s"]\noutput_dir = "split"\n'
    )
    generation = parse_cli_args(
        cli.build_parser(), ["csfsgenerate", "-c", str(previous)]
    )
    assert generation.rcsfs_out == Path("old.c")
    assert generation.rcsfs_parquet == Path("old.parquet")
    assert generation.descriptor == Path("old_descriptors.parquet")
    assert generation.generation["active_space"] == "1s"
    split = parse_cli_args(cli.build_parser(), ["csfs-split", "-c", str(previous)])
    assert split.split_csfs_parquet == Path("old.parquet")
    assert split.active_spaces == ["AS1=1s"]


def test_conflicting_old_and_new_names_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "conflict.toml"
    config.write_text(
        "[csfsgenerate]\ncore = 0\ninactive_core = 1\n"
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 0\n"
    )
    with pytest.raises(SystemExit) as exc:
        parse_cli_args(cli.build_parser(), ["csfsgenerate", "-c", str(config)])
    assert exc.value.code == 2
