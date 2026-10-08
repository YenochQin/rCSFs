"""The single CLI configuration boundary, including all command surfaces."""

import tomllib
from pathlib import Path

import pytest

from rcsfs import cli
from rcsfs._cli_config import parse_cli_args


@pytest.mark.parametrize(
    ("command", "section"),
    [
        ("csfsgenerate", "csfsgenerate"),
        ("gen-descriptors", "gen-descriptors"),
        ("zero-first", "zero-first"),
        ("csfs-split", "csfs-split"),
        ("split-active", "csfs-split"),
        ("rcsfsplit", "csfs-split"),
        ("interacting", "interacting"),
        ("restore-csfs", "restore-csfs"),
    ],
)
def test_init_config_creates_only_selected_command_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    section: str,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert cli.main([command, "init-config"]) == 0
    assert capsys.readouterr().out == "Created rcsfs.toml\n"

    generated = tmp_path / "rcsfs.toml"
    contents = generated.read_text(encoding="utf-8")
    assert tomllib.loads(contents) == {}
    assert f"# [{section}]" in contents
    if command == "csfsgenerate":
        assert '# conf = "e1_vv1"' in contents
        assert "# as = 6" in contents
        assert "# json = false" not in contents
    for other in (
        "csfsgenerate",
        "gen-descriptors",
        "zero-first",
        "csfs-split",
        "interacting",
        "restore-csfs",
    ):
        if other != section:
            assert f"# [{other}]" not in contents
    assert cli.main([command, "init-config"]) == 0
    assert capsys.readouterr().out == f"[{section}] template already exists\n"
    assert generated.read_text(encoding="utf-8") == contents


def test_init_config_adds_another_command_without_changing_existing_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = "[csfsgenerate]\ninactive_core = 0\n"
    default = tmp_path / "rcsfs.toml"
    default.write_text(original, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert cli.main(["gen-descriptors", "init-config"]) == 0
    updated = default.read_text(encoding="utf-8")
    assert updated.startswith(original)
    assert "# [gen-descriptors]" in updated
    assert tomllib.loads(updated) == {"csfsgenerate": {"inactive_core": 0}}


def test_init_config_preserves_existing_command_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = "[interacting]\nreference = 'reference.c'\ncandidates = 'all.c'\n"
    default = tmp_path / "rcsfs.toml"
    default.write_text(original, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert cli.main(["interacting", "init-config"]) == 0
    assert default.read_text(encoding="utf-8") == original


def test_init_config_rejects_invalid_existing_file_without_modifying_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = "invalid = [\n"
    default = tmp_path / "rcsfs.toml"
    default.write_text(original, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert cli.main(["gen-descriptors", "init-config"]) == 1
    assert default.read_text(encoding="utf-8") == original


def test_init_config_reports_path_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "rcsfs.toml").mkdir()
    monkeypatch.chdir(tmp_path)

    assert cli.main(["csfsgenerate", "init-config"]) == 1
    assert "Cannot create rcsfs.toml" in capsys.readouterr().err


def test_top_level_init_config_is_not_a_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        cli.main(["init-config"])

    assert exc.value.code == 2
    assert not (tmp_path / "rcsfs.toml").exists()


def test_init_config_help_does_not_create_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        cli.main(["csfsgenerate", "init-config", "--help"])

    assert exc.value.code == 0
    assert not (tmp_path / "rcsfs.toml").exists()


@pytest.mark.parametrize(
    "command",
    [
        "csfsgenerate",
        "gen-descriptors",
        "zero-first",
        "csfs-split",
        "interacting",
        "restore-csfs",
    ],
)
def test_init_config_example_values_are_accepted_by_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main([command, "init-config"]) == 0
    config = tmp_path / "rcsfs.toml"
    lines = config.read_text(encoding="utf-8").splitlines()
    if command == "csfsgenerate":
        assert not any("scratch_dir" in line for line in lines)
    section_start = lines.index(f"# [{command}]")
    active = "\n".join(line.removeprefix("# ") for line in lines[section_start:]) + "\n"
    config.write_text(active, encoding="utf-8")

    args = parse_cli_args(
        cli.build_parser(), [command, "-c"] if command == "csfsgenerate" else [command]
    )
    assert args.config == Path("rcsfs.toml")


def test_csfsgenerate_rejects_scratch_dir_in_toml(tmp_path: Path) -> None:
    config = tmp_path / "generation.toml"
    config.write_text(
        '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "2s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'scratch_dir = "elsewhere"\n'
    )

    with pytest.raises(SystemExit) as exc:
        parse_cli_args(cli.build_parser(), ["csfsgenerate", "-c", str(config)])

    assert exc.value.code == 2


def test_legacy_generation_rejects_scratch_dir_in_toml(tmp_path: Path) -> None:
    config = tmp_path / "legacy.toml"
    config.write_text(
        '[generate]\ncore = 0\nreferences = ["1s(2,*)"]\n'
        'active_orbitals = "2s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'scratch_dir = "elsewhere"\n[output]\ncsf = "out.c"\n'
    )

    with pytest.raises(SystemExit) as exc:
        parse_cli_args(cli.build_parser(), ["csfsgenerate", "-c", str(config)])

    assert exc.value.code == 2


def test_csfsgenerate_rejects_scratch_dir_flag() -> None:
    with pytest.raises(SystemExit) as exc:
        parse_cli_args(
            cli.build_parser(), ["csfsgenerate", "--scratch-dir", "elsewhere"]
        )

    assert exc.value.code == 2


def test_regular_command_does_not_create_default_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_run_interacting", lambda args: 0)

    assert cli.main(["interacting", "reference.c", "candidates.c"]) == 0
    assert not (tmp_path / "rcsfs.toml").exists()


def test_csfsgenerate_stays_interactive_without_default_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PromptReached(Exception):
        pass

    def stop_at_prompt(message: str) -> str:
        raise PromptReached

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_prompt", stop_at_prompt)

    with pytest.raises(PromptReached):
        cli.main(["csfsgenerate"])
    assert not (tmp_path / "rcsfs.toml").exists()


def test_csfsgenerate_ignores_existing_config_without_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PromptReached(Exception):
        pass

    config = tmp_path / "rcsfs.toml"
    config.write_text("invalid = [", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli, "_prompt", lambda message: (_ for _ in ()).throw(PromptReached)
    )

    with pytest.raises(PromptReached):
        cli.main(["csfsgenerate"])
    assert config.read_text(encoding="utf-8") == "invalid = ["


def test_csfsgenerate_bare_config_flag_requires_default_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["csfsgenerate", "--config"])
    assert exc.value.code == 2
    assert not (tmp_path / "rcsfs.toml").exists()


def test_csfsgenerate_explicit_config_ignores_default_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = tmp_path / "rcsfs.toml"
    default.write_text("invalid = [", encoding="utf-8")
    custom = tmp_path / "custom.toml"
    custom.write_text(
        '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'rcsfs_out = "custom.c"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert cli.main(["csfsgenerate", "--config", str(custom)]) == 0
    assert (tmp_path / "custom.c").is_file()
    assert default.read_text(encoding="utf-8") == "invalid = ["
    assert custom.read_text(encoding="utf-8").startswith("[csfsgenerate]")


def test_top_level_config_runs_generation_then_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'rcsfs_out = "generated.c"\ngenerate_descriptors = true\n'
        'generation_storage = "memory"\n'
        '\n[csfs-split]\nsplit_csfs_parquet = "generated.parquet"\n'
        'csfs_header = "generated_header.toml"\nactive_spaces = ["AS1=1s"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert cli.main(["-c", "rcsfs.toml"]) == 0
    assert (tmp_path / "generated.c").is_file()
    assert (tmp_path / "generated.parquet").is_file()
    assert (tmp_path / "generatedAS1.c").is_file()


def test_top_level_config_builds_parquet_for_split_without_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'rcsfs_out = "generated.c"\ngenerate_descriptors = false\n'
        '\n[csfs-split]\nsplit_csfs_parquet = "generated.parquet"\n'
        'csfs_header = "generated_header.toml"\nactive_spaces = ["AS1=1s"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert cli.main(["-c", "rcsfs.toml"]) == 0
    assert (tmp_path / "generated.c").is_file()
    assert (tmp_path / "generated.parquet").is_file()
    assert (tmp_path / "generated_header.toml").is_file()
    assert (tmp_path / "generatedAS1.c").is_file()
    assert not (tmp_path / "generated_descriptors.parquet").exists()

    outputs = [
        tmp_path / name
        for name in (
            "generated.c",
            "generated.parquet",
            "generated_header.toml",
            "generatedAS1.c",
        )
    ]
    for output in outputs:
        output.write_bytes(b"old result")
    assert cli.main(["-c", "rcsfs.toml"]) == 0
    assert all(output.read_bytes() != b"old result" for output in outputs)


def test_conf_and_as_name_raw_generation_and_split_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "rcsfs.toml"
    config.write_text(
        'conf = "e1_vv1_"\n'
        "[csfsgenerate]\nas = 2\ninactive_core = 0\n"
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 0\n"
        '[csfs-split]\nactive_spaces = ["AS1=1s", "AS2=1s"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert cli.main(["-c", "rcsfs.toml"]) == 0
    outputs = [
        tmp_path / name
        for name in (
            "e1_vv1_as2raw.c",
            "e1_vv1_as2raw.parquet",
            "e1_vv1_as2raw_header.toml",
            "e1_vv1_as1raw.c",
        )
    ]
    assert all(path.is_file() for path in outputs)
    assert not list(tmp_path.glob("rcsfs-split-*"))
    for path in outputs:
        path.write_bytes(b"old")
    assert cli.main(["-c", "rcsfs.toml"]) == 0
    assert all(path.read_bytes() != b"old" for path in outputs)
    assert config.read_text(encoding="utf-8").startswith('conf = "e1_vv1_"')


@pytest.mark.parametrize(
    "spaces",
    [
        '[csfs-split]\nactive_spaces = ["AS0=1s"]\n',
        '[csfs-split.active_spaces]\nAS0 = "1s"\n',
    ],
)
def test_as_zero_names_mr_generation_and_split_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spaces: str
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        'conf = "mr_"\n'
        "[csfsgenerate]\nas = 0\ninactive_core = 0\n"
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 0\n" + spaces,
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert cli.main(["-c", "rcsfs.toml"]) == 0
    for name in (
        "mr_as0raw.c",
        "mr_as0raw.parquet",
        "mr_as0raw_header.toml",
    ):
        assert (tmp_path / name).is_file()


@pytest.mark.parametrize("section", ["csfs-split", "split-active"])
@pytest.mark.parametrize("command", ["csfs-split", "split-active", "rcsfsplit"])
def test_active_space_table_preserves_order_and_cli_override(
    tmp_path: Path, section: str, command: str
) -> None:
    config = tmp_path / "rcsfs.toml"
    config.write_text(
        f'[{section}]\nsplit_csfs_parquet = "all.parquet"\n'
        'csfs_header = "head.toml"\njson = true\n'
        f'[{section}.active_spaces]\nAS3 = "6s,6p"\nAS0 = "3s,3p"\nAS1 = "4s,4p"\n',
        encoding="utf-8",
    )
    args = parse_cli_args(cli.build_parser(), [command, "-c", str(config)])
    assert args.active_spaces == ["AS3=6s,6p", "AS0=3s,3p", "AS1=4s,4p"]
    assert args.json is True
    args = parse_cli_args(
        cli.build_parser(), [command, "-c", str(config), "--space", "AS2=5s"]
    )
    assert args.active_spaces == ["AS2=5s"]


@pytest.mark.parametrize(
    "entries, error",
    [
        ("", "nonempty table"),
        ('AS0 = ""', "active_spaces.AS0"),
        ('AS0 = "   "', "active_spaces.AS0"),
        ("AS0 = 1", "active_spaces.AS0"),
        ('AS0 = ["1s"]', "active_spaces.AS0"),
        ('"../AS0" = "1s"', "invalid active_spaces label"),
    ],
)
def test_invalid_active_space_table_fails_before_processing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], entries: str, error: str
) -> None:
    config = tmp_path / "rcsfs.toml"
    config.write_text(
        '[csfs-split]\nsplit_csfs_parquet = "all.parquet"\n'
        'csfs_header = "head.toml"\n[csfs-split.active_spaces]\n' + entries + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["-c", str(config)])
    assert exc.value.code == 2
    assert error in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == [config]


def test_conf_and_as_use_last_generator_for_implicit_split_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        'conf = "calc_"\n'
        "[[csfsgenerate]]\nas = 1\ninactive_core = 0\n"
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 0\n"
        "[[csfsgenerate]]\nas = 2\ninactive_core = 0\n"
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 0\n"
        '[csfs-split]\nactive_spaces = ["AS1=1s"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert cli.main(["-c", "rcsfs.toml"]) == 0
    assert (tmp_path / "calc_as1raw.c").is_file()
    assert (tmp_path / "calc_as2raw.parquet").is_file()
    assert (tmp_path / "calc_as1raw.c").is_file()
    assert not (tmp_path / "calc_as1raw.parquet").exists()


def test_conf_naming_works_with_separate_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        'conf = "calc_"\n[csfsgenerate]\nas = 2\ninactive_core = 0\n'
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 0\ngenerate_parquet = true\n"
        '[csfs-split]\nactive_spaces = ["AS1=1s", "AS2=1s"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert cli.main(["csfsgenerate", "-c"]) == 0
    assert cli.main(["csfs-split"]) == 0
    assert (tmp_path / "calc_as2raw.parquet").is_file()
    assert (tmp_path / "calc_as1raw.c").is_file()
    assert (tmp_path / "calc_as2raw.c").is_file()
    assert (tmp_path / "calc_as2raw.c").is_file()


@pytest.mark.parametrize("conf", ["calc", "calc_"])
def test_conf_naming_respects_explicit_input_paths(tmp_path: Path, conf: str) -> None:
    config = tmp_path / "rcsfs.toml"
    config.write_text(
        f'conf = "{conf}"\n[csfsgenerate]\nas = 6\ninactive_core = 0\n'
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        'j_min = 0\nj_max = 0\nexcitations = 0\nrcsfs_out = "custom.c"\n'
        '[csfs-split]\nactive_spaces = ["AS1=1s"]\n'
        'prefix = "custom"\n',
        encoding="utf-8",
    )
    generated = parse_cli_args(cli.build_parser(), ["csfsgenerate", "-c", str(config)])
    split = parse_cli_args(cli.build_parser(), ["csfs-split", "-c", str(config)])

    assert generated.rcsfs_out == Path("custom.c")
    assert split.split_csfs_parquet == Path("custom.parquet")
    assert split.csfs_header == Path("custom_header.toml")
    assert list(cli._split_targets(split)) == [Path("calc_as1raw.c")]

    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "[csfs-split]\n",
            '[csfs-split]\nsplit_csfs_parquet = "other.parquet"\n'
            'csfs_header = "other_header.toml"\n',
        ),
        encoding="utf-8",
    )
    split = parse_cli_args(cli.build_parser(), ["csfs-split", "-c", str(config)])
    assert split.split_csfs_parquet == Path("other.parquet")
    assert split.csfs_header == Path("other_header.toml")


def test_conf_naming_is_not_replaced_by_legacy_split_prefix(tmp_path: Path) -> None:
    config = tmp_path / "rcsfs.toml"
    config.write_text(
        'conf = "calc_"\n[csfsgenerate]\nas = 3\ninactive_core = 0\n'
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 0\n"
        '[csfs-split]\nactive_spaces = ["AS1=1s"]\n'
        'prefix = "split"\n',
        encoding="utf-8",
    )

    args = parse_cli_args(cli.build_parser(), ["csfs-split", "-c", str(config)])
    assert list(cli._split_targets(args)) == [Path("calc_as1raw.c")]


@pytest.mark.parametrize(
    "conf, level", [('"../escape"', "6"), ('"calc_"', "-1"), ('"calc_"', "true")]
)
def test_conf_and_as_reject_invalid_values(
    tmp_path: Path, conf: str, level: str
) -> None:
    config = tmp_path / "rcsfs.toml"
    config.write_text(
        f"conf = {conf}\n[csfsgenerate]\nas = {level}\ninactive_core = 0\n"
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 0\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        parse_cli_args(cli.build_parser(), ["csfsgenerate", "-c", str(config)])
    assert exc.value.code == 2


def test_as_without_conf_or_output_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "rcsfs.toml"
    config.write_text(
        "[csfsgenerate]\nas = 6\ninactive_core = 0\n"
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        "j_min = 0\nj_max = 0\nexcitations = 0\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        parse_cli_args(cli.build_parser(), ["csfsgenerate", "-c", str(config)])
    assert exc.value.code == 2


def test_top_level_config_runs_repeated_generators_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        '[[csfsgenerate]]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'rcsfs_out = "first.c"\n'
        "\n[[csfsgenerate]]\ninactive_core = 0\n"
        'reference_configuration = ["1s(2,*)"]\nactive_space = "1s"\n'
        'j_min = 0\nj_max = 0\nexcitations = 0\nrcsfs_out = "second.c"\n'
        '\n[csfs-split]\nsplit_csfs_parquet = "second.parquet"\n'
        'csfs_header = "second_header.toml"\nactive_spaces = ["AS1=1s"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert cli.main(["-c", "rcsfs.toml"]) == 0
    assert (tmp_path / "first.c").is_file()
    assert not (tmp_path / "first.parquet").exists()
    assert (tmp_path / "second.c").is_file()
    assert (tmp_path / "second.parquet").is_file()
    assert (tmp_path / "secondAS1.c").is_file()
    assert not (tmp_path / "second_descriptors.parquet").exists()


def test_top_level_config_rejects_repeated_generator_output_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = (
        '[[csfsgenerate]]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'rcsfs_out = "same.c"\n'
    )
    (tmp_path / "rcsfs.toml").write_text(entry + "\n" + entry, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        cli.main(["-c", "rcsfs.toml"])
    assert exc.value.code == 2
    assert not (tmp_path / "same.c").exists()


def test_top_level_config_rejects_shared_default_descriptor_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = (
        '[[csfsgenerate]]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        "generate_descriptors = true\n"
    )
    config = tmp_path / "rcsfs.toml"
    config.write_text(
        'conf = "calc"\n' + entry + "as = 0\n" + entry + "as = 1\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        cli.main(["-c", str(config)])
    assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == [config]


def test_top_level_config_rejects_unmatched_split_inputs_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'rcsfs_out = "generated.c"\n'
        '\n[csfs-split]\nsplit_csfs_parquet = "other.parquet"\n'
        'csfs_header = "other_header.toml"\nactive_spaces = ["AS1=1s"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        cli.main(["-c", "rcsfs.toml"])
    assert exc.value.code == 2
    assert not (tmp_path / "generated.c").exists()


def test_csfsgenerate_can_write_parquet_without_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'rcsfs_out = "generated.c"\ngenerate_parquet = true\n'
        'generation_storage = "memory"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert cli.main(["csfsgenerate", "-c"]) == 0
    assert (tmp_path / "generated.c").is_file()
    assert (tmp_path / "generated.parquet").is_file()
    assert (tmp_path / "generated_header.toml").is_file()
    assert not (tmp_path / "generated_descriptors.parquet").exists()


def test_estimate_only_without_parquet_does_not_generate_csf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'rcsfs_out = "generated.c"\nestimate_only = true\n'
        'generation_storage = "disk"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert cli.main(["csfsgenerate", "-c"]) == 1
    assert not (tmp_path / "generated.c").exists()


def test_top_level_config_runs_all_tables_in_order_and_stops_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        '[interacting]\nreference = "reference.c"\ncandidates = "candidates.c"\n'
        '\n[restore-csfs]\ndescriptors = "descriptors.parquet"\n'
        'header = "header.toml"\noutput = "restored.c"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []
    should_fail = False

    def fail_interacting(args: object) -> int:
        calls.append("interacting")
        return 1 if should_fail else 0

    def restore(args: object) -> int:
        calls.append("restore-csfs")
        return 0

    monkeypatch.setattr(cli, "_run_interacting", fail_interacting)
    monkeypatch.setattr(cli, "_run_restore_csfs", restore)

    assert cli.main(["--config", "rcsfs.toml"]) == 0
    assert calls == ["interacting", "restore-csfs"]

    calls.clear()
    should_fail = True
    assert cli.main(["--config", "rcsfs.toml"]) == 1
    assert calls == ["interacting"]


def test_csfsgenerate_interactive_output_cannot_replace_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = "invalid = ["
    config = tmp_path / "rcsfs.toml"
    config.write_text(original, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    answers = iter(["*", "0", "1s(2,*)", "", "1s", "0,0", "0", "n"])
    monkeypatch.setattr(cli, "_prompt", lambda message: next(answers))

    assert cli.main(["csfsgenerate", "rcsfs.toml"]) == 1
    assert config.read_text(encoding="utf-8") == original


def test_cli_with_explicit_config_does_not_create_default_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "custom.toml").write_text(
        "[interacting]\nreference = 'reference.c'\ncandidates = 'candidates.c'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_run_interacting", lambda args: 0)

    assert cli.main(["interacting", "--config", "custom.toml"]) == 0
    assert not (tmp_path / "rcsfs.toml").exists()


def test_cli_help_does_not_create_default_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        cli.main(["interacting", "--help"])

    assert exc.value.code == 0
    assert not (tmp_path / "rcsfs.toml").exists()


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
            '[zero-first]\nzero_csf = "zero.c"\nfull_csf = "full.c"\n',
            {"zero_csf": Path("zero.c")},
        ),
        (
            "csfs-split",
            '[csfs-split]\nsplit_csfs_parquet = "all.parquet"\ncsfs_header = "all_header.toml"\nactive_spaces = ["as1=5s,4p", "as2=6s,5p"]\n',
            {
                "split_csfs_parquet": Path("all.parquet"),
                "csfs_header": Path("all_header.toml"),
                "active_spaces": ["as1=5s,4p", "as2=6s,5p"],
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
    args = parse_cli_args(
        cli.build_parser(), [command, "-c"] if command == "csfsgenerate" else [command]
    )
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
        'active_spaces = ["as1=5s"]\njson = true\n'
    )
    monkeypatch.chdir(tmp_path)
    args = parse_cli_args(
        cli.build_parser(),
        ["rcsfsplit", "--space", "as2=6s"],
    )
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


def test_default_config_runs_generator_with_config_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rcsfs.toml").write_text(
        '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'rcsfs_out = "generated.c"\n'
    )
    monkeypatch.chdir(tmp_path)
    assert cli.main(["csfsgenerate", "-c"]) == 0
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
        'generation_storage = "disk"\nthreads = 8\n'
        '[csfs-split]\nsplit_csfs_parquet = "rcsfs_3exc.parquet"\n'
        'csfs_header = "rcsfs_3exc_header.toml"\n'
        'active_spaces = ["AS5=5s,5p,5d,5f,5g", "AS6=6s,6p,6d,6f,6g"]\n'
        ""
    )
    monkeypatch.chdir(tmp_path)
    generation = parse_cli_args(cli.build_parser(), ["csfsgenerate", "-c"])
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
        'space = ["AS1=1s"]\n'
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


pytestmark = pytest.mark.usefixtures("cli_cwd")
