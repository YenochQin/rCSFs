"""The single CLI configuration boundary, including all command surfaces."""

import tomllib
from pathlib import Path

import _native_cli as cli
import pytest


@pytest.mark.parametrize(
    "section",
    [
        "[info]\ninput_parquet = 12\n",
        '[info]\ninput_parquet = "missing.parquet"\njson = "yes"\n',
        '[convert]\ninput_csf = "missing.c"\nnum_workers = 0\n',
        '[convert]\ninput_csf = "missing.c"\nchunk_size = -1\n',
        '[info]\ninput_parquet = "missing.parquet"\nunknown = true\n',
        "[csfs-split]\nactive_spaces = []\n",
        "[csfs-split]\nactive_spaces = [12]\n",
        '[csfs-split]\nactive_spaces = ["AS1"]\n',
        '[csfs-split]\nactive_spaces = ["AS1=  "]\n',
        '[csfs-split]\nactive_spaces = [" =1s"]\n',
        '[csfs-split.active_spaces]\nAS1 = "  "\n',
        "[csfs-split.active_spaces]\nAS1 = 1\n",
    ],
)
def test_invalid_batch_is_rejected_before_any_output(
    tmp_path: Path, section: str
) -> None:
    config = tmp_path / "batch.toml"
    config.write_text(
        '[csfsgenerate]\ninactive_core = 0\nreference_configuration = ["1s(2,*)"]\n'
        'active_space = "1s"\nj_min = 0\nj_max = 0\nexcitations = 0\n'
        'rcsfs_out = "generated.c"\n' + section,
        encoding="utf-8",
    )
    result = cli.run(["--config", str(config)], cwd=tmp_path)
    assert result.returncode == 2, result.stderr
    assert "error:" in result.stderr
    assert list(tmp_path.iterdir()) == [config]


def test_cli_positional_and_flag_override_toml(tmp_path: Path) -> None:
    import json

    fixture = Path(__file__).parent / "fixtures" / "sample.csf"
    config = tmp_path / "custom.toml"
    config.write_text(
        '[convert]\ninput_csf = "missing.c"\noutput_parquet = "ignored.parquet"\n'
        "num_workers = 1\njson = false\n",
        encoding="utf-8",
    )
    result = cli.run(
        ["convert", str(fixture), "actual.parquet", "--config", str(config), "--json"],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["success"] is True
    assert (tmp_path / "actual.parquet").is_file()
    assert not (tmp_path / "ignored.parquet").exists()


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
        assert '# conf = "model"' in contents
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
    assert "cannot read rcsfs.toml" in capsys.readouterr().err


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

    assert cli.main(["csfsgenerate", "-c"]) == 0
    assert not (tmp_path / "generated.c").exists()


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


pytestmark = pytest.mark.usefixtures("cli_cwd")
