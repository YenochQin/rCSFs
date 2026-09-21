"""Generation CLI output safety and metadata regressions."""

import json
import tomllib
from pathlib import Path

import polars as pl
import pytest

from rcsfs import (
    cli,
    get_parquet_info,
    read_peel_subshells,
    restore_csfs_from_descriptors,
)
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


def test_config_generation_v2_metadata_and_roundtrip(tmp_path: Path) -> None:
    config = config_file(tmp_path)
    assert cli.main(["csfsgenerate", "--config", str(config)]) == 0
    frame = pl.read_parquet(tmp_path / "out_descriptors.parquet")
    metadata = tomllib.loads((tmp_path / "out_descriptors.toml").read_text())
    assert metadata == {
        "format_version": 2,
        "encoding": "parquet",
        "normalized": False,
        "record_count": frame.height,
        "subshells": read_peel_subshells(tmp_path / "out_header.toml"),
    }
    assert frame.height == 1
    assert frame.columns == [
        f"sub{index}_{channel}"
        for index in range(len(metadata["subshells"]))
        for channel in ("n", "2j", "v", "2k")
    ] + ["total_two_j", "parity"]
    assert frame.dtypes == [pl.Int32] * frame.width
    assert frame["total_two_j"].to_list() == [0]
    assert frame["parity"].to_list() == [1]
    parquet_metadata = get_parquet_info(tmp_path / "out_descriptors.parquet")[
        "key_value_metadata"
    ]
    assert parquet_metadata["descriptor_version"] == "2"
    assert parquet_metadata["channels_per_subshell"] == "4"
    restored = tmp_path / "restored.c"
    stats = restore_csfs_from_descriptors(
        tmp_path / "out_descriptors.parquet", tmp_path / "out_header.toml", restored
    )
    assert stats["record_count"] == frame.height
    assert restored.read_bytes() == (tmp_path / "out.c").read_bytes()


def test_config_generation_rejects_v2_normalization_without_outputs(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    config = config_file(tmp_path, normalize=True)
    assert cli.main(["csfsgenerate", "--config", str(config)]) == 1
    assert (
        "normalize is not supported by reversible V2 descriptors"
        in capfd.readouterr().err
    )
    assert set(tmp_path.iterdir()) == {config}


def test_config_generation_json_contains_stage_stats(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    config = config_file(tmp_path)
    assert cli.main(["csfsgenerate", "--config", str(config), "--json"]) == 0
    payload = json.loads(capfd.readouterr().out)
    assert payload["success"] is True
    assert payload["resource_stats"]["memory_budget_mib"] is None
    assert [stage["name"] for stage in payload["stage_stats"]] == [
        "enumeration",
        "workload_planning",
        "csf_generation",
        "deduplication",
        "descriptor_merge",
        "csf_restore",
    ]
    plan_stats = payload["plan_stats"]
    assert plan_stats["task_count"] >= 1
    assert plan_stats["unique_occupations"] == payload["unique_occupations"]
    # The counted estimate is what the schedule and the capacity model trust, so
    # it has to agree with what generation actually produced.
    assert plan_stats["estimated_total_records"] == payload["generated_count"]
    per_task = plan_stats["estimated_records_per_task"]
    assert per_task["count"] == plan_stats["task_count"]
    assert per_task["minimum"] <= per_task["p50"] <= per_task["p95"]
    assert per_task["p95"] <= per_task["maximum"]


def test_config_generation_reads_memory_budget(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    config = config_file(tmp_path)
    config.write_text(
        config.read_text().replace(
            "[generate]\n", "[generate]\nmemory_budget_mib = 64\n", 1
        )
    )
    assert cli.main(["csfsgenerate", "--config", str(config), "--json"]) == 0
    payload = json.loads(capfd.readouterr().out)
    assert payload["resource_stats"]["memory_budget_mib"] == 64


def test_cli_memory_budget_overrides_toml(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    config = config_file(tmp_path)
    config.write_text(
        config.read_text().replace(
            "[generate]\n", "[generate]\nmemory_budget_mib = 64\n", 1
        )
    )
    assert (
        cli.main(
            [
                "csfsgenerate",
                "--config",
                str(config),
                "--memory-budget-mib",
                "1",
            ]
        )
        == 1
    )
    assert "memory budget exceeded" in capfd.readouterr().err


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
    original = cli.generate_disk_outputs_from_transcript
    sentinel = tmp_path / "out.parquet"

    def racing_writer(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        sentinel.write_bytes(b"another writer")
        return result

    monkeypatch.setattr(cli, "generate_disk_outputs_from_transcript", racing_writer)
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


def test_estimate_only_reports_capacity_without_writing(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    config = config_file(tmp_path)
    assert cli.main(["csfsgenerate", "--config", str(config), "--estimate-only", "--json"]) == 0
    payload = json.loads(capfd.readouterr().out)
    assert payload["success"] is True
    assert payload["estimate_only"] is True
    # An estimate must not create scratch, staging or output files.
    assert set(tmp_path.iterdir()) == {config}
    assert payload["unique_occupations"] == payload["plan_stats"]["unique_occupations"]
    assert payload["pre_deduplication_records"] > 0
    assert payload["bytes"]["required_scratch"] > payload["bytes"]["scratch_peak"]
    assert payload["bytes"]["required_output"] > payload["bytes"]["staged_outputs"]
    assert payload["assumptions"]
    # Scratch is not bound to an input hash or format version, so a failed run
    # cannot be resumed from it.
    assert payload["failure_recovery"] == "restart"
    assert payload["space_checks"]


def test_estimate_matches_the_run_it_predicts(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    config = config_file(tmp_path)
    assert cli.main(["csfsgenerate", "--config", str(config), "--estimate-only", "--json"]) == 0
    estimate = json.loads(capfd.readouterr().out)
    assert cli.main(["csfsgenerate", "--config", str(config), "--json"]) == 0
    generated = json.loads(capfd.readouterr().out)
    assert estimate["plan_stats"]["task_count"] == generated["plan_stats"]["task_count"]
    assert estimate["pre_deduplication_records"] == generated["generated_count"]
    assert (
        estimate["plan_stats"]["estimated_total_records"]
        == generated["plan_stats"]["estimated_total_records"]
    )


def test_estimate_only_rejects_the_in_memory_path(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """An estimate describes the disk path; asking for it elsewhere is an error."""
    config = config_file(tmp_path)
    assert (
        cli.main(
            [
                "csfsgenerate",
                "--config",
                str(config),
                "--estimate-only",
                "--generation-storage",
                "memory",
            ]
        )
        == 1
    )
    captured = capfd.readouterr()
    assert "estimate_only covers the disk descriptor path" in captured.err
    assert set(tmp_path.iterdir()) == {config}


def test_destination_space_check_refuses_an_impossible_requirement(
    tmp_path: Path,
) -> None:
    """A destination that cannot hold the artifacts is refused, not discovered late."""
    estimate = {"bytes": {"staged_outputs": 10, "required_output": 100}}
    with pytest.raises(ValueError, match="Not enough free space"):
        cli._check_destination_space(estimate, {tmp_path / "out.c": 1 << 62})


def test_destination_space_check_reports_an_unreadable_directory(
    tmp_path: Path,
) -> None:
    """An unmeasurable destination is reported as unchecked rather than sufficient."""
    estimate = {"bytes": {"staged_outputs": 10, "required_output": 100}}
    checks = cli._check_destination_space(estimate, {tmp_path / "missing" / "out.c": 1})
    assert checks == [
        {
            "path": str(tmp_path / "missing"),
            "required_bytes": 10,
            "free_bytes": None,
            "sufficient": None,
        }
    ]
