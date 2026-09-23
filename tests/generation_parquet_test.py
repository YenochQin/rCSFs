"""Generation CLI output safety and metadata regressions."""

import json
import tomllib
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from rcsfs import (
    CsfGenerationEstimate,
    cli,
    estimate_disk_generation,
    generate_disk_outputs_from_transcript,
    get_parquet_info,
    read_peel_subshells,
    restore_csfs_from_descriptors,
)
from rcsfs._rcsfs import generate_csfs_from_transcript

FIXTURES = Path(__file__).parent / "fixtures"


def test_generation_estimate_type_names_the_deduplication_strategy() -> None:
    """The public type must describe every field the extension always returns."""
    assert "deduplication" in CsfGenerationEstimate.__required_keys__


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
        "setup",
        "enumeration",
        "workload_planning",
        "csf_generation",
        "deduplication",
        "header_write",
        "final_encoding_read",
        "final_encoding_select",
        "final_encoding_prepare",
        "final_encoding_descriptor_encode",
        "final_encoding_descriptor_write",
        "final_encoding_csf_outputs_encode",
        "final_encoding_csf_outputs_write",
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


def test_segment_codec_reaches_the_run_and_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The codec knob is a benchmark experiment, so the run must name what it used.

    A recorded codec the writer ignored would make P2a's compression ratios
    fiction; the extension reports the codec it actually wrote with, and an
    unusable value fails the run rather than falling back silently.
    """
    transcript = "* ! Orbital order\n0\n2s(2,*)2p(1,*)\n\n3s,3p,3d\n1,3\n1\nn\n"

    def run(label: str) -> tuple[dict[str, object], list[tuple[int, ...]]]:
        # Each run gets its own set: the API refuses to write over staged output.
        directory = tmp_path / label
        directory.mkdir()
        stats = generate_disk_outputs_from_transcript(
            transcript,
            directory / "out.c",
            directory / "out.parquet",
            directory / "descriptors.parquet",
            directory / "header.toml",
            directory / "scratch",
        )
        return stats, pl.read_parquet(directory / "descriptors.parquet").rows()

    default, uncompressed_rows = run("default")
    assert default["segment_codec"] == "none", "the default must stay uncompressed"

    monkeypatch.setenv("RCSFS_SEGMENT_CODEC", "zstd")
    compressed, compressed_rows = run("zstd")
    assert compressed["segment_codec"] == "zstd"
    # Compression is a storage decision: the published rows must not move.
    assert compressed_rows == uncompressed_rows
    assert compressed["record_count"] == default["record_count"]

    monkeypatch.setenv("RCSFS_SEGMENT_CODEC", "gzip")
    with pytest.raises(ValueError, match="invalid segment codec"):
        run("invalid")


def test_estimate_names_the_segment_codec_it_assumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The estimate prices uncompressed segments, so it says which codec it assumed."""
    transcript = "* ! Orbital order\n0\n2s(2,*)2p(1,*)\n\n3s,3p,3d\n1,3\n1\nn\n"
    estimate = estimate_disk_generation(transcript)
    assert estimate["segment_codec"] == "none"
    assert any("uncompressed" in line for line in estimate["assumptions"])

    monkeypatch.setenv("RCSFS_SEGMENT_CODEC", "zstd")
    compressed = estimate_disk_generation(transcript)
    assert compressed["segment_codec"] == "zstd"
    # A compressed run writes fewer segment bytes, so the model stays an upper
    # bound rather than being tuned to a ratio that has not been measured here.
    assert compressed["bytes"]["segments"] == estimate["bytes"]["segments"]


def test_deduplication_strategy_is_reported_and_selectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`duplicate_count` must be readable for what it is.

    The default path skips the exact comparison because the internal generator
    cannot repeat a row, so its zero is a fact about the proof; the exact path
    measures it. A run has to say which one produced the number.
    """
    transcript = "* ! Orbital order\n0\n2s(2,*)2p(1,*)\n\n3s,3p,3d\n1,3\n1\nn\n"

    def run(label: str) -> dict[str, object]:
        directory = tmp_path / label
        directory.mkdir()
        return generate_disk_outputs_from_transcript(
            transcript,
            directory / "out.c",
            directory / "out.parquet",
            directory / "descriptors.parquet",
            directory / "header.toml",
            directory / "scratch",
        )

    monkeypatch.delenv("RCSFS_DEDUPLICATION", raising=False)
    verified = run("verified")
    assert verified["deduplication"] == "verified_unique"
    assert verified["duplicate_count"] == 0

    monkeypatch.setenv("RCSFS_DEDUPLICATION", "exact")
    exact = run("exact")
    assert exact["deduplication"] == "exact"
    assert exact["duplicate_count"] == 0
    # The strategy changes how the number was obtained, not the published rows.
    assert exact["record_count"] == verified["record_count"]
    assert exact["block_count"] == verified["block_count"]

    monkeypatch.setenv("RCSFS_DEDUPLICATION", "trusted")
    with pytest.raises(ValueError, match="invalid de-duplication strategy"):
        run("invalid")


def test_the_estimate_prices_the_strategy_the_run_will_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-flight that reserves buckets a run never writes is wrong, not safe.

    The verified path writes no root or recursive bucket, so pricing them would
    multiply the scratch requirement of the very mode the disk path defaults to.
    """
    transcript = "* ! Orbital order\n0\n2s(2,*)2p(1,*)\n\n3s,3p,3d\n1,3\n1\nn\n"
    monkeypatch.delenv("RCSFS_DEDUPLICATION", raising=False)
    verified = estimate_disk_generation(transcript)
    assert verified["deduplication"] == "verified_unique"
    assert verified["bytes"]["root_buckets"] == 0
    assert verified["bytes"]["recursive_buckets"] == 0
    assert verified["bytes"]["segments"] > 0
    assert verified["bytes"]["survivor_bitsets"] > 0
    assert any("verified path" in line for line in verified["assumptions"])

    monkeypatch.setenv("RCSFS_DEDUPLICATION", "exact")
    exact = estimate_disk_generation(transcript)
    assert exact["deduplication"] == "exact"
    assert exact["bytes"]["root_buckets"] > 0
    assert exact["bytes"]["recursive_buckets"] > 0
    assert exact["bytes"]["segments"] == verified["bytes"]["segments"]
    assert any("exact path" in line for line in exact["assumptions"])
    assert (
        exact["bytes"]["scratch_peak"] > verified["bytes"]["scratch_peak"]
    ), "the exact path writes buckets the verified path does not"


def test_the_final_encoding_bounds_the_memory_owned_by_its_writers(
    tmp_path: Path,
) -> None:
    """A budget must be spent on what the run is about to hold, not after.

    The one-pass final encoding holds a batch's selected columns, its formatted
    three-line records, two live Parquet writers and one parallel descriptor row
    group at once. Each of those has to
    be reserved *before* it exists and for what it really is, or a low budget
    fails only after the allocation it was meant to prevent and then reports a
    managed peak lower than the process held. The ladder below is the observable
    form of that: as the budget rises the refusal moves from the generation
    batch to the CSF Parquet writer, then to the final-encoding batch and
    parallel row group; only a budget covering all of them runs.
    """
    transcript = (FIXTURES / "o1_cc1as1.rcsfgenerate").read_text(encoding="utf-8")

    def run(budget_mib: int) -> tuple[str, float, Path | None, Path | None]:
        directory = tmp_path / f"budget-{budget_mib}"
        directory.mkdir()
        csf_parquet = directory / "out.parquet"
        descriptor_parquet = directory / "descriptors.parquet"
        try:
            stats = generate_disk_outputs_from_transcript(
                transcript,
                directory / "out.c",
                csf_parquet,
                descriptor_parquet,
                directory / "header.toml",
                directory / "scratch",
                threads=1,
                memory_budget_mib=budget_mib,
            )
        except (OSError, ValueError) as error:
            return str(error), 0.0, None, None
        resource_stats = stats["resource_stats"]
        return (
            "ok",
            resource_stats["peak_managed_bytes"] / 2**20,
            descriptor_parquet,
            csf_parquet,
        )

    # Far too small: the generation stage's own batch reservation refuses first.
    message, _, _, _ = run(8)
    assert "descriptor generation batch" in message, message

    # Enough for generation, not for the final encoding: the refusal must name a
    # reservation this stage makes, which is what proves the charge exists and
    # happens before the allocation.
    for budget in (12, 16, 20, 24, 28, 36, 40):
        message, _, _, _ = run(budget)
        assert message != "ok", f"{budget} MiB was accepted; the charges are too small"
        assert any(
            label in message
            for label in (
                "CSF Parquet encoding",
                "final descriptor encoding",
                "final-encoding source batch",
                "final-encoding batch",
                "parallel descriptor row group",
            )
        ), f"{budget} MiB refused for an unrelated reason: {message}"

    # And the two writers are charged separately: a budget that covers one
    # writer's allowance plus the batch is not automatically enough for both.
    outcome, peak, descriptor_parquet, csf_parquet = run(48)
    assert outcome == "ok", outcome
    assert 40.0 <= peak <= 48.0, peak
    assert descriptor_parquet is not None
    assert csf_parquet is not None
    for path in (descriptor_parquet, csf_parquet):
        metadata = pq.ParquetFile(path).metadata
        assert metadata.num_row_groups > 1
        assert all(
            metadata.row_group(index).num_rows <= 8_192
            for index in range(metadata.num_row_groups)
        )


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


def test_estimate_checks_every_volume_it_is_given(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """The estimate owns the space model, including the volumes the CLI adds."""
    config = config_file(tmp_path)
    transcript = "* ! Orbital order\n0\n1s(2,*)\n\n1s\n0,0\n0\nn\n"
    report = estimate_disk_generation(
        transcript,
        scratch_dir=tmp_path,
        staging_dir=tmp_path,
        destinations={
            "csf_text": tmp_path / "out.c",
            "descriptor": tmp_path / "out_descriptors.parquet",
        },
    )
    del capfd, config
    checks = report["space_checks"]
    # Scratch, staging and both destinations share one volume here, so they are
    # checked as a single requirement rather than four independent ones.
    assert len(checks) == 1
    check = checks[0]
    assert check["required_bytes"] >= report["bytes"]["required_scratch"]
    assert check["free_bytes"] is not None
    assert check["sufficient"] is True


def test_an_unmeasurable_volume_is_reported_as_unchecked(tmp_path: Path) -> None:
    """An estimate reports what it found; refusing is the run's decision.

    A volume this platform cannot measure must not be reported as sufficient
    either, which is why `sufficient` is None rather than True.
    """
    transcript = "* ! Orbital order\n0\n1s(2,*)\n\n1s\n0,0\n0\nn\n"
    absent = tmp_path / "absent" / "scratch"
    report = estimate_disk_generation(transcript, scratch_dir=absent)
    # The reported path is the directory the free-space figure would describe.
    assert report["space_checks"] == [
        {
            "path": str(absent.parent),
            "required_bytes": report["bytes"]["required_scratch"],
            "free_bytes": None,
            "sufficient": None,
        }
    ]


def test_the_run_api_carries_the_space_policy(tmp_path: Path) -> None:
    """The generation API takes the opt-out the CLI flag maps onto.

    The refusal itself is a Rust unit test over `check_space`: this platform
    reports free space everywhere, so a Python test cannot construct the
    unmeasurable or insufficient case without a dedicated filesystem.
    """
    transcript = "* ! Orbital order\n0\n1s(2,*)\n\n1s\n0,0\n0\nn\n"
    stats = generate_disk_outputs_from_transcript(
        transcript,
        tmp_path / "out.c",
        tmp_path / "out.parquet",
        tmp_path / "out_descriptors.parquet",
        tmp_path / "out_header.toml",
        tmp_path / "scratch",
        allow_unchecked_space=True,
    )
    assert stats["success"] is True


def test_estimate_and_generation_reject_the_same_low_budget(tmp_path: Path) -> None:
    """An estimate must not succeed where the run it predicts would fail.

    The registered B2 input enumerates ~17 MiB of occupation arena, so a 1 MiB
    budget rejects both at enumeration. Charging the estimate an unlimited arena
    would have reported success for a run that cannot start.
    """
    transcript = (Path(__file__).parent / "fixtures" / "b2_cc1_fullas_2exc.rcsfgenerate").read_text()
    with pytest.raises(OSError, match="memory budget exceeded"):
        estimate_disk_generation(transcript, memory_budget_mib=1)
    with pytest.raises(OSError, match="memory budget exceeded"):
        generate_disk_outputs_from_transcript(
            transcript,
            tmp_path / "out.c",
            tmp_path / "out.parquet",
            tmp_path / "out_descriptors.parquet",
            tmp_path / "out_header.toml",
            tmp_path / "scratch",
            memory_budget_mib=1,
        )
    # Neither attempt may leave an artifact behind. The API owns the scratch
    # directory it was asked to create; the CLI removes its own on failure.
    assert [
        path
        for path in tmp_path.iterdir()
        if path.name != "scratch"
    ] == []


def test_cli_accepts_the_unchecked_space_opt_out(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """The opt-out is plumbed through, not silently ignored."""
    config = config_file(tmp_path)
    assert (
        cli.main(
            [
                "csfsgenerate",
                "--config",
                str(config),
                "--estimate-only",
                "--json",
                "--allow-unchecked-space",
            ]
        )
        == 0
    )
    payload = json.loads(capfd.readouterr().out)
    # Space is measurable here, so the checks still ran and still passed.
    assert payload["space_checks"]
    assert all(check["sufficient"] is True for check in payload["space_checks"])
