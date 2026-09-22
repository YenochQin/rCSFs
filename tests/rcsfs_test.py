import shutil
from pathlib import Path

import pytest

from rcsfs import (
    convert_csfs,
    generate_csfs_from_transcript,
    generate_descriptors_from_parquet,
    generate_disk_outputs_from_transcript,
    get_parquet_info,
    read_peel_subshells,
    restore_csfs_from_descriptors,
    select_interacting_csfs,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_CSF = FIXTURES_DIR / "sample.csf"
E1_CC1AS1_TRANSCRIPT = FIXTURES_DIR / "e1_cc1as1.rcsfgenerate"


def test_descriptor_compression_keeps_legacy_positional_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rcsfs

    calls: dict[str, object] = {}

    def fake_generate(**kwargs: object) -> dict[str, object]:
        calls.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(rcsfs, "_generate_descriptors_from_parquet", fake_generate)
    result = rcsfs.generate_descriptors_from_parquet(
        "input.parquet", "output.parquet", ["1s"], None, False, "snappy"
    )
    assert result["success"] is True
    assert calls["compression"] == "snappy"
    assert calls["descriptor_version"] == 2


def test_end_to_end_public_python_api_defaults_to_v2(tmp_path: Path) -> None:
    csf_parquet = tmp_path / "sample.parquet"

    stats = convert_csfs(SAMPLE_CSF, csf_parquet, chunk_size=90, num_workers=2)

    assert stats["success"] is True
    assert stats["csf_count"] > 0
    assert Path(stats["header_file"]).exists()

    parquet_info = get_parquet_info(csf_parquet)
    assert parquet_info["num_rows"] == stats["csf_count"]
    assert parquet_info["num_columns"] == 4
    assert "UNCOMPRESSED" in parquet_info["compression"]
    assert "created_by" in parquet_info

    peel_subshells = read_peel_subshells(stats["header_file"])
    assert peel_subshells

    descriptor_parquet = tmp_path / "descriptors.parquet"
    descriptor_stats = generate_descriptors_from_parquet(
        csf_parquet,
        descriptor_parquet,
        peel_subshells=peel_subshells,
        num_workers=2,
        normalize=False,
        header_path=stats["header_file"],
    )

    expected_channels = 4
    expected_size = expected_channels * len(peel_subshells) + 2
    assert descriptor_stats["success"] is True
    assert descriptor_stats["csf_count"] == stats["csf_count"]
    assert descriptor_stats["descriptor_count"] == stats["csf_count"]
    assert descriptor_stats["descriptor_size"] == expected_size
    assert descriptor_stats["descriptor_version"] == 2
    assert descriptor_stats["channels_per_subshell"] == expected_channels

    descriptor_info = get_parquet_info(descriptor_parquet)
    assert descriptor_info["num_rows"] == stats["csf_count"]
    assert "ZSTD" in descriptor_info["compression"]
    kv = descriptor_info["key_value_metadata"]
    assert kv["descriptor_version"] == "2"
    assert kv["channels_per_subshell"] == str(expected_channels)
    assert "source_header_sha256" in kv


def test_select_interacting_csfs_public_api_is_explicitly_non_exact(
    tmp_path: Path,
) -> None:
    source = Path(__file__).parent / "fixtures" / "complete.csf"
    candidates = tmp_path / "candidates.csf"
    candidates.write_bytes(source.read_bytes())
    output = tmp_path / "selected.csf"

    stats = select_interacting_csfs(source, candidates, output, num_workers=2)

    assert stats["exact"] is False
    assert stats["method"] == "structural_upper_bound"
    assert stats["hamiltonian"] == "dirac_coulomb"
    assert stats["block_count"] == 2
    assert stats["reference_count"] == 2
    assert stats["candidate_count"] == 2
    assert stats["exact_reference_skipped"] == 2
    assert stats["selected_count"] == 0
    assert stats["output_count"] == 2
    assert [block["parity"] for block in stats["blocks"]] == ["-", "+"]
    assert output.read_bytes() == source.read_bytes()

    with pytest.raises(FileExistsError):
        _ = select_interacting_csfs(source, candidates, output)


@pytest.mark.parametrize("num_workers", [0, -1, -(2**63), 2**64, 10**30])
def test_select_interacting_csfs_rejects_non_positive_workers(
    tmp_path: Path, num_workers: int
) -> None:
    """Every unusable worker count is a ``ValueError``, per the public API.

    Negative and oversized integers must not escape as ``OverflowError`` from
    the native argument conversion before the documented check runs.
    """
    source = FIXTURES_DIR / "complete.csf"
    candidates = tmp_path / "candidates.csf"
    candidates.write_bytes(source.read_bytes())
    output = tmp_path / "selected.csf"

    with pytest.raises(ValueError):
        _ = select_interacting_csfs(source, candidates, output, num_workers=num_workers)
    assert not output.exists()


@pytest.mark.parametrize("num_workers", ["4", 1.5, object()])
def test_select_interacting_csfs_rejects_non_integer_workers(
    tmp_path: Path, num_workers: object
) -> None:
    source = FIXTURES_DIR / "complete.csf"
    candidates = tmp_path / "candidates.csf"
    candidates.write_bytes(source.read_bytes())

    with pytest.raises(TypeError):
        _ = select_interacting_csfs(
            source,
            candidates,
            tmp_path / "selected.csf",
            num_workers=num_workers,  # pyright: ignore[reportArgumentType]
        )


def test_select_interacting_csfs_accepts_extended_candidate_peel_list(
    tmp_path: Path,
) -> None:
    """The reference peel list only has to be a prefix of the candidate's.

    This relaxation of GRASP's identical-peel-list rule lets one reference
    space be reused against a candidate space that appends correlation
    orbitals. The candidate header becomes the output header.
    """
    reference = FIXTURES_DIR / "complete.csf"
    lines = reference.read_text().splitlines(keepends=True)
    # Line 4 is the peel subshell list; append an unoccupied orbital.
    assert lines[3].split() == ["4f-", "4f", "5d-", "5d"]
    lines[3] = lines[3].rstrip("\n") + "   6s \n"
    candidates = tmp_path / "candidates.csf"
    _ = candidates.write_text("".join(lines))
    output = tmp_path / "selected.csf"

    stats = select_interacting_csfs(reference, candidates, output, num_workers=2)

    assert stats["exact"] is False
    assert stats["block_count"] == 2
    assert stats["reference_count"] == 2
    # Records are unchanged, so each candidate still duplicates a reference.
    assert stats["exact_reference_skipped"] == 2
    assert stats["selected_count"] == 0
    assert stats["output_count"] == 2
    # The wider candidate header is what gets published.
    assert output.read_text().splitlines()[3].split() == [
        "4f-",
        "4f",
        "5d-",
        "5d",
        "6s",
    ]


def test_select_interacting_csfs_requires_distinct_inputs(tmp_path: Path) -> None:
    """One file cannot serve as both reference and candidate."""
    source = FIXTURES_DIR / "complete.csf"
    shared = tmp_path / "shared.csf"
    shared.write_bytes(source.read_bytes())
    output = tmp_path / "selected.csf"

    with pytest.raises(ValueError, match="different files"):
        _ = select_interacting_csfs(shared, shared, output)
    assert not output.exists()


def test_generate_csfs_from_transcript_matches_registered_e1_cc1as1(
    tmp_path: Path,
) -> None:
    transcript = E1_CC1AS1_TRANSCRIPT.read_text()
    output_csf = tmp_path / "out.c"

    stats = generate_csfs_from_transcript(transcript, output_csf)

    assert stats["success"] is True
    assert stats["output_file"] == str(output_csf)
    assert stats["record_count"] == 452_373
    assert stats["block_count"] == 7
    assert output_csf.exists()


def test_generate_csfs_from_transcript_reports_failure(tmp_path: Path) -> None:
    stats = generate_csfs_from_transcript("not a valid transcript", tmp_path / "out.c")

    assert stats["success"] is False
    assert "error" in stats


def test_normalize_path_tolerates_normalization_errors(tmp_path: Path) -> None:
    csf_parquet = tmp_path / "sample.parquet"
    convert_stats = convert_csfs(SAMPLE_CSF, csf_parquet, chunk_size=90, num_workers=2)

    bad_subshells = ["xyz"]
    normalized_output = tmp_path / "normalized_bad_subshells.parquet"
    normalized_stats = generate_descriptors_from_parquet(
        csf_parquet,
        normalized_output,
        peel_subshells=bad_subshells,
        num_workers=2,
        normalize=True,
        descriptor_version=1,
    )

    assert normalized_stats["success"] is True
    assert normalized_stats["csf_count"] == convert_stats["csf_count"]
    assert normalized_stats["descriptor_count"] == convert_stats["csf_count"]
    assert normalized_stats["descriptor_size"] == 3 * len(bad_subshells)

    normalized_info = get_parquet_info(normalized_output)
    assert normalized_info["num_rows"] == convert_stats["csf_count"]
    assert "ZSTD" in normalized_info["compression"]


@pytest.mark.parametrize(
    "transcript",
    [
        "* ! Orbital order\n0\n2p(2,*)\n\n3s,3p,3d\n0,4\n2\nn\n",
        "* ! Orbital order\n0\n3d(4,*)\n\n3s,3p,3d\n0,4\n0\nn\n",
    ],
)
def test_generated_descriptors_match_text_pipeline(
    tmp_path: Path, transcript: str
) -> None:
    import tomllib

    import polars as pl

    for normalize in (False, True):
        work = tmp_path / str(normalize)
        work.mkdir()
        csf = work / "out.c"
        from rcsfs import cli

        direct_path = work / "direct.parquet"
        stats = generate_csfs_from_transcript(
            transcript,
            csf,
            normalize=normalize,
            threads=2,
        )
        assert stats["success"], stats
        converted = convert_csfs(csf, work / "text.parquet")
        assert converted["success"], converted
        shells = read_peel_subshells(converted["header_file"])
        derived = generate_descriptors_from_parquet(
            work / "text.parquet",
            work / "features.parquet",
            shells,
            normalize=normalize,
            # csfsgenerate --generate-descriptors still hardcodes V1
            # (plan step 13 is deferred); match it here so the two
            # pipelines are directly comparable.
            descriptor_version=1,
        )
        assert derived["success"], derived
        lines = iter(["*", *transcript.splitlines()[1:]])
        from unittest.mock import patch

        with patch.object(
            cli, "_prompt", side_effect=lambda _, lines=lines: next(lines)
        ):
            argv = [
                "csfsgenerate",
                str(work / "cli.c"),
                "--generate-descriptors",
                "--descriptor-parquet",
                str(direct_path),
            ]
            if normalize:
                argv.append("--normalize")
            assert cli.main(argv) == 0
        direct = pl.read_parquet(direct_path)
        expected = pl.read_parquet(work / "features.parquet")
        assert direct.shape == expected.shape
        for actual_row, expected_row in zip(
            direct.iter_rows(), expected.iter_rows(), strict=True
        ):
            if normalize:
                assert actual_row == pytest.approx(expected_row, abs=1e-7)
            else:
                assert actual_row == expected_row
        assert (
            tomllib.loads(direct_path.with_suffix(".toml").read_text())["subshells"]
            == shells
        )


def test_restore_csfs_cli_roundtrip(tmp_path: Path) -> None:
    """V2 descriptors round-trip through `rcsfs restore-csfs` byte for byte.

    Built from a generator transcript rather than `fixtures/complete.csf`:
    that fixture predates `validate_record` and prints a seniority digit for
    a many-electron manifold GRASP's own state table never labels, so it is
    not a legal V2 record.
    """
    from rcsfs import cli

    transcript = "* ! Orbital order\n0\n2p(2,*)\n\n3s,3p,3d\n0,4\n2\nn\n"
    source = tmp_path / "generated.c"
    generation_stats = generate_csfs_from_transcript(transcript, source, threads=2)
    assert generation_stats["success"] is True
    source_lines = source.read_text().splitlines()
    source_lines.insert(8, " *")
    source.write_text("\n".join(source_lines) + "\n")

    csf_parquet = tmp_path / "complete.parquet"
    stats = convert_csfs(source, csf_parquet, num_workers=2)
    assert stats["success"] is True
    header_path = Path(stats["header_file"])
    peel_subshells = read_peel_subshells(header_path)
    _ = shutil.copyfile(header_path, csf_parquet.with_name("complete_header.toml"))

    descriptor_parquet = tmp_path / "complete_desc.parquet"
    descriptor_stats = generate_descriptors_from_parquet(
        csf_parquet,
        descriptor_parquet,
        peel_subshells=peel_subshells,
        descriptor_version=2,
    )
    assert descriptor_stats["success"] is True

    parquet_info = get_parquet_info(descriptor_parquet)
    kv = parquet_info["key_value_metadata"]
    assert kv["descriptor_version"] == "2"
    assert kv["channels_per_subshell"] == "4"
    assert set(kv) >= {
        "descriptor_version",
        "channels_per_subshell",
        "subshell_count",
        "peel_subshells",
        "missing_sentinel",
        "normalized",
        "feature_columns",
        "global_columns",
        "source_header_sha256",
        "source_header_filename",
    }

    restored = tmp_path / "restored.c"
    exit_code = cli.main(
        [
            "restore-csfs",
            "--descriptors",
            str(descriptor_parquet),
            "--header",
            str(header_path),
            "--output",
            str(restored),
        ]
    )
    assert exit_code == 0
    assert restored.read_bytes() == source.read_bytes()

    # Direct API call, restoring a subset by index, still matches source order.
    subset_restored = tmp_path / "restored_subset.c"
    subset_stats = restore_csfs_from_descriptors(
        descriptor_parquet, header_path, subset_restored, indices=[1]
    )
    assert subset_stats["success"] is True
    assert subset_stats["record_count"] == 1


@pytest.mark.parametrize("threads", [1, 2])
def test_disk_generation_scratch_layout_and_v2_roundtrip(
    tmp_path: Path, threads: int, capfd: pytest.CaptureFixture[str]
) -> None:
    transcript = "* ! Orbital order\n0\n2s(2,*)2p(1,*)\n\n3s,3p\n1,3\n1\nn\n"
    csf = tmp_path / "generated.c"
    csf_parquet = tmp_path / "generated.parquet"
    descriptors = tmp_path / "generated_descriptors.parquet"
    header = tmp_path / "generated_header.toml"
    scratch = tmp_path / "scratch"

    stats = generate_disk_outputs_from_transcript(
        transcript,
        csf,
        csf_parquet,
        descriptors,
        header,
        scratch,
        threads=threads,
        memory_budget_mib=64,
    )

    assert stats["success"] is True
    assert stats["resource_stats"]["memory_budget_mib"] == 64
    assert stats["resource_stats"]["peak_managed_bytes"] <= 64 * 1024 * 1024
    assert stats["record_count"] > 0
    assert "Range progress:" in capfd.readouterr().err
    assert [stage["name"] for stage in stats["stage_stats"]] == [
        "enumeration",
        "workload_planning",
        "csf_generation",
        "deduplication",
        "final_encoding_prepare",
        "final_encoding_descriptor_encode",
        "final_encoding_descriptor_write",
        "final_encoding_csf_outputs_encode",
        "final_encoding_csf_outputs_write",
    ]
    assert stats["plan_stats"]["estimated_total_records"] == stats["generated_count"]
    assert all(stage["elapsed_millis"] >= 0 for stage in stats["stage_stats"])
    assert all(
        stage["cpu_millis"] is None or stage["cpu_millis"] >= 0
        for stage in stats["stage_stats"]
    )
    assert list((scratch / "ranges").glob("range-*/*.arrow"))
    assert not (scratch / "ranges" / "ranges").exists()
    assert get_parquet_info(csf_parquet)["num_rows"] == stats["record_count"]
    descriptor_info = get_parquet_info(descriptors)
    assert descriptor_info["num_rows"] == stats["record_count"]
    assert descriptor_info["key_value_metadata"]["descriptor_version"] == "2"

    restored = tmp_path / "restored.c"
    restored_stats = restore_csfs_from_descriptors(descriptors, header, restored)
    assert restored_stats["record_count"] == stats["record_count"]
    assert restored.read_bytes() == csf.read_bytes()


def test_disk_generation_rejects_budget_before_unbounded_bucket_buffers(
    tmp_path: Path,
) -> None:
    transcript = "* ! Orbital order\n0\n1s(2,*)\n\n1s\n0,0\n0\nn\n"
    with pytest.raises(OSError, match="memory budget exceeded"):
        generate_disk_outputs_from_transcript(
            transcript,
            tmp_path / "generated.c",
            tmp_path / "generated.parquet",
            tmp_path / "generated_descriptors.parquet",
            tmp_path / "generated_header.toml",
            tmp_path / "scratch",
            threads=1,
            memory_budget_mib=1,
        )


def test_config_generation_uses_disk_v2_pipeline(tmp_path: Path) -> None:
    """A descriptor-producing TOML run uses bounded disk generation by default."""
    from rcsfs import cli

    csf = tmp_path / "calculation.c"
    parquet = tmp_path / "calculation.parquet"
    descriptors = tmp_path / "calculation_descriptors.parquet"
    config = tmp_path / "gencsfs.toml"
    config.write_text(
        f"""
[generate]
order = "*"
core = 0
references = ["2p(2,*)"]
active_orbitals = "3s,3p,3d"
j_min = 0
j_max = 4
excitations = 2
continue_lists = false

[output]
generate_descriptors = true
csf = "{csf}"
parquet = "{parquet}"
descriptor_parquet = "{descriptors}"
normalize = false
""",
        encoding="utf-8",
    )

    assert cli.main(["csfsgenerate", "--config", str(config)]) == 0
    assert csf.exists()
    assert parquet.exists()
    assert descriptors.exists()
    assert (tmp_path / "calculation_header.toml").exists()
    assert (
        get_parquet_info(descriptors)["key_value_metadata"]["descriptor_version"] == "2"
    )
