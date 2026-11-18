from pathlib import Path

import pytest

from rcsfs import (
    convert_csfs,
    descriptor_header_path,
    generate_csfs_from_transcript,
    get_parquet_info,
    read_csfs,
    read_peel_subshells,
    restore_csfs_from_descriptors,
    select_csfs,
    select_interacting_csfs,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
E1_CC1AS1_TRANSCRIPT = FIXTURES_DIR / "e1_cc1as1.rcsfgenerate"


def test_end_to_end_public_python_api_defaults_to_v2(tmp_path: Path) -> None:
    source = FIXTURES_DIR / "descriptor_multiblock.c"
    output = tmp_path / "basis_descriptors.parquet"
    stats = convert_csfs(source, output)
    assert stats["record_count"] == 2
    header = Path(stats["header_file"])
    info = get_parquet_info(output)
    assert info["num_rows"] == 2
    assert info["num_columns"] == 4 * len(read_peel_subshells(header)) + 2
    assert "ZSTD" in info["compression"]
    assert info["key_value_metadata"]["descriptor_version"] == "2"
    assert info["key_value_metadata"]["channels_per_subshell"] == "4"
    selected = tmp_path / "selected_descriptors.parquet"
    select_csfs(output, selected, [1])
    restored = tmp_path / "restored.c"
    restore_csfs_from_descriptors(selected, descriptor_header_path(selected), restored)
    reimported = tmp_path / "reimported_descriptors.parquet"
    convert_csfs(restored, reimported)
    assert read_csfs(reimported)[1].equals(read_csfs(selected)[1])
    assert read_csfs(reimported)[0]["block_info"]["block_lengths"] == [1]


def _interaction_inputs(tmp_path: Path) -> tuple[Path, Path]:
    text = FIXTURES_DIR / "descriptor_multiblock.c"
    reference = tmp_path / "reference_descriptors.parquet"
    candidates = tmp_path / "candidates_descriptors.parquet"
    convert_csfs(text, reference)
    convert_csfs(text, candidates)
    return reference, candidates


def test_select_interacting_csfs_public_api_is_explicitly_non_exact(
    tmp_path: Path,
) -> None:
    source, candidates = _interaction_inputs(tmp_path)
    output = tmp_path / "selected_descriptors.parquet"

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
    assert [block["parity"] for block in stats["blocks"]] == ["+", "-"]
    assert read_csfs(output)[1].equals(read_csfs(source)[1])

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
    source, candidates = _interaction_inputs(tmp_path)
    output = tmp_path / "selected_descriptors.parquet"

    with pytest.raises(ValueError):
        _ = select_interacting_csfs(source, candidates, output, num_workers=num_workers)
    assert not output.exists()


@pytest.mark.parametrize("num_workers", ["4", 1.5, object()])
def test_select_interacting_csfs_rejects_non_integer_workers(
    tmp_path: Path, num_workers: object
) -> None:
    source, candidates = _interaction_inputs(tmp_path)

    with pytest.raises(TypeError):
        _ = select_interacting_csfs(
            source,
            candidates,
            tmp_path / "selected_descriptors.parquet",
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
    fixture = FIXTURES_DIR / "descriptor_multiblock.c"
    text = tmp_path / "extended.c"
    lines = fixture.read_text().splitlines()
    lines[3] += "   6s"
    text.write_text("\n".join(lines) + "\n")
    reference = tmp_path / "reference_descriptors.parquet"
    candidates = tmp_path / "candidates_descriptors.parquet"
    output = tmp_path / "selected_descriptors.parquet"
    convert_csfs(fixture, reference)
    convert_csfs(text, candidates)

    stats = select_interacting_csfs(reference, candidates, output, num_workers=2)
    assert stats["exact_reference_skipped"] == 2
    assert read_peel_subshells(descriptor_header_path(output)) == ["2s", "2p", "6s"]
    assert read_csfs(output)[1].equals(read_csfs(candidates)[1])


def test_select_interacting_csfs_requires_distinct_inputs(tmp_path: Path) -> None:
    """One file cannot serve as both reference and candidate."""
    shared, _ = _interaction_inputs(tmp_path)
    output = tmp_path / "selected_descriptors.parquet"

    with pytest.raises(ValueError, match="different files"):
        _ = select_interacting_csfs(shared, shared, output)
    assert not output.exists()


def test_generate_csfs_from_transcript_matches_registered_e1_cc1as1(
    tmp_path: Path,
) -> None:
    transcript = E1_CC1AS1_TRANSCRIPT.read_text()
    output_csf = tmp_path / "out_descriptors.parquet"

    stats = generate_csfs_from_transcript(transcript, output_csf)

    assert stats["success"] is True
    assert stats["output_file"] == str(output_csf)
    assert stats["record_count"] == 452_373
    assert stats["block_count"] == 7
    assert output_csf.exists()


def test_generate_csfs_from_transcript_reports_failure(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        generate_csfs_from_transcript(
            "not a valid transcript", tmp_path / "out_descriptors.parquet"
        )
    assert not list(tmp_path.iterdir())
