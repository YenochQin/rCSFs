from pathlib import Path

import pytest

from rcsfs import (
    convert_csfs,
    generate_csfs_from_transcript,
    generate_descriptors_from_parquet,
    get_parquet_info,
    read_peel_subshells,
)


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_CSF = FIXTURES_DIR / "sample.csf"
E1_CC1AS1_TRANSCRIPT = FIXTURES_DIR / "e1_cc1as1.rcsfgenerate"


def test_end_to_end_public_python_api(tmp_path: Path) -> None:
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
    )

    assert descriptor_stats["success"] is True
    assert descriptor_stats["csf_count"] == stats["csf_count"]
    assert descriptor_stats["descriptor_count"] == stats["csf_count"]
    assert descriptor_stats["descriptor_size"] == 3 * len(peel_subshells)

    descriptor_info = get_parquet_info(descriptor_parquet)
    assert descriptor_info["num_rows"] == stats["csf_count"]
    assert "ZSTD" in descriptor_info["compression"]


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
    import polars as pl
    import tomllib

    for normalize in (False, True):
        work = tmp_path / str(normalize)
        work.mkdir()
        csf = work / "out.c"
        csv = work / "direct.csv"
        stats = generate_csfs_from_transcript(
            transcript,
            csf,
            descriptor_path=csv,
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
        )
        assert derived["success"], derived
        direct = pl.read_csv(csv, has_header=False)
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
            tomllib.loads(csv.with_suffix(".toml").read_text())["subshells"] == shells
        )
