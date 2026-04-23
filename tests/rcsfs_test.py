from pathlib import Path

from rcsfs import (
    convert_csfs,
    generate_descriptors_from_parquet,
    get_parquet_info,
    read_peel_subshells,
)


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_CSF = FIXTURES_DIR / "sample.csf"


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
