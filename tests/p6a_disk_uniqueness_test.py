"""P6a: registered fixtures generate without ever de-duplicating a row.

``tests/p6a_uniqueness_test.rs`` argues the structural property and checks it
exhaustively on bounded systems; a debug build cannot afford a full registered
input record by record. This file adds the other half of the evidence: each
registered fixture is generated completely through the *production* disk path,
where the exact de-duplication stage counts every row it removes.

That count is a real measurement over the whole run, so a change that
reintroduces a duplicate fails here with the number instead of passing
silently. The same run pins the end-of-generation invariant: the CSFs actually
produced equal the number the workload planner counted before the run.

The structural argument for why these runs cannot de-duplicate anything is
``docs/V2_GENERATION_UNIQUENESS.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rcsfs import generate_disk_outputs_from_transcript, get_parquet_info

FIXTURES = Path(__file__).parent / "fixtures"

#: Registered inputs and the number of CSFs each produces. Both are complete
#: ``rcsfgenerate`` transcripts that need no GRASP checkout; ``o1`` reaches odd
#: total 2J, ``e1`` even, so neither parity is left untested.
REGISTERED = [
    ("o1_cc1as1", 89_786),
    ("e1_cc1as1", 452_373),
]


@pytest.mark.parametrize(("name", "records"), REGISTERED)
def test_a_registered_run_generates_without_duplicates(
    name: str, records: int, tmp_path: Path
) -> None:
    transcript = (FIXTURES / f"{name}.rcsfgenerate").read_text(encoding="utf-8")
    csf = tmp_path / "calculation.c"
    csf_parquet = tmp_path / "calculation.parquet"
    descriptors = tmp_path / "calculation_descriptors.parquet"
    header = tmp_path / "calculation_header.toml"

    stats = generate_disk_outputs_from_transcript(
        transcript,
        csf,
        csf_parquet,
        descriptors,
        header,
        tmp_path / "scratch",
    )

    assert stats["success"] is True
    # `.get` because these statistics are optional in the shared TypedDict; a
    # missing key fails the comparison rather than passing quietly.
    assert stats.get("generated_count") == records
    # The measurement this test exists for: the de-duplication chain walked
    # every row of the run and removed none of them.
    assert stats.get("duplicate_count") == 0
    assert stats.get("record_count") == records
    # An unpublished artifact is not evidence of a duplicate-free run.
    assert get_parquet_info(descriptors)["num_rows"] == records
    assert descriptors.stat().st_size > 0
    assert csf.stat().st_size > 0
    assert csf_parquet.stat().st_size > 0
    assert header.is_file()
    # P5a counted the records the run went on to produce, which is what makes
    # the capacity estimate and the schedule trustworthy.
    plan_stats = stats.get("plan_stats")
    assert plan_stats is not None
    assert plan_stats["estimated_total_records"] == records
