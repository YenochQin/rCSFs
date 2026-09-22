"""P6a: registered fixtures generate without ever de-duplicating a row.

``tests/p6a_uniqueness_test.rs`` argues the structural property and checks it
exhaustively on bounded systems; a debug build cannot afford a full registered
input record by record. This file adds the other half of the evidence: each
registered fixture is generated completely through the *production* disk path,
where the exact de-duplication stage compares every row of a symmetry block
against every other and counts what it removes.

That count is a real measurement over the whole run, so a change that
reintroduces a duplicate fails here with the number instead of passing
silently. The runs below therefore ask for ``exact`` explicitly: the default
strategy is the proof-backed one, whose ``duplicate_count`` is zero *by
construction* and would make this evidence vacuous. The same runs pin the
end-of-generation invariant — the CSFs actually produced equal the number the
workload planner counted before the run.

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


def run_fixture(
    name: str,
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    strategy: str | None = None,
) -> tuple[dict[str, object], Path, Path]:
    """Generate one registered fixture into its own output set.

    The strategy is set explicitly in both directions, so the run that is meant
    to exercise the default cannot inherit a strategy from the outer
    environment.
    """
    if strategy is None:
        monkeypatch.delenv("RCSFS_DEDUPLICATION", raising=False)
    else:
        monkeypatch.setenv("RCSFS_DEDUPLICATION", strategy)
    directory.mkdir()
    transcript = (FIXTURES / f"{name}.rcsfgenerate").read_text(encoding="utf-8")
    csf = directory / "calculation.c"
    csf_parquet = directory / "calculation.parquet"
    descriptors = directory / "calculation_descriptors.parquet"
    header = directory / "calculation_header.toml"
    stats = generate_disk_outputs_from_transcript(
        transcript,
        csf,
        csf_parquet,
        descriptors,
        header,
        directory / "scratch",
    )
    return stats, csf, descriptors


@pytest.mark.parametrize(("name", "records"), REGISTERED)
def test_a_registered_run_generates_without_duplicates(
    name: str, records: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stats, csf, descriptors = run_fixture(name, tmp_path / "exact", monkeypatch, "exact")

    assert stats["success"] is True
    assert stats.get("deduplication") == "exact"
    # `.get` because these statistics are optional in the shared TypedDict; a
    # missing key fails the comparison rather than passing quietly.
    assert stats.get("generated_count") == records
    # The measurement this test exists for: the comparison walked every row of
    # the run and removed none of them.
    assert stats.get("duplicate_count") == 0
    assert stats.get("record_count") == records
    # An unpublished artifact is not evidence of a duplicate-free run.
    assert get_parquet_info(descriptors)["num_rows"] == records
    assert descriptors.stat().st_size > 0
    assert csf.stat().st_size > 0
    # P5a counted the records the run went on to produce, which is what makes
    # the capacity estimate and the schedule trustworthy.
    plan_stats = stats.get("plan_stats")
    assert plan_stats is not None
    assert plan_stats["estimated_total_records"] == records


def test_the_verified_path_publishes_what_the_exact_path_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The skipped comparison must not change a single published byte.

    ``o1`` is the smaller registered fixture; the same differential runs on B1
    and B2 in ``docs/benchmarks/v2_disk_generation_dedup_20260922.md``, where it
    also measures what the skipped round trip cost.
    """
    exact, exact_csf, exact_descriptors = run_fixture(
        "o1_cc1as1", tmp_path / "exact", monkeypatch, "exact"
    )
    verified, verified_csf, verified_descriptors = run_fixture(
        "o1_cc1as1", tmp_path / "verified", monkeypatch
    )

    assert verified.get("deduplication") == "verified_unique"
    assert verified.get("duplicate_count") == 0
    assert verified.get("deduplication") != exact.get("deduplication")
    # Same counts, same block structure, same published files.
    for key in ("generated_count", "record_count", "descriptor_count", "block_count"):
        assert verified.get(key) == exact.get(key), f"{key} differs between the paths"
    assert verified_csf.read_bytes() == exact_csf.read_bytes()
    assert verified_descriptors.read_bytes() == exact_descriptors.read_bytes()
