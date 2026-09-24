# Target machine V2 generation measurements (2026-09-24)

This campaign used a 48-logical-CPU Linux host with 810,775,953,408 bytes of
physical memory. The test directory is on an NVMe-backed XFS filesystem, as
confirmed separately with `findmnt -T` on the host. The JSON reports record
`filesystem.type: null` because the current metadata helper parses the macOS
`mount` syntax, not Linux's `type xfs` syntax; free and total byte counts came
from a separate disk-usage call and are unaffected. The page cache was not
cleared, and no competing workload or CPU quota was recorded.

The registered B1/B2 transcripts were measured with one warm-up and three
measurements per combination, in a fresh process for each call. The timed
region includes generation and artifact publication, and excludes cleanup.
Every run completed without a resource refusal. Source was clean: the initial
thread/budget reports name `7da458b`; the codec/thread reports name `d032be5`.
The intervening commit changed only the plan document, and all five reports
name the same loaded extension SHA-256,
`27303f2ac1c7cfb4793dbeb333e829d5eb29fca843e0a19551ec0e2e8f3941ee`.

Raw reports:

- [B1, 1/2/4/8 threads and 1024/8192 MiB](v2_target_xfs_threads_b1_20260924.json)
- [B2, 1/2/4/8 threads and 1024/8192 MiB](v2_target_xfs_threads_b2_20260924.json)
- [B1, 8/16/32 threads and three codecs](v2_target_xfs_codec_threads_b1_20260924.json)
- [B2, 8/16/32 threads and three codecs](v2_target_xfs_codec_threads_b2_20260924.json)
- [B3 count, planner and capacity estimate only](v2_target_xfs_capacity_b3_20260924.json)

## Thread and codec results

All figures below are median end-to-end seconds at an 8192 MiB managed-memory
budget. The 8-thread figures are from the same campaign as the 16/32-thread
figures, so the comparison does not mix sessions.

| Input | Segment codec | 8 threads | 16 threads | 32 threads | Scratch peak at 32 threads |
| --- | --- | ---: | ---: | ---: | ---: |
| B1 | none | 9.234 | 8.329 | **7.413** | 1,073 MiB |
| B1 | lz4 | 10.692 | 9.922 | 8.881 | 25.5 MiB |
| B1 | zstd | 10.436 | 9.287 | 7.859 | 12.5 MiB |
| B2 | none | **2.395** | 2.620 | 2.621 | 506 MiB |
| B2 | lz4 | 3.027 | 2.831 | 2.956 | 8.3 MiB |
| B2 | zstd | 2.735 | 2.526 | 2.490 | 4.5 MiB |

B1 uncompressed improves by 20% in elapsed time from 8 to 32 threads, with
most of the gain in generation and final-encoding preparation. B2 is already
short enough that more threads do not improve the end-to-end result. The
same-machine 8-thread timings in the earlier report were 9.605/2.567 seconds
for B1/B2; this session measured 9.234/2.395 seconds with the same extension,
so cross-session variation should not be mistaken for a source change.

At 32 threads, zstd cuts scratch by about 99% relative to uncompressed
segments. B1 takes 6% longer with zstd, while lz4 takes 20% longer. B2's
32-thread zstd result is slightly faster than 32-thread none, but the fastest
B2 combination is 8-thread none and the absolute differences are small. Keep
`none` as the throughput default; zstd is the useful option when scratch
capacity matters. These measurements do not justify changing the capacity
model to assume compression: it must still preflight a safe upper bound.

Across all codec/thread combinations, each input has one CSF text digest, one
CSF Parquet digest, one header digest and the registered record count. B1's
descriptor Parquet bytes differ between thread counts because its row-group
layout changes; within a fixed thread count, its digest is identical across
codecs. The physical descriptor digest is not a logical-row comparison. B2's
descriptor digest is identical in all nine combinations. At 32 threads the
largest observed single-process RSS in the uncompressed runs is below 0.52 GB
for either input, and managed peaks are below 0.29 GB. Neither approached the
8192 MiB managed budget.

## B3 capacity and failure paths

The B3 estimate finished in 24.65 seconds without generating any CSF file:
7,027,846 configurations, 5,811,925,522 pre-deduplication records and 1,454
planned tasks. Task p50/p95/max is 3,999,140 / 3,999,948 / 4,000,000 records,
with no unsplittable task. The uncompressed model estimates 1.866 TB for CSF
text alone, 5.853 TB of scratch peak, and 10.195 TB required on the shared
scratch/staging/output volume after coexistence and safety allowance. That
volume reported 1.388 TB free. Its `space_checks[0].sufficient` is `false`,
so a full B3 generation must not be started there. Segment compression cannot
make the final text fit on that volume.

The user also ran the following focused failure tests on this host and supplied
their terminal output: `tests/publication_test.py` (4 passed), the two selected
budget-refusal tests in `tests/generation_parquet_test.py` (2 passed, 30
deselected), and the Rust tests
`failed_segment_merge_does_not_publish_a_descriptor` and
`failed_deduplicated_merge_does_not_publish_a_descriptor` (1 passed each).
The filtered-out Rust suites report zero tests as expected. These tests cover
simulated publication races/cross-volume errors, bounded-memory rejection and
failed merge cleanup. They do not inject a stalled consumer, actual target
filesystem exhaustion, or a real cross-volume publication failure.

Target-machine samples now exist for P0b/P5a/P1.5 and the P2a codec decision.
P0b's low/mid/high budget matrix and slow-consumer fault injection remain
incomplete; root/recursive bucket ratios for the exact path were not measured
because this campaign used `verified_unique`. The full four-excitation B4
input remains blocked by the reported capacity shortage, independent of the
successful three-excitation GRASP text comparison.
