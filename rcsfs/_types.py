"""Shared public result types for the Python and native API boundaries."""

from typing import Literal, NotRequired, TypedDict

type InteractionHamiltonian = Literal[
    "dirac_coulomb",
    "dirac_coulomb_breit",
]
type InteractionMethod = Literal["structural_upper_bound"]


class CsfHeaderInfo(TypedDict):
    """The five source header lines retained from a CSF file."""

    header_lines: list[str]


class CsfBlockInfo(TypedDict):
    """Symmetry-block metadata extracted from ``*`` separators."""

    block_lengths: list[int]
    block_count: int


class CsfDataStats(TypedDict):
    """In-memory CSF parsing statistics matching the TOML sidecar schema."""

    csf_count: int
    total_lines: int
    truncated_count: int


class CsfHeaderData(TypedDict):
    """Header dictionary matching the TOML written by :func:`convert_csfs`."""

    header_info: CsfHeaderInfo
    block_info: CsfBlockInfo
    conversion_stats: CsfDataStats


class ConversionStats(TypedDict):
    """Statistics returned from CSF conversion operations."""

    success: bool
    input_file: NotRequired[str]
    output_file: NotRequired[str]
    parquet_file: NotRequired[str]
    header_file: NotRequired[str]
    max_line_len: NotRequired[int]
    chunk_size: NotRequired[int]
    error: NotRequired[str]
    total_lines: NotRequired[int]
    csf_count: NotRequired[int]
    truncated_count: NotRequired[int]


class ParquetInfo(TypedDict):
    """Metadata returned for a Parquet file."""

    file_path: str
    file_size: int
    num_rows: int
    num_columns: int
    compression: str
    created_by: str
    key_value_metadata: dict[str, str | None]


class DescriptorGenerationStats(TypedDict):
    """Statistics returned from batch descriptor generation."""

    success: bool
    input_file: str
    output_file: str
    csf_count: int
    descriptor_count: int
    orbital_count: int
    descriptor_size: int
    descriptor_version: int
    channels_per_subshell: int


class CsfRestoreStats(TypedDict):
    """Statistics returned from restoring CSFs out of a V2 descriptor file."""

    success: bool
    output_file: str
    record_count: int
    output_bytes: int


class PartitionStats(TypedDict):
    """Statistics returned from a zero-first partition operation."""

    success: bool
    zero_parquet: NotRequired[str]
    full_parquet: NotRequired[str]
    output_file: NotRequired[str]
    block_count: NotRequired[int]
    zero_csf_count: NotRequired[int]
    full_csf_count: NotRequired[int]
    output_csf_count: NotRequired[int]
    first_order_count: NotRequired[int]
    error: NotRequired[str]


class CsfGenerationStageStats(TypedDict):
    """Coarse logical measurements for one disk-generation stage."""

    name: str
    elapsed_millis: int
    cpu_millis: int | None
    input_records: int
    output_records: int
    input_bytes: int
    output_bytes: int


class CsfGenerationResourceStats(TypedDict):
    """Managed-memory accounting for a generation run."""

    memory_budget_mib: int | None
    budget_bytes: int | None
    peak_managed_bytes: int
    current_managed_bytes: int
    occupation_bytes: int


class CsfGenerationRecordDistribution(TypedDict):
    """Distribution of the counted record estimate across scheduled tasks."""

    count: int
    total: int
    minimum: int
    p50: int
    p95: int
    maximum: int


class CsfGenerationPlanStats(TypedDict):
    """Counted workload and the schedule the disk path derived from it.

    The estimates are pre-deduplication record counts taken before generation
    starts. ``estimated_total_records`` is not a promise: ``generated_count``
    reports what was actually produced, and the two are expected to agree.
    """

    task_count: int
    target_records_per_task: int
    estimated_records_per_task: CsfGenerationRecordDistribution
    estimated_total_records: int
    unique_occupations: int
    zero_record_configurations: int
    unsplittable_tasks: int
    unsplittable_records: int


#: Codec for the temporary Arrow IPC segments. Selected through the
#: ``RCSFS_SEGMENT_CODEC`` environment variable, which is a benchmark knob
#: rather than a public option; ``"none"`` is the default.
type CsfSegmentCodec = Literal["none", "lz4", "zstd"]

#: How the published descriptor was made unique. ``"verified_unique"`` is the
#: default for the internal generation path, which is proven not to repeat a
#: record: ``duplicate_count`` is then zero *by construction*. ``"exact"``
#: compares every row against every other row of its symmetry block, so its
#: ``duplicate_count`` is a measurement. Selected through
#: ``RCSFS_DEDUPLICATION``, a verification knob rather than a public option.
type CsfDeduplication = Literal["verified_unique", "exact"]


class CsfGenerationEstimateBytes(TypedDict):
    """Estimated size of one path of a disk generation run, in bytes.

    ``scratch_peak`` covers the Arrow segments, de-duplication buckets, the
    recursive repartition allowance and the survivor bitsets, which coexist.
    ``staged_outputs`` is the set the CLI publishes from, and it exists at the
    same time as the published files because publication copies.
    ``required_*`` are those values with the model's safety margin applied.
    """

    segments: int
    root_buckets: int
    recursive_buckets: int
    survivor_bitsets: int
    scratch_peak: int
    descriptor: int
    csf_text: int
    csf_parquet: int
    staged_outputs: int
    required_scratch: int
    required_output: int


class CsfGenerationSpaceCheck(TypedDict):
    """One volume's requirement: the largest simultaneous sum of its contents."""

    path: str
    required_bytes: int
    #: ``None`` when the platform did not report free space; the surrounding
    #: call fails in that case unless unchecked space was accepted explicitly.
    free_bytes: int | None
    sufficient: bool | None


class CsfGenerationEstimate(TypedDict):
    """Counted workload and capacity estimate for a transcript.

    Produced without creating a scratch directory or writing any output. Every
    estimate is an upper bound derived from the ratios listed in
    ``assumptions``; the ratios come from the registered benchmark inputs and
    the unmeasured terms are assumed conservatively rather than as zero.
    """

    success: bool
    unique_occupations: int
    pre_deduplication_records: int
    peel_subshells: int
    v2_columns: int
    #: The codec the predicted run would write its temporary segments with. The
    #: byte model is uncompressed, so a compressed run stays inside the
    #: estimate rather than exceeding it.
    segment_codec: CsfSegmentCodec
    #: The de-duplication strategy whose temporary-space requirements were
    #: priced by this estimate.
    deduplication: CsfDeduplication
    enumeration_millis: int
    planning_millis: int
    plan_stats: CsfGenerationPlanStats
    bytes: CsfGenerationEstimateBytes
    assumptions: list[str]
    #: One entry per volume the estimate was asked about, with the sum of the
    #: requirements that coexist on it. Absent when no paths were supplied.
    space_checks: NotRequired[list[CsfGenerationSpaceCheck]]
    #: ``"restart"``: scratch is not bound to an input hash or format version,
    #: so a failed run cannot continue from it.
    failure_recovery: Literal["restart"]


class CsfGenerationStats(TypedDict):
    """Statistics returned from transcript-driven CSF generation."""

    success: bool
    output_file: NotRequired[str]
    descriptor_file: NotRequired[str]
    record_count: NotRequired[int]
    block_count: NotRequired[int]
    unique_occupations: NotRequired[int]
    descriptor_count: NotRequired[int]
    generated_count: NotRequired[int]
    duplicate_count: NotRequired[int]
    csf_bytes: NotRequired[int]
    descriptor_bytes: NotRequired[int]
    #: The codec the temporary segments were written with, so a report can be
    #: checked against the run that produced it.
    segment_codec: NotRequired[CsfSegmentCodec]
    #: How `duplicate_count` was obtained: measured (`exact`) or zero by
    #: construction (`verified_unique`).
    deduplication: NotRequired[CsfDeduplication]
    stage_stats: NotRequired[list[CsfGenerationStageStats]]
    resource_stats: NotRequired[CsfGenerationResourceStats]
    plan_stats: NotRequired[CsfGenerationPlanStats]
    error: NotRequired[str]


class InteractionBlockStats(TypedDict):
    """Per-symmetry-block structural interaction statistics."""

    block_index: int
    total_two_j: int
    parity: Literal["+", "-"]
    reference_count: int
    candidate_count: int
    exact_reference_skipped: int
    selected_count: int
    rejected_count: int
    output_count: int


class InteractionStats(TypedDict):
    """Statistics from conservative structural interaction selection."""

    exact: Literal[False]
    hamiltonian: InteractionHamiltonian
    method: InteractionMethod
    reference_file: str
    candidate_file: str
    output_file: str
    block_count: int
    reference_count: int
    candidate_count: int
    exact_reference_skipped: int
    selected_count: int
    rejected_count: int
    output_count: int
    output_bytes: int
    blocks: list[InteractionBlockStats]
