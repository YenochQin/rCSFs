//! Bounded V2 descriptor generation through ordered on-disk Arrow segments.
//!
//! This module owns the storage adapter: range scheduling and the segment
//! writers, the exact de-duplication chain, and the segment-level merge that
//! the one-pass final encoding (`super::final_encoding`) or the two-pass
//! reference path consumes. Keeping the range writer here lets the angular
//! generator use `GeneratedRecordSink` without learning about Arrow, scratch
//! paths or final publication.

use anyhow::{Context, Result, ensure};
use arrow::array::{Array, Int32Array, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::{FileWriter, IpcWriteOptions};
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::{
    ArtifactKind, CapacityEstimate, SpaceCheck, SpacePolicy, SpaceRole, check_space, preflight_run,
};
use super::{
    DeduplicationStrategy, EnumeratedConfiguration, EnumeratedOccupations, ExcitationRequest,
    GeneratedRecordRef, GeneratedRecordSink, GenerationOptions, GenerationPlan, Parity, PlanStats,
    PlannedTask, RecordSelection, ResourceBudget, ResourcePermit, ResourceStats, SegmentCodec,
    Subshell, SubshellOccupation, TaskSpan, enumerate_occupations_with_budget, estimate_capacity,
    estimate_workload, generate_configuration_records, plan_generation, report_plan,
    request_targets,
};
use crate::atomic_output::{create_temporary_output, publish_temporary_output};
use crate::complete_csf::OccupiedSubshell;
use crate::descriptor_schema::{
    DescriptorLayout, DescriptorVersion, output_kv_metadata, output_schema, validate_record,
};
use crate::descriptor_v2::write_feature_row;

const DEFAULT_DEDUP_BUCKET_COUNT: usize = 256;
const DEFAULT_DEDUP_MAX_ROWS_PER_BUCKET: usize = 65_536;
const DEDUP_RECURSIVE_BUCKET_COUNT: usize = 16;
const MAX_DEDUP_SPLIT_DEPTH: usize = 16;

const PROGRESS_REPORT_INTERVAL: Duration = Duration::from_secs(2);

/// A coarse stage measurement exposed through the disk-generation result.
///
/// Byte counters describe files observed at the stage boundary. They are
/// logical file bytes, not a claim about physical device I/O or page-cache
/// traffic.
#[derive(Clone, Debug)]
pub(crate) struct StageStats {
    pub(crate) name: &'static str,
    pub(crate) elapsed_millis: u128,
    pub(crate) cpu_millis: Option<u128>,
    pub(crate) input_records: usize,
    pub(crate) output_records: usize,
    pub(crate) input_bytes: u64,
    pub(crate) output_bytes: u64,
}

#[derive(Debug)]
struct StageTimer {
    wall: Instant,
    cpu_millis: Option<u128>,
}

impl StageTimer {
    fn start() -> Self {
        Self {
            wall: Instant::now(),
            cpu_millis: process_cpu_millis(),
        }
    }

    fn finish(
        self,
        name: &'static str,
        input_records: usize,
        output_records: usize,
        input_bytes: u64,
        output_bytes: u64,
    ) -> StageStats {
        let cpu_millis = process_cpu_millis()
            .zip(self.cpu_millis)
            .map(|(end, start)| end.saturating_sub(start));
        StageStats {
            name,
            elapsed_millis: self.wall.elapsed().as_millis(),
            cpu_millis,
            input_records,
            output_records,
            input_bytes,
            output_bytes,
        }
    }
}

#[cfg(unix)]
pub(crate) fn process_cpu_millis() -> Option<u128> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage initializes the rusage structure when it returns 0.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return None;
    }
    // SAFETY: the successful getrusage call initialized `usage`.
    let usage = unsafe { usage.assume_init() };
    let user_micros = u128::try_from(usage.ru_utime.tv_sec)
        .ok()?
        .checked_mul(1_000_000)?
        .checked_add(u128::try_from(usage.ru_utime.tv_usec).ok()?)?;
    let system_micros = u128::try_from(usage.ru_stime.tv_sec)
        .ok()?
        .checked_mul(1_000_000)?
        .checked_add(u128::try_from(usage.ru_stime.tv_usec).ok()?)?;
    Some((user_micros + system_micros) / 1_000)
}

#[cfg(not(unix))]
fn process_cpu_millis() -> Option<u128> {
    None
}

#[derive(Debug)]
struct RangeProgress {
    total: usize,
    started: AtomicUsize,
    completed: AtomicUsize,
    records: AtomicUsize,
    last_report: Mutex<Instant>,
}

impl RangeProgress {
    fn new(total: usize) -> Self {
        Self {
            total,
            started: AtomicUsize::new(0),
            completed: AtomicUsize::new(0),
            records: AtomicUsize::new(0),
            last_report: Mutex::new(Instant::now() - PROGRESS_REPORT_INTERVAL),
        }
    }

    fn range_started(&self) {
        self.started.fetch_add(1, Ordering::Relaxed);
        self.report(false);
    }

    fn range_completed(&self, records: usize) {
        self.completed.fetch_add(1, Ordering::Relaxed);
        self.records.fetch_add(records, Ordering::Relaxed);
        self.report(false);
    }

    fn finish(&self) {
        self.report(true);
    }

    fn report(&self, force: bool) {
        let Ok(mut last_report) = self.last_report.lock() else {
            return;
        };
        if !force && last_report.elapsed() < PROGRESS_REPORT_INTERVAL {
            return;
        }
        *last_report = Instant::now();
        eprintln!(
            "Range progress: started {}/{}; completed {}/{}; generated {} CSFs",
            self.started.load(Ordering::Relaxed),
            self.total,
            self.completed.load(Ordering::Relaxed),
            self.total,
            self.records.load(Ordering::Relaxed),
        );
    }
}

/// A deterministic, contiguous slice of enumerated configurations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GenerationRange {
    pub(crate) ordinal: u32,
    pub(crate) start_configuration: usize,
    pub(crate) end_configuration: usize,
}

/// Compatibility alias for the pre-P1 name used by the Rust maintenance
/// tests. New production code passes the shared [`GenerationOptions`] directly.
pub(crate) type StreamingGenerationOptions = GenerationOptions;

/// One finalized Arrow IPC segment.  The file carries V2 columns followed by
/// `range_ordinal` and `local_ordinal`; the latter two are storage ordering
/// columns and never appear in the final descriptor Parquet file.
#[derive(Clone, Debug)]
pub(crate) struct DescriptorSegment {
    pub(crate) path: PathBuf,
    pub(crate) range_ordinal: u32,
    pub(crate) local_start: u64,
    pub(crate) total_two_j: u16,
    pub(crate) parity: Parity,
    pub(crate) record_count: usize,
    pub(crate) byte_count: u64,
    /// Maximum rows in an IPC record batch. Final encoding uses this bound to
    /// reserve decode memory before asking the reader to materialize a batch.
    pub(crate) rows_per_batch: usize,
}

/// Completed generation stage before deduplication.
#[derive(Debug)]
pub(crate) struct SegmentGeneration {
    pub(crate) layout: DescriptorLayout,
    pub(crate) peel_subshells: Vec<String>,
    pub(crate) core_subshells: Vec<Subshell>,
    /// The counted workload and the schedule derived from it.
    pub(crate) plan: GenerationPlan,
    pub(crate) segments: Vec<DescriptorSegment>,
    pub(crate) unique_occupations: usize,
    pub(crate) record_count: usize,
    /// The codec the segments were actually written with, so a report can be
    /// checked against the run that produced it.
    pub(crate) segment_codec: SegmentCodec,
    pub(crate) stage_stats: Vec<StageStats>,
    pub(crate) budget: ResourceBudget,
    pub(crate) resource_stats: ResourceStats,
}

/// Result of the ordered segment merge.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct SegmentMergeStats {
    pub(crate) record_count: usize,
    pub(crate) block_lengths: Vec<usize>,
}

/// Limits for the exact, on-disk V2 de-duplication stage.
///
/// Only a leaf bucket is held in memory while comparing complete descriptor
/// rows.  A bucket over `max_rows_per_bucket` is repartitioned with an
/// independent deterministic hash until it fits or the explicit depth limit
/// reports a bounded failure instead of exhausting process memory.
#[derive(Clone, Debug)]
pub(crate) struct DeduplicationOptions {
    pub(crate) bucket_count: usize,
    pub(crate) max_rows_per_bucket: usize,
}

impl Default for DeduplicationOptions {
    fn default() -> Self {
        Self {
            bucket_count: DEFAULT_DEDUP_BUCKET_COUNT,
            max_rows_per_bucket: DEFAULT_DEDUP_MAX_ROWS_PER_BUCKET,
        }
    }
}

/// Disk-backed survivors selected from the pre-deduplication segments.
///
/// The bitset is indexed by the stable ordinal inside its `(2J, parity)`
/// block.  Keeping it separate from the descriptor payload lets the final
/// pass preserve the first source occurrence without retaining any CSF rows
/// in memory.
#[derive(Debug)]
pub(crate) struct DeduplicatedSegments {
    pub(crate) layout: DescriptorLayout,
    pub(crate) peel_subshells: Vec<String>,
    pub(crate) segments: Vec<DescriptorSegment>,
    pub(crate) survivor_bitsets: BTreeMap<(u16, bool), SurvivorBitsetFile>,
    pub(crate) generated_count: usize,
    pub(crate) unique_count: usize,
    pub(crate) duplicate_count: usize,
    pub(crate) hash_collision_count: usize,
    pub(crate) block_lengths: Vec<usize>,
    pub(crate) temporary_bytes_written: u64,
    pub(crate) budget: ResourceBudget,
}

/// Results from the disk generation pipeline before the caller publishes its
/// private staging files as one output set.
#[derive(Debug)]
pub(crate) struct DiskGenerationStats {
    pub(crate) unique_occupations: usize,
    pub(crate) generated_count: usize,
    pub(crate) unique_count: usize,
    pub(crate) duplicate_count: usize,
    pub(crate) block_count: usize,
    pub(crate) csf_bytes: u64,
    pub(crate) descriptor_bytes: u64,
    /// The codec the temporary segments were written with. The byte model
    /// above describes uncompressed segments regardless, so a compressed run
    /// is smaller than this accounting suggests.
    pub(crate) segment_codec: &'static str,
    /// How the published rows were made unique, so `duplicate_count` can be
    /// read for what it is: measured under `exact`, zero by construction under
    /// `verified_unique`.
    pub(crate) deduplication: &'static str,
    pub(crate) stage_stats: Vec<StageStats>,
    pub(crate) resource_stats: ResourceStats,
    pub(crate) plan_stats: PlanStats,
}

/// Generate, make unique according to the construction-selected strategy, and
/// restore a V2 descriptor through private staging paths. The caller owns final
/// publication, which keeps this deep module independent of CLI output naming
/// and gives it all-or-nothing semantics when multiple products are requested.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_disk_outputs_from_transcript(
    transcript: &str,
    csf_output: &Path,
    csf_parquet_output: &Path,
    descriptor_output: &Path,
    header_output: &Path,
    scratch_dir: &Path,
    threads: Option<usize>,
) -> Result<DiskGenerationStats> {
    let options = GenerationOptions::from_api(threads, None, Some(scratch_dir.to_path_buf()))?;
    generate_disk_outputs_from_transcript_with_options(
        transcript,
        csf_output,
        csf_parquet_output,
        descriptor_output,
        header_output,
        &options,
    )
}

/// Disk generation entry point with the shared execution options.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_disk_outputs_from_transcript_with_options(
    transcript: &str,
    csf_output: &Path,
    csf_parquet_output: &Path,
    descriptor_output: &Path,
    header_output: &Path,
    options: &GenerationOptions,
) -> Result<DiskGenerationStats> {
    options.validate()?;
    let scratch_dir = options
        .scratch_dir
        .as_deref()
        .context("disk generation requires a scratch directory")?;
    ensure!(
        !csf_output.exists()
            && !csf_parquet_output.exists()
            && !descriptor_output.exists()
            && !header_output.exists(),
        "disk generation staging outputs must not already exist"
    );
    ensure!(
        !scratch_dir.exists(),
        "disk generation scratch directory already exists: {}",
        scratch_dir.display()
    );
    let mut stage_stats = Vec::new();
    eprintln!("Parsing transcript and enumerating configurations...");
    // Parsing and scratch creation are part of the call the caller times, so
    // they are part of what the stages must account for.
    let setup_timer = StageTimer::start();
    let request = ExcitationRequest::from_transcript(transcript)?;
    fs::create_dir(scratch_dir).with_context(|| {
        format!(
            "failed to create disk generation scratch directory {}",
            scratch_dir.display()
        )
    })?;
    stage_stats.push(setup_timer.finish("setup", 0, 0, 0, 0));
    eprintln!("Generating CSFs and V2 descriptors to disk segments...");
    // Refuse a run whose estimate does not fit the scratch or staging volumes,
    // while the scratch directory still holds nothing but the empty root.
    let outputs = [
        csf_output,
        csf_parquet_output,
        descriptor_output,
        header_output,
    ];
    let generated =
        generate_v2_descriptor_segments_checked(&request, scratch_dir, options, |estimate| {
            for check in preflight_run(
                estimate,
                scratch_dir,
                &outputs,
                options.allow_unchecked_space,
            )? {
                report_space_check(&check);
            }
            Ok(())
        })?;
    stage_stats.extend(generated.stage_stats.iter().cloned());
    eprintln!(
        "Generated {} CSFs across {} symmetry blocks",
        generated.record_count,
        count_blocks(&generated.segments)
    );
    print_j_value_summary(&generated.segments);

    // The strategy is decided by the construction, never by a caller: every
    // record here comes from the internal generator, which the uniqueness proof
    // covers end to end. A caller can only ask for the slower strategy that
    // re-checks the proof.
    let dedup_timer = StageTimer::start();
    let generated_bytes = generated
        .segments
        .iter()
        .try_fold(0u64, |total, segment| total.checked_add(segment.byte_count))
        .context("generated segment byte count overflow")?;
    let deduplicated = match options.deduplication {
        DeduplicationStrategy::VerifiedUnique => {
            eprintln!("Verifying generated CSFs are unique (proof-backed path)...");
            verified_unique_deduplication(&generated, &scratch_dir.join("dedup"))?
        }
        DeduplicationStrategy::Exact => {
            eprintln!("Deduplicating descriptor segments...");
            deduplicate_v2_descriptor_segments(
                &generated,
                &scratch_dir.join("dedup"),
                &DeduplicationOptions::default(),
            )?
        }
    };
    let dedup_bytes = deduplicated.temporary_bytes_written;
    stage_stats.push(dedup_timer.finish(
        "deduplication",
        generated.record_count,
        deduplicated.unique_count,
        generated_bytes,
        dedup_bytes,
    ));
    if options.deduplication == DeduplicationStrategy::Exact {
        eprintln!(
            "Deduplicated: {} unique CSFs ({} duplicates removed)",
            deduplicated.unique_count, deduplicated.duplicate_count
        );
    } else {
        // Say what was and was not checked: the count below is zero because the
        // generator cannot repeat a record, not because a comparison found none.
        eprintln!(
            "Verified {} CSFs unique by construction (exact comparison not run; \
             set RCSFS_DEDUPLICATION=exact to measure it)",
            deduplicated.unique_count
        );
    }

    eprintln!("Writing generation header...");
    let header_timer = StageTimer::start();
    let header_lines = generated_header_lines(&generated.core_subshells, &generated.peel_subshells);
    write_generation_header(header_output, header_lines.clone(), &deduplicated)?;
    stage_stats.push(
        header_timer.finish(
            "header_write",
            deduplicated.unique_count,
            deduplicated.unique_count,
            0,
            fs::metadata(header_output)
                .map(|metadata| metadata.len())
                .unwrap_or(0),
        ),
    );

    eprintln!("Building final descriptor and CSF outputs from segments...");
    let final_stats = super::final_encoding::build_final_outputs_from_segments(
        &deduplicated,
        &header_lines,
        descriptor_output,
        csf_output,
        csf_parquet_output,
        options.threads,
    )?;
    let descriptor_bytes = final_stats.descriptor_bytes;
    let csf_bytes = final_stats.csf_bytes;
    // Seven entries cover segment read/decompression, selection, preparation,
    // and each artifact family split into encode and write. Their walls sum to
    // the whole pass, which makes the pass checkable against end-to-end time.
    for (name, elapsed_millis, cpu_millis) in final_stats.phase_entries() {
        let is_input_phase = matches!(
            name,
            "final_encoding_read" | "final_encoding_select" | "final_encoding_prepare"
        );
        let is_descriptor = name.starts_with("final_encoding_descriptor");
        let output_bytes = if is_input_phase {
            0
        } else if is_descriptor {
            descriptor_bytes
        } else {
            final_stats
                .csf_bytes
                .checked_add(final_stats.csf_parquet_bytes)
                .context("final output byte count overflow")?
        };
        stage_stats.push(StageStats {
            name,
            elapsed_millis,
            cpu_millis,
            input_records: deduplicated.unique_count,
            output_records: final_stats.record_count,
            input_bytes: if name == "final_encoding_read" {
                dedup_bytes
            } else {
                0
            },
            output_bytes,
        });
    }
    eprintln!("Generation complete!");
    Ok(DiskGenerationStats {
        unique_occupations: generated.unique_occupations,
        generated_count: deduplicated.generated_count,
        unique_count: deduplicated.unique_count,
        duplicate_count: deduplicated.duplicate_count,
        block_count: deduplicated.block_lengths.len(),
        csf_bytes,
        descriptor_bytes,
        segment_codec: generated.segment_codec.name(),
        deduplication: options.deduplication.name(),
        stage_stats,
        resource_stats: generated
            .budget
            .snapshot(generated.resource_stats.occupation_bytes),
        plan_stats: generated.plan.stats(),
    })
}

/// What a run is expected to cost, derived without writing anything.
///
/// This is the scriptable pre-flight: it enumerates and counts exactly as
/// generation does, then reports the schedule and the byte estimate. It creates
/// no scratch directory and publishes no file, so it can be run before a
/// decision about a multi-hour job.
#[derive(Debug)]
pub(crate) struct DiskEstimate {
    pub(crate) layout: DescriptorLayout,
    pub(crate) plan_stats: PlanStats,
    pub(crate) capacity: CapacityEstimate,
    /// The codec the predicted run would write its segments with. The byte
    /// model is uncompressed, so a compressed run stays inside the estimate.
    pub(crate) segment_codec: &'static str,
    /// The de-duplication strategy the estimate priced. The scratch model
    /// depends on it: the verified path writes no bucket at all.
    pub(crate) deduplication: &'static str,
    pub(crate) enumeration_millis: u128,
    pub(crate) planning_millis: u128,
}

/// Where a run will put its data, for the pre-flight's space checks.
///
/// The roles are what make the check meaningful: paths sharing a volume have
/// their requirements added, and each phase is compared by its maximum.
#[derive(Clone, Debug, Default)]
pub(crate) struct SpaceLayout {
    pub(crate) scratch_dir: Option<PathBuf>,
    /// Where the staged artifact set is written.
    pub(crate) staging_dir: Option<PathBuf>,
    /// Where each artifact is published; publication copies, so these bytes
    /// coexist with the staged set.
    pub(crate) destinations: Vec<(PathBuf, ArtifactKind)>,
}

/// Count the workload and estimate the capacity of a transcript.
pub(crate) fn estimate_disk_generation(
    transcript: &str,
    options: &GenerationOptions,
) -> Result<DiskEstimate> {
    estimate_disk_generation_with_layout(transcript, options, &SpaceLayout::default())
        .map(|(estimate, _checks)| estimate)
}

/// Count the workload, estimate the capacity, and check the volumes.
///
/// The space checks run over the same estimate the generation path uses, so a
/// prediction and the run it predicts cannot disagree about what fits.
pub(crate) fn estimate_disk_generation_with_layout(
    transcript: &str,
    options: &GenerationOptions,
    layout: &SpaceLayout,
) -> Result<(DiskEstimate, Vec<SpaceCheck>)> {
    options.validate()?;
    let enumeration_timer = StageTimer::start();
    let request = ExcitationRequest::from_transcript(transcript)?;
    // The estimate must fail the same way the run it predicts would, so the
    // same budget gates the occupation arena here. Charging an unlimited arena
    // would let a low-budget estimate succeed and the run then fail at
    // enumeration.
    let (occupations, _charge) =
        enumerate_occupations_with_budget(&request, Some(&options.budget))?;
    ensure!(
        !occupations.configurations.is_empty(),
        "occupation enumeration produced no configurations"
    );
    let peel = precompute_peel_subshells(&occupations);
    ensure!(
        !peel.is_empty(),
        "no occupied Peel subshells were enumerated"
    );
    let enumeration_millis = enumeration_timer.wall.elapsed().as_millis();
    let planning_timer = StageTimer::start();
    let workload = estimate_workload(&request, &occupations, options.threads)?;
    let plan = plan_generation(
        &request,
        &occupations,
        workload,
        options.threads,
        options.records_per_task,
    )?;
    let capacity = estimate_capacity(
        peel.len(),
        plan.workload.total_records,
        u64::try_from(request_targets(&request)?.len())
            .context("2J target count exceeds u64")?
            .saturating_mul(2),
        options.deduplication,
    )?;
    let estimate = DiskEstimate {
        layout: DescriptorLayout::new(DescriptorVersion::V2, peel.len()),
        plan_stats: plan.stats(),
        capacity,
        segment_codec: options.segment_codec.name(),
        deduplication: options.deduplication.name(),
        enumeration_millis,
        planning_millis: planning_timer.wall.elapsed().as_millis(),
    };
    let mut entries: Vec<(PathBuf, SpaceRole)> = Vec::new();
    if let Some(scratch) = &layout.scratch_dir {
        entries.push((scratch.clone(), SpaceRole::Scratch));
    }
    if let Some(staging) = &layout.staging_dir {
        entries.push((staging.clone(), SpaceRole::Staging));
    }
    for (destination, kind) in &layout.destinations {
        entries.push((
            destination.clone(),
            SpaceRole::Published {
                artifact: *kind,
                bytes: estimate.capacity.artifact_bytes(*kind),
            },
        ));
    }
    // Reporting, not enforcing: an estimate exists to answer whether a run
    // would fit, so an impossible requirement is a result it must be able to
    // report. Callers that are about to write use `SpacePolicy::Require`
    // instead -- the generation path does, and so does the CLI.
    let checks = if entries.is_empty() {
        Vec::new()
    } else {
        check_space(&estimate.capacity, &entries, SpacePolicy::Report)?
    };
    Ok((estimate, checks))
}

/// Report one pre-flight space check on stderr.
///
/// A check the platform cannot answer is reported as unchecked rather than as
/// sufficient: an unknown value is not a clearance.
fn report_space_check(check: &SpaceCheck) {
    match (check.free_bytes, check.sufficient) {
        (Some(free), Some(sufficient)) => eprintln!(
            "Pre-flight: {} needs {} bytes, {} available{}",
            check.path,
            check.required_bytes,
            free,
            if sufficient { "" } else { " (INSUFFICIENT)" }
        ),
        _ => eprintln!(
            "Pre-flight: {} needs {} bytes; free space is not reported on this platform",
            check.path, check.required_bytes
        ),
    }
}

/// Determine the deterministic Peel table needed before any V2 row is written.
///
/// This scans the already-enumerated relativistic occupations rather than the
/// active-orbital declaration.  It is therefore the same union the existing
/// memory path derives from generated configuration chunks, except for the
/// rare configuration that produces no target-J record; Phase 5's used-orbital
/// bitmap will close that remaining header-equivalence edge case.
pub(crate) fn precompute_peel_subshells(occupations: &EnumeratedOccupations) -> Vec<Subshell> {
    let mut used = HashSet::new();
    for configuration in &occupations.configurations {
        for occupation in &configuration.occupations {
            if occupation.electrons > 0 {
                used.insert(occupation.subshell);
            }
        }
    }
    let mut peel = used.into_iter().collect::<Vec<_>>();
    peel.sort_by_key(|shell| (shell.n(), shell.l(), shell.kappa() < 0));
    peel
}

/// Generate V2 descriptor rows into ordered Arrow IPC segments.
///
/// `scratch_dir` is an operation-owned directory.  Each range creates a
/// unique `ranges/range-XXXXXX` child and uses `create_new` for every segment,
/// so a stale or concurrent run fails before it can overwrite data.
pub(crate) fn generate_v2_descriptor_segments(
    request: &ExcitationRequest,
    scratch_dir: &Path,
    options: &StreamingGenerationOptions,
) -> Result<SegmentGeneration> {
    generate_v2_descriptor_segments_checked(request, scratch_dir, options, |_| Ok(()))
}

/// Generate V2 segments, offering the counted capacity estimate to the caller
/// before a single row is written.
///
/// Enumeration and counting happen once. `preflight` therefore receives an
/// estimate derived from the same counts that scheduled the work, and can
/// refuse the run while the scratch directory is still empty instead of after
/// the segments have filled a filesystem.
pub(crate) fn generate_v2_descriptor_segments_checked(
    request: &ExcitationRequest,
    scratch_dir: &Path,
    options: &StreamingGenerationOptions,
    preflight: impl FnOnce(&CapacityEstimate) -> Result<()>,
) -> Result<SegmentGeneration> {
    options.validate()?;
    let enumerate_timer = StageTimer::start();
    let (occupations, occupation_charge) =
        enumerate_occupations_with_budget(request, Some(&options.budget))?;
    let occupation_bytes = occupation_charge
        .as_ref()
        .map_or(0, |charge| charge.bytes());
    let enumeration_stats = enumerate_timer.finish(
        "enumeration",
        0,
        occupations.configurations.len(),
        0,
        occupation_bytes,
    );
    ensure!(
        !occupations.configurations.is_empty(),
        "occupation enumeration produced no configurations"
    );
    let peel = precompute_peel_subshells(&occupations);
    ensure!(
        !peel.is_empty(),
        "no occupied Peel subshells were enumerated"
    );
    let peel_subshells = peel.iter().map(ToString::to_string).collect::<Vec<_>>();
    let layout = DescriptorLayout::new(DescriptorVersion::V2, peel_subshells.len());
    let global_indices = peel
        .iter()
        .enumerate()
        .map(|(index, &shell)| Ok((shell, u16::try_from(index)?)))
        .collect::<Result<HashMap<_, _>>>()?;
    let ranges_root = scratch_dir.join("ranges");
    let planning_timer = StageTimer::start();
    let workload = estimate_workload(request, &occupations, options.threads)?;
    let plan = plan_generation(
        request,
        &occupations,
        workload,
        options.threads,
        options.records_per_task,
    )?;
    report_plan(&plan);
    let planning_stats = planning_timer.finish(
        "workload_planning",
        occupations.configurations.len(),
        plan.tasks.len(),
        0,
        // Planning reads the occupation arena and writes no file of its own, so
        // both byte counters stay zero. The counted record estimate is reported
        // through `plan_stats`, which is not a file-byte measurement.
        0,
    );
    // The capacity estimate is complete before any segment exists. Symmetry
    // blocks are not known yet, so the bitset term uses the largest number the
    // requested 2J range can produce.
    let capacity = estimate_capacity(
        peel_subshells.len(),
        plan.workload.total_records,
        u64::try_from(request_targets(request)?.len())
            .context("2J target count exceeds u64")?
            .saturating_mul(2),
        options.deduplication,
    )?;
    preflight(&capacity)?;
    let generation_timer = StageTimer::start();
    fs::create_dir_all(&ranges_root).with_context(|| {
        format!(
            "failed to create scratch directory {}",
            ranges_root.display()
        )
    })?;
    let progress = Arc::new(RangeProgress::new(plan.tasks.len()));

    let run_task = || {
        plan.tasks
            .par_iter()
            .map(|task| {
                generate_task_segments(
                    task,
                    request.min_two_j,
                    request.max_two_j,
                    &occupations.configurations,
                    &occupations.core_subshells,
                    layout,
                    &peel_subshells,
                    &global_indices,
                    &ranges_root,
                    options,
                    &progress,
                )
            })
            .collect::<Result<Vec<_>>>()
    };
    let range_results = match options.threads {
        Some(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .context("failed to build descriptor range thread pool")?
            .install(run_task)?,
        None => run_task()?,
    };
    progress.finish();

    let mut segments = Vec::new();
    let mut record_count = 0usize;
    for result in range_results {
        record_count = record_count
            .checked_add(result.record_count)
            .context("descriptor record count overflow")?;
        segments.extend(result.segments);
    }
    ensure!(
        record_count > 0,
        "no CSFs generated for the requested 2J range"
    );
    // The plan claims to cover every counted record. Comparing the generated
    // total against the counted total turns a scheduling mistake that drops
    // work into a failure here, instead of a shorter output file that only a
    // byte-for-byte comparison would notice.
    ensure!(
        u64::try_from(record_count).context("descriptor record count exceeds u64")?
            == plan.workload.total_records,
        "generation produced {record_count} CSFs but the workload plan counted \
         {}; the schedule does not cover every record",
        plan.workload.total_records
    );
    let segment_bytes = segments
        .iter()
        .try_fold(0u64, |total, segment| total.checked_add(segment.byte_count))
        .context("generated segment byte count overflow")?;
    // The occupation arena is no longer needed once all range workers have
    // completed. Release its reservation before later stages acquire bucket
    // and merge buffers.
    drop(occupation_charge);
    let resource_stats = options.budget.snapshot(occupation_bytes);
    Ok(SegmentGeneration {
        layout,
        peel_subshells,
        core_subshells: occupations.core_subshells,
        segments,
        unique_occupations: occupations.configurations.len(),
        record_count,
        segment_codec: options.segment_codec,
        stage_stats: vec![
            enumeration_stats,
            planning_stats,
            generation_timer.finish(
                "csf_generation",
                occupations.configurations.len(),
                record_count,
                0,
                segment_bytes,
            ),
        ],
        budget: options.budget.clone(),
        resource_stats,
        plan,
    })
}

/// Merge all pre-deduplication segments to a final V2 Parquet descriptor.
///
/// Blocks are ordered by `(2J, parity)` exactly like the existing writer;
/// segments within a block are ordered by range ordinal and local ordinal.
/// The temporary output guard guarantees that any decode/write/publication
/// error leaves no destination descriptor behind.
pub(crate) fn merge_v2_descriptor_segments(
    generated: &SegmentGeneration,
    output_path: &Path,
) -> Result<SegmentMergeStats> {
    ensure!(
        !generated.segments.is_empty(),
        "cannot merge an empty segment set"
    );
    ensure!(
        !output_path.exists(),
        "descriptor output already exists: {}",
        output_path.display()
    );
    let schema = output_schema(generated.layout, false)?;
    let _memory_permit = crate::csf_output::descriptor_writer_permit(
        &generated.budget,
        generated.layout,
        "descriptor merge",
    )?;
    let properties = WriterProperties::builder()
        .set_compression(
            crate::csfs_descriptor::parquet_batch::parse_compression(None)
                .expect("default compression is valid"),
        )
        .set_dictionary_enabled(true)
        .set_max_row_group_row_count(Some(crate::csf_output::ROW_GROUP_ROWS))
        .set_key_value_metadata(Some(output_kv_metadata(
            generated.layout,
            &generated.peel_subshells,
            false,
            None,
            None,
        )))
        .build();
    let (temporary, file) = create_temporary_output(output_path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(properties))
        .context("failed to create final descriptor Parquet writer")?;

    let mut blocks = BTreeMap::<(u16, bool), Vec<&DescriptorSegment>>::new();
    for segment in &generated.segments {
        blocks
            .entry(block_key(segment.total_two_j, segment.parity))
            .or_default()
            .push(segment);
    }

    let mut total = 0usize;
    let mut block_lengths = Vec::with_capacity(blocks.len());
    for segments in blocks.values_mut() {
        segments.sort_by_key(|segment| (segment.range_ordinal, segment.local_start));
        let mut block_count = 0usize;
        for segment in segments {
            block_count = block_count
                .checked_add(copy_segment_to_parquet(
                    segment,
                    generated.layout,
                    schema.clone(),
                    &mut writer,
                )?)
                .context("descriptor block count overflow")?;
        }
        ensure!(block_count > 0, "descriptor block cannot be empty");
        total = total
            .checked_add(block_count)
            .context("descriptor record count overflow")?;
        block_lengths.push(block_count);
    }
    ensure!(
        total == generated.record_count,
        "segment merge wrote {total} rows but generation reported {}",
        generated.record_count
    );
    writer
        .close()
        .context("failed to close final descriptor Parquet writer")?;
    publish_temporary_output(temporary.path(), output_path, false)?;
    Ok(SegmentMergeStats {
        record_count: total,
        block_lengths,
    })
}

/// Select the first exact occurrence of every V2 descriptor row on disk.
///
/// Rows are first partitioned per symmetry block by a stable 128-bit digest.
/// The digest is only a routing key: every possible duplicate is confirmed by
/// equality over all `4M + 2` V2 integers, including seniority and the
/// distinction between `MISSING` and a printed zero.  The resulting on-disk
/// bitsets are consumed by [`merge_v2_deduplicated_segments`] in a separate,
/// stable source-order pass.
pub(crate) fn deduplicate_v2_descriptor_segments(
    generated: &SegmentGeneration,
    scratch_dir: &Path,
    options: &DeduplicationOptions,
) -> Result<DeduplicatedSegments> {
    validate_deduplication_options(options)?;
    ensure!(
        !generated.segments.is_empty(),
        "cannot deduplicate an empty segment set"
    );
    fs::create_dir(scratch_dir).with_context(|| {
        format!(
            "failed to create de-duplication scratch directory {}",
            scratch_dir.display()
        )
    })?;

    let mut blocks = grouped_segments(&generated.segments);
    let mut survivor_bitsets = BTreeMap::new();
    let mut generated_count = 0usize;
    let mut unique_count = 0usize;
    let mut duplicate_count = 0usize;
    let mut hash_collision_count = 0usize;
    let mut temporary_bytes_written = 0u64;
    let mut block_lengths = Vec::with_capacity(blocks.len());

    for (&key, segments) in &mut blocks {
        segments.sort_by_key(|segment| (segment.range_ordinal, segment.local_start));
        let expected_rows = segments.iter().try_fold(0usize, |total, segment| {
            total
                .checked_add(segment.record_count)
                .context("descriptor block record count overflow")
        })?;
        let block_dir = scratch_dir.join(block_directory_name(key));
        fs::create_dir(&block_dir).with_context(|| {
            format!("failed to create bucket directory {}", block_dir.display())
        })?;
        let mut buckets = BucketWriters::new_with_budget(
            &block_dir,
            "root",
            options.bucket_count,
            generated.layout.row_len(),
            Some(&generated.budget),
        )?;
        let mut ordinal = 0u64;
        for segment in segments.iter().copied() {
            visit_segment_rows(segment, generated.layout, |row| {
                let hash = stable_row_hash(row);
                buckets.push(
                    root_bucket_index(&hash, options.bucket_count),
                    ordinal,
                    hash,
                    row,
                )?;
                ordinal = ordinal.checked_add(1).context("block ordinal overflow")?;
                Ok(())
            })?;
        }
        ensure!(
            usize::try_from(ordinal).ok() == Some(expected_rows),
            "source segments contain {ordinal} rows, expected {expected_rows}"
        );
        let bucket_files = buckets.finish()?;
        temporary_bytes_written = temporary_bytes_written
            .checked_add(bucket_files_size(&bucket_files)?)
            .context("de-duplication bucket byte count overflow")?;
        let bitset_path = block_dir.join("survivors.bitset");
        let mut bitset =
            SurvivorBitset::new_with_budget(bitset_path, expected_rows, Some(&generated.budget))?;
        let mut block_stats = DeduplicationStats::default();
        for bucket in bucket_files {
            process_bucket(
                bucket,
                generated.layout.row_len(),
                options.max_rows_per_bucket,
                0,
                &mut bitset,
                &mut block_stats,
                &generated.budget,
            )?;
        }
        ensure!(
            block_stats.generated_count == expected_rows,
            "de-duplication read {} rows, expected {expected_rows}",
            block_stats.generated_count
        );
        ensure!(
            block_stats.generated_count
                == block_stats
                    .unique_count
                    .checked_add(block_stats.duplicate_count)
                    .context("de-duplication count overflow")?,
            "de-duplication counts are inconsistent"
        );
        let file = bitset.finish()?;
        temporary_bytes_written = temporary_bytes_written
            .checked_add(fs::metadata(&file.path)?.len())
            .context("survivor bitset byte count overflow")?;
        generated_count = generated_count
            .checked_add(block_stats.generated_count)
            .context("generated descriptor count overflow")?;
        unique_count = unique_count
            .checked_add(block_stats.unique_count)
            .context("unique descriptor count overflow")?;
        duplicate_count = duplicate_count
            .checked_add(block_stats.duplicate_count)
            .context("duplicate descriptor count overflow")?;
        hash_collision_count = hash_collision_count
            .checked_add(block_stats.hash_collision_count)
            .context("hash collision count overflow")?;
        block_lengths.push(block_stats.unique_count);
        survivor_bitsets.insert(key, file);
    }
    ensure!(
        generated_count == generated.record_count,
        "de-duplication processed {generated_count} rows but generation reported {}",
        generated.record_count
    );
    Ok(DeduplicatedSegments {
        layout: generated.layout,
        peel_subshells: generated.peel_subshells.clone(),
        segments: generated.segments.clone(),
        survivor_bitsets,
        generated_count,
        unique_count,
        duplicate_count,
        hash_collision_count,
        block_lengths,
        temporary_bytes_written,
        budget: generated.budget.clone(),
    })
}

/// The verified path: every generated row survives, by the uniqueness proof.
///
/// The internal generator cannot emit the same V2 row twice
/// (`docs/V2_GENERATION_UNIQUENESS.md`), so a run that reaches this function
/// with the generator's own records has nothing to remove. What it still does
/// is the part that is not a comparison: the merge reads the segments back in
/// publication order, so the survivor bitsets have to exist and have to cover
/// exactly the rows the merge will visit. They are written full instead of
/// being filled in by a bucket comparison, which removes the root-bucket round
/// trip — every row written twice and read twice — from the run.
///
/// What this path does *not* do is measure anything: `duplicate_count` is zero
/// by construction here, not because zero were found. A caller that wants the
/// count measured asks for [`DeduplicationStrategy::Exact`].
pub(crate) fn verified_unique_deduplication(
    generated: &SegmentGeneration,
    scratch_dir: &Path,
) -> Result<DeduplicatedSegments> {
    ensure!(
        !generated.segments.is_empty(),
        "cannot verify an empty segment set"
    );
    fs::create_dir(scratch_dir).with_context(|| {
        format!(
            "failed to create de-duplication scratch directory {}",
            scratch_dir.display()
        )
    })?;

    let mut blocks = grouped_segments(&generated.segments);
    let mut survivor_bitsets = BTreeMap::new();
    let mut block_lengths = Vec::with_capacity(blocks.len());
    let mut temporary_bytes_written = 0u64;
    for (&key, segments) in &mut blocks {
        segments.sort_by_key(|segment| (segment.range_ordinal, segment.local_start));
        let rows = segments.iter().try_fold(0usize, |total, segment| {
            total
                .checked_add(segment.record_count)
                .context("descriptor block record count overflow")
        })?;
        let block_dir = scratch_dir.join(block_directory_name(key));
        fs::create_dir(&block_dir).with_context(|| {
            format!("failed to create bucket directory {}", block_dir.display())
        })?;
        let path = block_dir.join("survivors.bitset");
        let mut bitset = SurvivorBitset::new_with_budget(path, rows, Some(&generated.budget))?;
        // Every ordinal the merge will visit must be marked: a missed bit would
        // silently drop a record rather than being caught later, because this
        // path has no comparison to disagree with it.
        for ordinal in 0..rows {
            bitset.mark(u64::try_from(ordinal).context("ordinal exceeds u64")?)?;
        }
        let file = bitset.finish()?;
        temporary_bytes_written = temporary_bytes_written
            .checked_add(fs::metadata(&file.path)?.len())
            .context("survivor bitset byte count overflow")?;
        ensure!(
            file.unique_count == rows,
            "the verified survivor bitset covers {} of {rows} rows",
            file.unique_count
        );
        block_lengths.push(rows);
        survivor_bitsets.insert(key, file);
    }
    ensure!(
        generated_count(&block_lengths) == generated.record_count,
        "the verified survivor bitsets cover {} rows but generation reported {}",
        generated_count(&block_lengths),
        generated.record_count
    );
    Ok(DeduplicatedSegments {
        layout: generated.layout,
        peel_subshells: generated.peel_subshells.clone(),
        segments: generated.segments.clone(),
        survivor_bitsets,
        generated_count: generated.record_count,
        unique_count: generated.record_count,
        duplicate_count: 0,
        hash_collision_count: 0,
        block_lengths,
        temporary_bytes_written,
        budget: generated.budget.clone(),
    })
}

fn generated_count(block_lengths: &[usize]) -> usize {
    block_lengths.iter().copied().sum()
}

/// Write unique V2 descriptors selected by
/// [`deduplicate_v2_descriptor_segments`] in their original GRASP order.
pub(crate) fn merge_v2_deduplicated_segments(
    deduplicated: &DeduplicatedSegments,
    output_path: &Path,
) -> Result<SegmentMergeStats> {
    ensure!(
        !output_path.exists(),
        "descriptor output already exists: {}",
        output_path.display()
    );
    let schema = output_schema(deduplicated.layout, false)?;
    let _memory_permit = crate::csf_output::descriptor_writer_permit(
        &deduplicated.budget,
        deduplicated.layout,
        "deduplicated descriptor merge",
    )?;
    let properties = crate::csf_output::descriptor::properties(
        deduplicated.layout,
        &deduplicated.peel_subshells,
        deduplicated.generated_count,
        deduplicated.unique_count,
        deduplicated.duplicate_count,
        &deduplicated.block_lengths,
    )?;
    let (temporary, file) = create_temporary_output(output_path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(properties))
        .context("failed to create de-duplicated descriptor Parquet writer")?;

    let mut blocks = grouped_segments(&deduplicated.segments);
    let mut total = 0usize;
    let mut block_lengths = Vec::with_capacity(blocks.len());
    for (&key, segments) in &mut blocks {
        segments.sort_by_key(|segment| (segment.range_ordinal, segment.local_start));
        let bitset_file = deduplicated
            .survivor_bitsets
            .get(&key)
            .context("missing survivor bitset for descriptor block")?;
        let bitset = SurvivorBitset::read(bitset_file)?;
        let mut ordinal = 0u64;
        let mut block_count = 0usize;
        for segment in segments {
            block_count = block_count
                .checked_add(copy_surviving_segment_to_parquet(
                    segment,
                    deduplicated.layout,
                    schema.clone(),
                    &mut writer,
                    &bitset,
                    &mut ordinal,
                )?)
                .context("unique descriptor block count overflow")?;
        }
        ensure!(
            usize::try_from(ordinal).ok() == Some(bitset.bit_len),
            "survivor bitset length does not match source block"
        );
        ensure!(
            block_count == bitset_file.unique_count,
            "survivor bitset selected {block_count} rows, expected {}",
            bitset_file.unique_count
        );
        total = total
            .checked_add(block_count)
            .context("unique descriptor total overflow")?;
        block_lengths.push(block_count);
    }
    ensure!(
        total == deduplicated.unique_count,
        "unique merge wrote {total} rows, expected {}",
        deduplicated.unique_count
    );
    ensure!(
        block_lengths == deduplicated.block_lengths,
        "unique merge block lengths do not match de-duplication result"
    );
    writer
        .close()
        .context("failed to close de-duplicated descriptor Parquet writer")?;
    publish_temporary_output(temporary.path(), output_path, false)?;
    Ok(SegmentMergeStats {
        record_count: total,
        block_lengths,
    })
}

#[derive(Clone, Debug)]
pub(crate) struct SurvivorBitsetFile {
    path: PathBuf,
    pub(crate) bit_len: usize,
    pub(crate) unique_count: usize,
}

pub(crate) struct SurvivorBitset {
    path: PathBuf,
    pub(crate) bit_len: usize,
    bytes: Vec<u8>,
    _memory_permit: Option<ResourcePermit>,
}

impl SurvivorBitset {
    fn new(path: PathBuf, bit_len: usize) -> Result<Self> {
        Self::new_with_budget(path, bit_len, None)
    }

    fn new_with_budget(
        path: PathBuf,
        bit_len: usize,
        budget: Option<&ResourceBudget>,
    ) -> Result<Self> {
        let byte_len = bit_len
            .checked_add(7)
            .context("survivor bitset size overflow")?
            / 8;
        let memory_permit = budget
            .map(|budget| {
                budget.try_reserve(
                    u64::try_from(byte_len).context("survivor bitset bytes exceed u64")?,
                    "survivor bitset",
                )
            })
            .transpose()?;
        Ok(Self {
            path,
            bit_len,
            bytes: vec![0; byte_len],
            _memory_permit: memory_permit,
        })
    }

    pub(crate) fn read(file: &SurvivorBitsetFile) -> Result<Self> {
        let expected_bytes = file
            .bit_len
            .checked_add(7)
            .context("survivor bitset size overflow")?
            / 8;
        let bytes = fs::read(&file.path)
            .with_context(|| format!("failed to read survivor bitset {}", file.path.display()))?;
        ensure!(
            bytes.len() == expected_bytes,
            "survivor bitset {} has {} bytes, expected {expected_bytes}",
            file.path.display(),
            bytes.len()
        );
        Ok(Self {
            path: file.path.clone(),
            bit_len: file.bit_len,
            bytes,
            _memory_permit: None,
        })
    }

    fn mark(&mut self, ordinal: u64) -> Result<()> {
        let ordinal = usize::try_from(ordinal).context("survivor ordinal exceeds address space")?;
        ensure!(
            ordinal < self.bit_len,
            "survivor ordinal {ordinal} exceeds bitset length {}",
            self.bit_len
        );
        self.bytes[ordinal / 8] |= 1 << (ordinal % 8);
        Ok(())
    }

    pub(crate) fn contains(&self, ordinal: u64) -> Result<bool> {
        let ordinal = usize::try_from(ordinal).context("survivor ordinal exceeds address space")?;
        ensure!(
            ordinal < self.bit_len,
            "survivor ordinal {ordinal} exceeds bitset length {}",
            self.bit_len
        );
        Ok(self.bytes[ordinal / 8] & (1 << (ordinal % 8)) != 0)
    }

    fn finish(self) -> Result<SurvivorBitsetFile> {
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(&self.path)
            .with_context(|| format!("failed to create survivor bitset {}", self.path.display()))?;
        file.write_all(&self.bytes)
            .with_context(|| format!("failed to write survivor bitset {}", self.path.display()))?;
        file.flush()
            .with_context(|| format!("failed to flush survivor bitset {}", self.path.display()))?;
        Ok(SurvivorBitsetFile {
            path: self.path,
            bit_len: self.bit_len,
            unique_count: self
                .bytes
                .iter()
                .map(|byte| byte.count_ones() as usize)
                .sum(),
        })
    }
}

#[derive(Default)]
struct DeduplicationStats {
    generated_count: usize,
    unique_count: usize,
    duplicate_count: usize,
    hash_collision_count: usize,
    temporary_bytes_written: u64,
}

#[derive(Clone, Debug)]
struct BucketFile {
    path: PathBuf,
    record_count: usize,
}

struct BucketWriters {
    row_len: usize,
    writers: Vec<BufWriter<File>>,
    files: Vec<BucketFile>,
    record_buffer: Vec<u8>,
    _memory_permit: Option<ResourcePermit>,
}

impl BucketWriters {
    fn new(directory: &Path, label: &str, count: usize, row_len: usize) -> Result<Self> {
        Self::new_with_budget(directory, label, count, row_len, None)
    }

    fn new_with_budget(
        directory: &Path,
        label: &str,
        count: usize,
        row_len: usize,
        budget: Option<&ResourceBudget>,
    ) -> Result<Self> {
        let buffer_bytes = count
            .checked_mul(8 * 1024)
            .and_then(|value| value.checked_add(bucket_record_byte_len(row_len).ok()?))
            .context("de-duplication bucket writer memory byte count overflow")?;
        let memory_permit = budget
            .map(|budget| {
                budget.try_reserve(
                    u64::try_from(buffer_bytes)
                        .context("de-duplication bucket writer bytes exceed u64")?,
                    "de-duplication bucket writers",
                )
            })
            .transpose()?;
        let mut writers = Vec::with_capacity(count);
        let mut files = Vec::with_capacity(count);
        for index in 0..count {
            let path = directory.join(format!("{label}-{index:03}.bucket"));
            let file = File::options()
                .write(true)
                .create_new(true)
                .open(&path)
                .with_context(|| format!("failed to create bucket {}", path.display()))?;
            writers.push(BufWriter::new(file));
            files.push(BucketFile {
                path,
                record_count: 0,
            });
        }
        Ok(Self {
            row_len,
            writers,
            files,
            record_buffer: Vec::with_capacity(bucket_record_byte_len(row_len)?),
            _memory_permit: memory_permit,
        })
    }

    fn push(&mut self, index: usize, ordinal: u64, hash: [u8; 16], row: &[i32]) -> Result<()> {
        ensure!(
            row.len() == self.row_len,
            "bucket row width {} does not match {}",
            row.len(),
            self.row_len
        );
        self.record_buffer.clear();
        self.record_buffer.extend_from_slice(&ordinal.to_le_bytes());
        self.record_buffer.extend_from_slice(&hash);
        for value in row {
            self.record_buffer.extend_from_slice(&value.to_le_bytes());
        }
        let writer = self
            .writers
            .get_mut(index)
            .context("bucket index exceeds bucket count")?;
        writer.write_all(&self.record_buffer)?;
        let bucket = self
            .files
            .get_mut(index)
            .expect("writer and bucket metadata have matching lengths");
        bucket.record_count = bucket
            .record_count
            .checked_add(1)
            .context("bucket record count overflow")?;
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<BucketFile>> {
        for writer in &mut self.writers {
            writer
                .flush()
                .context("failed to flush de-duplication bucket")?;
        }
        Ok(self
            .files
            .into_iter()
            .filter(|bucket| bucket.record_count > 0)
            .collect())
    }
}

fn bucket_files_size(files: &[BucketFile]) -> Result<u64> {
    files.iter().try_fold(0u64, |total, bucket| {
        total
            .checked_add(fs::metadata(&bucket.path)?.len())
            .context("bucket byte count overflow")
    })
}

struct BucketReader {
    reader: BufReader<File>,
    row_len: usize,
    remaining: usize,
    record_buffer: Vec<u8>,
}

impl BucketReader {
    fn open(bucket: &BucketFile, row_len: usize) -> Result<Self> {
        let file = File::open(&bucket.path)
            .with_context(|| format!("failed to open bucket {}", bucket.path.display()))?;
        Ok(Self {
            reader: BufReader::new(file),
            row_len,
            remaining: bucket.record_count,
            record_buffer: vec![0; bucket_record_byte_len(row_len)?],
        })
    }

    fn next(&mut self, row: &mut Vec<i32>) -> Result<Option<(u64, [u8; 16])>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.reader.read_exact(&mut self.record_buffer)?;
        let ordinal_bytes = self.record_buffer[..8]
            .try_into()
            .expect("bucket ordinal field has fixed width");
        let hash = self.record_buffer[8..24]
            .try_into()
            .expect("bucket hash field has fixed width");
        row.clear();
        row.try_reserve(self.row_len)?;
        for bytes in self.record_buffer[24..].chunks_exact(4) {
            row.push(i32::from_le_bytes(
                bytes
                    .try_into()
                    .expect("bucket integer field has fixed width"),
            ));
        }
        self.remaining -= 1;
        Ok(Some((u64::from_le_bytes(ordinal_bytes), hash)))
    }
}

fn bucket_record_byte_len(row_len: usize) -> Result<usize> {
    row_len
        .checked_mul(std::mem::size_of::<i32>())
        .and_then(|row_bytes| row_bytes.checked_add(24))
        .context("bucket record byte length overflow")
}

fn process_bucket(
    bucket: BucketFile,
    row_len: usize,
    max_rows_per_bucket: usize,
    depth: usize,
    bitset: &mut SurvivorBitset,
    stats: &mut DeduplicationStats,
    budget: &ResourceBudget,
) -> Result<()> {
    if bucket.record_count <= max_rows_per_bucket {
        return deduplicate_bucket(bucket, row_len, bitset, stats, budget, max_rows_per_bucket);
    }
    ensure!(
        depth < MAX_DEDUP_SPLIT_DEPTH,
        "bucket {} has {} rows after {depth} recursive splits; reduce the root bucket size or max_rows_per_bucket",
        bucket.path.display(),
        bucket.record_count
    );
    let split_directory = bucket.path.with_extension(format!("split-{depth:02}"));
    fs::create_dir(&split_directory).with_context(|| {
        format!(
            "failed to create recursive bucket directory {}",
            split_directory.display()
        )
    })?;
    let mut children = BucketWriters::new_with_budget(
        &split_directory,
        "child",
        DEDUP_RECURSIVE_BUCKET_COUNT,
        row_len,
        Some(budget),
    )?;
    let mut reader = BucketReader::open(&bucket, row_len)?;
    let mut row = Vec::with_capacity(row_len);
    while let Some((ordinal, hash)) = reader.next(&mut row)? {
        let child = recursive_bucket_index(&row, depth);
        children.push(child, ordinal, hash, &row)?;
    }
    fs::remove_file(&bucket.path)
        .with_context(|| format!("failed to remove split bucket {}", bucket.path.display()))?;
    let child_files = children.finish()?;
    stats.temporary_bytes_written = stats
        .temporary_bytes_written
        .checked_add(bucket_files_size(&child_files)?)
        .context("recursive bucket byte count overflow")?;
    for child in child_files {
        process_bucket(
            child,
            row_len,
            max_rows_per_bucket,
            depth + 1,
            bitset,
            stats,
            budget,
        )?;
    }
    Ok(())
}

fn deduplicate_bucket(
    bucket: BucketFile,
    row_len: usize,
    bitset: &mut SurvivorBitset,
    stats: &mut DeduplicationStats,
    budget: &ResourceBudget,
    max_rows_per_bucket: usize,
) -> Result<()> {
    let row_bytes = max_rows_per_bucket
        .checked_mul(row_len)
        .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
        .and_then(|value| value.checked_add(max_rows_per_bucket.checked_mul(64)?))
        .context("de-duplication bucket memory byte count overflow")?;
    let _memory_permit = budget.try_reserve(
        u64::try_from(row_bytes).context("de-duplication bucket bytes exceed u64")?,
        "de-duplication bucket",
    )?;
    let mut reader = BucketReader::open(&bucket, row_len)?;
    let mut rows_by_hash = HashMap::<[u8; 16], Vec<Vec<i32>>>::new();
    let mut row = Vec::with_capacity(row_len);
    while let Some((ordinal, hash)) = reader.next(&mut row)? {
        stats.generated_count = stats
            .generated_count
            .checked_add(1)
            .context("de-duplication generated count overflow")?;
        let candidates = rows_by_hash.entry(hash).or_default();
        if candidates.iter().any(|candidate| candidate == &row) {
            stats.duplicate_count = stats
                .duplicate_count
                .checked_add(1)
                .context("duplicate descriptor count overflow")?;
            continue;
        }
        if !candidates.is_empty() {
            stats.hash_collision_count = stats
                .hash_collision_count
                .checked_add(1)
                .context("hash collision count overflow")?;
        }
        candidates.push(row.clone());
        bitset.mark(ordinal)?;
        stats.unique_count = stats
            .unique_count
            .checked_add(1)
            .context("unique descriptor count overflow")?;
    }
    fs::remove_file(&bucket.path)
        .with_context(|| format!("failed to remove consumed bucket {}", bucket.path.display()))?;
    Ok(())
}

fn stable_row_hash(row: &[i32]) -> [u8; 16] {
    let mut hasher = Sha256::new();
    hasher.update(b"rCSFs/CSFDescriptorV2/dedup/v1\\0");
    for value in row {
        hasher.update(value.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut hash = [0; 16];
    hash.copy_from_slice(&digest[..16]);
    hash
}

fn root_bucket_index(hash: &[u8; 16], bucket_count: usize) -> usize {
    let mut bytes = [0; std::mem::size_of::<usize>()];
    bytes.copy_from_slice(&hash[..std::mem::size_of::<usize>()]);
    usize::from_le_bytes(bytes) & (bucket_count - 1)
}

fn recursive_bucket_index(row: &[i32], depth: usize) -> usize {
    let mut hasher = Sha256::new();
    hasher.update(b"rCSFs/CSFDescriptorV2/dedup/split/v1\\0");
    hasher.update(
        u64::try_from(depth)
            .expect("usize fits in u64")
            .to_le_bytes(),
    );
    for value in row {
        hasher.update(value.to_le_bytes());
    }
    usize::from(hasher.finalize()[0]) & (DEDUP_RECURSIVE_BUCKET_COUNT - 1)
}

fn validate_deduplication_options(options: &DeduplicationOptions) -> Result<()> {
    ensure!(
        options.bucket_count.is_power_of_two() && options.bucket_count > 0,
        "bucket_count must be a non-zero power of two"
    );
    ensure!(
        options.max_rows_per_bucket > 0,
        "max_rows_per_bucket must be greater than 0"
    );
    Ok(())
}

pub(crate) fn grouped_segments(
    source: &[DescriptorSegment],
) -> BTreeMap<(u16, bool), Vec<&DescriptorSegment>> {
    let mut blocks = BTreeMap::new();
    for segment in source {
        blocks
            .entry(block_key(segment.total_two_j, segment.parity))
            .or_insert_with(Vec::new)
            .push(segment);
    }
    blocks
}

fn block_directory_name((total_two_j, odd): (u16, bool)) -> String {
    let parity = if odd { "odd" } else { "even" };
    format!("j{total_two_j:04}-{parity}")
}

fn count_blocks(segments: &[DescriptorSegment]) -> usize {
    let mut blocks = std::collections::HashSet::new();
    for segment in segments {
        blocks.insert((segment.total_two_j, segment.parity == Parity::Odd));
    }
    blocks.len()
}

fn print_j_value_summary(segments: &[DescriptorSegment]) {
    let mut blocks = BTreeMap::<(u16, bool), usize>::new();
    for segment in segments {
        *blocks
            .entry(block_key(segment.total_two_j, segment.parity))
            .or_insert(0) += segment.record_count;
    }

    eprintln!("\nCSFs per J value:");
    for ((total_two_j, odd), count) in blocks {
        let parity = if odd { "odd" } else { "even" };
        let j_value = f64::from(total_two_j) / 2.0;
        eprintln!("  J = {:4.1} ({:>4}): {:10} CSFs", j_value, parity, count);
    }
    eprintln!();
}

fn generated_header_lines(core_subshells: &[Subshell], peel_subshells: &[String]) -> [String; 5] {
    [
        "Core subshells:".to_owned(),
        format_header_labels(core_subshells.iter().map(ToString::to_string)),
        "Peel subshells:".to_owned(),
        format_header_labels(peel_subshells.iter().cloned()),
        "CSF(s):".to_owned(),
    ]
}

fn format_header_labels(labels: impl IntoIterator<Item = String>) -> String {
    labels
        .into_iter()
        .map(|label| {
            let display = if label.ends_with('-') {
                label
            } else {
                format!("{label} ")
            };
            format!("{display:>5}")
        })
        .collect::<String>()
        .trim_end()
        .to_owned()
}

fn write_generation_header(
    path: &Path,
    header_lines: [String; 5],
    deduplicated: &DeduplicatedSegments,
) -> Result<()> {
    use crate::csfs_conversion::{BlockInfo, ConversionStats, HeaderData, HeaderInfo};

    let header = HeaderData {
        header_info: HeaderInfo {
            header_lines: header_lines.into(),
        },
        block_info: BlockInfo {
            block_lengths: deduplicated.block_lengths.clone(),
            block_count: deduplicated.block_lengths.len(),
        },
        conversion_stats: ConversionStats {
            csf_count: deduplicated.unique_count,
            total_lines: deduplicated
                .unique_count
                .checked_mul(3)
                .context("CSF line count overflow")?,
            truncated_count: 0,
        },
    };
    let text = toml::to_string_pretty(&header).context("failed to serialize generation header")?;
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("failed to create header {}", path.display()))?;
    file.write_all(text.as_bytes())
        .with_context(|| format!("failed to write header {}", path.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush header {}", path.display()))?;
    Ok(())
}

fn validate_options(options: &StreamingGenerationOptions) -> Result<()> {
    ensure!(options.threads != Some(0), "threads must be greater than 0");
    ensure!(
        options.records_per_task != Some(0),
        "records_per_task must be greater than 0"
    );
    ensure!(
        options.rows_per_batch > 0,
        "rows_per_batch must be greater than 0"
    );
    ensure!(
        options.rows_per_segment > 0,
        "rows_per_segment must be greater than 0"
    );
    Ok(())
}

struct RangeResult {
    segments: Vec<DescriptorSegment>,
    record_count: usize,
}

#[allow(clippy::too_many_arguments)]
fn generate_task_segments(
    task: &PlannedTask,
    min_two_j: u16,
    max_two_j: u16,
    configurations: &[EnumeratedConfiguration],
    core_subshells: &[Subshell],
    layout: DescriptorLayout,
    peel_subshells: &[String],
    global_indices: &HashMap<Subshell, u16>,
    ranges_root: &Path,
    options: &StreamingGenerationOptions,
    progress: &RangeProgress,
) -> Result<RangeResult> {
    progress.range_started();
    let range_dir = ranges_root.join(format!("range-{:06}", task.ordinal));
    fs::create_dir(&range_dir)
        .with_context(|| format!("failed to create range directory {}", range_dir.display()))?;
    let mut writer = RangeSegmentWriter::new(task.ordinal, layout, range_dir, options)?;
    // Keep recursion on the task worker. The old nested collector retained
    // complete branch CSF vectors and could exhaust a caller-supplied budget
    // before the segment writer had a chance to flush.
    let branch_count = 1;
    let plan_span = task.span.configurations();
    for configuration in &configurations[plan_span] {
        let configuration_bytes = configuration
            .occupations
            .len()
            .checked_mul(std::mem::size_of::<super::SubshellOccupation>())
            .and_then(|value| value.checked_add(4096))
            .context("generation configuration byte count overflow")?;
        let _configuration_permit = options.budget.try_reserve(
            u64::try_from(configuration_bytes)
                .context("generation configuration bytes exceed u64")?,
            "generation configuration copy",
        )?;
        let local_to_global = local_to_global_indices(&configuration.occupations, global_indices)?;
        let mut sink =
            DescriptorBatchSink::new(layout, peel_subshells, local_to_global, &mut writer);
        let selection = task_selection(task)?;
        generate_configuration_records(
            core_subshells,
            &configuration.occupations,
            min_two_j,
            max_two_j,
            selection,
            branch_count,
            &mut sink,
        )?;
    }
    let result = writer.finish()?;
    progress.range_completed(result.record_count);
    Ok(result)
}

/// The record selection a planned task generates.
///
/// A multi-configuration task applies its selection to each configuration in
/// turn; the task span guarantees this walks the configurations in the same
/// order the unsplit path would.
fn task_selection(task: &PlannedTask) -> Result<RecordSelection<'_>> {
    match &task.span {
        TaskSpan::Configurations { .. } => Ok(RecordSelection::All),
        TaskSpan::Targets { targets, .. } => Ok(RecordSelection::Targets(targets)),
        TaskSpan::StatePrefixes {
            target,
            branch_count,
            branches,
            ..
        } => Ok(RecordSelection::StatePrefixes {
            target: *target,
            branch_count: *branch_count,
            branches: branches.clone(),
        }),
    }
}

fn local_to_global_indices(
    configuration: &[SubshellOccupation],
    global_indices: &HashMap<Subshell, u16>,
) -> Result<Vec<u16>> {
    configuration
        .iter()
        .filter(|entry| entry.electrons > 0)
        .map(|entry| {
            global_indices
                .get(&entry.subshell)
                .copied()
                .with_context(|| {
                    format!(
                        "subshell {} is absent from global Peel table",
                        entry.subshell
                    )
                })
        })
        .collect()
}

struct DescriptorBatchSink<'a> {
    layout: DescriptorLayout,
    peel_subshells: &'a [String],
    local_to_global: Vec<u16>,
    remapped_occupied: Vec<OccupiedSubshell>,
    row: Vec<i32>,
    range_writer: &'a mut RangeSegmentWriter,
}

impl<'a> DescriptorBatchSink<'a> {
    fn new(
        layout: DescriptorLayout,
        peel_subshells: &'a [String],
        local_to_global: Vec<u16>,
        range_writer: &'a mut RangeSegmentWriter,
    ) -> Self {
        Self {
            layout,
            peel_subshells,
            local_to_global,
            remapped_occupied: Vec::new(),
            row: vec![0; layout.row_len()],
            range_writer,
        }
    }
}

impl GeneratedRecordSink for DescriptorBatchSink<'_> {
    fn push(&mut self, record: GeneratedRecordRef<'_>) -> Result<()> {
        self.remapped_occupied.clear();
        self.remapped_occupied.try_reserve(record.occupied.len())?;
        for occupied in record.occupied {
            let local_index = usize::from(occupied.subshell_index);
            let global_index = *self.local_to_global.get(local_index).with_context(|| {
                format!("local subshell index {local_index} exceeds configuration Peel table")
            })?;
            self.remapped_occupied.push(OccupiedSubshell {
                subshell_index: global_index,
                occupation: occupied.occupation,
                state: occupied.state,
            });
        }
        validate_record(
            self.peel_subshells,
            &self.remapped_occupied,
            record.couplings,
        )?;
        write_feature_row(
            self.layout,
            &self.remapped_occupied,
            record.couplings,
            record.total_two_j,
            record.parity,
            &mut self.row,
        )?;
        self.range_writer
            .push(record.total_two_j, record.parity, &self.row)
    }
}

struct RangeSegmentWriter {
    range_ordinal: u32,
    layout: DescriptorLayout,
    schema: SchemaRef,
    range_dir: PathBuf,
    rows_per_batch: usize,
    rows_per_segment: usize,
    codec: SegmentCodec,
    budget: ResourceBudget,
    blocks: BTreeMap<(u16, bool), BlockSegmentWriter>,
}

impl RangeSegmentWriter {
    fn new(
        range_ordinal: u32,
        layout: DescriptorLayout,
        range_dir: PathBuf,
        options: &StreamingGenerationOptions,
    ) -> Result<Self> {
        Ok(Self {
            range_ordinal,
            layout,
            schema: segment_schema(layout)?,
            range_dir,
            rows_per_batch: options.rows_per_batch,
            rows_per_segment: options.rows_per_segment,
            codec: options.segment_codec,
            budget: options.budget.clone(),
            blocks: BTreeMap::new(),
        })
    }

    fn push(&mut self, total_two_j: u16, parity: Parity, row: &[i32]) -> Result<()> {
        ensure!(
            row.len() == self.layout.row_len(),
            "descriptor row width {} does not match {}",
            row.len(),
            self.layout.row_len()
        );
        let key = block_key(total_two_j, parity);
        let writer = self.blocks.entry(key).or_insert_with(|| {
            BlockSegmentWriter::new(
                self.range_ordinal,
                total_two_j,
                parity,
                self.layout.row_len(),
                self.schema.clone(),
                self.range_dir.clone(),
                self.rows_per_batch,
                self.rows_per_segment,
                self.codec,
                self.budget.clone(),
            )
        });
        writer.push(row)
    }

    fn finish(self) -> Result<RangeResult> {
        let mut segments = Vec::new();
        let mut record_count = 0usize;
        for writer in self.blocks.into_values() {
            let result = writer.finish()?;
            record_count = record_count
                .checked_add(result.0)
                .context("range descriptor record count overflow")?;
            segments.extend(result.1);
        }
        Ok(RangeResult {
            segments,
            record_count,
        })
    }
}

struct BlockSegmentWriter {
    range_ordinal: u32,
    total_two_j: u16,
    parity: Parity,
    row_len: usize,
    schema: SchemaRef,
    range_dir: PathBuf,
    rows_per_batch: usize,
    rows_per_segment: usize,
    codec: SegmentCodec,
    budget: ResourceBudget,
    next_part: u32,
    next_local_ordinal: u64,
    current: Option<OpenSegment>,
    segments: Vec<DescriptorSegment>,
}

struct OpenSegment {
    path: PathBuf,
    local_start: u64,
    record_count: usize,
    columns: Vec<Vec<i32>>,
    local_ordinals: Vec<u64>,
    writer: FileWriter<BufWriter<File>>,
    _memory_permit: ResourcePermit,
}

impl BlockSegmentWriter {
    #[allow(clippy::too_many_arguments)]
    fn new(
        range_ordinal: u32,
        total_two_j: u16,
        parity: Parity,
        row_len: usize,
        schema: SchemaRef,
        range_dir: PathBuf,
        rows_per_batch: usize,
        rows_per_segment: usize,
        codec: SegmentCodec,
        budget: ResourceBudget,
    ) -> Self {
        Self {
            range_ordinal,
            total_two_j,
            parity,
            row_len,
            schema,
            range_dir,
            rows_per_batch,
            rows_per_segment,
            codec,
            budget,
            next_part: 0,
            next_local_ordinal: 0,
            current: None,
            segments: Vec::new(),
        }
    }

    fn push(&mut self, row: &[i32]) -> Result<()> {
        if self
            .current
            .as_ref()
            .is_some_and(|segment| segment.record_count == self.rows_per_segment)
        {
            self.finish_current()?;
        }
        if self.current.is_none() {
            self.open_next()?;
        }
        let current = self.current.as_mut().expect("segment was opened");
        for (column, &value) in current.columns.iter_mut().zip(row) {
            column.push(value);
        }
        current.local_ordinals.push(self.next_local_ordinal);
        current.record_count += 1;
        self.next_local_ordinal += 1;
        if current.local_ordinals.len() == self.rows_per_batch {
            Self::flush_current(
                current,
                self.range_ordinal,
                self.schema.clone(),
                self.rows_per_batch,
            )?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<(usize, Vec<DescriptorSegment>)> {
        self.finish_current()?;
        let count = self.segments.iter().try_fold(0usize, |count, segment| {
            count
                .checked_add(segment.record_count)
                .context("block descriptor record count overflow")
        })?;
        Ok((count, self.segments))
    }

    fn open_next(&mut self) -> Result<()> {
        // Ensure the range directory exists before creating files
        if !self.range_dir.exists() {
            fs::create_dir_all(&self.range_dir).with_context(|| {
                format!(
                    "failed to create range directory {}",
                    self.range_dir.display()
                )
            })?;
        }

        let parity = match self.parity {
            Parity::Even => "even",
            Parity::Odd => "odd",
        };
        let path = self.range_dir.join(format!(
            "j{:04}-{parity}-{:03}.arrow",
            self.total_two_j, self.next_part
        ));
        self.next_part = self
            .next_part
            .checked_add(1)
            .context("too many block segments")?;
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("failed to create segment {}", path.display()))?;
        let row_bytes = self
            .rows_per_batch
            .checked_mul(self.row_len)
            .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
            .context("descriptor batch byte count overflow")?;
        let ordinal_bytes = self
            .rows_per_batch
            .checked_mul(std::mem::size_of::<u64>())
            .context("descriptor ordinal byte count overflow")?;
        // Arrow arrays are materialized while a batch is flushed, so reserve
        // space for both the mutable column buffers and the temporary arrays.
        let managed_bytes = row_bytes
            .checked_add(ordinal_bytes)
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_add(4096))
            .context("descriptor batch managed byte count overflow")?;
        let memory_permit = self.budget.try_reserve(
            u64::try_from(managed_bytes).context("descriptor batch bytes exceed u64")?,
            "descriptor generation batch",
        )?;
        let mut columns = Vec::with_capacity(self.row_len);
        for _ in 0..self.row_len {
            let mut column = Vec::new();
            column.try_reserve_exact(self.rows_per_batch)?;
            columns.push(column);
        }
        let local_ordinals = Vec::with_capacity(self.rows_per_batch);
        let writer = FileWriter::try_new_with_options(
            BufWriter::new(file),
            self.schema.as_ref(),
            segment_write_options(self.codec)?,
        )
        .with_context(|| format!("failed to open Arrow segment {}", path.display()))?;
        self.current = Some(OpenSegment {
            path,
            local_start: self.next_local_ordinal,
            record_count: 0,
            columns,
            local_ordinals,
            writer,
            _memory_permit: memory_permit,
        });
        Ok(())
    }

    fn finish_current(&mut self) -> Result<()> {
        let Some(mut current) = self.current.take() else {
            return Ok(());
        };
        Self::flush_current(
            &mut current,
            self.range_ordinal,
            self.schema.clone(),
            self.rows_per_batch,
        )?;
        current.writer.finish().with_context(|| {
            format!("failed to finish Arrow segment {}", current.path.display())
        })?;
        let byte_count = fs::metadata(&current.path)?.len();
        self.segments.push(DescriptorSegment {
            path: current.path,
            range_ordinal: self.range_ordinal,
            local_start: current.local_start,
            total_two_j: self.total_two_j,
            parity: self.parity,
            record_count: current.record_count,
            byte_count,
            rows_per_batch: self.rows_per_batch,
        });
        Ok(())
    }

    fn flush_current(
        current: &mut OpenSegment,
        range_ordinal: u32,
        schema: SchemaRef,
        rows_per_batch: usize,
    ) -> Result<()> {
        let row_count = current.local_ordinals.len();
        if row_count == 0 {
            return Ok(());
        }
        let mut arrays: Vec<Arc<dyn Array>> = current
            .columns
            .iter_mut()
            .map(|column| {
                let values = std::mem::replace(column, Vec::with_capacity(rows_per_batch));
                Arc::new(Int32Array::from(values)) as Arc<dyn Array>
            })
            .collect();
        arrays.push(Arc::new(UInt32Array::from(vec![range_ordinal; row_count])));
        arrays.push(Arc::new(UInt64Array::from(std::mem::replace(
            &mut current.local_ordinals,
            Vec::with_capacity(rows_per_batch),
        ))));
        let batch = RecordBatch::try_new(schema, arrays)
            .context("failed to construct Arrow descriptor segment batch")?;
        current
            .writer
            .write(&batch)
            .with_context(|| format!("failed to write Arrow segment {}", current.path.display()))?;
        Ok(())
    }
}

/// The Arrow IPC write options a segment codec asks for.
///
/// Both codecs are compiled in (the `arrow-ipc` `lz4` and `zstd` features), so
/// a selected codec genuinely compresses; a value Arrow could not honour would
/// fail with Arrow's own error instead of silently writing plain segments that
/// the statistics would still describe as compressed.
fn segment_write_options(codec: SegmentCodec) -> Result<IpcWriteOptions> {
    let compression = match codec {
        SegmentCodec::None => None,
        SegmentCodec::Lz4Frame => Some(arrow_ipc::CompressionType::LZ4_FRAME),
        SegmentCodec::Zstd => Some(arrow_ipc::CompressionType::ZSTD),
    };
    IpcWriteOptions::default()
        .try_with_compression(compression)
        .map_err(|error| anyhow::anyhow!("segment codec {} is not usable: {error}", codec.name()))
}

fn segment_schema(layout: DescriptorLayout) -> Result<SchemaRef> {
    let schema = output_schema(layout, false)?;
    let mut fields = schema.fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(Field::new(
        "range_ordinal",
        DataType::UInt32,
        false,
    )));
    fields.push(Arc::new(Field::new(
        "local_ordinal",
        DataType::UInt64,
        false,
    )));
    Ok(Arc::new(Schema::new(fields)))
}

fn block_key(total_two_j: u16, parity: Parity) -> (u16, bool) {
    (total_two_j, parity == Parity::Odd)
}

fn copy_segment_to_parquet(
    segment: &DescriptorSegment,
    layout: DescriptorLayout,
    output_schema: SchemaRef,
    writer: &mut ArrowWriter<File>,
) -> Result<usize> {
    let file = File::open(&segment.path)
        .with_context(|| format!("failed to open segment {}", segment.path.display()))?;
    let reader = FileReader::try_new(file, None)
        .with_context(|| format!("failed to read Arrow segment {}", segment.path.display()))?;
    let expected_columns = layout
        .row_len()
        .checked_add(2)
        .context("segment column count overflow")?;
    let mut count = 0usize;
    let mut next_local = segment.local_start;
    for batch in reader {
        let batch = batch
            .with_context(|| format!("failed to decode segment {}", segment.path.display()))?;
        ensure!(
            batch.num_columns() == expected_columns,
            "segment {} has {} columns, expected {expected_columns}",
            segment.path.display(),
            batch.num_columns()
        );
        let range_column = batch
            .column(layout.row_len())
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("segment range_ordinal column is not UInt32")?;
        let local_column = batch
            .column(layout.row_len() + 1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .context("segment local_ordinal column is not UInt64")?;
        for row in 0..batch.num_rows() {
            ensure!(
                range_column.value(row) == segment.range_ordinal,
                "segment range ordinal does not match file metadata"
            );
            ensure!(
                local_column.value(row) == next_local,
                "segment local ordinal is not contiguous"
            );
            next_local += 1;
        }
        let columns = batch.columns()[..layout.row_len()].to_vec();
        let output = RecordBatch::try_new(output_schema.clone(), columns)
            .context("failed to construct final descriptor batch")?;
        writer
            .write(&output)
            .context("failed to write final descriptor batch")?;
        count = count
            .checked_add(batch.num_rows())
            .context("segment record count overflow")?;
    }
    ensure!(
        count == segment.record_count,
        "segment {} contains {count} rows, expected {}",
        segment.path.display(),
        segment.record_count
    );
    Ok(count)
}

fn visit_segment_rows(
    segment: &DescriptorSegment,
    layout: DescriptorLayout,
    mut visit: impl FnMut(&[i32]) -> Result<()>,
) -> Result<usize> {
    let file = File::open(&segment.path)
        .with_context(|| format!("failed to open segment {}", segment.path.display()))?;
    let reader = FileReader::try_new(file, None)
        .with_context(|| format!("failed to read Arrow segment {}", segment.path.display()))?;
    let expected_columns = layout
        .row_len()
        .checked_add(2)
        .context("segment column count overflow")?;
    let mut count = 0usize;
    let mut next_local = segment.local_start;
    let mut row_values = vec![0; layout.row_len()];
    for batch in reader {
        let batch = batch
            .with_context(|| format!("failed to decode segment {}", segment.path.display()))?;
        ensure!(
            batch.num_columns() == expected_columns,
            "segment {} has {} columns, expected {expected_columns}",
            segment.path.display(),
            batch.num_columns()
        );
        let data_columns = batch.columns()[..layout.row_len()]
            .iter()
            .map(|column| {
                column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .context("segment descriptor column is not Int32")
            })
            .collect::<Result<Vec<_>>>()?;
        let range_column = batch
            .column(layout.row_len())
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("segment range_ordinal column is not UInt32")?;
        let local_column = batch
            .column(layout.row_len() + 1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .context("segment local_ordinal column is not UInt64")?;
        for row in 0..batch.num_rows() {
            ensure!(
                range_column.value(row) == segment.range_ordinal,
                "segment range ordinal does not match file metadata"
            );
            ensure!(
                local_column.value(row) == next_local,
                "segment local ordinal is not contiguous"
            );
            next_local += 1;
            for (value, column) in row_values.iter_mut().zip(&data_columns) {
                *value = column.value(row);
            }
            visit(&row_values)?;
            count = count
                .checked_add(1)
                .context("segment record count overflow")?;
        }
    }
    ensure!(
        count == segment.record_count,
        "segment {} contains {count} rows, expected {}",
        segment.path.display(),
        segment.record_count
    );
    Ok(count)
}

fn copy_surviving_segment_to_parquet(
    segment: &DescriptorSegment,
    layout: DescriptorLayout,
    output_schema: SchemaRef,
    writer: &mut ArrowWriter<File>,
    survivors: &SurvivorBitset,
    next_ordinal: &mut u64,
) -> Result<usize> {
    let file = File::open(&segment.path)
        .with_context(|| format!("failed to open segment {}", segment.path.display()))?;
    let reader = FileReader::try_new(file, None)
        .with_context(|| format!("failed to read Arrow segment {}", segment.path.display()))?;
    let expected_columns = layout
        .row_len()
        .checked_add(2)
        .context("segment column count overflow")?;
    let mut count = 0usize;
    let mut source_count = 0usize;
    let mut next_local = segment.local_start;
    for batch in reader {
        let batch = batch
            .with_context(|| format!("failed to decode segment {}", segment.path.display()))?;
        ensure!(
            batch.num_columns() == expected_columns,
            "segment {} has {} columns, expected {expected_columns}",
            segment.path.display(),
            batch.num_columns()
        );
        let range_column = batch
            .column(layout.row_len())
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("segment range_ordinal column is not UInt32")?;
        let local_column = batch
            .column(layout.row_len() + 1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .context("segment local_ordinal column is not UInt64")?;
        let mut selected_columns = (0..layout.row_len())
            .map(|_| Vec::with_capacity(batch.num_rows()))
            .collect::<Vec<_>>();
        let data_columns = batch.columns()[..layout.row_len()]
            .iter()
            .map(|column| {
                column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .context("segment descriptor column is not Int32")
            })
            .collect::<Result<Vec<_>>>()?;
        for row in 0..batch.num_rows() {
            ensure!(
                range_column.value(row) == segment.range_ordinal,
                "segment range ordinal does not match file metadata"
            );
            ensure!(
                local_column.value(row) == next_local,
                "segment local ordinal is not contiguous"
            );
            next_local += 1;
            source_count = source_count
                .checked_add(1)
                .context("segment source count overflow")?;
            if survivors.contains(*next_ordinal)? {
                for (selected, column) in selected_columns.iter_mut().zip(&data_columns) {
                    selected.push(column.value(row));
                }
                count = count
                    .checked_add(1)
                    .context("surviving segment count overflow")?;
            }
            *next_ordinal = next_ordinal
                .checked_add(1)
                .context("survivor ordinal overflow")?;
        }
        if !selected_columns[0].is_empty() {
            let arrays = selected_columns
                .into_iter()
                .map(|column| Arc::new(Int32Array::from(column)) as Arc<dyn Array>)
                .collect();
            let output = RecordBatch::try_new(output_schema.clone(), arrays)
                .context("failed to construct surviving descriptor batch")?;
            writer
                .write(&output)
                .context("failed to write surviving descriptor batch")?;
        }
    }
    ensure!(
        source_count == segment.record_count,
        "segment {} contains {source_count} rows, expected {}",
        segment.path.display(),
        segment.record_count
    );
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csf_generation::final_encoding::build_final_outputs_from_segments;
    use crate::csf_generation::{GenerationRequest, enumerate_occupations, generate_csfs};
    use crate::descriptor_v2::write_feature_row;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn transcript() -> &'static str {
        "* ! Orbital order\n0\n2s(2,*)2p(1,*)\n\n3s,3p\n1,3\n1\nn\n"
    }

    fn temporary_directory(label: &str) -> PathBuf {
        let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rcsfs-streaming-{label}-{}-{sequence}",
            std::process::id()
        ))
    }

    fn read_rows(path: &Path) -> Result<Vec<Vec<i32>>> {
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;
        let mut rows = Vec::new();
        for batch in reader {
            let batch = batch?;
            let columns = batch
                .columns()
                .iter()
                .map(|column| {
                    column
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .context("final descriptor column is not Int32")
                })
                .collect::<Result<Vec<_>>>()?;
            for row in 0..batch.num_rows() {
                rows.push(columns.iter().map(|column| column.value(row)).collect());
            }
        }
        Ok(rows)
    }

    fn expected_rows(
        request: &ExcitationRequest,
        peel_subshells: &[String],
    ) -> Result<Vec<Vec<i32>>> {
        let occupations = enumerate_occupations(request)?;
        let global_indices = peel_subshells
            .iter()
            .enumerate()
            .map(|(index, label)| Ok((label.parse::<Subshell>()?, u16::try_from(index)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        let layout = DescriptorLayout::new(DescriptorVersion::V2, peel_subshells.len());
        let mut blocks = BTreeMap::<(u16, bool), Vec<Vec<i32>>>::new();
        for configuration in occupations.configurations {
            let generation_request = GenerationRequest {
                core_subshells: occupations.core_subshells.clone(),
                configuration: configuration.occupations,
                min_two_j: request.min_two_j,
                max_two_j: request.max_two_j,
            };
            let file = generate_csfs(&generation_request)?;
            for record in &file.records {
                let remapped = file
                    .occupied(record)?
                    .iter()
                    .map(|occupied| {
                        let local = file
                            .subshells
                            .get(usize::from(occupied.subshell_index))
                            .context("generated local subshell is missing")?
                            .parse::<Subshell>()?;
                        Ok(OccupiedSubshell {
                            subshell_index: *global_indices
                                .get(&local)
                                .context("generated subshell is absent from global Peel")?,
                            occupation: occupied.occupation,
                            state: occupied.state,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut row = vec![0; layout.row_len()];
                write_feature_row(
                    layout,
                    &remapped,
                    file.couplings(record)?,
                    record.total_two_j,
                    record.parity,
                    &mut row,
                )?;
                blocks
                    .entry(block_key(record.total_two_j, record.parity))
                    .or_default()
                    .push(row);
            }
        }
        Ok(blocks.into_values().flatten().collect())
    }

    fn copy_segment_into_later_range(
        source: &DescriptorSegment,
        layout: DescriptorLayout,
        destination: &Path,
        range_ordinal: u32,
    ) -> DescriptorSegment {
        let input = File::open(&source.path).unwrap();
        let reader = FileReader::try_new(input, None).unwrap();
        let schema = segment_schema(layout).unwrap();
        let output = File::options()
            .write(true)
            .create_new(true)
            .open(destination)
            .unwrap();
        let mut writer = FileWriter::try_new_buffered(output, schema.as_ref()).unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            let mut columns = batch.columns()[..layout.row_len()].to_vec();
            columns.push(Arc::new(UInt32Array::from(vec![
                range_ordinal;
                batch.num_rows()
            ])));
            columns.push(batch.column(layout.row_len() + 1).clone());
            writer
                .write(&RecordBatch::try_new(schema.clone(), columns).unwrap())
                .unwrap();
        }
        writer.finish().unwrap();
        DescriptorSegment {
            path: destination.to_path_buf(),
            range_ordinal,
            local_start: source.local_start,
            total_two_j: source.total_two_j,
            parity: source.parity,
            record_count: source.record_count,
            byte_count: fs::metadata(destination).unwrap().len(),
            rows_per_batch: source.rows_per_batch,
        }
    }

    /// A configuration rich enough to be divided along its state-prefix tree:
    /// five configurations, 96 records, 38 in the largest one.
    fn prefix_split_transcript() -> &'static str {
        "* ! Orbital order\n0\n5g(4,*)\n\n5g\n0,8\n1\nn\n"
    }

    #[test]
    fn range_segments_merge_to_the_same_v2_rows_at_each_thread_count() {
        // This transcript reaches the state-prefix split, which the registered
        // small transcripts never did: every configuration there produces at
        // most one record per 2J, so no split was ever scheduled and a broken
        // split could not be observed.
        let request = ExcitationRequest::from_transcript(prefix_split_transcript()).unwrap();
        let root = temporary_directory("merge");
        fs::create_dir(&root).unwrap();
        // Thread count and scheduling granularity are execution choices: the
        // published rows must be identical for every combination, including the
        // smallest task target that still fits one configuration per task and
        // the state-prefix splitting it forces.
        let mut schedules = vec![
            (1usize, 1u64),
            (2, 1),
            (num_cpus::get().max(1), 1),
            (1, 2),
            (1, 3),
            (1, 5),
            (1, u64::MAX),
        ];
        schedules.sort_unstable();
        schedules.dedup();
        let mut results = Vec::new();
        let mut prefix_tasks = 0usize;
        for (threads, records_per_task) in schedules {
            let label = format!("t{threads}-r{records_per_task}");
            let scratch = root.join(format!("scratch-{label}"));
            let output = root.join(format!("descriptors-{label}.parquet"));
            let generated = generate_v2_descriptor_segments(
                &request,
                &scratch,
                &StreamingGenerationOptions {
                    threads: Some(threads),
                    records_per_task: Some(records_per_task),
                    rows_per_batch: 2,
                    rows_per_segment: 3,
                    ..StreamingGenerationOptions::default()
                },
            )
            .unwrap();
            assert!(!generated.segments.is_empty());
            assert!(
                generated
                    .segments
                    .iter()
                    .all(|segment| segment.byte_count > 0)
            );
            prefix_tasks += generated
                .plan
                .tasks
                .iter()
                .filter(|task| matches!(task.span, TaskSpan::StatePrefixes { .. }))
                .count();
            let merge = merge_v2_descriptor_segments(&generated, &output).unwrap();
            assert_eq!(merge.record_count, generated.record_count);
            results.push((
                read_rows(&output).unwrap(),
                expected_rows(&request, &generated.peel_subshells).unwrap(),
            ));
        }
        assert!(
            prefix_tasks > 0,
            "no schedule exercised the state-prefix split"
        );
        assert!(
            results
                .iter()
                .all(|(rows, comparison)| rows == comparison && rows == &results[0].0),
            "scheduling changed the published rows"
        );
        fs::remove_dir_all(root).unwrap();
    }

    /// A plan accounts for the counted workload exactly once and walks the
    /// configurations in enumeration order.
    ///
    /// Splitting one configuration across several tasks is allowed, so the
    /// configurations a task touches do not partition the input on their own.
    /// What must hold is that the estimated records add up to the counted
    /// total — a task that is computed and then not scheduled would silently
    /// lose those records — and that no task reaches backwards.
    #[test]
    fn a_plan_accounts_for_every_counted_record_in_order() {
        for target in [None, Some(1), Some(3)] {
            let request = ExcitationRequest::from_transcript(transcript()).unwrap();
            let occupations = enumerate_occupations(&request).unwrap();
            let workload = estimate_workload(&request, &occupations, Some(1)).unwrap();
            let plan = plan_generation(&request, &occupations, workload, Some(1), target).unwrap();
            let scheduled = plan
                .tasks
                .iter()
                .try_fold(0u64, |total, task| {
                    total.checked_add(task.estimated_records)
                })
                .unwrap();
            assert_eq!(scheduled, plan.workload.total_records);
            let mut previous = 0;
            for task in &plan.tasks {
                let span = task.span.configurations();
                assert!(
                    span.start >= previous && span.end > span.start,
                    "task spans must advance in configuration order: {:?}",
                    task.span
                );
                previous = span.start;
            }
            assert_eq!(
                plan.tasks.last().unwrap().span.configurations().end,
                occupations.configurations.len()
            );
        }
    }

    /// A rejected pre-flight must leave the scratch directory without a single
    /// segment: the whole point is to fail before hours of writing.
    #[test]
    fn a_refused_preflight_writes_no_segment() {
        let request = ExcitationRequest::from_transcript(transcript()).unwrap();
        let root = temporary_directory("preflight");
        fs::create_dir(&root).unwrap();
        let scratch = root.join("scratch");
        let error = generate_v2_descriptor_segments_checked(
            &request,
            &scratch,
            &StreamingGenerationOptions {
                threads: Some(1),
                ..StreamingGenerationOptions::default()
            },
            |estimate| {
                assert!(estimate.scratch_peak_bytes > 0);
                anyhow::bail!("refused by the test pre-flight")
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("refused by the test pre-flight"));
        assert!(
            !scratch.exists(),
            "a refused pre-flight must not create the scratch directory"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_segment_merge_does_not_publish_a_descriptor() {
        let request = ExcitationRequest::from_transcript(transcript()).unwrap();
        let root = temporary_directory("failed-merge");
        fs::create_dir(&root).unwrap();
        let generated = generate_v2_descriptor_segments(
            &request,
            &root.join("scratch"),
            &StreamingGenerationOptions {
                threads: Some(1),
                records_per_task: Some(1),
                rows_per_batch: 2,
                rows_per_segment: 3,
                ..StreamingGenerationOptions::default()
            },
        )
        .unwrap();
        fs::remove_file(&generated.segments[0].path).unwrap();
        let output = root.join("descriptors.parquet");
        assert!(merge_v2_descriptor_segments(&generated, &output).is_err());
        assert!(!output.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn precomputed_peel_matches_the_enumerated_nonzero_union() {
        let request = ExcitationRequest::from_transcript(transcript()).unwrap();
        let occupations = enumerate_occupations(&request).unwrap();
        let peel = precompute_peel_subshells(&occupations);
        assert!(peel.windows(2).all(|pair| {
            (pair[0].n(), pair[0].l(), pair[0].kappa() < 0)
                < (pair[1].n(), pair[1].l(), pair[1].kappa() < 0)
        }));
        assert!(peel.iter().all(|shell| {
            occupations
                .configurations
                .iter()
                .flat_map(|configuration| &configuration.occupations)
                .any(|occupation| occupation.electrons > 0 && occupation.subshell == *shell)
        }));
    }

    #[test]
    fn disk_deduplication_keeps_first_rows_across_segments_and_recursive_buckets() {
        let request = ExcitationRequest::from_transcript(transcript()).unwrap();
        let root = temporary_directory("dedup");
        fs::create_dir(&root).unwrap();
        let generated = generate_v2_descriptor_segments(
            &request,
            &root.join("ranges"),
            &StreamingGenerationOptions {
                threads: Some(1),
                records_per_task: Some(1),
                rows_per_batch: 2,
                rows_per_segment: 3,
                ..StreamingGenerationOptions::default()
            },
        )
        .unwrap();
        let baseline_output = root.join("baseline.parquet");
        merge_v2_descriptor_segments(&generated, &baseline_output).unwrap();
        let baseline_rows = read_rows(&baseline_output).unwrap();

        let duplicate = copy_segment_into_later_range(
            &generated.segments[0],
            generated.layout,
            &root.join("later-range.arrow"),
            999_999,
        );
        let duplicated = SegmentGeneration {
            layout: generated.layout,
            peel_subshells: generated.peel_subshells.clone(),
            core_subshells: generated.core_subshells.clone(),
            plan: generated.plan.clone(),
            segments: generated
                .segments
                .iter()
                .cloned()
                .chain(std::iter::once(duplicate.clone()))
                .collect(),
            unique_occupations: generated.unique_occupations,
            record_count: generated.record_count + duplicate.record_count,
            segment_codec: generated.segment_codec,
            stage_stats: generated.stage_stats.clone(),
            budget: generated.budget.clone(),
            resource_stats: generated.resource_stats.clone(),
        };
        let deduplicated = deduplicate_v2_descriptor_segments(
            &duplicated,
            &root.join("dedup"),
            &DeduplicationOptions {
                bucket_count: 1,
                max_rows_per_bucket: 2,
            },
        )
        .unwrap();
        assert_eq!(deduplicated.generated_count, duplicated.record_count);
        assert_eq!(deduplicated.unique_count, baseline_rows.len());
        assert_eq!(deduplicated.duplicate_count, duplicate.record_count);
        let output = root.join("unique.parquet");
        let merge = merge_v2_deduplicated_segments(&deduplicated, &output).unwrap();
        assert_eq!(merge.record_count, baseline_rows.len());
        assert_eq!(read_rows(&output).unwrap(), baseline_rows);
        fs::remove_dir_all(root).unwrap();
    }

    /// One run's published result: CSF text, descriptor rows, record count,
    /// block count.
    type Published = (Vec<u8>, Vec<Vec<i32>>, usize, usize);

    /// The verified path must publish exactly what the exact path publishes.
    ///
    /// This is the differential the proof needs: the same transcript, generated
    /// and published twice, once with the root-bucket comparison and once
    /// without it. If the two disagree on a byte, the proof or the fast path is
    /// wrong, and the exact path is what decides.
    #[test]
    fn the_verified_path_publishes_what_the_exact_path_publishes() {
        let root = temporary_directory("verified-differential");
        fs::create_dir(&root).unwrap();
        let mut previous: Option<Published> = None;
        for strategy in [
            DeduplicationStrategy::VerifiedUnique,
            DeduplicationStrategy::Exact,
        ] {
            let directory = root.join(strategy.name());
            fs::create_dir(&directory).unwrap();
            let csf = directory.join("out.c");
            let csf_parquet = directory.join("out.parquet");
            let descriptors = directory.join("descriptors.parquet");
            let header = directory.join("header.toml");
            let stats = generate_disk_outputs_from_transcript_with_options(
                prefix_split_transcript(),
                &csf,
                &csf_parquet,
                &descriptors,
                &header,
                &GenerationOptions {
                    threads: Some(1),
                    records_per_task: Some(3),
                    rows_per_batch: 2,
                    rows_per_segment: 3,
                    deduplication: strategy,
                    scratch_dir: Some(directory.join("scratch")),
                    ..GenerationOptions::default()
                },
            )
            .unwrap();
            assert_eq!(stats.deduplication, strategy.name());
            // Both paths must agree on the statistics, not only on the bytes:
            // the verified path reports zero duplicates by construction and the
            // exact path must find zero here for that to be the same statement.
            assert_eq!(stats.duplicate_count, 0);
            assert_eq!(stats.generated_count, stats.unique_count);
            let publication: Published = (
                fs::read(&csf).unwrap(),
                read_rows(&descriptors).unwrap(),
                stats.generated_count,
                stats.block_count,
            );
            if let Some((other_csf, other_rows, other_records, other_blocks)) = &previous {
                assert_eq!(
                    &publication.0, other_csf,
                    "{strategy:?} changed the CSF text"
                );
                assert_eq!(
                    &publication.1, other_rows,
                    "{strategy:?} changed the descriptor rows"
                );
                assert_eq!(
                    publication.2, *other_records,
                    "{strategy:?} changed the record count"
                );
                assert_eq!(
                    publication.3, *other_blocks,
                    "{strategy:?} changed the block count"
                );
            }
            previous = Some(publication);
        }
        fs::remove_dir_all(root).unwrap();
    }

    /// And the two paths must be genuinely different: on a segment set that
    /// repeats a row — the shape the P6a control produces — the verified path
    /// keeps both copies while the exact path removes one.
    ///
    /// Nothing reachable through a transcript looks like this, which is exactly
    /// why the fast path is limited to the internal generator. The test states
    /// the boundary instead of leaving it implicit: if the fast path were ever
    /// selected for a repeated input, this is the difference it would make.
    #[test]
    fn a_repeated_segment_is_only_removed_by_the_exact_path() {
        let request = ExcitationRequest::from_transcript(transcript()).unwrap();
        let root = temporary_directory("verified-boundary");
        fs::create_dir(&root).unwrap();
        let generated = generate_v2_descriptor_segments(
            &request,
            &root.join("ranges"),
            &StreamingGenerationOptions {
                threads: Some(1),
                ..StreamingGenerationOptions::default()
            },
        )
        .unwrap();
        let duplicate = copy_segment_into_later_range(
            &generated.segments[0],
            generated.layout,
            &root.join("later-range.arrow"),
            999_999,
        );
        let duplicated = SegmentGeneration {
            layout: generated.layout,
            peel_subshells: generated.peel_subshells.clone(),
            core_subshells: generated.core_subshells.clone(),
            plan: generated.plan.clone(),
            segments: generated
                .segments
                .iter()
                .cloned()
                .chain(std::iter::once(duplicate.clone()))
                .collect(),
            unique_occupations: generated.unique_occupations,
            record_count: generated.record_count + duplicate.record_count,
            segment_codec: generated.segment_codec,
            stage_stats: generated.stage_stats.clone(),
            budget: generated.budget.clone(),
            resource_stats: generated.resource_stats.clone(),
        };

        let verified =
            verified_unique_deduplication(&duplicated, &root.join("verified-dedup")).unwrap();
        assert_eq!(verified.duplicate_count, 0);
        assert_eq!(verified.unique_count, duplicated.record_count);

        let exact = deduplicate_v2_descriptor_segments(
            &duplicated,
            &root.join("exact-dedup"),
            &DeduplicationOptions::default(),
        )
        .unwrap();
        assert_eq!(exact.duplicate_count, duplicate.record_count);
        assert_eq!(
            exact.unique_count + exact.duplicate_count,
            verified.unique_count,
            "the two paths must disagree by exactly the repeated rows"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn complete_row_comparison_preserves_v2_distinctions_after_hash_collision() {
        let root = temporary_directory("collision");
        fs::create_dir(&root).unwrap();
        let bucket_dir = root.join("buckets");
        fs::create_dir(&bucket_dir).unwrap();
        let mut buckets = BucketWriters::new(&bucket_dir, "forced", 1, 6).unwrap();
        let collision = [0; 16];
        let seniority_one = [2, 1, 1, -1, 2, 1];
        let seniority_three = [2, 1, 3, -1, 2, 1];
        let explicit_zero = [2, 1, 1, 0, 2, 1];
        buckets.push(0, 0, collision, &seniority_one).unwrap();
        buckets.push(0, 1, collision, &seniority_three).unwrap();
        buckets.push(0, 2, collision, &explicit_zero).unwrap();
        buckets.push(0, 3, collision, &seniority_one).unwrap();
        let bucket = buckets.finish().unwrap().pop().unwrap();
        let mut bitset = SurvivorBitset::new(root.join("survivors.bitset"), 4).unwrap();
        let mut stats = DeduplicationStats::default();
        let budget = ResourceBudget::unlimited();
        deduplicate_bucket(bucket, 6, &mut bitset, &mut stats, &budget, 4).unwrap();
        assert_eq!(stats.generated_count, 4);
        assert_eq!(stats.unique_count, 3);
        assert_eq!(stats.duplicate_count, 1);
        assert_eq!(stats.hash_collision_count, 2);
        assert!(bitset.contains(0).unwrap());
        assert!(bitset.contains(1).unwrap());
        assert!(bitset.contains(2).unwrap());
        assert!(!bitset.contains(3).unwrap());
        bitset.finish().unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_deduplicated_merge_does_not_publish_a_descriptor() {
        let request = ExcitationRequest::from_transcript(transcript()).unwrap();
        let root = temporary_directory("failed-deduplicated-merge");
        fs::create_dir(&root).unwrap();
        let generated = generate_v2_descriptor_segments(
            &request,
            &root.join("ranges"),
            &StreamingGenerationOptions {
                threads: Some(1),
                records_per_task: Some(1),
                rows_per_batch: 2,
                rows_per_segment: 3,
                ..StreamingGenerationOptions::default()
            },
        )
        .unwrap();
        let deduplicated = deduplicate_v2_descriptor_segments(
            &generated,
            &root.join("dedup"),
            &DeduplicationOptions::default(),
        )
        .unwrap();
        fs::remove_file(&generated.segments[0].path).unwrap();
        let output = root.join("unique.parquet");
        assert!(merge_v2_deduplicated_segments(&deduplicated, &output).is_err());
        assert!(!output.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn disk_pipeline_restores_the_same_csf_text_as_memory_generation() {
        let root = temporary_directory("disk-pipeline");
        fs::create_dir(&root).unwrap();
        let expected = root.join("expected.c");
        super::super::generate_csfs_from_transcript(transcript(), &expected, Some(1)).unwrap();
        let generated = root.join("generated.c");
        let csf_parquet = root.join("generated.parquet");
        let descriptors = root.join("generated_descriptors.parquet");
        let header = root.join("generated_header.toml");
        let stats = generate_disk_outputs_from_transcript(
            transcript(),
            &generated,
            &csf_parquet,
            &descriptors,
            &header,
            &root.join("scratch"),
            Some(1),
        )
        .unwrap();
        assert_eq!(fs::read(&generated).unwrap(), fs::read(&expected).unwrap());
        assert_eq!(stats.unique_count, read_rows(&descriptors).unwrap().len());
        assert!(stats.duplicate_count <= stats.generated_count);
        assert!(csf_parquet.exists());
        fs::remove_dir_all(root).unwrap();
    }

    /// The combined final-encoding pass must publish exactly what the
    /// two-pass path (descriptor merge, then descriptor read-back) published.
    ///
    /// This is P4's differential: the old tail is kept as the reference
    /// implementation precisely so the new one-pass tail can be compared
    /// against it. The descriptor Parquet and the CSF text are compared byte
    /// for byte — same writers, same order — while the CSF Parquet is compared
    /// by logical rows, because its row-group boundaries follow the batch
    /// layout of whichever path produced it and byte equality is not part of
    /// its contract.
    ///
    /// The thread count is a scheduling choice, so the same pass is run at
    /// 1/2/4/8 threads here: the published CSF text and the descriptor's
    /// logical V2 rows must be identical for every one of them.
    #[test]
    fn the_combined_final_pass_publishes_what_merge_and_restore_published() {
        let root = temporary_directory("final-encoding-differential");
        fs::create_dir(&root).unwrap();
        let mut publications = Vec::new();
        for (label, combined) in [("two-pass", false), ("combined", true)] {
            let directory = root.join(label);
            fs::create_dir(&directory).unwrap();
            let request = ExcitationRequest::from_transcript(prefix_split_transcript()).unwrap();
            let generated = generate_v2_descriptor_segments(
                &request,
                &directory.join("ranges"),
                &StreamingGenerationOptions {
                    threads: Some(1),
                    records_per_task: Some(3),
                    rows_per_batch: 2,
                    rows_per_segment: 5,
                    ..StreamingGenerationOptions::default()
                },
            )
            .unwrap();
            let deduplicated =
                verified_unique_deduplication(&generated, &directory.join("dedup")).unwrap();
            let header_lines =
                generated_header_lines(&generated.core_subshells, &generated.peel_subshells);
            write_generation_header(
                &directory.join("header.toml"),
                header_lines.clone(),
                &deduplicated,
            )
            .unwrap();
            let csf = directory.join("out.c");
            let csf_parquet = directory.join("out.parquet");
            let descriptors = directory.join("descriptors.parquet");
            if combined {
                let stats = build_final_outputs_from_segments(
                    &deduplicated,
                    &header_lines,
                    &descriptors,
                    &csf,
                    &csf_parquet,
                    Some(2),
                )
                .unwrap();
                assert_eq!(stats.record_count, deduplicated.unique_count);
                // Every phase is reported, and the seven of them together are
                // the pass: segment reading, selection, parallel preparation,
                // and each artifact family split into compute and write halves.
                let entries = stats.phase_entries();
                assert_eq!(
                    entries.map(|(name, _, _)| name),
                    [
                        "final_encoding_read",
                        "final_encoding_select",
                        "final_encoding_prepare",
                        "final_encoding_descriptor_encode",
                        "final_encoding_descriptor_write",
                        "final_encoding_csf_outputs_encode",
                        "final_encoding_csf_outputs_write",
                    ]
                );
                let pass: u128 = entries.iter().map(|(_, millis, _)| millis).sum();
                assert!(pass <= stats.record_count as u128 * 1000);
            } else {
                merge_v2_deduplicated_segments(&deduplicated, &descriptors).unwrap();
                crate::csfs_descriptor::restore_v2_descriptor_parquet_to_outputs(
                    &descriptors,
                    &directory.join("header.toml"),
                    &csf,
                    Some(&csf_parquet),
                )
                .unwrap();
            }
            publications.push((
                fs::read(&csf).unwrap(),
                read_rows(&descriptors).unwrap(),
                ParquetRecordBatchReaderBuilder::try_new(File::open(&descriptors).unwrap())
                    .unwrap()
                    .metadata()
                    .file_metadata()
                    .key_value_metadata()
                    .cloned(),
                read_csf_parquet_rows(&csf_parquet).unwrap(),
            ));
        }
        let (old_csf, old_descriptors, old_metadata, old_parquet_rows) = &publications[0];
        let (new_csf, new_descriptors, new_metadata, new_parquet_rows) = &publications[1];
        assert_eq!(new_csf, old_csf, "the combined pass changed the CSF text");
        assert_eq!(
            new_descriptors, old_descriptors,
            "the combined pass changed the descriptor V2 rows"
        );
        assert_eq!(
            new_metadata, old_metadata,
            "the combined pass changed the descriptor metadata"
        );
        assert_eq!(
            new_parquet_rows, old_parquet_rows,
            "the combined pass changed the CSF Parquet rows"
        );
        fs::remove_dir_all(root).unwrap();
    }

    /// The thread count is a scheduling choice: it must not reach the artifacts.
    ///
    /// The combined pass parallelizes preparation across `threads`, and the
    /// ordered publication side is what keeps that invisible. This runs the same
    /// input at 1/2/4/8 threads and compares the published CSF text byte for
    /// byte and the descriptor Parquet row for row — the evidence the plan asks
    /// for before its thread-invariance item can be closed.
    #[test]
    fn the_combined_final_pass_is_thread_invariant() {
        let root = temporary_directory("final-encoding-threads");
        fs::create_dir(&root).unwrap();
        let mut publications: Vec<(Vec<u8>, Vec<Vec<i32>>, Vec<(u64, String, String, String)>)> =
            Vec::new();
        for threads in [1usize, 2, 4, 8] {
            let directory = root.join(format!("threads-{threads}"));
            fs::create_dir(&directory).unwrap();
            let request = ExcitationRequest::from_transcript(prefix_split_transcript()).unwrap();
            let generated = generate_v2_descriptor_segments(
                &request,
                &directory.join("ranges"),
                &StreamingGenerationOptions {
                    threads: Some(threads),
                    records_per_task: Some(3),
                    rows_per_batch: 2,
                    rows_per_segment: 5,
                    ..StreamingGenerationOptions::default()
                },
            )
            .unwrap();
            let deduplicated =
                verified_unique_deduplication(&generated, &directory.join("dedup")).unwrap();
            let header_lines =
                generated_header_lines(&generated.core_subshells, &generated.peel_subshells);
            write_generation_header(
                &directory.join("header.toml"),
                header_lines.clone(),
                &deduplicated,
            )
            .unwrap();
            let csf = directory.join("out.c");
            let csf_parquet = directory.join("out.parquet");
            let descriptors = directory.join("descriptors.parquet");
            let stats = build_final_outputs_from_segments(
                &deduplicated,
                &header_lines,
                &descriptors,
                &csf,
                &csf_parquet,
                Some(threads),
            )
            .unwrap();
            assert_eq!(stats.record_count, deduplicated.unique_count);
            publications.push((
                fs::read(&csf).unwrap(),
                read_rows(&descriptors).unwrap(),
                read_csf_parquet_rows(&csf_parquet).unwrap(),
            ));
        }
        let (csf, rows, parquet_rows) = &publications[0];
        for (index, (other_csf, other_rows, other_parquet_rows)) in
            publications.iter().enumerate().skip(1)
        {
            assert_eq!(
                other_csf, csf,
                "threads changed the CSF text at run {index}"
            );
            assert_eq!(
                other_rows, rows,
                "threads changed the descriptor V2 rows at run {index}"
            );
            assert_eq!(
                other_parquet_rows, parquet_rows,
                "threads changed the CSF Parquet rows at run {index}"
            );
        }
        fs::remove_dir_all(root).unwrap();
    }

    /// Read the logical rows of a three-line CSF Parquet file.
    fn read_csf_parquet_rows(path: &Path) -> Result<Vec<(u64, String, String, String)>> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let file = File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        let mut rows = Vec::new();
        for batch in reader {
            let batch = batch?;
            let index = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .context("CSF Parquet idx column is not UInt64")?;
            let lines = (1..4)
                .map(|column| {
                    Ok(batch
                        .column(column)
                        .as_any()
                        .downcast_ref::<arrow::array::StringArray>()
                        .context("CSF Parquet line column is not Utf8")?
                        .to_owned())
                })
                .collect::<Result<Vec<_>>>()?;
            for row in 0..batch.num_rows() {
                rows.push((
                    index.value(row),
                    lines[0].value(row).to_owned(),
                    lines[1].value(row).to_owned(),
                    lines[2].value(row).to_owned(),
                ));
            }
        }
        Ok(rows)
    }

    /// Every codec must reach the writer, be readable again, and be reported.
    ///
    /// Recording a codec the writer ignored is the failure this guards: the
    /// reported bytes would then describe compressed segments while the files
    /// are plain, and the compression experiment would "prove" a ratio that
    /// does not exist. The rows are repetitive and numerous enough that
    /// framing overhead cannot mask the effect.
    #[test]
    fn each_segment_codec_is_reported_readable_and_actually_compresses() {
        let root = temporary_directory("segment-codec");
        fs::create_dir(&root).unwrap();
        let layout = DescriptorLayout::new(DescriptorVersion::V2, 3);
        let row_len = layout.row_len();
        let rows = 20_000;
        let mut sizes = BTreeMap::new();
        for codec in [
            SegmentCodec::None,
            SegmentCodec::Lz4Frame,
            SegmentCodec::Zstd,
        ] {
            let directory = root.join(codec.name());
            fs::create_dir(&directory).unwrap();
            let mut writer = BlockSegmentWriter::new(
                0,
                0,
                Parity::Even,
                row_len,
                segment_schema(layout).unwrap(),
                directory,
                128,
                rows,
                codec,
                ResourceBudget::unlimited(),
            );
            // A row a real run could produce: a handful of distinct values,
            // mostly MISSING, which is what the V2 payload looks like.
            let row = (0..row_len)
                .map(|index| if index % 5 == 0 { index as i32 } else { -1 })
                .collect::<Vec<_>>();
            for _ in 0..rows {
                writer.push(&row).unwrap();
            }
            let (record_count, segments) = writer.finish().unwrap();
            assert_eq!(record_count, rows);
            let byte_count = segments
                .iter()
                .map(|segment| segment.byte_count)
                .sum::<u64>();
            assert!(byte_count > 0);
            assert_eq!(read_segment_rows(&segments[0].path, 256), vec![row; 256]);
            sizes.insert(codec, byte_count);
        }
        let uncompressed = sizes[&SegmentCodec::None];
        for codec in [SegmentCodec::Lz4Frame, SegmentCodec::Zstd] {
            assert!(
                sizes[&codec] < uncompressed,
                "{} wrote {} bytes against {uncompressed} uncompressed, so the codec \
                 never reached the writer",
                codec.name(),
                sizes[&codec],
            );
        }
        fs::remove_dir_all(root).unwrap();
    }

    /// Read the first `limit` rows back out of an Arrow IPC segment.
    fn read_segment_rows(path: &Path, limit: usize) -> Vec<Vec<i32>> {
        let file = File::open(path).unwrap();
        let reader = FileReader::try_new(BufReader::new(file), None).unwrap();
        let mut rows = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let columns = batch
                .columns()
                .iter()
                .take(batch.num_columns() - 2)
                .map(|column| column.as_any().downcast_ref::<Int32Array>().unwrap())
                .collect::<Vec<_>>();
            for index in 0..batch.num_rows() {
                if rows.len() == limit {
                    return rows;
                }
                rows.push(columns.iter().map(|column| column.value(index)).collect());
            }
        }
        rows
    }

    /// A codec must change nothing but the bytes on disk: the same input
    /// publishes the same CSV text, descriptor rows and counts.
    #[test]
    fn the_segment_codec_changes_no_published_byte() {
        let root = temporary_directory("codec-equivalence");
        fs::create_dir(&root).unwrap();
        let transcript = prefix_split_transcript();
        let mut outputs = Vec::new();
        for codec in [
            SegmentCodec::None,
            SegmentCodec::Lz4Frame,
            SegmentCodec::Zstd,
        ] {
            let label = codec.name();
            let directory = root.join(label);
            fs::create_dir(&directory).unwrap();
            let csf = directory.join("out.c");
            let csf_parquet = directory.join("out.parquet");
            let descriptors = directory.join("descriptors.parquet");
            let header = directory.join("header.toml");
            let stats = generate_disk_outputs_from_transcript_with_options(
                transcript,
                &csf,
                &csf_parquet,
                &descriptors,
                &header,
                &GenerationOptions {
                    threads: Some(1),
                    records_per_task: Some(4),
                    rows_per_batch: 2,
                    rows_per_segment: 3,
                    segment_codec: codec,
                    scratch_dir: Some(directory.join("scratch")),
                    ..GenerationOptions::default()
                },
            )
            .unwrap();
            assert_eq!(
                stats.segment_codec,
                codec.name(),
                "the run did not report the codec it was given"
            );
            assert!(stats.duplicate_count == 0, "the input repeats no record");
            outputs.push((
                label,
                fs::read(&csf).unwrap(),
                read_rows(&descriptors).unwrap(),
                stats.generated_count,
            ));
        }
        let (_, first_csf, first_rows, first_count) = &outputs[0];
        for (label, csf, rows, count) in &outputs[1..] {
            assert_eq!(csf, first_csf, "{label} changed the CSF text");
            assert_eq!(rows, first_rows, "{label} changed the descriptor rows");
            assert_eq!(count, first_count, "{label} changed the record count");
        }
        fs::remove_dir_all(root).unwrap();
    }
}
