//! Bounded V2 descriptor generation through ordered on-disk Arrow segments.
//!
//! This module owns the Phase 2 storage adapter.  It deliberately has no
//! Python binding yet: Phase 4 will place it behind the multi-output
//! transaction.  Keeping the range writer here lets the angular generator use
//! `GeneratedRecordSink` without learning about Arrow, scratch paths or final
//! Parquet publication.

use anyhow::{Context, Result, ensure};
use arrow::array::{Array, Int32Array, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{
    EnumeratedConfiguration, EnumeratedOccupations, ExcitationRequest, GeneratedRecordRef,
    GeneratedRecordSink, GenerationRequest, Parity, Subshell, enumerate_occupations,
    generate_records_into,
};
use crate::atomic_output::{create_temporary_output, publish_temporary_output};
use crate::complete_csf::OccupiedSubshell;
use crate::descriptor_schema::{
    DescriptorLayout, DescriptorVersion, output_kv_metadata, output_schema, validate_record,
};
use crate::descriptor_v2::write_feature_row;

const DEFAULT_CONFIGURATIONS_PER_RANGE: usize = 4_096;
const DEFAULT_ROWS_PER_BATCH: usize = 8_192;
const DEFAULT_ROWS_PER_SEGMENT: usize = 131_072;
const DEFAULT_DEDUP_BUCKET_COUNT: usize = 256;
const DEFAULT_DEDUP_MAX_ROWS_PER_BUCKET: usize = 65_536;
const DEDUP_RECURSIVE_BUCKET_COUNT: usize = 16;
const MAX_DEDUP_SPLIT_DEPTH: usize = 16;

/// A deterministic, contiguous slice of enumerated configurations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GenerationRange {
    pub(crate) ordinal: u32,
    pub(crate) start_configuration: usize,
    pub(crate) end_configuration: usize,
}

/// Bounded-buffer settings for the internal streaming path.
///
/// These are crate-visible until Phase 4 introduces the small public
/// `GenerationOptions` interface.  Each setting is validated before any
/// segment is written, so an invalid plan cannot leave a partial final file.
#[derive(Clone, Debug)]
pub(crate) struct StreamingGenerationOptions {
    pub(crate) threads: Option<usize>,
    pub(crate) configurations_per_range: usize,
    pub(crate) rows_per_batch: usize,
    pub(crate) rows_per_segment: usize,
}

impl Default for StreamingGenerationOptions {
    fn default() -> Self {
        Self {
            threads: None,
            configurations_per_range: DEFAULT_CONFIGURATIONS_PER_RANGE,
            rows_per_batch: DEFAULT_ROWS_PER_BATCH,
            rows_per_segment: DEFAULT_ROWS_PER_SEGMENT,
        }
    }
}

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
}

/// Completed generation stage before deduplication.
#[derive(Debug)]
pub(crate) struct SegmentGeneration {
    pub(crate) layout: DescriptorLayout,
    pub(crate) peel_subshells: Vec<String>,
    pub(crate) core_subshells: Vec<Subshell>,
    pub(crate) ranges: Vec<GenerationRange>,
    pub(crate) segments: Vec<DescriptorSegment>,
    pub(crate) unique_occupations: usize,
    pub(crate) record_count: usize,
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
    survivor_bitsets: BTreeMap<(u16, bool), SurvivorBitsetFile>,
    pub(crate) generated_count: usize,
    pub(crate) unique_count: usize,
    pub(crate) duplicate_count: usize,
    pub(crate) hash_collision_count: usize,
    pub(crate) block_lengths: Vec<usize>,
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
}

/// Generate, exactly de-duplicate and restore a V2 descriptor through private
/// staging paths.  The caller owns final publication, which keeps this deep
/// module independent of CLI output naming and gives it all-or-nothing
/// semantics when multiple products are requested.
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
    let request = ExcitationRequest::from_transcript(transcript)?;
    fs::create_dir(scratch_dir).with_context(|| {
        format!(
            "failed to create disk generation scratch directory {}",
            scratch_dir.display()
        )
    })?;
    let generated = generate_v2_descriptor_segments(
        &request,
        &scratch_dir.join("ranges"),
        &StreamingGenerationOptions {
            threads,
            ..StreamingGenerationOptions::default()
        },
    )?;
    let deduplicated = deduplicate_v2_descriptor_segments(
        &generated,
        &scratch_dir.join("dedup"),
        &DeduplicationOptions::default(),
    )?;
    let header_lines = generated_header_lines(&generated.core_subshells, &generated.peel_subshells);
    write_generation_header(header_output, header_lines.clone(), &deduplicated)?;
    merge_v2_deduplicated_segments(&deduplicated, descriptor_output)?;
    crate::csfs_descriptor::restore_v2_descriptor_parquet_to_outputs(
        descriptor_output,
        header_output,
        csf_output,
        Some(csf_parquet_output),
    )?;
    Ok(DiskGenerationStats {
        unique_occupations: generated.unique_occupations,
        generated_count: deduplicated.generated_count,
        unique_count: deduplicated.unique_count,
        duplicate_count: deduplicated.duplicate_count,
        block_count: deduplicated.block_lengths.len(),
        csf_bytes: fs::metadata(csf_output)?.len(),
        descriptor_bytes: fs::metadata(descriptor_output)?.len(),
    })
}

/// Partition `configuration_count` inputs into fixed-size, stable ranges.
pub(crate) fn plan_generation_ranges(
    configuration_count: usize,
    configurations_per_range: usize,
) -> Result<Vec<GenerationRange>> {
    ensure!(
        configurations_per_range > 0,
        "configurations_per_range must be greater than 0"
    );
    let mut ranges = Vec::with_capacity(configuration_count.div_ceil(configurations_per_range));
    for start in (0..configuration_count).step_by(configurations_per_range) {
        let ordinal = u32::try_from(ranges.len()).context("too many generation ranges")?;
        ranges.push(GenerationRange {
            ordinal,
            start_configuration: start,
            end_configuration: (start + configurations_per_range).min(configuration_count),
        });
    }
    Ok(ranges)
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
    validate_options(options)?;
    let occupations = enumerate_occupations(request)?;
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
    let ranges = plan_generation_ranges(
        occupations.configurations.len(),
        options.configurations_per_range,
    )?;
    let ranges_root = scratch_dir.join("ranges");
    fs::create_dir_all(&ranges_root).with_context(|| {
        format!(
            "failed to create scratch directory {}",
            ranges_root.display()
        )
    })?;

    let run_range = || {
        ranges
            .par_iter()
            .map(|range| {
                generate_range_segments(
                    *range,
                    &occupations.configurations,
                    &occupations.core_subshells,
                    request.min_two_j,
                    request.max_two_j,
                    layout,
                    &peel_subshells,
                    &global_indices,
                    &ranges_root,
                    options,
                )
            })
            .collect::<Result<Vec<_>>>()
    };
    let range_results = match options.threads {
        Some(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .context("failed to build descriptor range thread pool")?
            .install(run_range)?,
        None => run_range()?,
    };

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
    Ok(SegmentGeneration {
        layout,
        peel_subshells,
        core_subshells: occupations.core_subshells,
        ranges,
        segments,
        unique_occupations: occupations.configurations.len(),
        record_count,
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
    let properties = WriterProperties::builder()
        .set_compression(
            crate::csfs_descriptor::parquet_batch::parse_compression(None)
                .expect("default compression is valid"),
        )
        .set_dictionary_enabled(true)
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
        let mut buckets = BucketWriters::new(
            &block_dir,
            "root",
            options.bucket_count,
            generated.layout.row_len(),
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
        let bitset_path = block_dir.join("survivors.bitset");
        let mut bitset = SurvivorBitset::new(bitset_path, expected_rows)?;
        let mut block_stats = DeduplicationStats::default();
        for bucket in bucket_files {
            process_bucket(
                bucket,
                generated.layout.row_len(),
                options.max_rows_per_bucket,
                0,
                &mut bitset,
                &mut block_stats,
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
    })
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
    let mut metadata = output_kv_metadata(
        deduplicated.layout,
        &deduplicated.peel_subshells,
        false,
        None,
        None,
    );
    metadata.extend([
        KeyValue::new(
            "generated_record_count".to_owned(),
            Some(deduplicated.generated_count.to_string()),
        ),
        KeyValue::new(
            "unique_record_count".to_owned(),
            Some(deduplicated.unique_count.to_string()),
        ),
        KeyValue::new(
            "duplicate_record_count".to_owned(),
            Some(deduplicated.duplicate_count.to_string()),
        ),
        KeyValue::new(
            "block_lengths".to_owned(),
            Some(format_usize_list(&deduplicated.block_lengths)),
        ),
    ]);
    let properties = WriterProperties::builder()
        .set_compression(
            crate::csfs_descriptor::parquet_batch::parse_compression(None)
                .expect("default compression is valid"),
        )
        .set_dictionary_enabled(true)
        .set_key_value_metadata(Some(metadata))
        .build();
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
struct SurvivorBitsetFile {
    path: PathBuf,
    bit_len: usize,
    unique_count: usize,
}

struct SurvivorBitset {
    path: PathBuf,
    bit_len: usize,
    bytes: Vec<u8>,
}

impl SurvivorBitset {
    fn new(path: PathBuf, bit_len: usize) -> Result<Self> {
        let byte_len = bit_len
            .checked_add(7)
            .context("survivor bitset size overflow")?
            / 8;
        Ok(Self {
            path,
            bit_len,
            bytes: vec![0; byte_len],
        })
    }

    fn read(file: &SurvivorBitsetFile) -> Result<Self> {
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

    fn contains(&self, ordinal: u64) -> Result<bool> {
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
}

impl BucketWriters {
    fn new(directory: &Path, label: &str, count: usize, row_len: usize) -> Result<Self> {
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
) -> Result<()> {
    if bucket.record_count <= max_rows_per_bucket {
        return deduplicate_bucket(bucket, row_len, bitset, stats);
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
    let mut children = BucketWriters::new(
        &split_directory,
        "child",
        DEDUP_RECURSIVE_BUCKET_COUNT,
        row_len,
    )?;
    let mut reader = BucketReader::open(&bucket, row_len)?;
    let mut row = Vec::with_capacity(row_len);
    while let Some((ordinal, hash)) = reader.next(&mut row)? {
        let child = recursive_bucket_index(&row, depth);
        children.push(child, ordinal, hash, &row)?;
    }
    fs::remove_file(&bucket.path)
        .with_context(|| format!("failed to remove split bucket {}", bucket.path.display()))?;
    for child in children.finish()? {
        process_bucket(
            child,
            row_len,
            max_rows_per_bucket,
            depth + 1,
            bitset,
            stats,
        )?;
    }
    Ok(())
}

fn deduplicate_bucket(
    bucket: BucketFile,
    row_len: usize,
    bitset: &mut SurvivorBitset,
    stats: &mut DeduplicationStats,
) -> Result<()> {
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

fn grouped_segments(
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

fn format_usize_list(values: &[usize]) -> String {
    let mut text = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            text.push(',');
        }
        text.push_str(&value.to_string());
    }
    text.push(']');
    text
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
        options.configurations_per_range > 0,
        "configurations_per_range must be greater than 0"
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
fn generate_range_segments(
    range: GenerationRange,
    configurations: &[EnumeratedConfiguration],
    core_subshells: &[Subshell],
    min_two_j: u16,
    max_two_j: u16,
    layout: DescriptorLayout,
    peel_subshells: &[String],
    global_indices: &HashMap<Subshell, u16>,
    ranges_root: &Path,
    options: &StreamingGenerationOptions,
) -> Result<RangeResult> {
    let range_dir = ranges_root.join(format!("range-{:06}", range.ordinal));
    fs::create_dir(&range_dir)
        .with_context(|| format!("failed to create range directory {}", range_dir.display()))?;
    let mut writer = RangeSegmentWriter::new(range.ordinal, layout, range_dir, options)?;
    let branch_count = if rayon::current_num_threads() > 1 {
        rayon::current_num_threads().saturating_mul(4)
    } else {
        1
    };
    for configuration in &configurations[range.start_configuration..range.end_configuration] {
        let generation_request = GenerationRequest {
            core_subshells: core_subshells.to_vec(),
            configuration: configuration.occupations.clone(),
            min_two_j,
            max_two_j,
        };
        let local_to_global = local_to_global_indices(&generation_request, global_indices)?;
        let mut sink =
            DescriptorBatchSink::new(layout, peel_subshells, local_to_global, &mut writer);
        generate_records_into(&generation_request, branch_count, &mut sink)?;
    }
    writer.finish()
}

fn local_to_global_indices(
    request: &GenerationRequest,
    global_indices: &HashMap<Subshell, u16>,
) -> Result<Vec<u16>> {
    request
        .configuration
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
        let columns = (0..self.row_len)
            .map(|_| Vec::with_capacity(self.rows_per_batch))
            .collect();
        let local_ordinals = Vec::with_capacity(self.rows_per_batch);
        let writer = FileWriter::try_new_buffered(file, self.schema.as_ref())
            .with_context(|| format!("failed to open Arrow segment {}", path.display()))?;
        self.current = Some(OpenSegment {
            path,
            local_start: self.next_local_ordinal,
            record_count: 0,
            columns,
            local_ordinals,
            writer,
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
    use crate::csf_generation::generate_csfs;
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
        }
    }

    #[test]
    fn range_segments_merge_to_the_same_v2_rows_at_each_thread_count() {
        let request = ExcitationRequest::from_transcript(transcript()).unwrap();
        let root = temporary_directory("merge");
        fs::create_dir(&root).unwrap();
        let mut thread_counts = vec![1, 2, num_cpus::get().max(1)];
        thread_counts.sort_unstable();
        thread_counts.dedup();
        let mut results = Vec::new();
        for threads in thread_counts {
            let scratch = root.join(format!("scratch-{threads}"));
            let output = root.join(format!("descriptors-{threads}.parquet"));
            let generated = generate_v2_descriptor_segments(
                &request,
                &scratch,
                &StreamingGenerationOptions {
                    threads: Some(threads),
                    configurations_per_range: 1,
                    rows_per_batch: 2,
                    rows_per_segment: 3,
                },
            )
            .unwrap();
            assert!(generated.segments.len() > 1);
            assert!(
                generated
                    .segments
                    .iter()
                    .all(|segment| segment.byte_count > 0)
            );
            let merge = merge_v2_descriptor_segments(&generated, &output).unwrap();
            assert_eq!(merge.record_count, generated.record_count);
            results.push((
                read_rows(&output).unwrap(),
                expected_rows(&request, &generated.peel_subshells).unwrap(),
            ));
        }
        let expected = &results[0].1;
        for (rows, comparison) in &results {
            assert_eq!(rows, comparison);
            assert_eq!(rows, expected);
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn range_planning_is_contiguous_and_rejects_zero_size() {
        assert!(plan_generation_ranges(4, 0).is_err());
        assert_eq!(
            plan_generation_ranges(5, 2).unwrap(),
            [
                GenerationRange {
                    ordinal: 0,
                    start_configuration: 0,
                    end_configuration: 2,
                },
                GenerationRange {
                    ordinal: 1,
                    start_configuration: 2,
                    end_configuration: 4,
                },
                GenerationRange {
                    ordinal: 2,
                    start_configuration: 4,
                    end_configuration: 5,
                },
            ]
        );
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
                configurations_per_range: 1,
                rows_per_batch: 2,
                rows_per_segment: 3,
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
                configurations_per_range: 1,
                rows_per_batch: 2,
                rows_per_segment: 3,
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
            ranges: generated.ranges.clone(),
            segments: generated
                .segments
                .iter()
                .cloned()
                .chain(std::iter::once(duplicate.clone()))
                .collect(),
            unique_occupations: generated.unique_occupations,
            record_count: generated.record_count + duplicate.record_count,
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
        deduplicate_bucket(bucket, 6, &mut bitset, &mut stats).unwrap();
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
                configurations_per_range: 1,
                rows_per_batch: 2,
                rows_per_segment: 3,
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
}
