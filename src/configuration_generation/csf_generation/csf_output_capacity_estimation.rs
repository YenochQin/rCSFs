//! Pre-flight capacity model for the V2 disk generation path.
//!
//! The disk pipeline writes several times the size of its final output before it
//! publishes anything: uncompressed Arrow segments, de-duplication buckets, the
//! survivor bitsets and the published artifacts all coexist. A run that cannot
//! finish should fail before the first large segment is written, not hours later
//! when a filesystem fills up.
//!
//! Scratch framing uses the registered B1/B2 measurements. Published bytes
//! depend on compression and include per-column schema and row-group overhead.

use anyhow::{Context, Result};

use super::DeduplicationStrategy;

/// Extra bytes per row for the Arrow IPC framing of a segment.
///
/// The earlier Int32 measurements were 3.3%; retain 10% as an explicit
/// assumption for Int8 until large-scale measurements are repeated.
const SEGMENT_OVERHEAD_NUMERATOR: u64 = 11;
const SEGMENT_OVERHEAD_DENOMINATOR: u64 = 10;

/// Bytes added to a root-bucket row beyond the V2 payload: a 16-byte digest and
/// an 8-byte stable ordinal, plus room for bucket framing.
const BUCKET_ROW_OVERHEAD_BYTES: u64 = 24;
const BUCKET_ROW_FRAMING_BYTES: u64 = 32;

/// Additional full repartitions assumed for oversized leaf buckets.
///
/// One level is assumed because the mechanism exists but was never triggered by
/// the registered inputs, and an unmeasured quantity may not be assumed to be
/// zero.
const RECURSIVE_REPARTITION_LEVELS: u64 = 1;

// With 8192-row groups and Zstd, the registered e1/o1 Int8 outputs used
// 0.044 and 0.043 bytes/cell including metadata. Also allow for short/wide
// files, where column headers and Parquet footers dominate the payload.
const DESCRIPTOR_HUNDREDTHS_PER_CELL: u64 = 5;
const BYTES_PER_COLUMN_CHUNK: u64 = 256;
const SCHEMA_BYTES_PER_COLUMN: u64 = 512;

/// Allowance for the companion header TOML.
const METADATA_BYTES: u64 = 64 * 1024;

/// Safety margin applied to every requirement.
const MARGIN_NUMERATOR: u64 = 5;
const MARGIN_DENOMINATOR: u64 = 4;

/// Estimated bytes for each path of one disk generation run.
#[derive(Clone, Debug)]
pub(crate) struct CapacityEstimate {
    pub(crate) peel_subshells: usize,
    pub(crate) v2_columns: u64,
    pub(crate) pre_deduplication_records: u64,
    pub(crate) segment_bytes: u64,
    pub(crate) root_bucket_bytes: u64,
    pub(crate) recursive_bucket_bytes: u64,
    pub(crate) survivor_bitset_bytes: u64,
    /// Scratch peak: segments, root buckets, one recursive repartition and the
    /// survivor bitsets.
    pub(crate) scratch_peak_bytes: u64,
    pub(crate) descriptor_bytes: u64,
    /// The staged output set the CLI publishes from.
    pub(crate) staged_output_bytes: u64,
    /// Scratch peak plus the safety margin, as required of the scratch volume.
    pub(crate) required_scratch_bytes: u64,
    /// Staged output set plus the safety margin, as required of each volume
    /// that receives a published artifact.
    pub(crate) required_output_bytes: u64,
    pub(crate) assumptions: Vec<String>,
}

/// One published artifact of a run.
///
/// A caller says *where* each artifact goes and the model charges that volume
/// the size the artifact actually takes, so the caller never repeats the
/// estimate's own numbers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ArtifactKind {
    Descriptor,
    /// The CSF header TOML. It is its own destination because it can be
    /// published to a different directory than the descriptor Parquet.
    Header,
}

impl ArtifactKind {
    /// The name a caller uses for this artifact.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Descriptor => "descriptor",
            Self::Header => "header",
        }
    }

    #[cfg(feature = "python")]
    pub(crate) fn from_name(name: &str) -> Result<Self> {
        match name {
            "descriptor" => Ok(Self::Descriptor),
            "header" => Ok(Self::Header),
            other => {
                anyhow::bail!("unknown published artifact {other:?}; expected descriptor or header")
            }
        }
    }
}

impl CapacityEstimate {
    /// One line per assumption, for reports that must show their derivation.
    pub(crate) fn assumption_lines(&self) -> &[String] {
        &self.assumptions
    }

    /// Published size of one artifact kind.
    ///
    pub(crate) fn artifact_bytes(&self, kind: ArtifactKind) -> u64 {
        match kind {
            ArtifactKind::Descriptor => self.descriptor_bytes,
            ArtifactKind::Header => METADATA_BYTES,
        }
    }
}

fn scaled(value: u64, numerator: u64, denominator: u64) -> Result<u64> {
    value
        .checked_mul(numerator)
        .map(|scaled| scaled / denominator)
        .context("capacity estimate overflow")
}

pub(crate) fn with_margin(value: u64) -> Result<u64> {
    value
        .checked_add(scaled(
            value,
            MARGIN_NUMERATOR - MARGIN_DENOMINATOR,
            MARGIN_DENOMINATOR,
        )?)
        .context("capacity estimate overflow")
}

/// Estimate the bytes one run needs, from the counted workload and the layout.
///
/// `blocks` is the number of `(2J, parity)` symmetry blocks, used only to size
/// the survivor bitsets.
///
/// `strategy` decides how much of the scratch the model may claim at all: the
/// verified path writes no root or recursive bucket, so pricing those buckets
/// for it would reserve several times the space the run uses. The model must
/// describe the run it is asked about — an upper bound on a run that will not
/// happen is not conservative, it is wrong in the other direction, because it
/// can refuse a job that fits.
pub(crate) fn estimate_capacity(
    peel_subshells: usize,
    pre_deduplication_records: u64,
    blocks: u64,
    strategy: DeduplicationStrategy,
) -> Result<CapacityEstimate> {
    let peel = u64::try_from(peel_subshells).context("peel subshell count exceeds u64")?;
    let v2_columns = peel
        .checked_mul(4)
        .and_then(|columns| columns.checked_add(2))
        .context("V2 column count overflow")?;
    // Each descriptor value occupies exactly one byte in Arrow/scratch.
    let row_bytes = v2_columns;
    let segment_row_bytes = scaled(
        row_bytes
            .checked_add(12)
            .context("segment row width overflow")?,
        SEGMENT_OVERHEAD_NUMERATOR,
        SEGMENT_OVERHEAD_DENOMINATOR,
    )?;
    let bucket_row_bytes = row_bytes
        .checked_add(BUCKET_ROW_OVERHEAD_BYTES)
        .and_then(|bytes| bytes.checked_add(BUCKET_ROW_FRAMING_BYTES))
        .context("bucket row width overflow")?;

    let segment_bytes = pre_deduplication_records
        .checked_mul(segment_row_bytes)
        .context("segment size estimate overflow")?;
    let (root_bucket_bytes, recursive_bucket_bytes) = match strategy {
        DeduplicationStrategy::Exact => {
            let root = pre_deduplication_records
                .checked_mul(bucket_row_bytes)
                .context("root bucket size estimate overflow")?;
            let recursive = root
                .checked_mul(RECURSIVE_REPARTITION_LEVELS)
                .context("recursive bucket size estimate overflow")?;
            (root, recursive)
        }
        // The verified path writes no bucket at all, so there is nothing to
        // allow for — not even a recursion level.
        DeduplicationStrategy::VerifiedUnique => (0, 0),
    };
    // One bit per surviving ordinal, plus a small header per symmetry block.
    let survivor_bitset_bytes = pre_deduplication_records
        .checked_div(8)
        .and_then(|bytes| bytes.checked_add(blocks.checked_mul(64)?))
        .context("survivor bitset size estimate overflow")?;
    let scratch_peak_bytes = segment_bytes
        .checked_add(root_bucket_bytes)
        .and_then(|bytes| bytes.checked_add(recursive_bucket_bytes))
        .and_then(|bytes| bytes.checked_add(survivor_bitset_bytes))
        .context("scratch peak estimate overflow")?;

    let row_groups = pre_deduplication_records.div_ceil(crate::csf_output::ROW_GROUP_ROWS as u64);
    let descriptor_bytes = pre_deduplication_records
        .checked_mul(DESCRIPTOR_HUNDREDTHS_PER_CELL)
        .map(|hundredths| hundredths.div_ceil(100))
        .and_then(|bytes| bytes.checked_add(row_groups.checked_mul(BYTES_PER_COLUMN_CHUNK)?))
        .and_then(|bytes| bytes.checked_add(SCHEMA_BYTES_PER_COLUMN))
        .and_then(|bytes| bytes.checked_mul(v2_columns))
        .and_then(|bytes| bytes.checked_add(4096))
        .context("descriptor size estimate overflow")?;
    let staged_output_bytes = descriptor_bytes
        .checked_add(METADATA_BYTES)
        .context("staged output size estimate overflow")?;

    let assumptions = vec![
        "record counts are pre-de-duplication: no record is assumed to be removed \
         (B1/B2 removed none, and a removing input only lowers the published sizes)"
            .to_owned(),
        match strategy {
            DeduplicationStrategy::VerifiedUnique => {
                "de-duplication is the verified path: the generator cannot repeat a row, so \
                 no root or recursive bucket is written and the scratch model covers segments \
                 and survivor bitsets only"
                    .to_owned()
            }
            DeduplicationStrategy::Exact => {
                "de-duplication is the exact path: the model covers one full root-bucket pass \
                 and one recursive repartition of an oversized leaf"
                    .to_owned()
            }
        },
        format!(
            "segments {segment_row_bytes} bytes/row = 1.1x the {row_bytes}-byte V2 row plus \
             two ordinals; 10% overhead is retained from the earlier Int32 measurements \
             and has not been rebenchmarked for Int8; segments are modeled uncompressed"
        ),
        format!(
            "root buckets {bucket_row_bytes} bytes/row = V2 row + 16-byte digest + 8-byte \
             ordinal + framing; measured exactly V2 row + 24"
        ),
        format!(
            "recursive bucketing assumes {RECURSIVE_REPARTITION_LEVELS} additional full \
             repartition; it was never triggered by the registered inputs"
        ),
        format!(
            "Int8 descriptor estimate: {DESCRIPTOR_HUNDREDTHS_PER_CELL}/100 bytes per cell, \
             {BYTES_PER_COLUMN_CHUNK} bytes per column per row group, \
             {SCHEMA_BYTES_PER_COLUMN} bytes per schema column and 4096 fixed bytes; \
             compressed sizes vary with the CSF data"
        ),
        "publishing copies the staged set to its destinations, so the staged set and the \
         published files exist at the same time"
            .to_owned(),
        format!("safety margin {MARGIN_NUMERATOR}/{MARGIN_DENOMINATOR} on every requirement"),
        "free space is checked with statvfs where the platform provides it; elsewhere it is \
         reported as unchecked rather than assumed sufficient"
            .to_owned(),
    ];

    Ok(CapacityEstimate {
        peel_subshells,
        v2_columns,
        pre_deduplication_records,
        segment_bytes,
        root_bucket_bytes,
        recursive_bucket_bytes,
        survivor_bitset_bytes,
        scratch_peak_bytes,
        descriptor_bytes,
        staged_output_bytes,
        required_scratch_bytes: with_margin(scratch_peak_bytes)?,
        required_output_bytes: with_margin(staged_output_bytes)?,
        assumptions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wider_layout_never_estimates_smaller_sizes() {
        let narrow = estimate_capacity(24, 1_000_000, 2, DeduplicationStrategy::Exact).unwrap();
        let wide = estimate_capacity(56, 1_000_000, 2, DeduplicationStrategy::Exact).unwrap();
        assert!(wide.segment_bytes > narrow.segment_bytes);
        assert!(wide.root_bucket_bytes > narrow.root_bucket_bytes);
        assert!(wide.descriptor_bytes > narrow.descriptor_bytes);
        assert!(wide.staged_output_bytes > narrow.staged_output_bytes);
    }

    #[test]
    fn int8_scratch_covers_raw_payload_and_keeps_output_assumptions() {
        // B1 shape: 98 Int8 columns, plus u32/u64 IPC ordinals or a
        // u64 ordinal and 16-byte bucket hash. Old Int32 scratch sizes no
        // longer describe these payloads. Published-output assumptions below
        // are retained from the historical measurements, not new benchmarks.
        let b1 = estimate_capacity(24, 2_695_762, 1, DeduplicationStrategy::Exact).unwrap();
        assert!(b1.segment_bytes >= 2_695_762 * (98 + 12));
        assert!(b1.root_bucket_bytes >= 2_695_762 * (98 + 24));
        assert!(b1.scratch_peak_bytes >= b1.segment_bytes + b1.root_bucket_bytes);
        assert!(b1.descriptor_bytes >= 2_594_699);
        // B2 shape: 226 Int8 columns over the historical record count.
        let b2 = estimate_capacity(56, 560_351, 1, DeduplicationStrategy::Exact).unwrap();
        assert!(b2.segment_bytes >= 560_351 * (226 + 12));
        assert!(b2.root_bucket_bytes >= 560_351 * (226 + 24));
        assert!(b2.scratch_peak_bytes >= b2.segment_bytes + b2.root_bucket_bytes);
        assert!(b2.descriptor_bytes >= 781_637);
    }

    /// A volume that is too small stops a run and is only reported by an
    /// estimate: the same numbers, two different questions.
    #[test]
    fn counting_overflow_is_an_error_not_an_estimate() {
        assert!(estimate_capacity(56, u64::MAX, 1, DeduplicationStrategy::Exact).is_err());
        assert!(estimate_capacity(usize::MAX, 1, 1, DeduplicationStrategy::Exact).is_err());
    }
}
