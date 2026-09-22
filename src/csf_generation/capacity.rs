//! Pre-flight capacity model for the V2 disk generation path.
//!
//! The disk pipeline writes several times the size of its final output before it
//! publishes anything: uncompressed Arrow segments, de-duplication buckets, the
//! survivor bitsets and the published artifacts all coexist. A run that cannot
//! finish should fail before the first large segment is written, not hours later
//! when a filesystem fills up.
//!
//! Every ratio below was measured on the registered `B1`/`B2` transcripts and is
//! recorded in `docs/benchmarks`. The model takes the *larger* end of each
//! measured range, and states every unmeasured quantity as an explicit
//! assumption rather than assuming zero, so the result is an upper bound with a
//! visible derivation instead of a point prediction.

use anyhow::{Context, Result, bail};

/// Extra bytes per row for the Arrow IPC framing of a segment.
///
/// Measured at 3.3% over the raw `4M + 2` integers plus the two storage
/// ordinals; rounded up to 10%.
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

/// Descriptor Parquet bytes per V2 integer per row, in hundredths of a byte.
///
/// Measured 0.98 hundredths (61 columns) and 0.62 hundredths (226 columns), so
/// one hundredth of a byte rounds both up while staying close to the measured
/// values.
const DESCRIPTOR_HUNDREDTHS_PER_INT_ROW: u64 = 1;

/// Published CSF text bytes per record.
///
/// Measured 320.05 (61 columns) and 300.77 (226 columns) bytes per record, so
/// the constant rounds the larger sample up to the next whole byte.
const CSF_TEXT_BYTES_PER_RECORD: u64 = 321;

/// Published CSF Parquet bytes per record.
///
/// Measured 27.51 (61 columns) and 72.60 (226 columns), so the constant rounds
/// the larger sample up to the next whole byte.
const CSF_PARQUET_BYTES_PER_RECORD: u64 = 73;

/// Allowance for one metadata file: the CSF header or the descriptor sidecar.
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
    pub(crate) csf_text_bytes: u64,
    pub(crate) csf_parquet_bytes: u64,
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
    CsfText,
    CsfParquet,
    Descriptor,
    /// The CSF header TOML. It is its own destination because it can be
    /// published to a different directory than the descriptor sidecar.
    Header,
    /// The descriptor sidecar TOML.
    DescriptorMetadata,
}

impl ArtifactKind {
    /// The name a caller uses for this artifact.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::CsfText => "csf_text",
            Self::CsfParquet => "csf_parquet",
            Self::Descriptor => "descriptor",
            Self::Header => "header",
            Self::DescriptorMetadata => "descriptor_metadata",
        }
    }

    pub(crate) fn from_name(name: &str) -> Result<Self> {
        match name {
            "csf_text" => Ok(Self::CsfText),
            "csf_parquet" => Ok(Self::CsfParquet),
            "descriptor" => Ok(Self::Descriptor),
            "header" => Ok(Self::Header),
            "descriptor_metadata" => Ok(Self::DescriptorMetadata),
            other => bail!(
                "unknown published artifact {other:?}; expected csf_text, csf_parquet, \
                 descriptor, header, or descriptor_metadata"
            ),
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
    /// The two metadata files get the whole allowance each. They are a few
    /// kilobytes against artifacts measured in gigabytes, so charging the
    /// allowance twice costs nothing and keeps the answer independent of which
    /// volume each one lands on.
    pub(crate) fn artifact_bytes(&self, kind: ArtifactKind) -> u64 {
        match kind {
            ArtifactKind::CsfText => self.csf_text_bytes,
            ArtifactKind::CsfParquet => self.csf_parquet_bytes,
            ArtifactKind::Descriptor => self.descriptor_bytes,
            ArtifactKind::Header | ArtifactKind::DescriptorMetadata => METADATA_BYTES,
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
pub(crate) fn estimate_capacity(
    peel_subshells: usize,
    pre_deduplication_records: u64,
    blocks: u64,
) -> Result<CapacityEstimate> {
    let peel = u64::try_from(peel_subshells).context("peel subshell count exceeds u64")?;
    let v2_columns = peel
        .checked_mul(4)
        .and_then(|columns| columns.checked_add(2))
        .context("V2 column count overflow")?;
    let row_bytes = v2_columns.checked_mul(4).context("V2 row width overflow")?;
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
    let root_bucket_bytes = pre_deduplication_records
        .checked_mul(bucket_row_bytes)
        .context("root bucket size estimate overflow")?;
    let recursive_bucket_bytes = root_bucket_bytes
        .checked_mul(RECURSIVE_REPARTITION_LEVELS)
        .context("recursive bucket size estimate overflow")?;
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

    let descriptor_bytes = pre_deduplication_records
        .checked_mul(v2_columns)
        .and_then(|int_rows| int_rows.checked_mul(DESCRIPTOR_HUNDREDTHS_PER_INT_ROW))
        .and_then(|hundredths| hundredths.checked_div(100))
        .context("descriptor size estimate overflow")?;
    let csf_text_bytes = pre_deduplication_records
        .checked_mul(CSF_TEXT_BYTES_PER_RECORD)
        .context("CSF text size estimate overflow")?;
    let csf_parquet_bytes = pre_deduplication_records
        .checked_mul(CSF_PARQUET_BYTES_PER_RECORD)
        .context("CSF Parquet size estimate overflow")?;
    let staged_output_bytes = descriptor_bytes
        .checked_add(csf_text_bytes)
        .and_then(|bytes| bytes.checked_add(csf_parquet_bytes))
        .and_then(|bytes| bytes.checked_add(METADATA_BYTES.checked_mul(2)?))
        .context("staged output size estimate overflow")?;

    let assumptions = vec![
        format!(
            "record counts are pre-de-duplication: no record is assumed to be removed \
             (B1/B2 removed none, and a removing input only lowers the published sizes)"
        ),
        format!(
            "segments {segment_row_bytes} bytes/row = 1.1x the {row_bytes}-byte V2 row plus \
             two ordinals; measured overhead was 3.3%"
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
            "descriptor {DESCRIPTOR_HUNDREDTHS_PER_INT_ROW}/100 bytes per V2 integer per row; \
             measured 0.0098 (61 columns) and 0.0062 (226 columns) bytes, so every shape is \
             rounded up"
        ),
        format!(
            "CSF text {CSF_TEXT_BYTES_PER_RECORD} bytes/record and CSF Parquet \
             {CSF_PARQUET_BYTES_PER_RECORD} bytes/record; measured 320.05/27.51 and 300.77/72.60, \
             rounded up"
        ),
        format!(
            "publishing copies the staged set to its destinations, so the staged set and the \
             published files exist at the same time"
        ),
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
        csf_text_bytes,
        csf_parquet_bytes,
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
        let narrow = estimate_capacity(24, 1_000_000, 2).unwrap();
        let wide = estimate_capacity(56, 1_000_000, 2).unwrap();
        assert!(wide.segment_bytes > narrow.segment_bytes);
        assert!(wide.root_bucket_bytes > narrow.root_bucket_bytes);
        assert!(wide.descriptor_bytes > narrow.descriptor_bytes);
        // The published CSF text does not depend on the Peel width in the model,
        // because the two measured samples did not increase with it.
        assert_eq!(wide.csf_text_bytes, narrow.csf_text_bytes);
    }

    #[test]
    fn the_estimate_matches_the_measured_registered_runs() {
        // B1: 2,695,762 records over a 24-subshell Peel table. Measured
        // 1,124,709,846 segment bytes, 1,121,773,963 bucket bytes and a
        // 2,232,842,742-byte scratch peak.
        let b1 = estimate_capacity(24, 2_695_762, 1).unwrap();
        assert!(b1.segment_bytes >= 1_124_709_846);
        assert!(b1.root_bucket_bytes >= 1_121_773_963);
        assert!(b1.scratch_peak_bytes >= 2_232_842_742);
        assert!(b1.descriptor_bytes >= 2_594_699);
        assert!(b1.csf_text_bytes >= 862_789_499);
        assert!(b1.csf_parquet_bytes >= 74_150_973);
        // B2: 560,351 records over 56 subshells. Measured 530,280,358 segment
        // bytes, 520,075,772 bucket bytes, 1,046,641,830-byte scratch peak,
        // 781,637 descriptor bytes, 168,534,272 text bytes and 40,680,076
        // Parquet bytes.
        let b2 = estimate_capacity(56, 560_351, 1).unwrap();
        assert!(b2.segment_bytes >= 530_280_358);
        assert!(b2.root_bucket_bytes >= 520_075_772);
        assert!(b2.scratch_peak_bytes >= 1_046_641_830);
        assert!(b2.descriptor_bytes >= 781_637);
        assert!(b2.csf_text_bytes >= 168_534_272);
        assert!(b2.csf_parquet_bytes >= 40_680_076);
    }

    /// A volume that is too small stops a run and is only reported by an
    /// estimate: the same numbers, two different questions.
    #[test]
    fn counting_overflow_is_an_error_not_an_estimate() {
        assert!(estimate_capacity(56, u64::MAX, 1).is_err());
        assert!(estimate_capacity(usize::MAX, 1, 1).is_err());
    }
}
