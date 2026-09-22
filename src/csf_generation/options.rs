//! Shared execution options and managed-memory accounting for CSF generation.

use anyhow::{Result, bail, ensure};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const MIB: u64 = 1024 * 1024;

/// Environment override for the temporary segments' Arrow IPC codec, for the
/// P2a compression experiment. It is intentionally not a public option: the
/// published interface exposes only user-meaningful settings, and compression
/// is a decision experiment until its measurements are in.
const SEGMENT_CODEC_ENV: &str = "RCSFS_SEGMENT_CODEC";

/// The compression applied to the temporary Arrow IPC segments.
///
/// The default stays uncompressed: compression is a decision experiment
/// (P2a), not a settled default, and the capacity pre-flight models segments
/// uncompressed. A selected codec must genuinely reach the writer — a value
/// that Arrow does not support fails the run instead of quietly writing plain
/// segments and reporting itself compressed.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum SegmentCodec {
    None,
    Lz4Frame,
    Zstd,
}

impl SegmentCodec {
    /// Parse the value a caller names the codec by.
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value.trim() {
            "none" => Ok(Self::None),
            "lz4" => Ok(Self::Lz4Frame),
            "zstd" => Ok(Self::Zstd),
            other => bail!(
                "invalid segment codec {other:?}; expected one of: none, lz4, zstd \
                 (lz4 and zstd compress the temporary Arrow IPC segments)"
            ),
        }
    }

    /// The codec this process was asked to use; unset means uncompressed.
    pub(crate) fn from_environment() -> Result<Self> {
        match std::env::var(SEGMENT_CODEC_ENV) {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::None),
            Err(error) => {
                Err(anyhow::Error::new(error).context(format!("cannot read {SEGMENT_CODEC_ENV}")))
            }
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Lz4Frame => "lz4",
            Self::Zstd => "zstd",
        }
    }
}

/// How the disk path makes the published descriptor unique.
///
/// The internal generation path is proven to emit pairwise-distinct rows
/// (`docs/V2_GENERATION_UNIQUENESS.md`), so for it the exact chain — partition
/// every row into root buckets, compare whole rows, keep the first — removes
/// nothing by construction. `VerifiedUnique` skips that round trip; `Exact` is
/// kept, and is the only strategy that can *measure* a duplicate.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum DeduplicationStrategy {
    /// Every record survives, because the generator cannot repeat one.
    ///
    /// No bucket is written and no row is compared: the survivor bitset is
    /// full. Records are still validated by the writer, and the merge still
    /// reads the segments back in publication order.
    VerifiedUnique,
    /// Compare every row against every other row of its symmetry block.
    Exact,
}

/// Environment override for the de-duplication strategy, for verification and
/// benchmark runs. Like the segment codec it is not a public option: the
/// strategy is decided by the construction, not by the caller, and the only
/// thing a caller can ask for is the slower strategy that re-checks the proof.
const DEDUPLICATION_ENV: &str = "RCSFS_DEDUPLICATION";

impl DeduplicationStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value.trim() {
            "verified_unique" => Ok(Self::VerifiedUnique),
            "exact" => Ok(Self::Exact),
            other => bail!(
                "invalid de-duplication strategy {other:?}; expected one of: \
                 verified_unique, exact"
            ),
        }
    }

    pub(crate) fn from_environment() -> Result<Self> {
        match std::env::var(DEDUPLICATION_ENV) {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::VerifiedUnique),
            Err(error) => {
                Err(anyhow::Error::new(error).context(format!("cannot read {DEDUPLICATION_ENV}")))
            }
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::VerifiedUnique => "verified_unique",
            Self::Exact => "exact",
        }
    }
}

/// Execution settings shared by the in-memory and disk generation paths.
///
/// The storage-specific batch constants stay internal defaults.  Callers only
/// provide the thread count, an optional managed-memory budget, and (for the
/// disk path) the operation-owned scratch directory.
#[derive(Clone, Debug)]
pub(crate) struct GenerationOptions {
    pub(crate) threads: Option<usize>,
    pub(crate) scratch_dir: Option<PathBuf>,
    /// Estimated pre-deduplication records one scheduling task may carry.
    /// `None` derives it from the counted total and the thread count.
    pub(crate) records_per_task: Option<u64>,
    /// Whether a volume whose free space cannot be measured is accepted.
    ///
    /// The default is `false`: a platform that cannot report free space would
    /// otherwise silently skip the pre-flight, which is the failure mode the
    /// pre-flight exists to prevent. Accepting it has to be an explicit choice.
    pub(crate) allow_unchecked_space: bool,
    pub(crate) rows_per_batch: usize,
    pub(crate) rows_per_segment: usize,
    /// Codec for the temporary Arrow IPC segments. Set through the
    /// environment for the P2a experiment; the default is uncompressed.
    pub(crate) segment_codec: SegmentCodec,
    /// How the published descriptor is made unique. The default is the
    /// verified path the uniqueness proof licenses; `Exact` re-checks it.
    pub(crate) deduplication: DeduplicationStrategy,
    pub(crate) budget: ResourceBudget,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            threads: None,
            scratch_dir: None,
            records_per_task: None,
            allow_unchecked_space: false,
            rows_per_batch: 8_192,
            rows_per_segment: 131_072,
            segment_codec: SegmentCodec::None,
            deduplication: DeduplicationStrategy::VerifiedUnique,
            budget: ResourceBudget::unlimited(),
        }
    }
}

impl GenerationOptions {
    pub(crate) fn from_api(
        threads: Option<usize>,
        memory_budget_mib: Option<usize>,
        scratch_dir: Option<PathBuf>,
    ) -> Result<Self> {
        Self::from_api_with_space_policy(threads, memory_budget_mib, scratch_dir, false)
    }

    /// Build the options, deciding explicitly what an unmeasurable volume means.
    pub(crate) fn from_api_with_space_policy(
        threads: Option<usize>,
        memory_budget_mib: Option<usize>,
        scratch_dir: Option<PathBuf>,
        allow_unchecked_space: bool,
    ) -> Result<Self> {
        ensure!(threads != Some(0), "threads must be greater than 0");
        let budget = ResourceBudget::new(memory_budget_mib)?;
        Ok(Self {
            threads,
            scratch_dir,
            budget,
            allow_unchecked_space,
            segment_codec: SegmentCodec::from_environment()?,
            deduplication: DeduplicationStrategy::from_environment()?,
            ..Self::default()
        })
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(self.threads != Some(0), "threads must be greater than 0");
        ensure!(
            self.records_per_task != Some(0),
            "records_per_task must be greater than 0"
        );
        ensure!(
            self.rows_per_batch > 0,
            "rows_per_batch must be greater than 0"
        );
        ensure!(
            self.rows_per_segment > 0,
            "rows_per_segment must be greater than 0"
        );
        Ok(())
    }
}

#[derive(Debug)]
struct ResourceState {
    limit_bytes: Option<u64>,
    current_bytes: AtomicU64,
    peak_bytes: AtomicU64,
}

/// A shared byte budget for buffers that are owned by the generation stages.
///
/// This is deliberately an accounting limit, not an RSS limit.  Allocator
/// metadata, thread stacks, Arrow/Parquet runtime buffers and the process
/// itself remain outside the managed total and are reported separately by the
/// caller's documentation.  On the registered B1/B2 inputs the managed peak was
/// 29-88 MiB against a 371-660 MiB process RSS, so the unmanaged remainder is
/// several times the accounted total; that gap is why this value must not be
/// described as a memory cap.
///
/// A reservation that would exceed the limit fails immediately with a resource
/// error. It deliberately does not wait or apply backpressure: a brief
/// concurrent overshoot is a real possibility when several workers reserve in
/// the same instant, and converting it into an unbounded wait would trade a
/// clear failure for a stall whose timing depends on worker completion order.
/// Callers that want headroom must ask for a larger budget.
#[derive(Clone, Debug)]
pub(crate) struct ResourceBudget {
    state: Arc<ResourceState>,
}

impl ResourceBudget {
    pub(crate) fn new(memory_budget_mib: Option<usize>) -> Result<Self> {
        let limit_bytes = memory_budget_mib
            .map(|mib| {
                u64::try_from(mib)
                    .ok()
                    .and_then(|value| value.checked_mul(MIB))
                    .ok_or_else(|| anyhow::anyhow!("memory_budget_mib is out of range"))
            })
            .transpose()?;
        Ok(Self {
            state: Arc::new(ResourceState {
                limit_bytes,
                current_bytes: AtomicU64::new(0),
                peak_bytes: AtomicU64::new(0),
            }),
        })
    }

    pub(crate) fn unlimited() -> Self {
        Self::new(None).expect("an unlimited resource budget is always valid")
    }

    pub(crate) fn try_reserve(&self, bytes: u64, label: &str) -> Result<ResourcePermit> {
        if bytes == 0 {
            return Ok(ResourcePermit {
                budget: self.clone(),
                bytes: 0,
            });
        }
        let mut current = self.state.current_bytes.load(Ordering::Relaxed);
        loop {
            let next = current
                .checked_add(bytes)
                .ok_or_else(|| anyhow::anyhow!("managed memory accounting overflow"))?;
            if let Some(limit) = self.state.limit_bytes {
                if next > limit {
                    bail!(
                        "memory budget exceeded while reserving {label}: requested {bytes} bytes, managed total would be {next} bytes, budget is {limit} bytes"
                    );
                }
            }
            match self.state.current_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.state.peak_bytes.fetch_max(next, Ordering::Relaxed);
                    return Ok(ResourcePermit {
                        budget: self.clone(),
                        bytes,
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }

    pub(crate) fn snapshot(&self, occupation_bytes: u64) -> ResourceStats {
        ResourceStats {
            memory_budget_mib: self
                .state
                .limit_bytes
                .and_then(|bytes| usize::try_from(bytes / MIB).ok()),
            budget_bytes: self.state.limit_bytes,
            peak_managed_bytes: self.state.peak_bytes.load(Ordering::Relaxed),
            current_managed_bytes: self.state.current_bytes.load(Ordering::Relaxed),
            occupation_bytes,
        }
    }
}

/// A scoped allocation reservation.  Cloning the budget does not duplicate
/// the reservation; all clones update the same counters.
#[derive(Debug)]
pub(crate) struct ResourcePermit {
    budget: ResourceBudget,
    bytes: u64,
}

impl ResourcePermit {
    pub(crate) fn resize(&mut self, bytes: u64, label: &str) -> Result<()> {
        if bytes > self.bytes {
            let extra = bytes - self.bytes;
            let extra_permit = self.budget.try_reserve(extra, label)?;
            self.bytes = bytes;
            // The temporary permit would release the extra bytes on drop.  It
            // is intentionally forgotten after the owner's byte count grows.
            std::mem::forget(extra_permit);
        } else if bytes < self.bytes {
            self.budget.release(self.bytes - bytes);
            self.bytes = bytes;
        }
        Ok(())
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for ResourcePermit {
    fn drop(&mut self) {
        self.budget.release(self.bytes);
    }
}

impl ResourceBudget {
    fn release(&self, bytes: u64) {
        if bytes != 0 {
            let previous = self.state.current_bytes.fetch_sub(bytes, Ordering::AcqRel);
            debug_assert!(previous >= bytes, "resource permit accounting underflow");
        }
    }
}

/// Stable resource information returned with a generation result.
#[derive(Clone, Debug)]
pub(crate) struct ResourceStats {
    pub(crate) memory_budget_mib: Option<usize>,
    pub(crate) budget_bytes: Option<u64>,
    pub(crate) peak_managed_bytes: u64,
    pub(crate) current_managed_bytes: u64,
    pub(crate) occupation_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_codec_names_round_trip_and_reject_the_rest() {
        for codec in [
            SegmentCodec::None,
            SegmentCodec::Lz4Frame,
            SegmentCodec::Zstd,
        ] {
            assert_eq!(SegmentCodec::parse(codec.name()).unwrap(), codec);
        }
        // Surrounding whitespace is tolerated because the value arrives from
        // the environment; anything else has to be one of the three names.
        assert_eq!(
            SegmentCodec::parse(" lz4 ").unwrap(),
            SegmentCodec::Lz4Frame
        );
        for invalid in ["", "gzip", "NONE", "lz4_frame", "zstd-3", "lz4,zstd"] {
            let error = SegmentCodec::parse(invalid).unwrap_err().to_string();
            assert!(
                error.contains("invalid segment codec"),
                "{invalid:?} was accepted: {error}"
            );
            assert!(
                error.contains("none, lz4, zstd"),
                "the error must list what is valid: {error}"
            );
        }
    }

    /// The default is the strategy the proof licenses, and the only other
    /// accepted value makes a run *re-check* the proof rather than skip a check.
    #[test]
    fn deduplication_names_round_trip_and_default_to_the_verified_path() {
        for strategy in [
            DeduplicationStrategy::VerifiedUnique,
            DeduplicationStrategy::Exact,
        ] {
            assert_eq!(
                DeduplicationStrategy::parse(strategy.name()).unwrap(),
                strategy
            );
        }
        assert_eq!(
            GenerationOptions::default().deduplication,
            DeduplicationStrategy::VerifiedUnique
        );
        for invalid in ["", "none", "skip", "trusted", "VERIFIED_UNIQUE"] {
            let error = DeduplicationStrategy::parse(invalid)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("invalid de-duplication strategy") && error.contains("exact"),
                "{invalid:?} was accepted, or its error does not name the valid values: {error}"
            );
        }
    }
}
