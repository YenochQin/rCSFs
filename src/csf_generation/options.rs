//! Shared execution options and managed-memory accounting for CSF generation.

use anyhow::{Result, bail, ensure};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const MIB: u64 = 1024 * 1024;

/// Compression used by private Arrow IPC segments. The default remains
/// uncompressed until the P2 benchmark establishes a better default.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SegmentCompression {
    None,
    Lz4,
    Zstd,
}

impl SegmentCompression {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Lz4 => "lz4",
            Self::Zstd => "zstd",
        }
    }

    pub(crate) fn from_environment() -> Result<Self> {
        let Ok(value) = std::env::var("RCSFS_SEGMENT_COMPRESSION") else {
            return Ok(Self::None);
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "none" | "uncompressed" => Ok(Self::None),
            "lz4" => Ok(Self::Lz4),
            "zstd" => Ok(Self::Zstd),
            other => bail!(
                "invalid RCSFS_SEGMENT_COMPRESSION {other:?}; expected none, lz4, or zstd"
            ),
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
    pub(crate) configurations_per_range: usize,
    pub(crate) rows_per_batch: usize,
    pub(crate) rows_per_segment: usize,
    pub(crate) budget: ResourceBudget,
    pub(crate) segment_compression: SegmentCompression,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            threads: None,
            scratch_dir: None,
            configurations_per_range: 4_096,
            rows_per_batch: 8_192,
            rows_per_segment: 131_072,
            budget: ResourceBudget::unlimited(),
            segment_compression: SegmentCompression::None,
        }
    }
}

impl GenerationOptions {
    pub(crate) fn from_api(
        threads: Option<usize>,
        memory_budget_mib: Option<usize>,
        scratch_dir: Option<PathBuf>,
    ) -> Result<Self> {
        ensure!(threads != Some(0), "threads must be greater than 0");
        let budget = ResourceBudget::new(memory_budget_mib)?;
        let segment_compression = SegmentCompression::from_environment()?;
        Ok(Self {
            threads,
            scratch_dir,
            budget,
            segment_compression,
            ..Self::default()
        })
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(self.threads != Some(0), "threads must be greater than 0");
        ensure!(
            self.configurations_per_range > 0,
            "configurations_per_range must be greater than 0"
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
/// caller's documentation.
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
