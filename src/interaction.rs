//! Conservative CSF interaction selection.
//!
//! This module deliberately implements only a structural upper bound on the
//! `rcsfinteract`/`ICHKQ2` occupation test.  It does not evaluate angular or
//! radial matrix elements, so a retained CSF is merely *possibly* interacting.
//! The [`InteractionStats::exact`] flag is consequently always `false`.

use crate::atomic_output::{
    ensure_distinct_inputs, ensure_new_output, ensure_output_does_not_alias_input,
    publish_staged_outputs,
};
use crate::complete_csf::Parity;
use crate::descriptor_dataset::{DatasetWriter, DescriptorRows, header_path};
use crate::descriptor_operations::remap;
use crate::descriptor_v3::parity_from_sign;
use anyhow::{Context, Result, ensure};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::path::Path;
use std::str::FromStr;

/// Hamiltonian requested by the caller.
///
/// Both modes currently use the same conservative occupation bound.  The mode
/// is retained in the result so later exact implementations cannot silently
/// lose the caller's physical-model choice.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HamiltonianMode {
    /// Request the Dirac–Coulomb Hamiltonian.
    DiracCoulomb,
    /// Request the Dirac–Coulomb Hamiltonian with the Breit interaction.
    DiracCoulombBreit,
}

impl HamiltonianMode {
    /// Canonical option string accepted by [`Self::parse`].
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DiracCoulomb => "dirac_coulomb",
            Self::DiracCoulombBreit => "dirac_coulomb_breit",
        }
    }

    /// Decode the public option spelling accepted at API boundaries.
    pub fn parse(value: &str) -> Result<Self> {
        value.parse()
    }
}

impl FromStr for HamiltonianMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "dirac_coulomb" => Ok(Self::DiracCoulomb),
            "dirac_coulomb_breit" => Ok(Self::DiracCoulombBreit),
            other => anyhow::bail!(
                "unknown hamiltonian {other:?}; expected \"dirac_coulomb\" or \"dirac_coulomb_breit\""
            ),
        }
    }
}

impl fmt::Display for HamiltonianMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Interaction-selection algorithm.
///
/// The explicit variant prevents this approximation from being mistaken for
/// a future exact `rcsfinteract` implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum InteractionMethod {
    /// Use occupation differences to bound which CSFs may interact.
    StructuralUpperBound,
}

impl InteractionMethod {
    /// Canonical option string accepted by [`Self::parse`].
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StructuralUpperBound => "structural_upper_bound",
        }
    }

    /// Decode the public option spelling accepted at API boundaries.
    pub fn parse(value: &str) -> Result<Self> {
        value.parse()
    }
}

impl FromStr for InteractionMethod {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "structural_upper_bound" => Ok(Self::StructuralUpperBound),
            other => anyhow::bail!(
                "unknown method {other:?}; only \"structural_upper_bound\" is supported"
            ),
        }
    }
}

impl fmt::Display for InteractionMethod {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Per-symmetry-block selection counts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InteractionBlockStats {
    /// Zero-based block index in both input files.
    pub block_index: usize,
    /// Twice the total angular momentum of all rows in the block.
    pub total_two_j: u16,
    /// Parity of all rows in the block.
    pub parity: Parity,
    /// Reference input rows, all written at the start of the output block.
    pub reference_count: usize,
    /// Candidate input rows before selection.
    pub candidate_count: usize,
    /// Candidate rows omitted because an identical reference CSF is already present.
    pub exact_reference_skipped: usize,
    /// Candidate rows appended after passing the occupation bound.
    pub selected_count: usize,
    /// Candidate rows excluded by the electron-count or occupation checks.
    pub rejected_count: usize,
    /// Output rows, the sum of reference and selected candidate rows in this block.
    pub output_count: usize,
}

/// Counts and provenance for one interaction-selection run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InteractionStats {
    /// `false`: structural selection is a conservative upper bound, not an
    /// exact Hamiltonian matrix-element calculation.
    pub exact: bool,
    /// Hamiltonian requested by the caller and recorded for provenance.
    pub mode: HamiltonianMode,
    /// Selection method used to decide which candidate rows to append.
    pub method: InteractionMethod,
    /// Number of aligned symmetry blocks processed.
    pub block_count: usize,
    /// Total reference rows written across the blocks.
    pub reference_count: usize,
    /// Total candidate input rows across the blocks.
    pub candidate_count: usize,
    /// Total candidates omitted because they matched a reference CSF exactly.
    pub exact_reference_skipped: usize,
    /// Total candidates retained in addition to the reference rows.
    pub selected_count: usize,
    /// Total candidates that failed the electron-count or occupation checks.
    pub rejected_count: usize,
    /// Total output rows, including the reference rows.
    pub output_count: usize,
    /// Published Parquet file size in bytes, excluding the TOML sidecar.
    pub output_bytes: u64,
    /// Per-block counts in the explicit input and output block order.
    pub blocks: Vec<InteractionBlockStats>,
}

const CANDIDATE_BATCH_ROWS: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Decision {
    ExactReference,
    Selected,
    Rejected,
}

/// Select candidate CSFs that may interact with at least one reference CSF.
///
/// This is a conservative, non-exact implementation of the occupation-level
/// `ICHKQ2` bound: within an aligned J/P block, a candidate is retained when
/// it has the same peel-electron count as a reference, differs in at most four
/// subshell occupations, every occupation difference has magnitude at most
/// two, and the sum of absolute occupation differences is at most four.
/// Passing this test does not imply a non-zero Hamiltonian matrix element.
///
/// Input constraints:
///
/// * core header lines must be byte-identical;
/// * the reference peel list must be a prefix of the candidate peel list;
/// * block counts and each corresponding block's J/P must match.
///
/// Each output block contains all reference CSFs in their original order, then
/// retained candidates in candidate order.  Every candidate exactly equal to
/// any reference CSF is skipped; other duplicate candidates are deliberately
/// preserved.  The candidate header is used so appended peel subshells remain
/// declared. Live memory scales with one reference block and bounded
/// candidate and I/O batches.
///
/// The descriptor/header pair is staged before publication. With
/// `overwrite == false`, publication refuses an existing destination.
pub fn select_interacting_csfs(
    reference_path: &Path,
    candidates_path: &Path,
    output_path: &Path,
    mode: HamiltonianMode,
    method: InteractionMethod,
    num_workers: Option<usize>,
    overwrite: bool,
) -> Result<InteractionStats> {
    ensure!(
        num_workers != Some(0),
        "num_workers must be greater than zero"
    );
    let reference_header = header_path(reference_path)?;
    let candidate_header = header_path(candidates_path)?;
    let output_header = header_path(output_path)?;
    if !overwrite {
        ensure_new_output(output_path)?;
        ensure_new_output(&output_header)?;
    }
    ensure_distinct_inputs(
        reference_path,
        candidates_path,
        "reference and candidate inputs must be different files",
    )?;
    for (input, role) in [
        (reference_path, "reference"),
        (candidates_path, "candidate"),
        (&reference_header, "reference header"),
        (&candidate_header, "candidate header"),
    ] {
        ensure_output_does_not_alias_input(output_path, input, role)?;
        ensure_output_does_not_alias_input(&output_header, input, role)?;
    }

    let mut reference = DescriptorRows::open(reference_path, &reference_header)?;
    let mut candidates = DescriptorRows::open(candidates_path, &candidate_header)?;
    ensure!(
        reference.header().lines[1] == candidates.header().lines[1],
        "core header mismatch: reference={:?}, candidates={:?}",
        reference.header().lines[1],
        candidates.header().lines[1]
    );
    ensure!(
        candidates
            .header()
            .peel
            .starts_with(&reference.header().peel),
        "reference peel subshells must be a prefix of candidate peel subshells"
    );
    let block_count = reference.header().block_lengths.len();
    ensure!(
        block_count == candidates.header().block_lengths.len(),
        "block count mismatch: reference={} candidates={}",
        block_count,
        candidates.header().block_lengths.len()
    );

    let indices = (0..reference.header().peel.len()).collect::<Vec<_>>();
    let width = candidates.header().layout().row_len();
    let mut header = candidates.header().clone();
    header.add_truncations_from(reference.header())?;
    header.electron_count = reference.header().electron_count;

    let parent = output_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let stage = tempfile::Builder::new()
        .prefix(".rcsfs-interacting-")
        .tempdir_in(parent)?;
    let staged = stage
        .path()
        .join(output_path.file_name().context("missing output filename")?);
    let staged_header = header_path(&staged)?;
    let mut writer = DatasetWriter::try_new(header, &staged, &staged_header, &[])?;
    let worker_pool = num_workers
        .map(|workers| ThreadPoolBuilder::new().num_threads(workers).build())
        .transpose()
        .context("failed to build interaction worker pool")?;
    let mut block_stats = Vec::with_capacity(block_count);

    for block_index in 0..block_count {
        let reference_count = reference.header().block_lengths[block_index];
        let candidate_count = candidates.header().block_lengths[block_index];
        let mut exact_references = HashSet::new();
        for row in reference.read_rows(reference_count) {
            let row = remap(&row?, &indices, width);
            writer.push(&row, block_index)?;
            ensure!(
                exact_references.insert(row),
                "reference block {block_index} contains duplicate CSFs"
            );
        }

        let (two_j, parity) = reference
            .block_symmetry()
            .context("empty reference block")?;
        let mut block = InteractionBlockStats {
            block_index,
            total_two_j: u16::try_from(two_j)?,
            parity: parity_from_sign(parity)?,
            reference_count,
            candidate_count,
            exact_reference_skipped: 0,
            selected_count: 0,
            rejected_count: 0,
            output_count: reference_count,
        };
        for batch_start in (0..candidate_count).step_by(CANDIDATE_BATCH_ROWS) {
            let count = (candidate_count - batch_start).min(CANDIDATE_BATCH_ROWS);
            let batch = candidates.read_rows(count).collect::<Result<Vec<_>>>()?;
            ensure!(
                candidates.block_symmetry() == Some((two_j, parity)),
                "J/P mismatch in block {block_index}: reference={:?} candidates={:?}",
                (two_j, parity),
                candidates.block_symmetry()
            );

            // The readers validate a constant electron count per dataset.
            if reference.header().electron_count != candidates.header().electron_count {
                block.rejected_count += count;
                continue;
            }

            let evaluate = || {
                batch
                    .par_iter()
                    .map(|row| decide_candidate(row, &exact_references))
                    .collect::<Vec<_>>()
            };
            let decisions = match &worker_pool {
                Some(pool) => pool.install(evaluate),
                None => evaluate(),
            };
            // Indexed parallel iteration preserves the input order.
            for (row, decision) in batch.iter().zip(decisions) {
                match decision {
                    Decision::ExactReference => block.exact_reference_skipped += 1,
                    Decision::Selected => {
                        writer.push(row, block_index)?;
                        block.selected_count += 1;
                    }
                    Decision::Rejected => block.rejected_count += 1,
                }
            }
        }
        block.output_count = reference_count
            .checked_add(block.selected_count)
            .context("block output count overflow")?;
        block_stats.push(block);
    }
    ensure!(
        reference.next_descriptor_row()?.is_none() && candidates.next_descriptor_row()?.is_none(),
        "unconsumed CSF blocks"
    );

    let reference_count = checked_sum(block_stats.iter().map(|block| block.reference_count))?;
    let candidate_count = checked_sum(block_stats.iter().map(|block| block.candidate_count))?;
    let exact_reference_skipped = checked_sum(
        block_stats
            .iter()
            .map(|block| block.exact_reference_skipped),
    )?;
    let selected_count = checked_sum(block_stats.iter().map(|block| block.selected_count))?;
    let rejected_count = checked_sum(block_stats.iter().map(|block| block.rejected_count))?;
    let output_count = checked_sum(block_stats.iter().map(|block| block.output_count))?;

    writer.finish()?;
    let output_bytes = fs::metadata(&staged)?.len();
    publish_staged_outputs(
        &[(&staged_header, &output_header), (&staged, output_path)],
        overwrite,
    )?;

    Ok(InteractionStats {
        exact: false,
        mode,
        method,
        block_count: block_stats.len(),
        reference_count,
        candidate_count,
        exact_reference_skipped,
        selected_count,
        rejected_count,
        output_count,
        output_bytes,
        blocks: block_stats,
    })
}

fn decide_candidate(candidate: &[i8], references: &HashSet<Vec<i8>>) -> Decision {
    if references.contains(candidate) {
        return Decision::ExactReference;
    }
    if references
        .iter()
        .any(|reference| occupation_upper_bound(occupations(reference), occupations(candidate)))
    {
        Decision::Selected
    } else {
        Decision::Rejected
    }
}

fn occupations(row: &[i8]) -> impl Iterator<Item = i8> + '_ {
    row[..row.len() - 2].iter().step_by(4).copied()
}

/// The rows use the same orbital table and contain the same number of electrons.
fn occupation_upper_bound(
    reference: impl IntoIterator<Item = i8>,
    candidate: impl IntoIterator<Item = i8>,
) -> bool {
    let mut absolute_difference = 0u8;
    for (reference_occupation, candidate_occupation) in reference.into_iter().zip(candidate) {
        let difference = reference_occupation.abs_diff(candidate_occupation);
        if difference > 2 {
            return false;
        }
        absolute_difference += difference;
        if absolute_difference > 4 {
            return false;
        }
    }
    true
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |total, value| {
        total
            .checked_add(value)
            .context("interaction count overflow")
    })
}

#[cfg(test)]
mod tests {
    use super::occupation_upper_bound;

    #[test]
    fn ichkq2_structural_limits_are_inclusive() {
        // Exactly two electrons moved: each active shell changes by two and
        // the total absolute occupation difference is the inclusive limit 4.
        assert!(occupation_upper_bound([2, 0], [0, 2]));
        assert!(!occupation_upper_bound([3, 0], [0, 3]));
        assert!(!occupation_upper_bound([2, 2, 0, 0], [0, 0, 2, 2]));
        assert!(!occupation_upper_bound(
            [1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 5]
        ));
    }
}
