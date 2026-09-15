//! Conservative CSF interaction selection.
//!
//! This module deliberately implements only a structural upper bound on the
//! `rcsfinteract`/`ICHKQ2` occupation test.  It does not evaluate angular or
//! radial matrix elements, so a retained CSF is merely *possibly* interacting.
//! The [`InteractionStats::exact`] flag is consequently always `false`.

use crate::complete_csf::{CompleteCsfFile, CsfRecord, Parity, SymmetryBlock};
use anyhow::{Context, Result, ensure};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Hamiltonian requested by the caller.
///
/// Both modes currently use the same conservative occupation bound.  The mode
/// is retained in the result so later exact implementations cannot silently
/// lose the caller's physical-model choice.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HamiltonianMode {
    DiracCoulomb,
    DiracCoulombBreit,
}

impl HamiltonianMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DiracCoulomb => "dirac_coulomb",
            Self::DiracCoulombBreit => "dirac_coulomb_breit",
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
pub enum InteractionMethod {
    StructuralUpperBound,
}

impl InteractionMethod {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StructuralUpperBound => "structural_upper_bound",
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
    pub total_two_j: u16,
    pub parity: Parity,
    pub reference_count: usize,
    pub candidate_count: usize,
    pub exact_reference_skipped: usize,
    pub selected_count: usize,
    pub rejected_count: usize,
    pub output_count: usize,
}

/// Counts and provenance for one interaction-selection run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InteractionStats {
    /// `false`: structural selection is a conservative upper bound, not an
    /// exact Hamiltonian matrix-element calculation.
    pub exact: bool,
    pub mode: HamiltonianMode,
    pub method: InteractionMethod,
    pub block_count: usize,
    pub reference_count: usize,
    pub candidate_count: usize,
    pub exact_reference_skipped: usize,
    pub selected_count: usize,
    pub rejected_count: usize,
    pub output_count: usize,
    pub output_bytes: u64,
    pub blocks: Vec<InteractionBlockStats>,
}

type ExactSubshellState = (u16, Option<u8>);
type ExactOccupation = (u16, u8, Option<ExactSubshellState>);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ExactCsf {
    occupied: Vec<ExactOccupation>,
    couplings: Vec<(u16, u16)>,
    total_two_j: u16,
    odd_parity: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Decision {
    ExactReference,
    Selected,
    Rejected,
}

struct ReferenceRecord {
    exact: ExactCsf,
    occupations: Vec<u16>,
    electron_count: u32,
}

struct TemporaryOutput {
    path: PathBuf,
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

static TEMPORARY_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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
/// declared.
///
/// Output is fully written to a same-directory temporary file before it is
/// published.  With `overwrite == false`, publication atomically refuses an
/// existing path.
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
    ensure!(
        overwrite || !output_path.exists(),
        "output file {} already exists (set overwrite=true to replace it)",
        output_path.display()
    );
    ensure!(
        reference_path != candidates_path,
        "reference and candidate inputs must be different files"
    );
    if reference_path.exists() && candidates_path.exists() {
        let reference_canonical = reference_path.canonicalize().with_context(|| {
            format!(
                "failed to resolve reference input {}",
                reference_path.display()
            )
        })?;
        let candidates_canonical = candidates_path.canonicalize().with_context(|| {
            format!(
                "failed to resolve candidate input {}",
                candidates_path.display()
            )
        })?;
        ensure!(
            reference_canonical != candidates_canonical
                && !same_file_identity(reference_path, candidates_path)?,
            "reference and candidate inputs must be different files"
        );
    }
    ensure_output_does_not_alias_input(output_path, reference_path, "reference")?;
    ensure_output_does_not_alias_input(output_path, candidates_path, "candidate")?;

    let reference = CompleteCsfFile::parse_path(reference_path)?;
    let candidates = CompleteCsfFile::parse_path(candidates_path)?;
    validate_inputs(&reference, &candidates)?;

    let shell_count = candidates.subshells.len();
    let mut selected_by_block = Vec::<Vec<usize>>::with_capacity(reference.blocks.len());
    let mut block_stats = Vec::with_capacity(reference.blocks.len());

    let mut evaluate = || -> Result<()> {
        for (block_index, (reference_block, candidate_block)) in
            reference.blocks.iter().zip(&candidates.blocks).enumerate()
        {
            let reference_range =
                block_range(reference_block.record_start, reference_block.record_len)?;
            let candidate_range =
                block_range(candidate_block.record_start, candidate_block.record_len)?;
            let references = reference.records[reference_range.clone()]
                .iter()
                .map(|record| reference_record(&reference, record, shell_count))
                .collect::<Result<Vec<_>>>()?;
            let exact_references = references
                .iter()
                .map(|record| record.exact.clone())
                .collect::<HashSet<_>>();
            ensure!(
                exact_references.len() == references.len(),
                "reference block {block_index} contains duplicate CSFs"
            );

            // `par_iter` is indexed, and collection into Vec retains candidate
            // order independently of worker completion order.
            let decisions = candidates.records[candidate_range.clone()]
                .par_iter()
                .map(|candidate| {
                    decide_candidate(
                        &candidates,
                        candidate,
                        shell_count,
                        &references,
                        &exact_references,
                    )
                })
                .collect::<Result<Vec<_>>>()?;

            let mut selected_indices = Vec::new();
            let mut exact_reference_skipped = 0usize;
            let mut selected_count = 0usize;
            let mut rejected_count = 0usize;
            for (offset, decision) in decisions.into_iter().enumerate() {
                match decision {
                    Decision::ExactReference => exact_reference_skipped += 1,
                    Decision::Selected => {
                        selected_count += 1;
                        selected_indices.push(candidate_range.start + offset);
                    }
                    Decision::Rejected => rejected_count += 1,
                }
            }
            let reference_count = reference_range.len();
            let candidate_count = candidate_range.len();
            block_stats.push(InteractionBlockStats {
                block_index,
                total_two_j: reference_block.total_two_j,
                parity: reference_block.parity,
                reference_count,
                candidate_count,
                exact_reference_skipped,
                selected_count,
                rejected_count,
                output_count: reference_count
                    .checked_add(selected_count)
                    .context("block output count overflow")?,
            });
            selected_by_block.push(selected_indices);
        }
        Ok(())
    };

    match num_workers {
        Some(workers) => ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .context("failed to build interaction worker pool")?
            .install(evaluate)?,
        None => evaluate()?,
    }

    let mut output = CompleteCsfFile {
        header_lines: candidates.header_lines.clone(),
        subshells: candidates.subshells.clone(),
        records: Vec::new(),
        occupied_subshells: Vec::new(),
        intermediate_couplings: Vec::new(),
        blocks: Vec::new(),
    };
    for (block_index, reference_block) in reference.blocks.iter().enumerate() {
        let reference_range =
            block_range(reference_block.record_start, reference_block.record_len)?;
        for record in &reference.records[reference_range] {
            output.append_generated_record(
                reference.occupied(record)?,
                reference.couplings(record)?,
                record.total_two_j,
                record.parity,
            )?;
        }
        for &record_index in &selected_by_block[block_index] {
            let record = &candidates.records[record_index];
            output.append_generated_record(
                candidates.occupied(record)?,
                candidates.couplings(record)?,
                record.total_two_j,
                record.parity,
            )?;
        }
    }

    // `append_generated_record` starts a new block only when J/P changes.
    // Complete-CSF text may nevertheless contain adjacent, explicitly
    // separated blocks with the same J/P. Rebuild the block table from the
    // paired input blocks so those file-level boundaries remain intact.
    let mut record_start = 0u64;
    output.blocks = block_stats
        .iter()
        .map(|block| {
            let record_len =
                u64::try_from(block.output_count).context("block output count exceeds u64")?;
            let output_block = SymmetryBlock {
                record_start,
                record_len,
                total_two_j: block.total_two_j,
                parity: block.parity,
            };
            record_start = record_start
                .checked_add(record_len)
                .context("output block range overflow")?;
            Ok(output_block)
        })
        .collect::<Result<Vec<_>>>()?;

    let (temporary, file) = create_temporary_output(output_path)?;
    let mut writer = BufWriter::new(file);
    output.write_to(&mut writer).with_context(|| {
        format!(
            "failed to write temporary output for {}",
            output_path.display()
        )
    })?;
    writer.flush()?;
    writer.get_ref().sync_all().with_context(|| {
        format!(
            "failed to sync temporary output for {}",
            output_path.display()
        )
    })?;
    drop(writer);
    let output_bytes = fs::metadata(&temporary.path)?.len();
    publish_temporary_output(&temporary.path, output_path, overwrite)?;

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

fn validate_inputs(reference: &CompleteCsfFile, candidates: &CompleteCsfFile) -> Result<()> {
    ensure!(
        reference.header_lines[1] == candidates.header_lines[1],
        "core header mismatch: reference={:?}, candidates={:?}",
        reference.header_lines[1],
        candidates.header_lines[1]
    );
    ensure!(
        reference.subshells.len() <= candidates.subshells.len()
            && reference.subshells == candidates.subshells[..reference.subshells.len()],
        "reference peel subshells must be a prefix of candidate peel subshells"
    );
    ensure!(
        reference.blocks.len() == candidates.blocks.len(),
        "block count mismatch: reference={} candidates={}",
        reference.blocks.len(),
        candidates.blocks.len()
    );
    for (index, (reference_block, candidate_block)) in
        reference.blocks.iter().zip(&candidates.blocks).enumerate()
    {
        ensure!(
            reference_block.total_two_j == candidate_block.total_two_j
                && reference_block.parity == candidate_block.parity,
            "J/P mismatch in block {}: reference=(2J={}, {:?}) candidates=(2J={}, {:?})",
            index,
            reference_block.total_two_j,
            reference_block.parity,
            candidate_block.total_two_j,
            candidate_block.parity
        );
    }
    Ok(())
}

fn block_range(start: u64, len: u64) -> Result<std::ops::Range<usize>> {
    let start = usize::try_from(start).context("block start exceeds platform address space")?;
    let len = usize::try_from(len).context("block length exceeds platform address space")?;
    let end = start
        .checked_add(len)
        .context("block record range overflow")?;
    Ok(start..end)
}

fn exact_csf(file: &CompleteCsfFile, record: &CsfRecord) -> Result<ExactCsf> {
    let occupied = file
        .occupied(record)?
        .iter()
        .map(|shell| {
            (
                shell.subshell_index,
                shell.occupation,
                shell.state.map(|state| (state.two_j, state.seniority)),
            )
        })
        .collect();
    let couplings = file
        .couplings(record)?
        .iter()
        .map(|coupling| (coupling.boundary, coupling.two_j))
        .collect();
    Ok(ExactCsf {
        occupied,
        couplings,
        total_two_j: record.total_two_j,
        odd_parity: record.parity == Parity::Odd,
    })
}

fn occupations(
    file: &CompleteCsfFile,
    record: &CsfRecord,
    shell_count: usize,
) -> Result<(Vec<u16>, u32)> {
    let mut values = vec![0u16; shell_count];
    let mut electron_count = 0u32;
    for shell in file.occupied(record)? {
        let index = usize::from(shell.subshell_index);
        let value = values
            .get_mut(index)
            .with_context(|| format!("subshell index {index} exceeds candidate peel list"))?;
        *value = value
            .checked_add(u16::from(shell.occupation))
            .context("occupation sum overflow")?;
        electron_count = electron_count
            .checked_add(u32::from(shell.occupation))
            .context("electron count overflow")?;
    }
    Ok((values, electron_count))
}

fn reference_record(
    file: &CompleteCsfFile,
    record: &CsfRecord,
    shell_count: usize,
) -> Result<ReferenceRecord> {
    let (occupations, electron_count) = occupations(file, record, shell_count)?;
    Ok(ReferenceRecord {
        exact: exact_csf(file, record)?,
        occupations,
        electron_count,
    })
}

fn decide_candidate(
    candidates: &CompleteCsfFile,
    candidate: &CsfRecord,
    shell_count: usize,
    references: &[ReferenceRecord],
    exact_references: &HashSet<ExactCsf>,
) -> Result<Decision> {
    let exact = exact_csf(candidates, candidate)?;
    if exact_references.contains(&exact) {
        return Ok(Decision::ExactReference);
    }
    let (candidate_occupations, candidate_electron_count) =
        occupations(candidates, candidate, shell_count)?;
    if references.iter().any(|reference| {
        occupation_upper_bound(
            &reference.occupations,
            reference.electron_count,
            &candidate_occupations,
            candidate_electron_count,
        )
    }) {
        Ok(Decision::Selected)
    } else {
        Ok(Decision::Rejected)
    }
}

fn occupation_upper_bound(
    reference: &[u16],
    reference_electrons: u32,
    candidate: &[u16],
    candidate_electrons: u32,
) -> bool {
    if reference_electrons != candidate_electrons || reference.len() != candidate.len() {
        return false;
    }
    let mut differing_shells = 0usize;
    let mut absolute_occupation_difference = 0u16;
    for (&reference_occupation, &candidate_occupation) in reference.iter().zip(candidate) {
        let difference = reference_occupation.abs_diff(candidate_occupation);
        if difference == 0 {
            continue;
        }
        if difference > 2 {
            return false;
        }
        absolute_occupation_difference += difference;
        if absolute_occupation_difference > 4 {
            return false;
        }
        differing_shells += 1;
        if differing_shells > 4 {
            return false;
        }
    }
    true
}

fn ensure_output_does_not_alias_input(
    output: &Path,
    input: &Path,
    input_description: &str,
) -> Result<()> {
    if output == input {
        anyhow::bail!(
            "output path must not be the {input_description} input path: {}",
            output.display()
        );
    }

    // If both names exist, canonicalization detects symlinks and metadata
    // identity detects hard links.  A missing output cannot alias an existing
    // input yet, so lexical equality above is sufficient for that case.
    if output.exists() {
        let output_canonical = output
            .canonicalize()
            .with_context(|| format!("failed to resolve output path {}", output.display()))?;
        let input_canonical = input.canonicalize().with_context(|| {
            format!(
                "failed to resolve {input_description} input {}",
                input.display()
            )
        })?;
        ensure!(
            output_canonical != input_canonical,
            "output path aliases the {input_description} input: {}",
            output.display()
        );
        ensure!(
            !same_file_identity(output, input)?,
            "output path is a hard link to the {input_description} input: {}",
            output.display()
        );
    }
    Ok(())
}

#[cfg(unix)]
fn same_file_identity(left: &Path, right: &Path) -> Result<bool> {
    use std::os::unix::fs::MetadataExt;

    let left = fs::metadata(left)?;
    let right = fs::metadata(right)?;
    Ok(left.dev() == right.dev() && left.ino() == right.ino())
}

#[cfg(not(unix))]
fn same_file_identity(_left: &Path, _right: &Path) -> Result<bool> {
    // Canonical paths catch ordinary names and symlinks on non-Unix targets.
    // Rust's standard library does not expose a portable file-ID comparison.
    Ok(false)
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |total, value| {
        total
            .checked_add(value)
            .context("interaction count overflow")
    })
}

fn create_temporary_output(output_path: &Path) -> Result<(TemporaryOutput, File)> {
    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = output_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("interaction-output");
    for _ in 0..128 {
        let sequence = TEMPORARY_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".{name}.rcsfs-{}-{sequence}.tmp",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((TemporaryOutput { path }, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create temporary output in {}", parent.display())
                });
            }
        }
    }
    anyhow::bail!(
        "failed to allocate a unique temporary output in {}",
        parent.display()
    )
}

fn publish_temporary_output(temporary: &Path, output: &Path, overwrite: bool) -> Result<()> {
    if overwrite {
        replace_output(temporary, output)?;
        return Ok(());
    }

    // A same-filesystem hard link is an atomic create-if-absent operation.
    // Unlike a preflight `exists` check, it also closes the publication race.
    fs::hard_link(temporary, output).with_context(|| {
        format!(
            "failed to publish output {} without overwriting an existing file",
            output.display()
        )
    })?;
    // Publication has succeeded. Cleanup is best-effort so callers never see
    // a failure for an output that is already complete and visible.
    let _ = fs::remove_file(temporary);
    Ok(())
}

#[cfg(not(windows))]
fn replace_output(temporary: &Path, output: &Path) -> Result<()> {
    fs::rename(temporary, output).with_context(|| {
        format!(
            "failed to replace output {} with completed temporary file",
            output.display()
        )
    })
}

#[cfg(windows)]
fn replace_output(temporary: &Path, output: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;

    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(
            existing_file_name: *const u16,
            new_file_name: *const u16,
            flags: u32,
        ) -> i32;
    }

    let temporary_wide = temporary
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let output_wide = output
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    // SAFETY: both pointers refer to live, NUL-terminated UTF-16 buffers for
    // the duration of the call. The flags request same-filesystem replacement.
    let succeeded = unsafe {
        MoveFileExW(
            temporary_wide.as_ptr(),
            output_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if succeeded == 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "failed to replace output {} with completed temporary file",
                output.display()
            )
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::occupation_upper_bound;

    #[test]
    fn ichkq2_structural_limits_are_inclusive() {
        // Exactly two electrons moved: each active shell changes by two and
        // the total absolute occupation difference is the inclusive limit 4.
        assert!(occupation_upper_bound(&[2, 0], 2, &[0, 2], 2));
        assert!(!occupation_upper_bound(
            &[1, 1, 1, 1, 1],
            5,
            &[0, 0, 0, 0, 0],
            0
        ));
        assert!(!occupation_upper_bound(&[3, 0], 3, &[0, 3], 3));
        assert!(!occupation_upper_bound(&[2, 2, 0, 0], 4, &[0, 0, 1, 3], 4));
        assert!(!occupation_upper_bound(
            &[1, 1, 1, 1, 1, 0],
            5,
            &[0, 0, 0, 0, 0, 5],
            5
        ));
    }
}
