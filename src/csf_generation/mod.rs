//! Serial generation for one explicit relativistic occupation configuration.
//!
//! This is the `GEN` stage of rcsfgenerate: subshell states and angular coupling
//! chains, followed by J block ordering. Excitations, nonrelativistic occupation
//! splitting, multiple references and existing-list expansion are separate stages
//! and are not inferred by this interface.

mod occupations;
mod pipeline;
mod states;
#[allow(dead_code)] // Phase 4 owns the public transaction/CLI wiring.
pub(crate) mod streaming;

pub use occupations::{
    EnumeratedConfiguration, EnumeratedOccupations, ExcitationRequest, OccupationMode, Orbital,
    ReferenceConfiguration, ReferenceSubshell, enumerate_occupations,
};
pub use pipeline::{
    TranscriptGenerationStats, WriteStats, generate_csfs_from_transcript, write_generated_csfs,
};

use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

use crate::complete_csf::{
    CompleteCsfFile, IntermediateCoupling, OccupiedSubshell, Parity, SubshellState,
};
pub(crate) use states::subshell_states;

const ORBITAL_LETTERS: &[u8] = b"spdfghiklmn";
const MAX_OCCUPIED_SUBSHELLS: usize = 20;

/// Relativistic orbital identity; kappa > 0 denotes j=l-1/2.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Subshell {
    n: u8,
    kappa: i8,
}

impl Subshell {
    /// GRASP's supported orbitals have n <= 15 and l <= 10.
    pub fn new(n: u8, kappa: i8) -> Result<Self> {
        ensure!(
            kappa != 0 && (-11..=10).contains(&kappa),
            "kappa must be in -11..=-1 or 1..=10"
        );
        let l = if kappa > 0 {
            kappa as u8
        } else {
            kappa.unsigned_abs() - 1
        };
        ensure!(
            (1..=15).contains(&n) && l < n,
            "subshell must satisfy 0 <= l < n <= 15"
        );
        Ok(Self { n, kappa })
    }

    pub fn n(self) -> u8 {
        self.n
    }
    pub fn kappa(self) -> i8 {
        self.kappa
    }
    pub fn l(self) -> u8 {
        if self.kappa > 0 {
            self.kappa as u8
        } else {
            self.kappa.unsigned_abs() - 1
        }
    }
    pub fn capacity(self) -> u8 {
        2 * self.kappa.unsigned_abs()
    }
}

impl fmt::Display for Subshell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}{}",
            self.n,
            char::from(ORBITAL_LETTERS[usize::from(self.l())]),
            if self.kappa > 0 { "-" } else { "" }
        )
    }
}

impl FromStr for Subshell {
    type Err = anyhow::Error;

    fn from_str(label: &str) -> Result<Self> {
        let (orbital, lower) = label
            .strip_suffix('-')
            .map_or((label, false), |s| (s, true));
        ensure!(
            orbital.is_ascii() && orbital.len() >= 2,
            "invalid subshell label {label:?}"
        );
        let (n, letter) = orbital.split_at(orbital.len() - 1);
        let l = ORBITAL_LETTERS
            .iter()
            .position(|&value| value == letter.as_bytes()[0])
            .with_context(|| format!("invalid orbital letter in {label:?}"))? as i8;
        ensure!(!lower || l > 0, "s- is not a relativistic subshell");
        let subshell = Self::new(n.parse()?, if lower { l } else { -l - 1 })?;
        ensure!(
            subshell.to_string() == label,
            "non-canonical subshell label {label:?}"
        );
        Ok(subshell)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubshellOccupation {
    pub subshell: Subshell,
    pub electrons: u8,
}

/// Explicit occupations in coupling order. Core subshells are fully occupied
/// and excluded from coupling. Zero occupations are omitted from output.
///
/// min/max_two_j must have the electron-number parity. The inclusive range
/// advances by two. Results are grouped by ascending 2J; within a block they
/// retain GEN's state-table order and ascending intermediate couplings.
#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub core_subshells: Vec<Subshell>,
    pub configuration: Vec<SubshellOccupation>,
    pub min_two_j: u16,
    pub max_two_j: u16,
}

/// One generated CSF backed by the generator's reusable integer buffers.
///
/// The slices remain valid only for the duration of [`GeneratedRecordSink::push`].
/// Sinks that retain records must copy or encode them before returning. Keeping this
/// contract explicit prevents the recursive generator from depending on any output
/// representation such as [`CompleteCsfFile`] or an Arrow writer.
#[derive(Clone, Copy)]
pub(crate) struct GeneratedRecordRef<'a> {
    pub(crate) occupied: &'a [OccupiedSubshell],
    pub(crate) couplings: &'a [IntermediateCoupling],
    pub(crate) total_two_j: u16,
    pub(crate) parity: Parity,
}

/// Receives generated CSFs in their compact integer representation.
///
/// This is the seam between angular-momentum recursion and storage. A sink may
/// append to an in-memory file, encode a disk segment, or collect test records;
/// it must preserve a record before [`Self::push`] returns if it needs it later.
pub(crate) trait GeneratedRecordSink {
    fn push(&mut self, record: GeneratedRecordRef<'_>) -> Result<()>;
}

/// The compatibility adapter used by the existing in-memory API.
pub(crate) struct CompleteCsfSink<'a> {
    output: &'a mut CompleteCsfFile,
}

impl<'a> CompleteCsfSink<'a> {
    pub(crate) fn new(output: &'a mut CompleteCsfFile) -> Self {
        Self { output }
    }
}

impl GeneratedRecordSink for CompleteCsfSink<'_> {
    fn push(&mut self, record: GeneratedRecordRef<'_>) -> Result<()> {
        self.output.append_generated_record(
            record.occupied,
            record.couplings,
            record.total_two_j,
            record.parity,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BufferedGeneratedRecord {
    occupied: Vec<OccupiedSubshell>,
    couplings: Vec<IntermediateCoupling>,
    total_two_j: u16,
    parity: Parity,
}

impl BufferedGeneratedRecord {
    fn copy_from(record: GeneratedRecordRef<'_>) -> Result<Self> {
        let mut occupied = Vec::new();
        occupied.try_reserve(record.occupied.len())?;
        occupied.extend_from_slice(record.occupied);
        let mut couplings = Vec::new();
        couplings.try_reserve(record.couplings.len())?;
        couplings.extend_from_slice(record.couplings);
        Ok(Self {
            occupied,
            couplings,
            total_two_j: record.total_two_j,
            parity: record.parity,
        })
    }

    fn as_ref(&self) -> GeneratedRecordRef<'_> {
        GeneratedRecordRef {
            occupied: &self.occupied,
            couplings: &self.couplings,
            total_two_j: self.total_two_j,
            parity: self.parity,
        }
    }
}

#[derive(Default)]
struct BufferedRecordSink {
    records: Vec<BufferedGeneratedRecord>,
}

impl GeneratedRecordSink for BufferedRecordSink {
    fn push(&mut self, record: GeneratedRecordRef<'_>) -> Result<()> {
        self.records.try_reserve(1)?;
        self.records
            .push(BufferedGeneratedRecord::copy_from(record)?);
        Ok(())
    }
}

struct PreparedGeneration {
    header_lines: [String; 5],
    subshells: Vec<String>,
    choices: Vec<Vec<SubshellState>>,
    occupied: Vec<OccupiedSubshell>,
    parity: Parity,
}

impl PreparedGeneration {
    fn empty_file(&self) -> CompleteCsfFile {
        CompleteCsfFile {
            header_lines: self.header_lines.clone(),
            subshells: self.subshells.clone(),
            records: Vec::new(),
            occupied_subshells: Vec::new(),
            intermediate_couplings: Vec::new(),
            blocks: Vec::new(),
        }
    }
}

/// Enumerate new CSFs without invoking Fortran or constructing text records.
///
/// An impossible target or empty occupation configuration yields an empty
/// result. Inspect `records.is_empty()` before export: empty CSF lists cannot
/// be written by the strict codec. Unsupported subshell state tables fail.
pub fn generate_csfs(request: &GenerationRequest) -> Result<CompleteCsfFile> {
    generate_csfs_impl(request, 1)
}

fn generate_csfs_impl(request: &GenerationRequest, branch_count: usize) -> Result<CompleteCsfFile> {
    let prepared = prepare_generation(request)?;
    let mut output = prepared.empty_file();
    if prepared.occupied.is_empty() {
        return Ok(output);
    }
    {
        let mut sink = CompleteCsfSink::new(&mut output);
        generate_prepared_records(
            &prepared,
            request.min_two_j,
            request.max_two_j,
            branch_count,
            &mut sink,
        )?;
    }
    Ok(output)
}

/// Generate one occupation configuration directly into a storage sink.
///
/// This preserves [`generate_csfs`]'s state-table and block traversal while
/// avoiding construction of a [`CompleteCsfFile`]. It is crate-visible for
/// range writers; public callers retain the compatibility API above.
#[allow(dead_code)] // Used by the staged range writer before Phase 4 exposes it.
pub(crate) fn generate_records_into(
    request: &GenerationRequest,
    branch_count: usize,
    sink: &mut impl GeneratedRecordSink,
) -> Result<()> {
    let prepared = prepare_generation(request)?;
    if prepared.occupied.is_empty() {
        return Ok(());
    }
    generate_prepared_records(
        &prepared,
        request.min_two_j,
        request.max_two_j,
        branch_count,
        sink,
    )
}

fn prepare_generation(request: &GenerationRequest) -> Result<PreparedGeneration> {
    ensure!(
        request.min_two_j <= request.max_two_j,
        "minimum 2J exceeds maximum 2J"
    );
    ensure!(
        printable_j(request.max_two_j),
        "target J exceeds GRASP's output field range"
    );
    let mut seen = HashSet::new();
    for &subshell in &request.core_subshells {
        ensure!(seen.insert(subshell), "duplicate core subshell {subshell}");
    }
    let mut occupied = Vec::new();
    let mut choices = Vec::new();
    let mut subshells = Vec::new();
    let mut electrons = 0u16;
    let mut parity_sum = 0u16;
    for entry in &request.configuration {
        ensure!(
            seen.insert(entry.subshell),
            "duplicate or core-overlapping subshell {}",
            entry.subshell
        );
        ensure!(
            entry.electrons <= entry.subshell.capacity(),
            "occupation exceeds capacity for {}",
            entry.subshell
        );
        if entry.electrons == 0 {
            continue;
        }
        ensure!(
            occupied.len() < MAX_OCCUPIED_SUBSHELLS,
            "GRASP supports at most 20 occupied peel subshells"
        );
        let states = subshell_states(u16::from(entry.subshell.capacity() - 1), entry.electrons)
            .with_context(|| {
                format!(
                    "unsupported occupation {}({})",
                    entry.subshell, entry.electrons
                )
            })?;
        occupied.push(OccupiedSubshell {
            subshell_index: u16::try_from(subshells.len())?,
            occupation: entry.electrons,
            state: None,
        });
        choices.push(states);
        subshells.push(entry.subshell.to_string());
        electrons += u16::from(entry.electrons);
        parity_sum += u16::from(entry.subshell.l()) * u16::from(entry.electrons);
    }
    ensure!(
        request.min_two_j % 2 == electrons % 2 && request.max_two_j % 2 == electrons % 2,
        "2J range must have the same parity as the peel electron count"
    );
    let parity = if parity_sum.is_multiple_of(2) {
        Parity::Even
    } else {
        Parity::Odd
    };
    let core_labels = request
        .core_subshells
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    Ok(PreparedGeneration {
        header_lines: [
            "Core subshells:".into(),
            header_orbitals(&core_labels),
            "Peel subshells:".into(),
            header_orbitals(&subshells),
            "CSF(s):".into(),
        ],
        subshells,
        choices,
        occupied,
        parity,
    })
}

fn generate_prepared_records(
    prepared: &PreparedGeneration,
    min_two_j: u16,
    max_two_j: u16,
    branch_count: usize,
    sink: &mut impl GeneratedRecordSink,
) -> Result<()> {
    // Prefixes follow the serial state-table traversal. Each subtree is disjoint;
    // indexed collection preserves that order independently of worker completion.
    let combinations = prepared
        .choices
        .iter()
        .fold(1usize, |n, states| n.saturating_mul(states.len()));
    if branch_count > 1 && combinations >= 64 {
        let mut prefixes = vec![Vec::new()];
        for states in &prepared.choices {
            if prefixes.len() >= branch_count {
                break;
            }
            prefixes = prefixes
                .into_iter()
                .flat_map(|prefix| {
                    states.iter().map(move |state| {
                        let mut next = prefix.clone();
                        next.push(*state);
                        next
                    })
                })
                .collect();
        }
        for target in (min_two_j..=max_two_j).step_by(2) {
            let branches = prefixes
                .par_iter()
                .map(|prefix| {
                    let mut branch_sink = BufferedRecordSink::default();
                    let mut selected = vec![
                        SubshellState {
                            two_j: 0,
                            seniority: None
                        };
                        prepared.occupied.len()
                    ];
                    selected[..prefix.len()].copy_from_slice(prefix);
                    let mut generator = Generator {
                        choices: &prepared.choices,
                        selected,
                        cumulative: vec![0; prepared.occupied.len()],
                        occupied: prepared.occupied.clone(),
                        printed_couplings: Vec::with_capacity(MAX_OCCUPIED_SUBSHELLS),
                        parity: prepared.parity,
                        sink: &mut branch_sink,
                    };
                    generator.select_states(prefix.len(), target)?;
                    Ok(branch_sink.records)
                })
                .collect::<Result<Vec<_>>>()?;
            for branch in branches {
                for record in &branch {
                    sink.push(record.as_ref())?;
                }
            }
        }
        return Ok(());
    }
    let mut generator = Generator {
        choices: &prepared.choices,
        selected: vec![
            SubshellState {
                two_j: 0,
                seniority: None
            };
            prepared.occupied.len()
        ],
        cumulative: vec![0; prepared.occupied.len()],
        occupied: prepared.occupied.clone(),
        printed_couplings: Vec::with_capacity(MAX_OCCUPIED_SUBSHELLS),
        parity: prepared.parity,
        sink,
    };
    for target in (min_two_j..=max_two_j).step_by(2) {
        generator.select_states(0, target)?;
    }
    Ok(())
}

/// Generate independent occupation configurations in parallel.
///
/// Rayon collects indexed parallel iterators in their original input order,
/// so callers can apply the same ordering and block merge logic as the
/// serial generator. Record-count arithmetic is checked for overflow.
pub fn generate_csfs_parallel(
    requests: &[GenerationRequest],

    threads: Option<usize>,
) -> Result<Vec<CompleteCsfFile>> {
    ensure!(threads != Some(0), "threads must be greater than 0");
    let run = || {
        requests
            .par_iter()
            .map(|request| {
                generate_csfs_impl(
                    request,
                    if rayon::current_num_threads() > 1 {
                        rayon::current_num_threads().saturating_mul(4)
                    } else {
                        1
                    },
                )
            })
            .collect::<Result<Vec<_>>>()
    };
    let results = match threads {
        Some(count) => rayon::ThreadPoolBuilder::new()
            .num_threads(count)
            .build()
            .context("failed to build rayon thread pool")?
            .install(run)?,
        None => run()?,
    };
    results.iter().try_fold(0usize, |total, file| {
        total
            .checked_add(file.records.len())
            .context("record count overflow")
    })?;
    Ok(results)
}

fn header_orbitals(labels: &[String]) -> String {
    labels
        .iter()
        .map(|label| {
            let display = if label.ends_with('-') {
                label.clone()
            } else {
                format!("{label} ")
            };
            format!("{display:>5}")
        })
        .collect::<String>()
        .trim_end()
        .to_owned()
}

fn printable_j(two_j: u16) -> bool {
    if two_j.is_multiple_of(2) {
        two_j <= 198
    } else {
        two_j <= 99
    }
}

struct Generator<'a, S: GeneratedRecordSink> {
    choices: &'a [Vec<SubshellState>],
    selected: Vec<SubshellState>,
    cumulative: Vec<u16>,
    occupied: Vec<OccupiedSubshell>,
    printed_couplings: Vec<IntermediateCoupling>,
    parity: Parity,

    sink: &'a mut S,
}

impl<S: GeneratedRecordSink> Generator<'_, S> {
    fn select_states(&mut self, index: usize, target: u16) -> Result<()> {
        if index == self.selected.len() {
            self.cumulative[0] = self.selected[0].two_j;
            if self.selected.len() == 1 {
                if self.cumulative[0] == target {
                    self.emit(target)?;
                }
                return Ok(());
            }
            return self.couple(1, target);
        }
        for state in &self.choices[index] {
            self.selected[index] = *state;
            self.select_states(index + 1, target)?;
        }
        Ok(())
    }

    fn couple(&mut self, next: usize, target: u16) -> Result<()> {
        let previous = self.cumulative[next - 1];
        let state_j = self.selected[next].two_j;
        let lower = previous.abs_diff(state_j);
        let upper = previous + state_j;
        if next == self.selected.len() - 1 {
            if (lower..=upper).contains(&target) && (target - lower).is_multiple_of(2) {
                self.cumulative[next] = target;
                self.emit(target)?;
            }
        } else {
            for value in (lower..=upper).step_by(2) {
                self.cumulative[next] = value;
                self.couple(next + 1, target)?;
            }
        }
        Ok(())
    }

    fn emit(&mut self, target: u16) -> Result<()> {
        for (index, entry) in self.occupied.iter_mut().enumerate() {
            let state = self.selected[index];
            entry.state = if state.two_j == 0 && self.choices[index].len() == 1 {
                None
            } else {
                Some(state)
            };
        }
        self.printed_couplings.clear();
        // Port of kopp2's FIRST flag. A visible J=0 state clears FIRST but
        // does not itself print an interior coupling once FIRST is cleared.
        let mut first = self.selected[0].two_j == 0 && self.choices[0].len() == 1;
        for index in 1..self.selected.len().saturating_sub(1) {
            let state_j = self.selected[index].two_j;
            if first && (state_j != 0 || self.choices[index].len() != 1) {
                first = false;
            } else if !first && state_j != 0 {
                let value = self.cumulative[index];
                ensure!(
                    printable_j(value),
                    "intermediate J exceeds GRASP's output field range"
                );
                self.printed_couplings.push(IntermediateCoupling {
                    boundary: u16::try_from(index + 1)?,
                    two_j: value,
                });
            }
        }
        self.sink.push(GeneratedRecordRef {
            occupied: &self.occupied,
            couplings: &self.printed_couplings,
            total_two_j: target,
            parity: self.parity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn four_d_shell_request() -> GenerationRequest {
        GenerationRequest {
            core_subshells: Vec::new(),
            configuration: ["3d", "4d", "5d", "6d"]
                .into_iter()
                .map(|label| SubshellOccupation {
                    subshell: label.parse().unwrap(),
                    electrons: 3,
                })
                .collect(),
            min_two_j: 0,
            max_two_j: 4,
        }
    }

    fn buffered_records(file: &CompleteCsfFile) -> Result<Vec<BufferedGeneratedRecord>> {
        file.records
            .iter()
            .map(|record| {
                Ok(BufferedGeneratedRecord {
                    occupied: file.occupied(record)?.to_vec(),
                    couplings: file.couplings(record)?.to_vec(),
                    total_two_j: record.total_two_j,
                    parity: record.parity,
                })
            })
            .collect()
    }

    #[test]
    fn generated_record_sink_matches_complete_csf_adapter() {
        let request = four_d_shell_request();
        for branch_count in [1, 4] {
            let prepared = prepare_generation(&request).unwrap();
            let mut sink = BufferedRecordSink::default();
            generate_prepared_records(
                &prepared,
                request.min_two_j,
                request.max_two_j,
                branch_count,
                &mut sink,
            )
            .unwrap();

            let complete = generate_csfs_impl(&request, branch_count).unwrap();
            assert_eq!(sink.records, buffered_records(&complete).unwrap());
        }
    }
}
