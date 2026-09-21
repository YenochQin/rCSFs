//! Serial generation for one explicit relativistic occupation configuration.
//!
//! This is the `GEN` stage of rcsfgenerate: subshell states and angular coupling
//! chains, followed by J block ordering. Excitations, nonrelativistic occupation
//! splitting, multiple references and existing-list expansion are separate stages
//! and are not inferred by this interface.

pub(crate) mod capacity;
mod occupations;
mod options;
mod pipeline;
#[allow(dead_code)] // The estimate-only CLI mode consumes part of this module.
pub(crate) mod planning;
mod states;
#[allow(dead_code)] // Phase 4 owns the public transaction/CLI wiring.
pub(crate) mod streaming;

pub(crate) use occupations::enumerate_occupations_with_budget;
pub use occupations::{
    EnumeratedConfiguration, EnumeratedOccupations, ExcitationRequest, OccupationMode, Orbital,
    ReferenceConfiguration, ReferenceSubshell, enumerate_occupations,
};
pub(crate) use options::{
    GenerationOptions, ResourceBudget, ResourcePermit, ResourceStats, SegmentCompression,
};
pub(crate) use capacity::{
    CapacityEstimate, SpaceCheck, estimate_capacity, preflight_run,
};
pub(crate) use planning::{
    GenerationPlan, PlanStats, PlannedTask, TaskSpan, estimate_workload, plan_generation,
    report_plan, request_targets,
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

pub(crate) struct PreparedGeneration {
    header_lines: [String; 5],
    pub(crate) subshells: Vec<String>,
    pub(crate) choices: Vec<Vec<SubshellState>>,
    pub(crate) occupied: Vec<OccupiedSubshell>,
    pub(crate) parity: Parity,
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
            RecordSelection::All,
            branch_count,
            &mut sink,
        )?;
    }
    Ok(output)
}

/// Which sub-sequence of a configuration's generation order a caller asks for.
///
/// Every variant is a subsequence of the serial state-table traversal, so a
/// caller that splits one configuration into several selections can publish
/// the original record order by consuming them in selection order.
#[derive(Clone, Debug)]
pub(crate) enum RecordSelection<'a> {
    /// Every 2J target of the request and every state branch.
    All,
    /// A subset of the request's 2J targets, in ascending order.
    Targets(&'a [u16]),
    /// One 2J target, restricted to a slice of the configuration's
    /// state-prefix tree. See [`state_prefixes`] for the split itself.
    StatePrefixes {
        target: u16,
        branch_count: usize,
        branches: std::ops::Range<usize>,
    },
}

/// Generate one occupation configuration directly into a storage sink.
///
/// This preserves [`generate_csfs`]'s state-table and block traversal while
/// avoiding construction of a [`CompleteCsfFile`]. It is crate-visible for
/// range writers; public callers retain the compatibility API above.
#[allow(dead_code)] // `Targets`/`StatePrefixes` are used by the disk task planner.
pub(crate) fn generate_records_into(
    request: &GenerationRequest,
    selection: RecordSelection<'_>,
    branch_count: usize,
    sink: &mut impl GeneratedRecordSink,
) -> Result<()> {
    generate_configuration_records(
        &request.core_subshells,
        &request.configuration,
        request.min_two_j,
        request.max_two_j,
        selection,
        branch_count,
        sink,
    )
}

/// Generate into a sink without materialising a [`GenerationRequest`] first.
///
/// The disk planner counts and schedules millions of configurations; letting
/// it borrow an enumeration entry avoids cloning every occupation vector.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_configuration_records(
    core_subshells: &[Subshell],
    configuration: &[SubshellOccupation],
    min_two_j: u16,
    max_two_j: u16,
    selection: RecordSelection<'_>,
    branch_count: usize,
    sink: &mut impl GeneratedRecordSink,
) -> Result<()> {
    let prepared = prepare_configuration(core_subshells, configuration, min_two_j, max_two_j)?;
    if prepared.occupied.is_empty() {
        return Ok(());
    }
    generate_prepared_records(&prepared, min_two_j, max_two_j, selection, branch_count, sink)
}

fn prepare_generation(request: &GenerationRequest) -> Result<PreparedGeneration> {
    prepare_configuration(
        &request.core_subshells,
        &request.configuration,
        request.min_two_j,
        request.max_two_j,
    )
}

fn prepare_configuration(
    core_subshells: &[Subshell],
    configuration: &[SubshellOccupation],
    min_two_j: u16,
    max_two_j: u16,
) -> Result<PreparedGeneration> {
    ensure!(min_two_j <= max_two_j, "minimum 2J exceeds maximum 2J");
    ensure!(
        printable_j(max_two_j),
        "target J exceeds GRASP's output field range"
    );
    let mut seen = HashSet::new();
    for &subshell in core_subshells {
        ensure!(seen.insert(subshell), "duplicate core subshell {subshell}");
    }
    let mut occupied = Vec::new();
    let mut choices = Vec::new();
    let mut subshells = Vec::new();
    let mut electrons = 0u16;
    let mut parity_sum = 0u16;
    for entry in configuration {
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
        min_two_j % 2 == electrons % 2 && max_two_j % 2 == electrons % 2,
        "2J range must have the same parity as the peel electron count"
    );
    let parity = if parity_sum.is_multiple_of(2) {
        Parity::Even
    } else {
        Parity::Odd
    };
    let core_labels = core_subshells
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
    selection: RecordSelection<'_>,
    branch_count: usize,
    sink: &mut impl GeneratedRecordSink,
) -> Result<()> {
    match selection {
        RecordSelection::All => {
            // Prefixes follow the serial state-table traversal. Each subtree is
            // disjoint; indexed collection preserves that order independently of
            // worker completion.
            let combinations = prepared
                .choices
                .iter()
                .fold(1usize, |n, states| n.saturating_mul(states.len()));
            if branch_count > 1 && combinations >= 64 {
                let prefixes = state_prefixes(prepared, branch_count);
                for target in (min_two_j..=max_two_j).step_by(2) {
                    let branches = prefixes
                        .par_iter()
                        .map(|prefix| {
                            let mut branch_sink = BufferedRecordSink::default();
                            let mut generator = Generator::new(prepared, &mut branch_sink);
                            generator.select_prefixed_states(prefix, target)?;
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
            let mut generator = Generator::new(prepared, sink);
            for target in (min_two_j..=max_two_j).step_by(2) {
                generator.select_states(0, target)?;
            }
            Ok(())
        }
        RecordSelection::Targets(targets) => {
            let mut generator = Generator::new(prepared, sink);
            for &target in targets {
                ensure!(
                    target >= min_two_j && target <= max_two_j && (target - min_two_j).is_multiple_of(2),
                    "target 2J {target} is outside the request's 2J range"
                );
                generator.select_states(0, target)?;
            }
            Ok(())
        }
        RecordSelection::StatePrefixes {
            target,
            branch_count,
            branches,
        } => {
            ensure!(
                target >= min_two_j && target <= max_two_j && (target - min_two_j).is_multiple_of(2),
                "target 2J {target} is outside the request's 2J range"
            );
            let prefixes = state_prefixes(prepared, branch_count);
            ensure!(
                branches.end <= prefixes.len(),
                "state-prefix slice {:?} exceeds a {}-prefix split",
                branches,
                prefixes.len()
            );
            let mut generator = Generator::new(prepared, sink);
            for prefix in &prefixes[branches] {
                generator.select_prefixed_states(prefix, target)?;
            }
            Ok(())
        }
    }
}

/// Split a configuration's state-selection tree into at most `branch_count`
/// prefixes, in the serial traversal order.
///
/// The split is depth-first and deterministic, so the returned order is the
/// lexicographic order of the serial state table. Partitions built from this
/// list (see [`RecordSelection::StatePrefixes`]) therefore concatenate back
/// into exactly the unsplit record order.
pub(crate) fn state_prefixes(
    prepared: &PreparedGeneration,
    branch_count: usize,
) -> Vec<Vec<SubshellState>> {
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
    prefixes
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

impl<'a, S: GeneratedRecordSink> Generator<'a, S> {
    fn new(prepared: &'a PreparedGeneration, sink: &'a mut S) -> Self {
        Generator {
            choices: &prepared.choices,
            selected: vec![
                SubshellState {
                    two_j: 0,
                    seniority: None,
                };
                prepared.occupied.len()
            ],
            cumulative: vec![0; prepared.occupied.len()],
            occupied: prepared.occupied.clone(),
            printed_couplings: Vec::with_capacity(MAX_OCCUPIED_SUBSHELLS),
            parity: prepared.parity,
            sink,
        }
    }

    /// Continue the traversal with `prefix`'s states already selected.
    ///
    /// Only the state selection is fixed: the coupling values below the prefix
    /// are still chosen by the walk, exactly as in the unsplit traversal.
    fn select_prefixed_states(&mut self, prefix: &[SubshellState], target: u16) -> Result<()> {
        ensure!(
            prefix.len() <= self.selected.len(),
            "state prefix is longer than the occupied subshell list"
        );
        self.selected[..prefix.len()].copy_from_slice(prefix);
        self.select_states(prefix.len(), target)
    }

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

/// Exact record counting for one prepared configuration.
///
/// [`Generator`] emits one record per (state selection, coupling chain) pair,
/// so the number of records is the number of such chains that reach the target.
/// Counting them by dynamic programming over the cumulative `2J` values costs
/// `O(subshells × 2J × states)` instead of walking every chain, which is what
/// makes a full-preflight workload estimate affordable.
///
/// The chain count is aggregation-safe: distinct chains produce distinct
/// records because the intermediate couplings are part of the stored row, so
/// duplicates in the state tables are counted rather than merged.
pub(crate) struct PreparedCounter {
    /// `multiplicity[i][two_j]` counts the entries of subshell `i`'s state table
    /// with that single-electron `2J`. Entries that share a `2J` but differ in
    /// seniority are distinct states and are counted separately.
    multiplicity: Vec<Vec<u64>>,
    /// `reach[i]` is the largest cumulative `2J` the first `i + 1` subshells
    /// can reach, and therefore the upper bound of index `i`'s ways vector.
    reach: Vec<u16>,
}

impl PreparedCounter {
    pub(crate) fn new(prepared: &PreparedGeneration) -> Self {
        let mut multiplicity = Vec::with_capacity(prepared.choices.len());
        let mut reach = Vec::with_capacity(prepared.choices.len());
        let mut running = 0u16;
        for states in &prepared.choices {
            let widest = states.iter().map(|state| state.two_j).max().unwrap_or(0);
            let mut counts = vec![0u64; usize::from(widest) + 1];
            for state in states {
                counts[usize::from(state.two_j)] += 1;
            }
            multiplicity.push(counts);
            running = running.saturating_add(widest);
            reach.push(running);
        }
        Self {
            multiplicity,
            reach,
        }
    }

    /// Largest cumulative `2J` the whole configuration can reach.
    pub(crate) fn max_two_j(&self) -> u16 {
        self.reach.last().copied().unwrap_or(0)
    }

    /// Whether the chain count equals the number of emitted records.
    ///
    /// [`Generator::emit`] rejects an *intermediate* coupling that exceeds
    /// GRASP's output field. Counting cannot see that per-selection `FIRST`
    /// flag, so it is exact only while no reachable intermediate value can
    /// exceed the limit. Unlike a rejection, counting more than the generator
    /// emits would under-report nothing: it can only overestimate, which is
    /// why an unsafe configuration is counted by running the generator instead
    /// (see [`count_configuration_records`]).
    pub(crate) fn chain_count_is_exact(&self) -> bool {
        let subshells = self.reach.len();
        // A coupling is printed only between the first and the last subshell,
        // so three or fewer occupied subshells never print one.
        subshells < 4 || (1..=subshells - 2).all(|index| self.reach[index] <= 99)
    }

    /// Number of records the generator would emit for one target.
    ///
    /// `prefix` restricts the state selection of the first `prefix.len()`
    /// subshells, matching [`Generator::select_prefixed_states`]; the coupling
    /// values below the prefix stay free, exactly as in the split traversal.
    pub(crate) fn count(&self, target: u16, prefix: &[SubshellState]) -> Result<u64> {
        ensure!(
            prefix.len() <= self.multiplicity.len(),
            "state prefix is longer than the occupied subshell list"
        );
        let width = usize::from(self.max_two_j()) + 1;
        if prefix.is_empty() && self.multiplicity.is_empty() {
            return Ok(0);
        }
        if usize::from(target) >= width {
            return Ok(0);
        }
        // Cumulative 2J after the subshells handled so far. Seeding index zero
        // with one empty chain makes the first subshell an ordinary coupling
        // step from `2J = 0`, which reproduces `cumulative[0] = selected[0]`.
        let mut ways = vec![0u64; width];
        ways[0] = 1;
        for index in 0..self.multiplicity.len() {
            let restricted = prefix.get(index).map(|state| state.two_j);
            ways = extend_ways(&ways, &self.multiplicity[index], restricted)?;
        }
        Ok(ways[usize::from(target)])
    }
}

/// Apply one subshell's coupling step to the cumulative-`2J` chain counts.
///
/// With `previous` the running `2J` and `j` the new state's single-electron
/// `2J`, [`Generator::couple`] reaches `current` only when
/// `|previous - j| <= current <= previous + j` and `current - |previous - j|`
/// is even. For fixed `(current, j)` that is a contiguous, fixed-parity range
/// of `previous`, so prefix sums over each parity of `ways` answer every
/// `(current, j)` pair in constant time instead of scanning the range.
///
/// `restricted` replaces the subshell's whole state table with a single state,
/// which is how a state-prefix slice fixes its selections.
fn extend_ways(ways: &[u64], multiplicity: &[u64], restricted: Option<u16>) -> Result<Vec<u64>> {
    let width = ways.len();
    let mut even_prefix = vec![0u64; width];
    let mut odd_prefix = vec![0u64; width];
    let (mut even, mut odd) = (0u64, 0u64);
    for (index, &value) in ways.iter().enumerate() {
        if index.is_multiple_of(2) {
            even = even
                .checked_add(value)
                .context("coupling chain count overflow")?;
        } else {
            odd = odd
                .checked_add(value)
                .context("coupling chain count overflow")?;
        }
        even_prefix[index] = even;
        odd_prefix[index] = odd;
    }
    let mut next = vec![0u64; width];
    for (state_two_j, &per_state) in multiplicity.iter().enumerate() {
        let state_two_j = u16::try_from(state_two_j).context("single-electron 2J exceeds u16")?;
        if restricted.is_some_and(|restricted| restricted != state_two_j) {
            continue;
        }
        let multiplicity = if restricted.is_some() { 1 } else { per_state };
        if multiplicity == 0 {
            continue;
        }
        let state_two_j = usize::from(state_two_j);
        for (current, slot) in next.iter_mut().enumerate() {
            let lower = current.abs_diff(state_two_j);
            if lower >= width {
                continue;
            }
            let upper = (current + state_two_j).min(width - 1);
            if lower > upper {
                continue;
            }
            // `|previous - j| + 2k == current` keeps `previous` in the parity
            // class of `current + j`.
            let prefix_sums = if (current + state_two_j).is_multiple_of(2) {
                &even_prefix
            } else {
                &odd_prefix
            };
            let window = prefix_sums[upper]
                - if lower == 0 { 0 } else { prefix_sums[lower - 1] };
            if window == 0 {
                continue;
            }
            *slot = slot
                .checked_add(
                    window
                        .checked_mul(multiplicity)
                        .context("coupling chain count overflow")?,
                )
                .context("coupling chain count overflow")?;
        }
    }
    Ok(next)
}

/// Count the records a configuration produces without materialising them.
///
/// The dynamic program is used when it is exact; otherwise the configuration is
/// generated into a counting sink, which applies the same state tables and the
/// same intermediate-J acceptance check as a real run.
pub(crate) fn count_configuration_records(
    core_subshells: &[Subshell],
    configuration: &[SubshellOccupation],
    min_two_j: u16,
    max_two_j: u16,
    targets: &[u16],
) -> Result<u64> {
    let prepared = prepare_configuration(core_subshells, configuration, min_two_j, max_two_j)?;
    if prepared.occupied.is_empty() {
        return Ok(0);
    }
    let counter = PreparedCounter::new(&prepared);
    if counter.chain_count_is_exact() {
        let mut total = 0u64;
        for &target in targets {
            total = total
                .checked_add(counter.count(target, &[])?)
                .context("configuration record count overflow")?;
        }
        return Ok(total);
    }
    let mut sink = CountingSink::default();
    generate_prepared_records(
        &prepared,
        min_two_j,
        max_two_j,
        RecordSelection::Targets(targets),
        1,
        &mut sink,
    )?;
    Ok(sink.records)
}

/// A sink that keeps only the number of records it was offered.
#[derive(Default)]
struct CountingSink {
    records: u64,
}

impl GeneratedRecordSink for CountingSink {
    fn push(&mut self, _record: GeneratedRecordRef<'_>) -> Result<()> {
        self.records = self
            .records
            .checked_add(1)
            .context("configuration record count overflow")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn four_d_shell_request() -> GenerationRequest {
        request(&[("3d", 3), ("4d", 3), ("5d", 3), ("6d", 3)], 0, 4)
    }

    fn request(entries: &[(&str, u8)], min_two_j: u16, max_two_j: u16) -> GenerationRequest {
        GenerationRequest {
            core_subshells: Vec::new(),
            configuration: entries
                .iter()
                .map(|&(label, electrons)| SubshellOccupation {
                    subshell: label.parse().unwrap(),
                    electrons,
                })
                .collect(),
            min_two_j,
            max_two_j,
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
                RecordSelection::All,
                branch_count,
                &mut sink,
            )
            .unwrap();

            let complete = generate_csfs_impl(&request, branch_count).unwrap();
            assert_eq!(sink.records, buffered_records(&complete).unwrap());
        }
    }

    /// Occupations whose state tables include the features that make counting
    /// non-trivial: duplicate single-electron 2J values with different
    /// seniority, labelled tables, a closed shell, several coupling chains per
    /// selection, and a configuration whose intermediate couplings exceed
    /// GRASP's output field.
    const COUNTING_COVERAGE: [(&[(&str, u8)], u16, u16); 8] = [
        (&[("5g", 4)], 0, 8),
        (&[("4f", 4)], 0, 8),
        (&[("3d", 6)], 0, 4),
        (&[("3d", 3), ("4d", 3)], 0, 12),
        (&[("5g", 4), ("5g-", 2)], 0, 12),
        (&[("4f", 4), ("5g", 4), ("5g-", 1)], 1, 11),
        (
            &[("4f", 3), ("4f-", 2), ("5g", 3), ("5g-", 2)],
            0,
            16,
        ),
        (
            &[("11n", 1), ("11n-", 1), ("11n", 1), ("11n-", 1), ("11n", 1), ("11n-", 1)],
            126,
            126,
        ),
    ];

    /// The counted records must agree with the generator record for record.
    ///
    /// Where the generator refuses the configuration, the counter must refuse
    /// it too: a count that overrode GRASP's intermediate-J limit would let the
    /// planner schedule work the generator cannot emit.
    #[test]
    fn counted_records_match_generated_records() {
        let mut exact_cases = 0;
        let mut rejected_cases = 0;
        for (entries, min_two_j, max_two_j) in COUNTING_COVERAGE {
            let request = request(entries, min_two_j, max_two_j);
            let targets = (min_two_j..=max_two_j).step_by(2).collect::<Vec<_>>();
            let counted = count_configuration_records(
                &request.core_subshells,
                &request.configuration,
                min_two_j,
                max_two_j,
                &targets,
            );
            let generated = generate_csfs(&request)
                .map(|file| u64::try_from(file.records.len()).expect("record count fits u64"));
            match (counted, generated) {
                (Ok(counted), Ok(generated)) => {
                    assert_eq!(
                        counted, generated,
                        "counter disagrees with the generator for {entries:?} over {min_two_j}..={max_two_j}"
                    );
                    exact_cases += 1;
                }
                (Err(_), Err(_)) => rejected_cases += 1,
                (counted, generated) => panic!(
                    "counter and generator disagree on acceptance for {entries:?}: {counted:?} against {generated:?}"
                ),
            }
        }
        assert_eq!(exact_cases, 7, "the coverage set lost an accepted case");
        assert_eq!(rejected_cases, 1, "the coverage set lost a rejected case");
    }

    /// A state-prefix slice must reproduce exactly the records of the
    /// unsplit traversal, in order, and the parts must partition it.
    #[test]
    fn state_prefix_slices_partition_the_unsplit_records() {
        let request = request(&[("4f", 3), ("4f-", 2), ("5g", 3), ("5g-", 2)], 0, 6);
        let whole = generate_csfs(&request).unwrap();
        let expected = buffered_records(&whole).unwrap();
        let prepared = prepare_generation(&request).unwrap();
        for branch_count in [2, 4, 8] {
            let prefixes = state_prefixes(&prepared, branch_count);
            let mut sliced = Vec::new();
            for &target in &(0..=6u16).step_by(2).collect::<Vec<_>>() {
                for prefix in &prefixes {
                    let mut sink = BufferedRecordSink::default();
                    let mut generator = Generator::new(&prepared, &mut sink);
                    generator.select_prefixed_states(prefix, target).unwrap();
                    sliced.extend(sink.records);
                }
            }
            assert_eq!(
                sliced, expected,
                "a {branch_count}-way state-prefix split changed the records or their order"
            );
        }
    }
}
