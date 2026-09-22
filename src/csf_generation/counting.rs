//! Exact record counting for one prepared configuration.
//!
//! The disk path schedules work by how many records each occupation
//! configuration will produce, so it needs that number before anything is
//! generated. [`count_configuration_records`] answers it without materialising
//! a record, and falls back to counting a real generation when the cheap
//! program cannot be trusted.
//!
//! This module is deliberately separate from the generator itself: it reads the
//! prepared configuration and the state-prefix split, and produces only
//! numbers.

use anyhow::{Context, Result, ensure};

use crate::complete_csf::SubshellState;

use super::{
    GeneratedRecordRef, GeneratedRecordSink, PreparedGeneration, RecordSelection, Subshell,
    SubshellOccupation, generate_prepared_records, prepare_configuration,
};

/// Exact record counting for one prepared configuration.
///
/// The generator emits one record per (state selection, coupling chain) pair,
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
                - if lower == 0 {
                    0
                } else {
                    prefix_sums[lower - 1]
                };
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
