//! Physical seniority and its reversible projection to GRASP printed labels.
//!
//! State order follows rcsfgenerate's JKVANT table. Unlike its SENIOR output
//! table, every entry here carries the actual seniority, including values
//! omitted from text. This covers the existing rCSFs generator state set;
//! it does not add further degeneracy labels or expand GRASP's capabilities.

use anyhow::{Result, bail, ensure};
use std::sync::OnceLock;

use crate::complete_csf::SubshellState;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PhysicalState {
    two_j: u16,
    seniority: u8,
}

// Shared by generation, descriptor encoding and restoration: avoid allocating
// and rebuilding this small immutable table for every CSF subshell.
static STATE_TABLES: [OnceLock<Vec<PhysicalState>>; 22 * 12] = [const { OnceLock::new() }; 22 * 12];

fn physical_states(two_j: u16, electrons: u8) -> Result<&'static [PhysicalState]> {
    ensure!(
        two_j > 0 && two_j % 2 == 1 && two_j <= 21,
        "single-electron 2j must be odd and in 1..=21"
    );
    let capacity = two_j + 1;
    ensure!(
        u16::from(electrons) <= capacity,
        "subshell occupation exceeds capacity"
    );
    let population = u16::from(electrons).min(capacity - u16::from(electrons));
    let slot = &STATE_TABLES[usize::from(two_j) * 12 + usize::from(population)];
    if let Some(states) = slot.get() {
        return Ok(states);
    }
    let states = build_physical_states(two_j, population)?;
    Ok(slot.get_or_init(|| states))
}

fn build_physical_states(two_j: u16, population: u16) -> Result<Vec<PhysicalState>> {
    let pairs: Vec<(u16, u8)> = match (two_j, population) {
        (_, 0) => vec![(0, 0)],
        (_, 1) => vec![(two_j, 1)],
        (_, 2) => (0..=2 * (two_j - 1))
            .step_by(4)
            .map(|j| (j, if j == 0 { 0 } else { 2 }))
            .collect(),
        (5, 3) => vec![(5, 1), (3, 3), (9, 3)],
        (7, 3) => vec![(7, 1), (3, 3), (5, 3), (9, 3), (11, 3), (15, 3)],
        (7, 4) => vec![
            (0, 0),
            (4, 2),
            (8, 2),
            (12, 2),
            (4, 4),
            (8, 4),
            (10, 4),
            (16, 4),
        ],
        (9, 3) => vec![
            (9, 1),
            (3, 3),
            (5, 3),
            (7, 3),
            (9, 3),
            (11, 3),
            (13, 3),
            (15, 3),
            (17, 3),
            (21, 3),
        ],
        (9, 4) => vec![
            (0, 0),
            (4, 2),
            (8, 2),
            (12, 2),
            (16, 2),
            (0, 4),
            (4, 4),
            (6, 4),
            (8, 4),
            (10, 4),
            (12, 4),
            (14, 4),
            (16, 4),
            (18, 4),
            (20, 4),
            (24, 4),
        ],
        (9, 5) => vec![
            (9, 1),
            (3, 3),
            (5, 3),
            (7, 3),
            (9, 3),
            (11, 3),
            (13, 3),
            (15, 3),
            (17, 3),
            (21, 3),
            (1, 5),
            (5, 5),
            (7, 5),
            (9, 5),
            (11, 5),
            (13, 5),
            (15, 5),
            (17, 5),
            (19, 5),
            (25, 5),
        ],
        _ => {
            bail!("GRASP has no state table for 2j={two_j}, electron/hole occupation={population}")
        }
    };
    Ok(pairs
        .into_iter()
        .map(|(two_j, seniority)| PhysicalState { two_j, seniority })
        .collect())
}

fn label(states: &[PhysicalState], state: PhysicalState) -> Option<u8> {
    states
        .iter()
        .any(|other| other.two_j == state.two_j && other.seniority != state.seniority)
        .then_some(state.seniority)
}

pub(crate) fn printed_subshell_states(two_j: u16, electrons: u8) -> Result<Vec<SubshellState>> {
    let states = physical_states(two_j, electrons)?;
    Ok(states
        .iter()
        .map(|&state| SubshellState {
            two_j: state.two_j,
            seniority: label(states, state),
        })
        .collect())
}

/// Decide whether a physical seniority must be printed in a CSF state field.
/// All angular momenta are doubled: `two_j` is single-electron j, while
/// `subshell_two_j` is the total angular momentum of the occupied subshell.
/// Returns `None` when J already distinguishes the state, otherwise `Some(v)`.
pub fn printed_seniority(
    two_j: u16,
    electrons: u8,
    subshell_two_j: u16,
    seniority: u8,
) -> Result<Option<u8>> {
    let states = physical_states(two_j, electrons)?;
    let state = PhysicalState {
        two_j: subshell_two_j,
        seniority,
    };
    ensure!(
        states.contains(&state),
        "illegal physical seniority {seniority} for 2j={two_j}, occupation={electrons}, subshell 2J={subshell_two_j}"
    );
    Ok(label(states, state))
}

/// Return the CSF printed-v value, using -1 when the label is omitted.
pub fn seniority_to_printed_v(
    two_j: u16,
    occupation: u8,
    subshell_two_j: u16,
    seniority: u8,
) -> Result<i32> {
    Ok(printed_seniority(two_j, occupation, subshell_two_j, seniority)?.map_or(-1, i32::from))
}

/// Recover actual seniority from a valid printed state (or an omitted state).
/// An omitted label is inferred only when it determines one physical seniority.
/// Missing information that leaves different seniorities possible is an error.
pub fn physical_seniority(
    two_j: u16,
    electrons: u8,
    printed_state: Option<SubshellState>,
) -> Result<u8> {
    let states = physical_states(two_j, electrons)?;
    let mut result = None;
    for &state in states {
        if let Some(printed) = printed_state
            && (printed.two_j != state.two_j || printed.seniority != label(states, state))
        {
            continue;
        }
        if let Some(previous) = result {
            ensure!(
                previous == state.seniority,
                "missing subshell state leaves seniority ambiguous for 2j={two_j}, occupation={electrons}"
            );
        }
        result = Some(state.seniority);
    }
    result.ok_or_else(|| anyhow::anyhow!("invalid printed subshell state {printed_state:?} for 2j={two_j}, occupation={electrons}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_depends_on_j_and_preserves_ambiguous_states() {
        assert_eq!(printed_seniority(7, 4, 8, 2).unwrap(), Some(2));
        assert_eq!(printed_seniority(7, 4, 8, 4).unwrap(), Some(4));
        assert_eq!(printed_seniority(7, 4, 12, 2).unwrap(), None);
        assert_eq!(printed_seniority(7, 4, 10, 4).unwrap(), None);
        assert_eq!(printed_seniority(9, 4, 0, 0).unwrap(), Some(0));
        assert!(printed_seniority(7, 4, 12, 4).is_err());
    }

    #[test]
    fn physical_values_include_empty_closed_single_and_paired_states() {
        for (j, q, sub_j, expected) in [
            (3, 0, 0, 0),
            (3, 4, 0, 0),
            (3, 1, 3, 1),
            (3, 3, 3, 1),
            (3, 2, 0, 0),
            (3, 2, 4, 2),
            (7, 5, 7, 1),
        ] {
            assert_eq!(
                physical_seniority(
                    j,
                    q,
                    Some(SubshellState {
                        two_j: sub_j,
                        seniority: None
                    })
                )
                .unwrap(),
                expected
            );
        }
        assert_eq!(physical_seniority(3, 4, None).unwrap(), 0);
        assert!(physical_seniority(7, 4, None).is_err());
    }

    #[test]
    fn every_supported_state_round_trips_through_printed_labels() {
        for j in (1..=21).step_by(2) {
            for q in 0..=u8::try_from(j + 1).unwrap() {
                let Ok(states) = physical_states(j, q) else {
                    continue;
                };
                for state in states {
                    let printed = printed_seniority(j, q, state.two_j, state.seniority).unwrap();
                    assert_eq!(
                        physical_seniority(
                            j,
                            q,
                            Some(SubshellState {
                                two_j: state.two_j,
                                seniority: printed
                            })
                        )
                        .unwrap(),
                        state.seniority
                    );
                    assert_eq!(
                        printed_seniority(
                            j,
                            u8::try_from(j + 1).unwrap() - q,
                            state.two_j,
                            state.seniority
                        )
                        .unwrap(),
                        printed
                    );
                }
            }
        }
    }
}
