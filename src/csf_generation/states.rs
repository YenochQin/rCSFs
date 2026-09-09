//! Ordered JKVANT/SENIOR tables from GRASP rcsfgenerate90/genb.f90.
//! Indexed by single-electron 2j and the smaller of electron/hole occupation.
//! -1 in SENIOR means no printed label, not seniority zero.

use anyhow::{Result, bail, ensure};

use crate::complete_csf::SubshellState;

pub(super) fn subshell_states(two_j: u16, electrons: u8) -> Result<Vec<SubshellState>> {
    let capacity = two_j + 1;
    ensure!(
        u16::from(electrons) <= capacity,
        "subshell occupation exceeds capacity"
    );
    let holes = capacity - u16::from(electrons);
    let population = u16::from(electrons).min(holes);
    let plain = |values: &[u16]| {
        values
            .iter()
            .map(|&two_j| SubshellState {
                two_j,
                seniority: None,
            })
            .collect()
    };
    Ok(match (two_j, population) {
        (_, 0) => plain(&[0]),
        (_, 1) => plain(&[two_j]),
        (_, 2) => (0..=2 * (two_j - 1))
            .step_by(4)
            .map(|two_j| SubshellState {
                two_j,
                seniority: None,
            })
            .collect(),
        (5, 3) => plain(&[5, 3, 9]),
        (7, 3) => plain(&[7, 3, 5, 9, 11, 15]),
        (7, 4) => labelled(&[0, 4, 8, 12, 4, 8, 10, 16], &[-1, 2, 2, -1, 4, 4, -1, -1]),
        (9, 3) => labelled(
            &[9, 3, 5, 7, 9, 11, 13, 15, 17, 21],
            &[1, -1, -1, -1, 3, -1, -1, -1, -1, -1],
        ),
        (9, 4) => labelled(
            &[0, 4, 8, 12, 16, 0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24],
            &[0, 2, 2, 2, 2, 4, 4, -1, 4, -1, 4, -1, 4, -1, -1, -1],
        ),
        (9, 5) => labelled(
            &[
                9, 3, 5, 7, 9, 11, 13, 15, 17, 21, 1, 5, 7, 9, 11, 13, 15, 17, 19, 25,
            ],
            &[
                1, -1, 3, 3, 3, 3, 3, 3, 3, -1, -1, 5, 5, 5, 5, 5, 5, 5, -1, -1,
            ],
        ),
        _ => {
            bail!("GRASP has no state table for 2j={two_j}, electron/hole occupation={population}")
        }
    })
}

fn labelled(values: &[u16], seniorities: &[i8]) -> Vec<SubshellState> {
    values
        .iter()
        .zip(seniorities)
        .map(|(&two_j, &seniority)| SubshellState {
            two_j,
            seniority: u8::try_from(seniority).ok(),
        })
        .collect()
}
