//! Encode complete jj-coupling chains and project them back to canonical GRASP text.

use anyhow::{Context, Result, bail, ensure};

use crate::complete_csf::{
    CompleteCsfFile, CsfRecord, IntermediateCoupling, OccupiedSubshell, Parity, SubshellState,
    prints_intermediate_coupling,
};
use crate::configuration_generation::Subshell;
use crate::descriptor_dataset::DatasetHeader;
use crate::descriptor_schema::{
    DescriptorLayout, DescriptorRowValidator, DescriptorVersion, decode_angular, empty_slot,
    encode_angular, valid_coupling, validate_record,
};
use crate::seniority::{PhysicalState, physical_seniority, physical_states, printed_seniority};

pub fn encode_v3(file: &CompleteCsfFile, record: &CsfRecord, row: &mut [i8]) -> Result<()> {
    let occupied = file.occupied(record)?;
    let layout = DescriptorLayout::new(DescriptorVersion::V3, file.subshells.len());
    let symmetry = (record.total_two_j, record.parity);
    if let Some(cumulative) = file.generated_cumulative(record)? {
        return write_generated_row(layout, &file.subshells, occupied, cumulative, symmetry, row);
    }
    if occupied.is_empty() {
        ensure!(
            !file.header_lines[1].trim().is_empty(),
            "an empty peel occupation requires a closed core"
        );
        return write_resolved_row(layout, occupied, std::iter::empty(), symmetry, row);
    }
    write_text_row(
        layout,
        &file.subshells,
        occupied,
        file.couplings(record)?,
        symmetry,
        row,
    )
}

pub(crate) fn write_generated_row(
    layout: DescriptorLayout,
    subshells: &[String],
    occupied: &[OccupiedSubshell],
    cumulative: &[u16],
    symmetry: (u16, Parity),
    row: &mut [i8],
) -> Result<()> {
    ensure!(
        occupied.len() == cumulative.len(),
        "incomplete generated coupling chain"
    );
    let states = occupied.iter().zip(cumulative).map(|(shell, &k)| {
        let orbital: Subshell = subshells
            .get(usize::from(shell.subshell_index))
            .context("invalid generated subshell index")?
            .parse()?;
        // The generator omits a state only when its table has a single J=0 entry.
        let j = shell.state.map_or(0, |state| state.two_j);
        let v = physical_seniority(
            u16::from(orbital.capacity() - 1),
            shell.occupation,
            shell.state,
        )?;
        let state = PhysicalState {
            two_j: j,
            seniority: v,
        };
        state.require_single_term(u16::from(orbital.capacity() - 1))?;
        Ok((state, k))
    });
    write_resolved_row(layout, occupied, states, symmetry, row)
}

fn write_resolved_row(
    layout: DescriptorLayout,
    occupied: &[OccupiedSubshell],
    states: impl Iterator<Item = Result<(PhysicalState, u16)>>,
    (total_two_j, parity): (u16, Parity),
    row: &mut [i8],
) -> Result<()> {
    ensure!(
        row.len() == layout.row_len(),
        "row buffer length {} does not match expected {}",
        row.len(),
        layout.row_len()
    );
    let (features, globals) = row.split_at_mut(layout.feature_len());
    let mut entries = occupied.iter().zip(states).peekable();
    let mut previous = 0;
    let mut cumulative_code = 1;
    for (index, slot) in features.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        *slot = empty_slot(cumulative_code);
        if entries
            .peek()
            .is_some_and(|(shell, _)| usize::from(shell.subshell_index) == index)
        {
            let (shell, state) = entries.next().expect("entry was present");
            let (state, k) = state?;
            ensure!(
                valid_coupling(previous, state.two_j, k),
                "invalid coupling at peel index {index}"
            );
            cumulative_code = encode_angular(k, "cumulative 2K")?;
            previous = k;
            *slot = [
                i8::try_from(shell.occupation)?,
                i8::try_from(state.seniority)?,
                encode_angular(state.two_j, "subshell 2J")?,
                cumulative_code,
            ];
        }
    }
    ensure!(
        entries.next().is_none(),
        "occupied orbitals do not follow the descriptor peel table"
    );
    globals[0] = encode_angular(total_two_j, "total 2J")?;
    ensure!(
        globals[0] == cumulative_code,
        "incorrect final angular momentum"
    );
    globals[1] = if parity == Parity::Even { 1 } else { -1 };
    Ok(())
}

#[derive(Clone, Copy)]
struct CouplingStep {
    state: PhysicalState,
    cumulative: u16,
    predecessor: usize,
    alternative_predecessor: Option<usize>,
}

/// Resolve all printed constraints jointly. At each position, paths meeting at
/// the same K have identical possible suffixes, so keep one path and record
/// whether a competing state or predecessor reached it.
pub(crate) fn write_text_row(
    layout: DescriptorLayout,
    subshells: &[String],
    occupied: &[OccupiedSubshell],
    couplings: &[IntermediateCoupling],
    symmetry: (u16, Parity),
    row: &mut [i8],
) -> Result<()> {
    validate_record(subshells, occupied, couplings)?;
    encode_angular(symmetry.0, "total 2J")?;
    let mut steps = Vec::with_capacity(occupied.len() + 1);
    steps.push(CouplingStep {
        state: PhysicalState {
            two_j: 0,
            seniority: 0,
        },
        cumulative: 0,
        predecessor: 0,
        alternative_predecessor: None,
    });
    let mut start = 0;
    let mut parity_exponent = 0usize;
    let mut by_k = Vec::new();
    let mut printed_couplings = couplings.iter().peekable();
    for (position, shell) in occupied.iter().enumerate() {
        let label = &subshells[usize::from(shell.subshell_index)];
        let orbital: Subshell = label.parse()?;
        parity_exponent += usize::from(shell.occupation) * usize::from(orbital.l());
        let candidates = physical_states(u16::from(orbital.capacity() - 1), shell.occupation)
            .with_context(|| format!("invalid state table for subshell {label}"))?;
        let fixed_k = if position + 1 == occupied.len() {
            Some(symmetry.0)
        } else if printed_couplings
            .peek()
            .is_some_and(|next| usize::from(next.boundary) == position + 1)
        {
            Some(
                printed_couplings
                    .next()
                    .expect("printed coupling was present")
                    .two_j,
            )
        } else {
            None
        };
        if let Some(k) = fixed_k {
            encode_angular(k, "cumulative 2K").with_context(|| format!("subshell {label}"))?;
        }
        by_k.fill(usize::MAX);
        let end = steps.len();
        for predecessor in start..end {
            let previous = steps[predecessor].cumulative;
            for &state in candidates {
                if shell
                    .state
                    .is_some_and(|printed| !state.matches_printed(printed))
                {
                    continue;
                }
                let lower = previous.abs_diff(state.two_j);
                let upper = previous + state.two_j;
                let (first, last) = match fixed_k {
                    Some(k) if valid_coupling(previous, state.two_j, k) => (k, k),
                    Some(_) => continue,
                    None => (lower, upper),
                };
                if by_k.len() <= usize::from(last) {
                    by_k.resize(usize::from(last) + 1, usize::MAX);
                }
                for cumulative in (first..=last).step_by(2) {
                    let index = &mut by_k[usize::from(cumulative)];
                    if *index == usize::MAX {
                        *index = steps.len();
                        steps.push(CouplingStep {
                            state,
                            cumulative,
                            predecessor,
                            alternative_predecessor: None,
                        });
                    } else {
                        steps[*index].alternative_predecessor = Some(predecessor);
                    }
                }
            }
        }
        ensure!(
            steps.len() > end,
            "no legal subshell state or coupling at {label}"
        );
        start = end;
    }
    ensure!(
        (symmetry.1 == Parity::Even) == parity_exponent.is_multiple_of(2),
        "incorrect parity"
    );
    let mut path = Vec::with_capacity(occupied.len());
    let mut index = start;
    for (position, _) in occupied.iter().enumerate().rev() {
        let step = steps[index];
        if let Some(alternative) = step.alternative_predecessor {
            let ambiguous_position = position - usize::from(alternative != step.predecessor);
            let label = &subshells[usize::from(occupied[ambiguous_position].subshell_index)];
            bail!(
                "missing subshell state or intermediate coupling leaves multiple chains at {label}"
            );
        }
        let label = &subshells[usize::from(occupied[position].subshell_index)];
        let orbital: Subshell = label.parse()?;
        step.state
            .require_single_term(u16::from(orbital.capacity() - 1))
            .with_context(|| format!("subshell {label}"))?;
        encode_angular(step.cumulative, "cumulative 2K")
            .with_context(|| format!("subshell {label}"))?;
        path.push((step.state, step.cumulative));
        index = step.predecessor;
    }
    write_resolved_row(
        layout,
        occupied,
        path.into_iter().rev().map(Ok),
        symmetry,
        row,
    )
}

pub(crate) fn parity_from_sign(value: i8) -> Result<Parity> {
    match value {
        1 => Ok(Parity::Even),
        -1 => Ok(Parity::Odd),
        other => bail!("invalid descriptor parity {other}"),
    }
}

/// Decode a V3 row into the state and coupling fields GRASP canonically prints.
pub fn decode_v3_into(
    row: &[i8],
    layout: DescriptorLayout,
    subshells: &[String],
    occupied: &mut Vec<OccupiedSubshell>,
    couplings: &mut Vec<IntermediateCoupling>,
) -> Result<(u16, Parity)> {
    ensure!(row.len() == layout.row_len(), "descriptor width mismatch");
    ensure!(
        subshells.len() == layout.subshell_count(),
        "subshell table does not match descriptor layout"
    );
    occupied.clear();
    couplings.clear();
    let mut seen_visible_state = false;
    let mut final_k = 0;
    let mut parity_exponent = 0usize;
    let slots = row[..layout.feature_len()].as_chunks::<4>().0;
    let last_occupied = slots.iter().rposition(|slot| slot[0] != 0);
    for (index, (&slot, label)) in slots.iter().zip(subshells).enumerate() {
        let [occupation, seniority, j_code, k_code] = slot;
        if occupation == 0 {
            ensure!(
                slot == empty_slot(encode_angular(final_k, "cumulative 2K")?),
                "invalid empty slot for {label}"
            );
            continue;
        }
        let orbital: Subshell = label.parse()?;
        let j = decode_angular(j_code, "subshell 2J+1")?;
        let k = decode_angular(k_code, "cumulative 2K+1")?;
        ensure!(valid_coupling(final_k, j, k), "invalid coupling at {label}");
        let occupation = u8::try_from(occupation)?;
        parity_exponent += usize::from(occupation) * usize::from(orbital.l());
        let seniority = u8::try_from(seniority)?;
        let single_two_j = u16::from(orbital.capacity() - 1);
        PhysicalState {
            two_j: j,
            seniority,
        }
        .require_single_term(single_two_j)?;
        let printed = printed_seniority(single_two_j, occupation, j, seniority)?;
        let state = (j != 0 || physical_states(single_two_j, occupation)?.len() != 1).then_some(
            SubshellState {
                two_j: j,
                seniority: printed,
            },
        );
        occupied.push(OccupiedSubshell {
            subshell_index: u16::try_from(index)?,
            occupation,
            state,
        });
        if prints_intermediate_coupling(&mut seen_visible_state, state)
            && Some(index) != last_occupied
        {
            couplings.push(IntermediateCoupling {
                boundary: u16::try_from(occupied.len())?,
                two_j: k,
            });
        }
        final_k = k;
    }
    let total_two_j = decode_angular(row[layout.feature_len()], "total 2J+1")?;
    ensure!(total_two_j == final_k, "incorrect final angular momentum");
    let parity = parity_from_sign(row[layout.feature_len() + 1])?;
    ensure!(
        (parity == Parity::Even) == parity_exponent.is_multiple_of(2),
        "incorrect parity"
    );
    Ok((total_two_j, parity))
}

/// Rebuild a [`CompleteCsfFile`] from decoded V3 rows and their source header.
///
/// `rows` holds every row in the descriptor file's original order. When
/// `indices` is `Some`, only those original row positions are restored, in
/// the given order; block boundaries are still derived correctly because
/// [`CompleteCsfFile::append_generated_record`] opens a new symmetry block
/// whenever `total_two_j`/parity changes between consecutive appended
/// records, which is correct for a full file or an arbitrary subset alike.
pub fn restore_file(
    rows: &[Vec<i8>],
    layout: DescriptorLayout,
    header_lines: [String; 5],
    block_lengths: Option<&[usize]>,
    indices: Option<&[u64]>,
) -> Result<CompleteCsfFile> {
    let header = DatasetHeader::from_lines(header_lines.clone())?;
    let validator = DescriptorRowValidator::new(&header.core, &header.peel)?;
    let mut electron_count = None;
    let subshells = header.peel;
    ensure!(
        subshells.len() == layout.subshell_count(),
        "header peel subshell count {} does not match layout subshell count {}",
        subshells.len(),
        layout.subshell_count()
    );

    let mut file = CompleteCsfFile::new_for_restore(header_lines, subshells.clone());
    let mut occupied_buf = Vec::new();
    let mut coupling_buf = Vec::new();

    let row_indices: Vec<usize> = match indices {
        Some(indices) => indices
            .iter()
            .map(|&index| {
                usize::try_from(index).context("row index exceeds platform address space")
            })
            .collect::<Result<Vec<_>>>()?,
        None => (0..rows.len()).collect(),
    };

    let full_restore = indices.is_none();
    let block_starts = if full_restore {
        let lengths = block_lengths.context("full restoration requires source block_lengths")?;
        ensure!(
            !lengths.is_empty(),
            "source block_lengths must not be empty"
        );
        ensure!(
            lengths
                .iter()
                .try_fold(0usize, |sum, &length| sum.checked_add(length))
                == Some(rows.len()),
            "source block_lengths do not cover all descriptor rows"
        );
        let mut starts = Vec::with_capacity(lengths.len());
        let mut start = 0usize;
        for &length in lengths {
            ensure!(length > 0, "source block_lengths contains an empty block");
            starts.push(start);
            start += length;
        }
        Some(starts)
    } else {
        None
    };

    for (output_position, row_index) in row_indices.into_iter().enumerate() {
        let row = rows
            .get(row_index)
            .with_context(|| format!("row index {row_index} exceeds available rows"))?;
        ensure!(
            row.len() == layout.row_len(),
            "row {row_index} length {} does not match expected {}",
            row.len(),
            layout.row_len()
        );
        validator
            .validate(row, &mut electron_count)
            .with_context(|| format!("invalid descriptor row {row_index}"))?;
        let (total_two_j, parity) = decode_v3_into(
            row,
            layout,
            &subshells,
            &mut occupied_buf,
            &mut coupling_buf,
        )?;
        let force_new_block = block_starts
            .as_ref()
            .is_some_and(|starts| starts.binary_search(&output_position).is_ok());
        file.append_restored_record(
            &occupied_buf,
            &coupling_buf,
            total_two_j,
            parity,
            force_new_block,
        )?;
    }

    ensure!(
        !file.records.is_empty(),
        "restore_file produced no CSF records"
    );
    Ok(file)
}

#[cfg(test)]
mod generated_row_tests {
    use super::*;

    #[test]
    fn generated_row_rejects_terms_that_need_w() {
        let layout = DescriptorLayout::new(DescriptorVersion::V3, 1);
        let orbitals = ["5g".to_owned()];
        for occupation in [4, 6] {
            for two_j in [8, 12] {
                for seniority in [2, 4] {
                    let shell = OccupiedSubshell {
                        subshell_index: 0,
                        occupation,
                        state: Some(SubshellState {
                            two_j,
                            seniority: Some(seniority),
                        }),
                    };
                    let result = write_generated_row(
                        layout,
                        &orbitals,
                        &[shell],
                        &[two_j],
                        (two_j, Parity::Even),
                        &mut [0; 6],
                    );
                    if seniority == 4 {
                        assert!(
                            result
                                .unwrap_err()
                                .to_string()
                                .contains("additional w label")
                        );
                    } else {
                        result.unwrap();
                    }
                }
            }
        }
    }
}
