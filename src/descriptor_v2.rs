//! V2 descriptor encode/decode and CSF restoration.
//!
//! Unlike V1's dense triplet, V2 stores every printed value verbatim (using
//! [`descriptor_schema::MISSING`] for anything GRASP did not print) so
//! `decode_v2_into` is the exact inverse of `encode_v2`: no line2 back-fill, no
//! folding the total `2J` into the last occupied subshell, and seniority is
//! never discarded.

use anyhow::{Context, Result, ensure};

use crate::complete_csf::{
    CompleteCsfFile, CsfRecord, IntermediateCoupling, OccupiedSubshell, Parity, SubshellState,
};
use crate::csfs_descriptor::parse_peel_subshells_from_header_lines;
use crate::descriptor_schema::{DescriptorLayout, DescriptorVersion, MISSING, validate_record};

/// Encode one already-validated CSF record into a caller-owned V2 row buffer.
///
/// `row.len()` must equal `layout.row_len()`. Validates the record through
/// [`validate_record`] first, so an encoded row is always a legal V2 record.
pub fn encode_v2(file: &CompleteCsfFile, record: &CsfRecord, row: &mut [i32]) -> Result<()> {
    let occupied = file.occupied(record)?;
    let couplings = file.couplings(record)?;
    validate_record(&file.subshells, occupied, couplings)?;

    let layout = DescriptorLayout::new(DescriptorVersion::V2, file.subshells.len());
    write_feature_row(
        layout,
        occupied,
        couplings,
        record.total_two_j,
        record.parity,
        row,
    )
}

/// Write occupied/coupling data into a caller-owned V2 row buffer.
///
/// Shared by [`encode_v2`] (integer-record producer) and
/// `csfs_descriptor::CSFDescriptorGenerator::parse_csf_v2_into` (text
/// producer) so the two producers cannot silently diverge on how a record's
/// fields land in the row (design doc §5.1/plan D9). Callers must validate
/// `occupied`/`couplings` themselves first.
pub(crate) fn write_feature_row(
    layout: DescriptorLayout,
    occupied: &[OccupiedSubshell],
    couplings: &[IntermediateCoupling],
    total_two_j: u16,
    parity: Parity,
    row: &mut [i32],
) -> Result<()> {
    ensure!(
        row.len() == layout.row_len(),
        "row buffer length {} does not match expected {}",
        row.len(),
        layout.row_len()
    );
    row.fill(MISSING);
    for index in 0..layout.subshell_count() {
        row[layout.slot(index) + layout.n_offset()] = 0;
    }

    for (position, shell) in occupied.iter().enumerate() {
        let index = usize::from(shell.subshell_index);
        let base = layout.slot(index);
        row[base + layout.n_offset()] = i32::from(shell.occupation);
        if let Some(state) = shell.state {
            row[base + layout.two_j_offset()] = i32::from(state.two_j);
            if let Some(seniority) = state.seniority {
                row[base + layout.seniority_offset()] = i32::from(seniority);
            }
        }
        // Boundary == 1-based occupied position (design doc §3.1). Position 0
        // and the last position always resolve to a boundary the writer never
        // prints (`2 <= boundary < field_count`), so their 2K slot stays MISSING.
        let boundary = u16::try_from(position + 1)?;
        if let Some(coupling) = couplings.iter().find(|value| value.boundary == boundary) {
            row[base + layout.two_k_offset()] = i32::from(coupling.two_j);
        }
    }

    let total_index = layout
        .total_two_j_index()
        .expect("V2 layout always carries total_two_j");
    let parity_index = layout
        .parity_index()
        .expect("V2 layout always carries parity");
    row[total_index] = i32::from(total_two_j);
    row[parity_index] = parity_sign(parity);
    Ok(())
}

fn parity_sign(parity: Parity) -> i32 {
    match parity {
        Parity::Even => 1,
        Parity::Odd => -1,
    }
}

fn parity_from_sign(value: i32) -> Result<Parity> {
    match value {
        1 => Ok(Parity::Even),
        -1 => Ok(Parity::Odd),
        other => Err(anyhow::anyhow!("invalid decoded parity value {other}")),
    }
}

/// Decode one V2 row into caller-owned buffers, avoiding any per-row heap
/// allocation beyond what `occupied`/`couplings` already reuse across calls.
///
/// Buffers are cleared and refilled in occupied-subshell order. The returned
/// `(u16, Parity)` pair is the record's total `2J` and parity, read verbatim
/// from the row's global columns.
pub fn decode_v2_into(
    row: &[i32],
    layout: DescriptorLayout,
    occupied: &mut Vec<OccupiedSubshell>,
    couplings: &mut Vec<IntermediateCoupling>,
) -> Result<(u16, Parity)> {
    ensure!(
        layout.version() == DescriptorVersion::V2,
        "decode_v2_into requires a V2 layout"
    );
    ensure!(
        row.len() == layout.row_len(),
        "row length {} does not match expected {}",
        row.len(),
        layout.row_len()
    );
    occupied.clear();
    couplings.clear();

    let mut occupied_position = 0u16;
    for index in 0..layout.subshell_count() {
        let base = layout.slot(index);
        let n_raw = row[base + layout.n_offset()];
        let two_j_raw = row[base + layout.two_j_offset()];
        let seniority_raw = row[base + layout.seniority_offset()];
        let two_k_raw = row[base + layout.two_k_offset()];
        if n_raw == 0 {
            ensure!(
                two_j_raw == MISSING && seniority_raw == MISSING && two_k_raw == MISSING,
                "unoccupied subshell {index} must be encoded as [0, -1, -1, -1]"
            );
            continue;
        }
        ensure!(n_raw > 0, "decoded occupation {n_raw} must be positive");
        let occupation = u8::try_from(n_raw).context("decoded occupation exceeds one byte")?;

        let state = if two_j_raw == MISSING {
            ensure!(
                seniority_raw == MISSING,
                "subshell {index} has seniority without a printed 2J"
            );
            None
        } else {
            let two_j = u16::try_from(two_j_raw).context("decoded 2J is negative")?;
            let seniority = if seniority_raw == MISSING {
                None
            } else {
                Some(u8::try_from(seniority_raw).context("decoded seniority is negative")?)
            };
            Some(SubshellState { two_j, seniority })
        };

        occupied.push(OccupiedSubshell {
            subshell_index: u16::try_from(index)?,
            occupation,
            state,
        });
        occupied_position += 1;

        if two_k_raw != MISSING {
            couplings.push(IntermediateCoupling {
                boundary: occupied_position,
                two_j: u16::try_from(two_k_raw).context("decoded 2K is negative")?,
            });
        }
    }

    let total_index = layout
        .total_two_j_index()
        .expect("V2 layout always carries total_two_j");
    let parity_index = layout
        .parity_index()
        .expect("V2 layout always carries parity");
    let total_raw = row[total_index];
    ensure!(total_raw != MISSING, "row is missing total_two_j");
    let total_two_j = u16::try_from(total_raw).context("decoded total 2J is negative")?;
    let parity = parity_from_sign(row[parity_index])?;

    Ok((total_two_j, parity))
}

/// Rebuild a [`CompleteCsfFile`] from decoded V2 rows and their source header.
///
/// `rows` holds every row in the descriptor file's original order. When
/// `indices` is `Some`, only those original row positions are restored, in
/// the given order; block boundaries are still derived correctly because
/// [`CompleteCsfFile::append_generated_record`] opens a new symmetry block
/// whenever `total_two_j`/parity changes between consecutive appended
/// records, which is correct for a full file or an arbitrary subset alike.
pub fn restore_file(
    rows: &[Vec<i32>],
    layout: DescriptorLayout,
    header_lines: [String; 5],
    block_lengths: Option<&[usize]>,
    indices: Option<&[u64]>,
) -> Result<CompleteCsfFile> {
    ensure!(
        layout.version() == DescriptorVersion::V2,
        "restore_file only supports V2 layouts"
    );
    let subshells = parse_peel_subshells_from_header_lines(&header_lines)?;
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
        let (total_two_j, parity) =
            decode_v2_into(row, layout, &mut occupied_buf, &mut coupling_buf)?;
        validate_record(&subshells, &occupied_buf, &coupling_buf)
            .with_context(|| format!("row {row_index} decoded to an invalid CSF record"))?;
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
mod tests {
    use super::*;
    use crate::csf_generation::{GenerationRequest, SubshellOccupation, generate_csfs};
    use std::io::Cursor;

    fn layout_for(file: &CompleteCsfFile) -> DescriptorLayout {
        DescriptorLayout::new(DescriptorVersion::V2, file.subshells.len())
    }

    #[test]
    fn encode_then_decode_round_trips_a_generated_record() {
        let request = GenerationRequest {
            core_subshells: Vec::new(),
            configuration: vec![SubshellOccupation {
                subshell: "4f".parse().unwrap(),
                electrons: 4,
            }],
            min_two_j: 4,
            max_two_j: 4,
        };
        let generated = generate_csfs(&request).unwrap();
        let layout = layout_for(&generated);

        for record in &generated.records {
            let mut row = vec![0i32; layout.row_len()];
            encode_v2(&generated, record, &mut row).unwrap();

            let mut occupied = Vec::new();
            let mut couplings = Vec::new();
            let (total_two_j, parity) =
                decode_v2_into(&row, layout, &mut occupied, &mut couplings).unwrap();

            assert_eq!(total_two_j, record.total_two_j);
            assert_eq!(parity, record.parity);
            assert_eq!(occupied.as_slice(), generated.occupied(record).unwrap());
            assert_eq!(couplings.as_slice(), generated.couplings(record).unwrap());
        }
    }

    #[test]
    fn printed_zero_differs_from_missing() {
        // Two 4f(4) states at 2J=4 have seniority 2 and 4 (design doc §2.2
        // collision sample); this asserts encode distinguishes an occupied
        // slot's absent state (MISSING) from an explicit printed value.
        let request = GenerationRequest {
            core_subshells: Vec::new(),
            configuration: vec![SubshellOccupation {
                subshell: "2p".parse().unwrap(),
                electrons: 1,
            }],
            min_two_j: 3,
            max_two_j: 3,
        };
        let generated = generate_csfs(&request).unwrap();
        let layout = layout_for(&generated);
        let record = &generated.records[0];
        let mut row = vec![0i32; layout.row_len()];
        encode_v2(&generated, record, &mut row).unwrap();

        // Single occupied subshell: n printed, 2J printed, seniority never
        // printed (single-electron state has no seniority label), 2K missing
        // (position 0 is also the last position here).
        assert_eq!(row[layout.slot(0) + layout.n_offset()], 1);
        assert_eq!(row[layout.slot(0) + layout.two_j_offset()], 3);
        assert_eq!(row[layout.slot(0) + layout.seniority_offset()], MISSING);
        assert_eq!(row[layout.slot(0) + layout.two_k_offset()], MISSING);
    }

    #[test]
    fn empty_slot_uses_zero_occupation_and_rejects_other_values() {
        let layout = DescriptorLayout::new(DescriptorVersion::V2, 2);
        let mut row = vec![MISSING; layout.row_len()];
        row[layout.slot(0) + layout.n_offset()] = 0;
        row[layout.slot(1) + layout.n_offset()] = 1;
        row[layout.slot(1) + layout.two_j_offset()] = 1;
        row[layout.total_two_j_index().unwrap()] = 1;
        row[layout.parity_index().unwrap()] = 1;

        let mut occupied = Vec::new();
        let mut couplings = Vec::new();
        decode_v2_into(&row, layout, &mut occupied, &mut couplings).unwrap();
        assert_eq!(occupied.len(), 1);
        assert_eq!(occupied[0].subshell_index, 1);

        row[layout.slot(0) + layout.two_j_offset()] = 3;
        assert!(decode_v2_into(&row, layout, &mut occupied, &mut couplings).is_err());
    }

    #[test]
    fn seniority_collision_distinguishes_4f4_states() {
        let request = GenerationRequest {
            core_subshells: Vec::new(),
            configuration: vec![SubshellOccupation {
                subshell: "4f".parse().unwrap(),
                electrons: 4,
            }],
            min_two_j: 4,
            max_two_j: 4,
        };
        let generated = generate_csfs(&request).unwrap();
        assert_eq!(generated.records.len(), 2);
        let layout = layout_for(&generated);

        let mut rows = Vec::new();
        for record in &generated.records {
            let mut row = vec![0i32; layout.row_len()];
            encode_v2(&generated, record, &mut row).unwrap();
            rows.push(row);
        }

        let seniority_offset = layout.slot(0) + layout.seniority_offset();
        assert_ne!(rows[0][seniority_offset], rows[1][seniority_offset]);
        assert_eq!(
            rows.iter().map(|r| r[seniority_offset]).collect::<Vec<_>>(),
            [2, 4]
        );

        for row in &rows {
            let mut occupied = Vec::new();
            let mut couplings = Vec::new();
            decode_v2_into(row, layout, &mut occupied, &mut couplings).unwrap();
            assert_eq!(occupied.len(), 1);
        }
        // Decoded records remain distinguishable: different rows decode to
        // different seniority values on the same occupied subshell.
        let mut occupied_a = Vec::new();
        let mut couplings_a = Vec::new();
        decode_v2_into(&rows[0], layout, &mut occupied_a, &mut couplings_a).unwrap();
        let mut occupied_b = Vec::new();
        let mut couplings_b = Vec::new();
        decode_v2_into(&rows[1], layout, &mut occupied_b, &mut couplings_b).unwrap();
        assert_ne!(occupied_a[0].state, occupied_b[0].state);
    }

    /// Build a multi-block file whose records occupy only a subset of a
    /// shared, wider peel subshell list (an "empty orbital position" per
    /// record), by merging two differently-shaped generation requests
    /// through the same union path production code uses
    /// (`write_generated_csfs`). Physically legal by construction, unlike
    /// `tests/fixtures/complete.csf`, which predates [`validate_record`] and
    /// contains a hand-written seniority/`2J` combination outside GRASP's
    /// real state tables (only ever exercised for byte-level text parsing).
    fn merged_multiblock_fixture() -> CompleteCsfFile {
        use crate::csf_generation::write_generated_csfs;
        use std::sync::atomic::{AtomicU64, Ordering};

        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let request_a = GenerationRequest {
            core_subshells: Vec::new(),
            configuration: vec![
                SubshellOccupation {
                    subshell: "2s".parse().unwrap(),
                    electrons: 2,
                },
                SubshellOccupation {
                    subshell: "2p-".parse().unwrap(),
                    electrons: 1,
                },
                SubshellOccupation {
                    subshell: "2p".parse().unwrap(),
                    electrons: 1,
                },
            ],
            min_two_j: 0,
            max_two_j: 4,
        };
        let request_b = GenerationRequest {
            core_subshells: Vec::new(),
            configuration: vec![
                SubshellOccupation {
                    subshell: "2s".parse().unwrap(),
                    electrons: 2,
                },
                SubshellOccupation {
                    subshell: "3s".parse().unwrap(),
                    electrons: 1,
                },
            ],
            min_two_j: 1,
            max_two_j: 1,
        };
        let chunks = vec![
            generate_csfs(&request_a).unwrap(),
            generate_csfs(&request_b).unwrap(),
        ];

        let path = std::env::temp_dir().join(format!(
            "descriptor_v2_test_{}_{}.csf",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        write_generated_csfs(&[], &chunks, &path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        CompleteCsfFile::parse_reader(Cursor::new(text)).unwrap()
    }

    #[test]
    fn roundtrip_is_byte_exact_for_fixture() {
        let mut parsed = merged_multiblock_fixture();
        let original_block = parsed.blocks[0];
        parsed.records.insert(1, parsed.records[0]);
        for block in &mut parsed.blocks[1..] {
            block.record_start += 1;
        }
        parsed.blocks.insert(
            1,
            crate::complete_csf::SymmetryBlock {
                record_start: 1,
                record_len: 1,
                total_two_j: original_block.total_two_j,
                parity: original_block.parity,
            },
        );
        assert!(
            parsed.blocks.len() >= 2,
            "fixture should span multiple blocks"
        );
        assert!(
            parsed
                .records
                .iter()
                .any(|record| parsed.occupied(record).unwrap().len() < parsed.subshells.len()),
            "fixture should include a record with an empty orbital position"
        );
        let layout = layout_for(&parsed);

        let mut rows = Vec::new();
        for record in &parsed.records {
            let mut row = vec![0i32; layout.row_len()];
            encode_v2(&parsed, record, &mut row).unwrap();
            rows.push(row);
        }
        assert!(rows.iter().any(|row| {
            (0..layout.subshell_count())
                .any(|index| row[layout.slot(index) + layout.n_offset()] == 0)
        }));
        assert!(rows.iter().all(|row| {
            (0..layout.subshell_count())
                .all(|index| row[layout.slot(index) + layout.n_offset()] >= 0)
        }));

        let header_lines = parsed.header_lines.clone();
        let block_lengths: Vec<usize> = parsed
            .blocks
            .iter()
            .map(|block| usize::try_from(block.record_len).unwrap())
            .collect();
        let restored =
            restore_file(&rows, layout, header_lines, Some(&block_lengths), None).unwrap();

        let mut original_bytes = Vec::new();
        parsed.write_to(&mut original_bytes).unwrap();
        let mut restored_bytes = Vec::new();
        restored.write_to(&mut restored_bytes).unwrap();
        assert_eq!(original_bytes, restored_bytes);
    }

    #[test]
    fn subset_restore_matches_source_selection() {
        let parsed = merged_multiblock_fixture();
        let layout = layout_for(&parsed);

        let mut rows = Vec::new();
        for record in &parsed.records {
            let mut row = vec![0i32; layout.row_len()];
            encode_v2(&parsed, record, &mut row).unwrap();
            rows.push(row);
        }

        let last = parsed.records.len() - 1;
        let indices = [last as u64];
        let restored = restore_file(
            &rows,
            layout,
            parsed.header_lines.clone(),
            None,
            Some(&indices),
        )
        .unwrap();
        assert_eq!(restored.records.len(), 1);

        let mut expected_bytes = Vec::new();
        parsed
            .write_record_to(&parsed.records[last], &mut expected_bytes)
            .unwrap();
        let mut restored_bytes = Vec::new();
        restored
            .write_record_to(&restored.records[0], &mut restored_bytes)
            .unwrap();
        assert_eq!(expected_bytes, restored_bytes);
    }

    #[test]
    fn decode_v2_into_rejects_non_v2_layout() {
        let layout = DescriptorLayout::new(DescriptorVersion::V1, 1);
        let row = vec![0i32; layout.row_len()];
        let mut occupied = Vec::new();
        let mut couplings = Vec::new();
        assert!(decode_v2_into(&row, layout, &mut occupied, &mut couplings).is_err());
    }

    #[test]
    fn decode_v2_into_rejects_missing_total_two_j() {
        let layout = DescriptorLayout::new(DescriptorVersion::V2, 1);
        let mut row = vec![MISSING; layout.row_len()];
        row[layout.slot(0) + layout.n_offset()] = 1;
        row[layout.parity_index().unwrap()] = 1;
        let mut occupied = Vec::new();
        let mut couplings = Vec::new();
        let error = decode_v2_into(&row, layout, &mut occupied, &mut couplings).unwrap_err();
        assert!(error.to_string().contains("missing total_two_j"));
    }
}
