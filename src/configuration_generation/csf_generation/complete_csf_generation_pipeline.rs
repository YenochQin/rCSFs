//! Ordering, merging and text/descriptor export for already-generated CSF
//! chunks, plus the transcript-to-file convenience entry point used by both
//! the `generate_transcript_csfs` example and the `rcsfs.generate_csfs_from_transcript`
//! PyO3 binding. Keeping this logic in one place means the example and the
//! Python binding cannot silently drift apart.

use anyhow::{Context, Result, ensure};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

use super::{
    EnumeratedOccupations, ExcitationRequest, GenerationRequest, Subshell, enumerate_occupations,
    generate_csfs_parallel,
};
use crate::complete_csf::{CompleteCsfFile, CsfRecord, Parity};
use crate::descriptor_dataset::{DatasetHeader, DatasetWriter, header_path};
use crate::descriptor_operations::{common_header, mapping, remap};
use crate::descriptor_schema::DescriptorLayout;
use crate::descriptor_schema::DescriptorVersion;
use crate::descriptor_v3::{encode_v3, write_generated_row};

fn header(labels: impl IntoIterator<Item = String>) -> String {
    labels
        .into_iter()
        .map(|label| {
            let display = if label.ends_with('-') {
                label
            } else {
                format!("{label} ")
            };
            format!("{display:>5}")
        })
        .collect::<String>()
        .trim_end()
        .to_owned()
}

/// Choose the same ordered orbital table for memory and disk generation and
/// check the common electron count before either path emits descriptor rows.
pub(super) fn enumerated_descriptor_basis(
    occupations: &EnumeratedOccupations,
) -> Result<(Vec<Subshell>, usize)> {
    let core_electrons = occupations
        .core_subshells
        .iter()
        .map(|shell| usize::from(shell.capacity()))
        .sum::<usize>();
    let mut electron_count = None;
    let mut used = HashSet::new();
    for configuration in &occupations.configurations {
        let mut electrons = core_electrons;
        for occupation in &configuration.occupations {
            if occupation.electrons > 0 {
                used.insert(occupation.subshell);
                electrons += usize::from(occupation.electrons);
            }
        }
        ensure!(
            electron_count.is_none_or(|expected| expected == electrons),
            "generation configurations have different electron counts"
        );
        electron_count = Some(electrons);
    }
    let electron_count =
        electron_count.context("occupation enumeration produced no configurations")?;
    ensure!(electron_count > 0, "empty electron basis");
    let mut peel = used.into_iter().collect::<Vec<_>>();
    peel.sort_by_key(|shell| (shell.n(), shell.l(), shell.kappa() < 0));
    Ok((peel, electron_count))
}

#[derive(Debug, serde::Serialize)]
pub struct GenerationBlockStats {
    pub total_two_j: u16,
    pub parity: &'static str,
    pub record_count: usize,
}

fn memory_j_value_summary(chunks: &[CompleteCsfFile]) -> Vec<GenerationBlockStats> {
    let mut blocks = BTreeMap::<(u16, bool), usize>::new();
    for chunk in chunks {
        for block in &chunk.blocks {
            *blocks
                .entry((block.total_two_j, block.parity == Parity::Odd))
                .or_insert(0) += usize::try_from(block.record_len).unwrap_or(0);
        }
    }

    crate::reporting::detail!("\nCSFs per J value:");
    for (&(total_two_j, odd), count) in &blocks {
        let parity = if odd { "odd" } else { "even" };
        let j_value = f64::from(total_two_j) / 2.0;
        crate::reporting::detail!("  J = {:4.1} ({:>4}): {:10} CSFs", j_value, parity, count);
    }
    blocks
        .into_iter()
        .map(|((total_two_j, odd), record_count)| GenerationBlockStats {
            total_two_j,
            parity: if odd { "-" } else { "+" },
            record_count,
        })
        .collect()
}

/// Statistics from ordering, merging and writing already-generated CSF chunks.
pub struct WriteStats {
    pub record_count: usize,
    pub block_count: usize,
    pub output_bytes: u64,
    pub descriptor_count: Option<usize>,
}

/// Order generated chunks into GRASP's final J/P block order and publish a
/// descriptor/header dataset. Text export is a separate explicit operation.
///
/// `output_path` must not already exist.
pub fn write_generated_csfs(
    core_subshells: &[Subshell],
    chunks: &[CompleteCsfFile],
    output_path: &Path,
) -> Result<WriteStats> {
    write_generated_csfs_with_peel(core_subshells, chunks, output_path, None, encode_v3)
}

fn write_generated_csfs_with_peel(
    core_subshells: &[Subshell],
    chunks: &[CompleteCsfFile],
    output_path: &Path,
    peel: Option<Vec<Subshell>>,
    encode: impl Fn(&CompleteCsfFile, &CsfRecord, &mut [i8]) -> Result<()>,
) -> Result<WriteStats> {
    let count = chunks.iter().try_fold(0usize, |count, chunk| {
        count
            .checked_add(chunk.records.len())
            .context("record count overflow")
    })?;
    ensure!(count > 0, "no CSFs generated");
    let mut order = BTreeMap::<(u16, bool), Vec<(usize, usize)>>::new();
    let mut used = HashSet::<Subshell>::new();
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        for (block_index, block) in chunk.blocks.iter().enumerate() {
            order
                .entry((block.total_two_j, block.parity == Parity::Odd))
                .or_default()
                .push((chunk_index, block_index));
        }
        if peel.is_none() && !chunk.records.is_empty() {
            for label in &chunk.subshells {
                used.insert(label.parse()?);
            }
        }
    }
    let peel = peel.unwrap_or_else(|| {
        let mut used = used.into_iter().collect::<Vec<_>>();
        used.sort_by_key(|shell| (shell.n(), shell.l(), shell.kappa() < 0));
        used
    });
    let headers = [
        "Core subshells:".to_owned(),
        header(core_subshells.iter().map(ToString::to_string)),
        "Peel subshells:".to_owned(),
        header(peel.iter().map(ToString::to_string)),
        "CSF(s):".to_owned(),
    ];
    let mut destination = DatasetHeader::from_lines(headers)?;
    let sorted_indices = destination
        .peel
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect::<HashMap<_, _>>();
    let populated = chunks.iter().filter(|chunk| !chunk.records.is_empty());
    if populated.clone().any(|chunk| {
        chunk
            .subshells
            .windows(2)
            .any(|pair| sorted_indices[pair[0].as_str()] >= sorted_indices[pair[1].as_str()])
    }) {
        let tables = populated
            .map(|chunk| {
                let mut lines = destination.lines.clone();
                lines[3] = header(chunk.subshells.iter().cloned());
                DatasetHeader::from_lines(lines)
            })
            .collect::<Result<Vec<_>>>()?;
        destination = common_header(&tables)?;
    }
    let header_output = header_path(output_path)?;
    let mut writer = DatasetWriter::new(destination.clone(), output_path, &header_output, &[])?;
    for (index, block_refs) in order.values().enumerate() {
        for &(chunk_index, block_index) in block_refs {
            let chunk = &chunks[chunk_index];
            let block = &chunk.blocks[block_index];
            let start = usize::try_from(block.record_start)?;
            let end = start
                .checked_add(usize::try_from(block.record_len)?)
                .context("block range overflow")?;
            let layout = DescriptorLayout::new(DescriptorVersion::V3, chunk.subshells.len());
            let mut row = vec![0; layout.row_len()];
            let indices = mapping(&chunk.subshells, &destination.peel)?;
            for record in &chunk.records[start..end] {
                encode(chunk, record, &mut row)?;
                writer.push(
                    &remap(&row, &indices, destination.layout().row_len()),
                    index,
                )?;
            }
        }
    }
    writer.finish()?;

    Ok(WriteStats {
        record_count: count,
        block_count: order.len(),
        output_bytes: fs::metadata(output_path)?.len(),
        descriptor_count: Some(count),
    })
}

/// Statistics for a full transcript-driven generation run.
pub struct TranscriptGenerationStats {
    pub unique_occupations: usize,
    pub record_count: usize,
    pub block_count: usize,
    pub output_bytes: u64,
    pub descriptor_count: Option<usize>,
    pub blocks: Vec<GenerationBlockStats>,
}

/// Parse an `rcsfgenerate.log`-format transcript, enumerate and generate the
/// resulting CSFs, and write them (and optionally their descriptors) to disk.
///
/// `transcript` is consumed purely in memory: this is the single entry point
/// backing `rcsfs.generate_csfs_from_transcript`, so an interactive caller
/// never needs to write its collected answers to a transcript file on disk.
pub fn generate_csfs_from_transcript(
    transcript: &str,
    output_path: &Path,

    threads: Option<usize>,
) -> Result<TranscriptGenerationStats> {
    crate::reporting::detail!("Parsing transcript and enumerating configurations...");
    let request = ExcitationRequest::from_transcript(transcript)?;
    let occupations = enumerate_occupations(&request)?;
    let (peel, _) = enumerated_descriptor_basis(&occupations)?;
    crate::reporting::detail!(
        "Enumerated {} unique occupation configurations",
        occupations.configurations.len()
    );

    crate::reporting::detail!("Generating CSFs in memory...");
    let requests = occupations
        .configurations
        .iter()
        .map(|configuration| GenerationRequest {
            core_subshells: occupations.core_subshells.clone(),
            configuration: configuration.occupations.clone(),
            min_two_j: request.min_two_j,
            max_two_j: request.max_two_j,
        })
        .collect::<Vec<_>>();
    let chunks = generate_csfs_parallel(&requests, threads)?;

    crate::reporting::detail!("Writing descriptor/header dataset...");
    let write_stats = write_generated_csfs_with_peel(
        &occupations.core_subshells,
        &chunks,
        output_path,
        Some(peel),
        // These chunks are private to this run and still have the original dense chain.
        |chunk, record, row| {
            write_generated_row(
                DescriptorLayout::new(DescriptorVersion::V3, chunk.subshells.len()),
                &chunk.subshells,
                chunk.occupied(record)?,
                chunk.generated_cumulative(record)?.unwrap_or_default(),
                (record.total_two_j, record.parity),
                row,
            )
        },
    )?;
    crate::reporting::detail!(
        "Generated {} CSFs across {} symmetry blocks",
        write_stats.record_count,
        write_stats.block_count
    );
    let blocks = memory_j_value_summary(&chunks);

    Ok(TranscriptGenerationStats {
        unique_occupations: occupations.configurations.len(),
        record_count: write_stats.record_count,
        block_count: write_stats.block_count,
        output_bytes: write_stats.output_bytes,
        descriptor_count: write_stats.descriptor_count,
        blocks,
    })
}
