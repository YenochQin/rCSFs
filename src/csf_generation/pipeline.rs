//! Ordering, merging and text/descriptor export for already-generated CSF
//! chunks, plus the transcript-to-file convenience entry point used by both
//! the `generate_transcript_csfs` example and the `rcsfs.generate_csfs_from_transcript`
//! PyO3 binding. Keeping this logic in one place means the example and the
//! Python binding cannot silently drift apart.

use anyhow::{Context, Result, ensure};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use super::{
    ExcitationRequest, GenerationRequest, Subshell, enumerate_occupations, generate_csfs_parallel,
};
use crate::complete_csf::{CompleteCsfFile, Parity};

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

fn print_memory_j_value_summary(chunks: &[CompleteCsfFile]) {
    let mut blocks = BTreeMap::<(u16, bool), usize>::new();
    for chunk in chunks {
        for block in &chunk.blocks {
            *blocks
                .entry((block.total_two_j, block.parity == Parity::Odd))
                .or_insert(0) += usize::try_from(block.record_len).unwrap_or(0);
        }
    }

    eprintln!("\nCSFs per J value:");
    for ((total_two_j, odd), count) in blocks {
        let parity = if odd { "odd" } else { "even" };
        let j_value = f64::from(total_two_j) / 2.0;
        eprintln!("  J = {:4.1} ({:>4}): {:10} CSFs", j_value, parity, count);
    }
    eprintln!();
}

/// Statistics from ordering, merging and writing already-generated CSF chunks.
pub struct WriteStats {
    pub record_count: usize,
    pub block_count: usize,
    pub output_bytes: u64,
    pub descriptor_count: Option<usize>,
}

/// Order generated chunks into GRASP's final J/P block order, merge them into
/// one CSF text file. Descriptor Parquet is produced by the Python pipeline.
///
/// `output_path` must not already exist.
pub fn write_generated_csfs(
    core_subshells: &[Subshell],
    chunks: &[CompleteCsfFile],
    output_path: &Path,
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
        if !chunk.records.is_empty() {
            for label in &chunk.subshells {
                used.insert(label.parse()?);
            }
        }
    }
    let mut used = used.into_iter().collect::<Vec<_>>();
    used.sort_by_key(|shell| (shell.n(), shell.l(), shell.kappa() < 0));
    let headers = [
        "Core subshells:".to_owned(),
        header(core_subshells.iter().map(ToString::to_string)),
        "Peel subshells:".to_owned(),
        header(used.iter().map(ToString::to_string)),
        "CSF(s):".to_owned(),
    ];
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)?;
    let mut writer = BufWriter::new(file);
    for line in headers {
        writeln!(writer, "{line}")?;
    }
    for (index, block_refs) in order.values().enumerate() {
        if index > 0 {
            writeln!(writer, " *")?;
        }
        for &(chunk_index, block_index) in block_refs {
            let chunk = &chunks[chunk_index];
            let block = &chunk.blocks[block_index];
            let start = usize::try_from(block.record_start)?;
            let end = start
                .checked_add(usize::try_from(block.record_len)?)
                .context("block range overflow")?;
            for record in &chunk.records[start..end] {
                chunk.write_record_to(record, &mut writer)?;
            }
        }
    }
    writer.flush()?;
    drop(writer);

    Ok(WriteStats {
        record_count: count,
        block_count: order.len(),
        output_bytes: fs::metadata(output_path)?.len(),
        descriptor_count: None,
    })
}

/// Statistics for a full transcript-driven generation run.
pub struct TranscriptGenerationStats {
    pub unique_occupations: usize,
    pub record_count: usize,
    pub block_count: usize,
    pub output_bytes: u64,
    pub descriptor_count: Option<usize>,
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
    eprintln!("Parsing transcript and enumerating configurations...");
    let request = ExcitationRequest::from_transcript(transcript)?;
    let occupations = enumerate_occupations(&request)?;
    eprintln!(
        "Enumerated {} unique occupation configurations",
        occupations.configurations.len()
    );

    eprintln!("Generating CSFs in memory...");
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

    eprintln!("Writing CSF text file...");
    let write_stats = write_generated_csfs(&occupations.core_subshells, &chunks, output_path)?;
    eprintln!(
        "Generated {} CSFs across {} symmetry blocks",
        write_stats.record_count, write_stats.block_count
    );
    print_memory_j_value_summary(&chunks);

    Ok(TranscriptGenerationStats {
        unique_occupations: occupations.configurations.len(),
        record_count: write_stats.record_count,
        block_count: write_stats.block_count,
        output_bytes: write_stats.output_bytes,
        descriptor_count: write_stats.descriptor_count,
    })
}
