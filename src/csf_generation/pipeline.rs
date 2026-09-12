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
use crate::descriptor_normalization::{infer_two_j_target, normalize_descriptor_per_csf};

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

/// Statistics from ordering, merging and writing already-generated CSF chunks.
pub struct WriteStats {
    pub record_count: usize,
    pub block_count: usize,
    pub output_bytes: u64,
    pub descriptor_count: Option<usize>,
}

/// Order generated chunks into GRASP's final J/P block order, merge them into
/// one CSF text file, and optionally export a descriptor CSV alongside it.
///
/// `output_path` and (if given) `descriptor_path` must not already exist.
pub fn write_generated_csfs(
    core_subshells: &[Subshell],
    chunks: &[CompleteCsfFile],
    output_path: &Path,
    descriptor_path: Option<&Path>,
    normalize: bool,
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
    let descriptor_subshells = used.iter().map(ToString::to_string).collect::<Vec<_>>();
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

    let mut descriptor_count = None;
    if let Some(path) = descriptor_path {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("descriptor output must be a new file: {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        let mut written = 0usize;
        for block_refs in order.values() {
            for &(chunk_index, block_index) in block_refs {
                let chunk = &chunks[chunk_index];
                let descriptor_indices = chunk
                    .subshells
                    .iter()
                    .map(|shell| {
                        descriptor_subshells
                            .iter()
                            .position(|global| global == shell)
                            .context("descriptor subshell missing from output header")
                    })
                    .collect::<Result<Vec<_>>>()?;
                let block = &chunk.blocks[block_index];
                let start = usize::try_from(block.record_start)?;
                let end = start + usize::try_from(block.record_len)?;
                for record in &chunk.records[start..end] {
                    let local = chunk.descriptor_for(record)?;
                    let mut descriptor = vec![0; descriptor_subshells.len() * 3];
                    for (local_index, &global_index) in descriptor_indices.iter().enumerate() {
                        descriptor[global_index * 3..global_index * 3 + 3]
                            .copy_from_slice(&local[local_index * 3..local_index * 3 + 3]);
                    }
                    if normalize {
                        let values = normalize_descriptor_per_csf(
                            &descriptor,
                            &descriptor_subshells,
                            infer_two_j_target(&descriptor),
                        )?;
                        writeln!(
                            writer,
                            "{}",
                            values
                                .iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join(",")
                        )?;
                    } else {
                        writeln!(
                            writer,
                            "{}",
                            descriptor
                                .iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join(",")
                        )?;
                    }
                    written += 1;
                }
            }
        }
        writer.flush()?;
        let metadata_path = path.with_extension("toml");
        let metadata = format!(
            "format_version = 1\nencoding = \"csv\"\nnormalized = {}\nrecord_count = {}\nsubshells = {:?}\n",
            normalize, count, descriptor_subshells
        );
        let mut metadata_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&metadata_path)
            .with_context(|| {
                format!(
                    "descriptor metadata must be a new file: {}",
                    metadata_path.display()
                )
            })?;
        metadata_file.write_all(metadata.as_bytes())?;
        descriptor_count = Some(written);
    }

    Ok(WriteStats {
        record_count: count,
        block_count: order.len(),
        output_bytes: fs::metadata(output_path)?.len(),
        descriptor_count,
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
    descriptor_path: Option<&Path>,
    normalize: bool,
) -> Result<TranscriptGenerationStats> {
    let request = ExcitationRequest::from_transcript(transcript)?;
    let occupations = enumerate_occupations(&request)?;
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
    let write_stats = write_generated_csfs(
        &occupations.core_subshells,
        &chunks,
        output_path,
        descriptor_path,
        normalize,
    )?;
    Ok(TranscriptGenerationStats {
        unique_occupations: occupations.configurations.len(),
        record_count: write_stats.record_count,
        block_count: write_stats.block_count,
        output_bytes: write_stats.output_bytes,
        descriptor_count: write_stats.descriptor_count,
    })
}
