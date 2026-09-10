//! Generate CSFs from an rcsfgenerate transcript using the Rust generator.
use _rcsfs::complete_csf::CompleteCsfFile;
use _rcsfs::csf_generation::{
    ExcitationRequest, GenerationRequest, Subshell, enumerate_occupations, generate_csfs_parallel,
};
use _rcsfs::descriptor_normalization::{infer_two_j_target, normalize_descriptor_per_csf};
use anyhow::{Context, Result, ensure};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

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

fn main() -> Result<()> {
    let total = Instant::now();
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(
        args.next()
            .context("usage: generate_transcript_csfs INPUT OUTPUT MAX_RECORDS")?,
    );
    let output = PathBuf::from(args.next().context("missing output path")?);
    let max_records = args
        .next()
        .context("missing global record limit")?
        .to_str()
        .context("invalid limit")?
        .parse::<usize>()?;
    let descriptor_output = args.next().map(PathBuf::from);
    let normalize = args.next().is_some_and(|arg| arg == "--normalize");
    ensure!(
        args.next().is_none(),
        "usage: generate_transcript_csfs INPUT OUTPUT MAX_RECORDS [DESCRIPTORS.csv] [--normalize]"
    );
    ensure!(max_records > 0, "record limit must be positive");
    let request = ExcitationRequest::from_transcript(&fs::read_to_string(input)?)?;
    let input_seconds = total.elapsed().as_secs_f64();
    let start = Instant::now();
    let occupations = enumerate_occupations(&request)?;
    let occupation_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let requests = occupations
        .configurations
        .iter()
        .map(|configuration| GenerationRequest {
            core_subshells: occupations.core_subshells.clone(),
            configuration: configuration.occupations.clone(),
            min_two_j: request.min_two_j,
            max_two_j: request.max_two_j,
            max_records,
        })
        .collect::<Vec<_>>();
    let threads = std::env::var("RCSFS_THREADS")
        .ok()
        .map(|value| value.parse())
        .transpose()?;
    let chunks = generate_csfs_parallel(&requests, max_records, threads)?;
    let count = chunks.iter().try_fold(0usize, |count, chunk| {
        count
            .checked_add(chunk.records.len())
            .context("record count overflow")
    })?;
    let generation_seconds = start.elapsed().as_secs_f64();
    ensure!(count > 0, "no CSFs generated");
    let start = Instant::now();
    let mut order = BTreeMap::<(u16, bool), Vec<(usize, usize)>>::new();
    let mut used = HashSet::<Subshell>::new();
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        for (block_index, block) in chunk.blocks.iter().enumerate() {
            order
                .entry((
                    block.total_two_j,
                    block.parity == _rcsfs::complete_csf::Parity::Odd,
                ))
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
        header(occupations.core_subshells.iter().map(ToString::to_string)),
        "Peel subshells:".to_owned(),
        header(used.iter().map(ToString::to_string)),
        "CSF(s):".to_owned(),
    ];
    let organization_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)?;
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
    if let Some(path) = descriptor_output {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("descriptor output must be a new file: {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        for block_refs in order.values() {
            for &(chunk_index, block_index) in block_refs {
                let chunk = &chunks[chunk_index];
                let block = &chunk.blocks[block_index];
                let start = usize::try_from(block.record_start)?;
                let end = start + usize::try_from(block.record_len)?;
                for record in &chunk.records[start..end] {
                    let descriptor = chunk.descriptor_for(record)?;
                    if normalize {
                        let values = normalize_descriptor_per_csf(
                            &descriptor,
                            &chunk.subshells,
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
                }
            }
        }
        writer.flush()?;
        let metadata_path = path.with_extension("toml");
        let metadata = format!(
            "format_version = 1\nencoding = \"csv\"\nnormalized = {}\nrecord_count = {}\nsubshells = {:?}\n",
            normalize,
            count,
            chunks
                .first()
                .map(|chunk| chunk.subshells.clone())
                .unwrap_or_default()
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
    }
    let output_seconds = start.elapsed().as_secs_f64();
    let total_seconds = total.elapsed().as_secs_f64();
    println!(
        "input_seconds = {input_seconds:.9}\noccupation_seconds = {occupation_seconds:.9}\ngeneration_seconds = {generation_seconds:.9}\norganization_seconds = {organization_seconds:.9}\noutput_seconds = {output_seconds:.9}\ntotal_seconds = {total_seconds:.9}"
    );
    println!(
        "unique_occupations = {}\nrecords = {count}\nblocks = {}\noutput_bytes = {}\ninteger_chunk_capacity_bytes = {}",
        occupations.configurations.len(),
        order.len(),
        fs::metadata(output)?.len(),
        chunks
            .iter()
            .map(CompleteCsfFile::allocated_bytes)
            .sum::<usize>()
            + chunks.capacity() * size_of::<CompleteCsfFile>()
    );
    // This capacity does not include occupation vectors, block-index nodes or allocator metadata.
    Ok(())
}
