//! Generate CSFs from an rcsfgenerate transcript using the Rust generator.
use _rcsfs::complete_csf::CompleteCsfFile;
use _rcsfs::csf_generation::{
    ExcitationRequest, GenerationRequest, enumerate_occupations, generate_csfs_parallel,
    write_generated_csfs,
};
use anyhow::{Context, Result, ensure};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let total = Instant::now();
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(
        args.next()
            .context("usage: generate_transcript_csfs INPUT OUTPUT")?,
    );
    let output = PathBuf::from(args.next().context("missing output path")?);
    let descriptor_output = args.next().map(PathBuf::from);
    let normalize = args.next().is_some_and(|arg| arg == "--normalize");
    ensure!(
        args.next().is_none(),
        "usage: generate_transcript_csfs INPUT OUTPUT [DESCRIPTORS.csv] [--normalize]"
    );
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
        })
        .collect::<Vec<_>>();
    let threads = std::env::var("RCSFS_THREADS")
        .ok()
        .map(|value| value.parse())
        .transpose()?;
    let chunks = generate_csfs_parallel(&requests, threads)?;
    let generation_seconds = start.elapsed().as_secs_f64();
    // Ordering, merging and file output are now one call (write_generated_csfs,
    // shared with the rcsfs.generate_csfs_from_transcript PyO3 binding), so
    // they are timed together rather than as separate organization/output phases.
    let start = Instant::now();
    let write_stats = write_generated_csfs(
        &occupations.core_subshells,
        &chunks,
        &output,
        descriptor_output.as_deref(),
        normalize,
    )?;
    let output_seconds = start.elapsed().as_secs_f64();
    let total_seconds = total.elapsed().as_secs_f64();
    println!(
        "input_seconds = {input_seconds:.9}\noccupation_seconds = {occupation_seconds:.9}\ngeneration_seconds = {generation_seconds:.9}\noutput_seconds = {output_seconds:.9}\ntotal_seconds = {total_seconds:.9}"
    );
    println!(
        "unique_occupations = {}\nrecords = {}\nblocks = {}\noutput_bytes = {}\ninteger_chunk_capacity_bytes = {}",
        occupations.configurations.len(),
        write_stats.record_count,
        write_stats.block_count,
        write_stats.output_bytes,
        chunks
            .iter()
            .map(CompleteCsfFile::allocated_bytes)
            .sum::<usize>()
            + chunks.capacity() * size_of::<CompleteCsfFile>()
    );
    // This capacity does not include occupation vectors, block-index nodes or allocator metadata.
    Ok(())
}
