use _rcsfs::complete_csf::CompleteCsfFile;
use anyhow::{Context, Result, ensure};
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let input = PathBuf::from(args.next().context("usage: roundtrip_csf INPUT OUTPUT")?);
    let output = PathBuf::from(args.next().context("usage: roundtrip_csf INPUT OUTPUT")?);
    ensure!(args.next().is_none(), "usage: roundtrip_csf INPUT OUTPUT");

    roundtrip(&input, &output)
}

pub fn roundtrip(input: &Path, output: &Path) -> Result<()> {
    let parsed = CompleteCsfFile::parse_path(input)?;
    // Atomic creation refuses existing files, including symlink/hard-link aliases
    // of the input, before any bytes can be overwritten.
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .with_context(|| format!("output must be a new file: {}", output.display()))?;
    parsed.write_to(BufWriter::new(file))?;
    let identical = files_equal(input, output)?;
    println!(
        "records={} blocks={} occupied_entries={} coupling_entries={} allocated_bytes={} byte_identical={identical}",
        parsed.records.len(),
        parsed.blocks.len(),
        parsed.occupied_subshells.len(),
        parsed.intermediate_couplings.len(),
        parsed.allocated_bytes(),
    );
    ensure!(identical, "round-trip output differs from input");
    Ok(())
}

fn files_equal(left: &Path, right: &Path) -> Result<bool> {
    let mut left = BufReader::new(File::open(left)?);
    let mut right = BufReader::new(File::open(right)?);
    let mut left_buffer = [0u8; 64 * 1024];
    let mut right_buffer = [0u8; 64 * 1024];
    loop {
        let left_len = left.read(&mut left_buffer)?;
        let right_len = right.read(&mut right_buffer)?;
        if left_len != right_len || left_buffer[..left_len] != right_buffer[..right_len] {
            return Ok(false);
        }
        if left_len == 0 {
            return Ok(true);
        }
    }
}
