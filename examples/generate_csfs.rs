//! Generate a fixed relativistic occupation configuration from a TOML request.
use _rcsfs::csf_generation::{GenerationRequest, Subshell, SubshellOccupation, generate_csfs};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::fs::{self, OpenOptions};
use std::io::BufWriter;
use std::path::PathBuf;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    #[serde(default)]
    core_subshells: Vec<String>,
    min_two_j: u16,
    max_two_j: u16,
    max_records: usize,
    occupations: Vec<Occupation>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Occupation {
    subshell: String,
    electrons: u8,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(
        args.next()
            .context("usage: generate_csfs REQUEST.toml OUTPUT.c")?,
    );
    let output = PathBuf::from(
        args.next()
            .context("usage: generate_csfs REQUEST.toml OUTPUT.c")?,
    );
    ensure!(
        args.next().is_none(),
        "usage: generate_csfs REQUEST.toml OUTPUT.c"
    );
    let input: Input = toml::from_str(&fs::read_to_string(input)?)?;
    let request = GenerationRequest {
        core_subshells: input
            .core_subshells
            .iter()
            .map(|label| label.parse::<Subshell>())
            .collect::<Result<_>>()?,
        configuration: input
            .occupations
            .iter()
            .map(|entry| {
                Ok(SubshellOccupation {
                    subshell: entry.subshell.parse()?,
                    electrons: entry.electrons,
                })
            })
            .collect::<Result<_>>()?,
        min_two_j: input.min_two_j,
        max_two_j: input.max_two_j,
        max_records: input.max_records,
    };
    let generated = generate_csfs(&request)?;
    ensure!(
        !generated.records.is_empty(),
        "no CSFs satisfy this request; no file was created"
    );
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .with_context(|| format!("output must be a new file: {}", output.display()))?;
    generated.write_to(BufWriter::new(file))?;
    println!(
        "records={} blocks={} allocated_bytes={}",
        generated.records.len(),
        generated.blocks.len(),
        generated.allocated_bytes()
    );
    Ok(())
}
