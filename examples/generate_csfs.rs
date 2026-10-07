//! Generate a fixed relativistic occupation configuration from a TOML request.
use _rcsfs::csf_generation::{GenerationRequest, Subshell, SubshellOccupation, generate_csfs};
use _rcsfs::descriptor_schema::{DescriptorLayout, DescriptorVersion};
use _rcsfs::descriptor_v2::encode_v2;
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    #[serde(default)]
    core_subshells: Vec<String>,
    min_two_j: u16,
    max_two_j: u16,

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
            .context("usage: generate_csfs REQUEST.toml OUTPUT.c [DESCRIPTORS.csv]")?,
    );
    let output = PathBuf::from(
        args.next()
            .context("usage: generate_csfs REQUEST.toml OUTPUT.c [DESCRIPTORS.csv]")?,
    );
    let descriptor_output = args.next().map(PathBuf::from);
    ensure!(
        args.next().is_none(),
        "usage: generate_csfs REQUEST.toml OUTPUT.c [DESCRIPTORS.csv]"
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
    if let Some(path) = descriptor_output {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("descriptor output must be a new file: {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        let layout = DescriptorLayout::new(DescriptorVersion::V2, generated.subshells.len());
        let columns = layout
            .feature_column_names()
            .into_iter()
            .chain(layout.global_column_names())
            .collect::<Vec<_>>();
        writeln!(writer, "{}", columns.join(","))?;
        let mut descriptor = vec![0; layout.row_len()];
        for record in &generated.records {
            encode_v2(&generated, record, &mut descriptor)?;
            let values = &descriptor;
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    write!(writer, ",")?;
                }
                write!(writer, "{value}")?;
            }
            writeln!(writer)?;
        }
        writer.flush()?;
    }
    println!(
        "records={} blocks={} allocated_bytes={}",
        generated.records.len(),
        generated.blocks.len(),
        generated.allocated_bytes()
    );
    Ok(())
}
