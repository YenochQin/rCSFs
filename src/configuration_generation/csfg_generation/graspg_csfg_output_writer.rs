use super::representative_occupation_planning::Plan;
use crate::complete_csf::{CompleteCsfFile, Parity};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Debug)]
pub struct CsfgBlockStats {
    pub two_j: u16,
    pub odd: bool,
    pub record_count: u64,
    pub expanded_csf_count: u64,
    pub label_csf_count: u64,
}

#[derive(Debug)]
pub struct CsfgStats {
    pub unique_occupations: usize,
    pub representative_occupations: usize,
    pub record_count: u64,
    pub label_csf_count: u64,
    pub expanded_csf_count: u64,
    pub type_counts: [u64; 5],
    pub blocks: Vec<CsfgBlockStats>,
    pub csfg_bytes: u64,
    pub label_bytes: u64,
    pub planning_seconds: f64,
    pub generation_seconds: f64,
    pub writing_seconds: f64,
}

fn header(labels: impl Iterator<Item = String>) -> String {
    labels
        .map(|s| {
            if s.ends_with('-') {
                format!("{s:>5}")
            } else {
                format!("{:>5}", format!("{s} "))
            }
        })
        .collect::<String>()
        .trim_end()
        .to_owned()
}

pub(super) fn write(
    plan: &Plan,
    chunks: &[CompleteCsfFile],
    csfg: &Path,
    label: &Path,
) -> Result<CsfgStats> {
    let mut order = BTreeMap::<(u16, bool), Vec<(usize, usize)>>::new();
    for (i, chunk) in chunks.iter().enumerate() {
        for (j, block) in chunk.blocks.iter().enumerate() {
            order
                .entry((block.total_two_j, block.parity == Parity::Odd))
                .or_default()
                .push((i, j));
        }
    }
    ensure!(
        !order.is_empty(),
        "no CSFGs generated for requested J range"
    );
    let mut stats = CsfgStats {
        unique_occupations: plan.unique_occupations,
        representative_occupations: chunks.len(),
        record_count: 0,
        label_csf_count: 0,
        expanded_csf_count: 0,
        type_counts: [0; 5],
        blocks: Vec::new(),
        csfg_bytes: 0,
        label_bytes: 0,
        planning_seconds: 0.,
        generation_seconds: 0.,
        writing_seconds: 0.,
    };
    let mut writer = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(csfg)?);
    writeln!(
        writer,
        "Core subshells:\n{}\nPeel subshells:\n{}\nCSF(s):",
        header(plan.core.iter().map(ToString::to_string)),
        header(plan.peel.iter().map(ToString::to_string))
    )?;
    for ((two_j, odd), refs) in order {
        if !stats.blocks.is_empty() {
            writeln!(writer, " *")?;
        }
        let mut block = CsfgBlockStats {
            two_j,
            odd,
            record_count: 0,
            expanded_csf_count: 0,
            label_csf_count: 0,
        };
        for (i, j) in refs {
            let chunk = &chunks[i];
            let span = &chunk.blocks[j];
            let template = &plan.templates[i];
            let expanded = span
                .record_len
                .checked_mul(template.multiplicity)
                .context("expanded CSF count overflow")?;
            block.record_count = block
                .record_count
                .checked_add(span.record_len)
                .context("record count overflow")?;
            block.expanded_csf_count = block
                .expanded_csf_count
                .checked_add(expanded)
                .context("expanded count overflow")?;
            stats.type_counts[template.kind - 1] = stats.type_counts[template.kind - 1]
                .checked_add(span.record_len)
                .context("type count overflow")?;
            if template.kind == 1 {
                block.label_csf_count += span.record_len;
            }
            let start = usize::try_from(span.record_start)?;
            let end = start + usize::try_from(span.record_len)?;
            for record in &chunk.records[start..end] {
                chunk.write_record_to(record, &mut writer)?;
            }
        }
        stats.record_count = stats
            .record_count
            .checked_add(block.record_count)
            .context("record count overflow")?;
        stats.label_csf_count += block.label_csf_count;
        stats.expanded_csf_count = stats
            .expanded_csf_count
            .checked_add(block.expanded_csf_count)
            .context("expanded count overflow")?;
        stats.blocks.push(block);
    }
    writer.flush()?;
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(label)?,
    );
    writeln!(
        writer,
        "{:12}   = nonsym",
        plan.core.len() + plan.labels.len()
    )?;
    writeln!(
        writer,
        "{}",
        header(
            plan.core
                .iter()
                .chain(&plan.labels)
                .map(ToString::to_string)
        )
    )?;
    writer.flush()?;
    stats.csfg_bytes = fs::metadata(csfg)?.len();
    stats.label_bytes = fs::metadata(label)?.len();
    Ok(stats)
}
