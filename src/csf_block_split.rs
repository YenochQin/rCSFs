//! Split one multi-block GRASP CSF list into one file per symmetry block.
//!
//! This mirrors the `.c`-file half of GRASP's `rasfsplit.f90`: the five
//! header lines are replicated into every output, and every `J^P` symmetry
//! block becomes its own single-block file named `<prefix>_<2J>.c`. The
//! sibling orbital file (`<stem>.w`) is byte-copied beside each output when
//! it exists and copying was requested. Mixing files (`.m`/`.cm`) are out of
//! scope for this module.
//!
//! The input is streamed once with one buffered writer per block, so memory
//! stays bounded by line and writer buffers rather than file size. Blocks
//! are keyed by their records' total `2J`, read from the final coupling
//! line; every record of a block must agree, and a `2J` seen in two blocks
//! is an error because both would map to the same `<prefix>_<2J>.c` name.
//! Real GRASP lists carry each `2J` at most once (their file naming already
//! separates parity groups), so this contract matches well-formed input.

use crate::atomic_output::{
    TemporaryOutput, create_temporary_output, ensure_output_does_not_alias_input,
    publish_temporary_output,
};
use crate::complete_csf::{
    Parity, BLOCK_SEPARATOR, parse_two_j, validate_header_labels,
};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Header lines every GRASP CSF list carries before its first record.
const HEADER_LINE_COUNT: usize = 5;
/// Width of one formatted CSF field, matching `complete_csf::FIELD_WIDTH`.
const FIELD_WIDTH: usize = 9;
/// Refuse pathological inputs before exhausting file descriptors. GRASP
/// lists carry one block per `J^P`, so real counts stay far below this.
const MAX_BLOCKS: usize = 10_000;

/// One output: a single symmetry block written as its own CSF file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JBlockOutputStats {
    pub output_file: String,
    /// Published `.w` copy for this output, when one was made.
    pub w_file: Option<String>,
    /// Zero-based position of the block inside the input file.
    pub block_index: usize,
    /// Total `2J` of every record in the block, e.g. `8` for `J = 4`.
    pub total_two_j: u16,
    pub parity: String,
    pub csf_count: usize,
}

/// Statistics from splitting one CSF list into per-block files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JBlockSplitStats {
    pub input_csf_count: usize,
    pub block_count: usize,
    pub outputs: Vec<JBlockOutputStats>,
}

/// A block whose records are streamed to its staged, not-yet-published file.
struct PendingOutput {
    final_path: PathBuf,
    temporary: TemporaryOutput,
    writer: BufWriter<File>,
    total_two_j: u16,
    parity: Parity,
    csf_count: usize,
}

/// A staged, not-yet-published `.w` copy aligned with one block output.
struct PendingCopy {
    final_path: PathBuf,
    temporary: TemporaryOutput,
}

fn parity_label(parity: Parity) -> &'static str {
    match parity {
        Parity::Even => "even",
        Parity::Odd => "odd",
    }
}

/// Read one LF-terminated ASCII line, mirroring `complete_csf`'s discipline.
fn next_line(reader: &mut impl BufRead, counter: &mut usize) -> Result<Option<String>> {
    let mut raw = String::new();
    if reader.read_line(&mut raw)? == 0 {
        return Ok(None);
    }
    *counter += 1;
    let number = *counter;
    ensure!(
        raw.ends_with('\n'),
        "line {number}: CSF lines must end with LF"
    );
    raw.pop();
    ensure!(
        !raw.ends_with('\r'),
        "line {number}: CRLF is not canonical; CSF lines must end with LF only"
    );
    ensure!(raw.is_ascii(), "line {number}: CSF text must be ASCII");
    Ok(Some(raw))
}

/// Read a record's total `2J` and parity from its three fixed-width lines.
///
/// Only the fields needed to name and validate the block are parsed; the
/// lines themselves are later copied verbatim.
fn record_symmetry(lines: &[(usize, String); 3]) -> Result<(u16, Parity)> {
    let (line1_number, line1) = &lines[0];
    let (line3_number, line3) = &lines[2];
    ensure!(
        !line1.is_empty() && line1.len().is_multiple_of(FIELD_WIDTH),
        "line {line1_number}: occupation line length {} is not a positive multiple of {FIELD_WIDTH}",
        line1.len()
    );
    let field_count = line1.len() / FIELD_WIDTH;
    ensure!(
        line3.len() == line1.len() + 2,
        "line {line3_number}: coupling line length {} must be occupation length + 2 ({})",
        line3.len(),
        line1.len() + 2
    );
    let parity = Parity::parse(line3.as_bytes()[line3.len() - 1])
        .with_context(|| format!("line {line3_number}: invalid final parity"))?;
    let total_field = line3[field_count * FIELD_WIDTH - 3..field_count * FIELD_WIDTH + 1].trim();
    let total_two_j = parse_two_j(total_field)
        .with_context(|| format!("line {line3_number}: invalid total J {total_field:?}"))?;
    Ok((total_two_j, parity))
}

/// Validate the output prefix once for both Rust and Python callers.
fn validate_prefix(prefix: &str) -> Result<()> {
    ensure!(
        !prefix.is_empty()
            && prefix != "."
            && prefix != ".."
            && !prefix.contains('/')
            && !prefix.contains('\\'),
        "prefix must be a nonempty filename stem without path separators"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)] // One parameter per streaming-scan input.
fn open_block_output(
    output_dir: &Path,
    prefix: &str,
    header_lines: &[String; HEADER_LINE_COUNT],
    input_csf: &Path,
    w_source: Option<&Path>,
    overwrite: bool,
    seen: &mut HashMap<u16, usize>,
    blocks: &mut Vec<PendingOutput>,
    total_two_j: u16,
    parity: Parity,
) -> Result<()> {
    if let Some(&first) = seen.get(&total_two_j) {
        let earlier = &blocks[first];
        bail!(
            "symmetry block {} (2J={}, {}) repeats the 2J of block {} (2J={}, {}); \
             both would map to {prefix}_{total_two_j}.c — split parity groups \
             into separate input files first",
            blocks.len(),
            total_two_j,
            parity_label(parity),
            first,
            earlier.total_two_j,
            parity_label(earlier.parity)
        );
    }
    seen.insert(total_two_j, blocks.len());
    let final_path = output_dir.join(format!("{prefix}_{total_two_j}.c"));
    ensure_output_does_not_alias_input(&final_path, input_csf, "CSF")?;
    if let Some(source) = w_source {
        ensure_output_does_not_alias_input(&final_path, source, "orbital")?;
    }
    ensure!(
        overwrite || !final_path.exists(),
        "output already exists: {} (set overwrite to replace it)",
        final_path.display()
    );
    let (temporary, file) = create_temporary_output(&final_path)?;
    let mut writer = BufWriter::new(file);
    for line in header_lines {
        writeln!(writer, "{line}")?;
    }
    blocks.push(PendingOutput {
        final_path,
        temporary,
        writer,
        total_two_j,
        parity,
        csf_count: 0,
    });
    ensure!(
        blocks.len() <= MAX_BLOCKS,
        "CSF file has more than {MAX_BLOCKS} symmetry blocks"
    );
    Ok(())
}

/// Split a multi-block CSF text file into one single-block file per `J^P`.
///
/// Every output receives the input's five header lines and exactly one
/// symmetry block, named `<prefix>_<2J>.c` inside `output_dir`. All outputs
/// are staged beside their destinations and published only after the whole
/// input has been read; publication of several files is not a transaction,
/// so a later failure reports what was already published. Existing outputs
/// are refused unless `overwrite` is set.
///
/// When `copy_w` is set and `<input stem>.w` exists, that orbital file is
/// byte-copied beside every output (`<prefix>_<2J>.w`); a missing `.w` is
/// not an error and is reported as `w_file: None`.
pub fn split_csfs_by_j(
    input_csf: &Path,
    output_dir: &Path,
    prefix: &str,
    copy_w: bool,
    overwrite: bool,
) -> Result<JBlockSplitStats> {
    validate_prefix(prefix)?;
    ensure!(
        output_dir.is_dir(),
        "output directory does not exist: {}",
        output_dir.display()
    );
    let input_file = File::open(input_csf)
        .with_context(|| format!("failed to open CSF file {}", input_csf.display()))?;
    let mut reader = BufReader::new(input_file);

    let w_candidate = input_csf.with_extension("w");
    let w_source: Option<PathBuf> = if copy_w && w_candidate.is_file() {
        Some(w_candidate)
    } else {
        None
    };

    // --- Header: five lines, labels as every GRASP writer emits them. ---
    let mut header_lines: [String; HEADER_LINE_COUNT] = Default::default();
    let mut line_number = 0usize;
    for (index, slot) in header_lines.iter_mut().enumerate() {
        let line = next_line(&mut reader, &mut line_number)
            .with_context(|| format!("missing CSF header line {}", index + 1))?
            .with_context(|| format!("failed to read header line {}", index + 1))?;
        *slot = line;
    }
    validate_header_labels(&header_lines)?;

    // --- Single streaming pass over the records. ---
    let mut blocks: Vec<PendingOutput> = Vec::new();
    let mut seen: HashMap<u16, usize> = HashMap::new();
    let mut pending_record: Vec<(usize, String)> = Vec::with_capacity(3);
    let mut in_block = false;
    while let Some(line) = next_line(&mut reader, &mut line_number)? {
        if line == BLOCK_SEPARATOR {
            ensure!(
                pending_record.is_empty(),
                "line {}: block separator interrupts a CSF record",
                line_number
            );
            ensure!(
                in_block,
                "line {}: block separator closes an empty symmetry block",
                line_number
            );
            in_block = false;
            continue;
        }
        ensure!(
            line.trim() != "*",
            "line {line_number}: block separator must be exactly {BLOCK_SEPARATOR:?}, found {line:?}"
        );
        pending_record.push((line_number, line));
        if pending_record.len() < 3 {
            continue;
        }
        let record: [(usize, String); 3] = std::mem::take(&mut pending_record)
            .try_into()
            .expect("exactly three record lines were collected");
        let (total_two_j, parity) = record_symmetry(&record)?;
        if !in_block {
            open_block_output(
                output_dir,
                prefix,
                &header_lines,
                input_csf,
                w_source.as_deref(),
                overwrite,
                &mut seen,
                &mut blocks,
                total_two_j,
                parity,
            )?;
            in_block = true;
        } else {
            let current = blocks
                .last()
                .expect("an open block always has an output");
            ensure!(
                total_two_j == current.total_two_j && parity == current.parity,
                "line {}: symmetry block {} mixes total J/parity values: \
                 (2J={}, {}) after (2J={}, {})",
                line_number,
                blocks.len() - 1,
                total_two_j,
                parity_label(parity),
                current.total_two_j,
                parity_label(current.parity)
            );
        }
        let output = blocks.last_mut().expect("an open block always has an output");
        for (_, record_line) in &record {
            writeln!(output.writer, "{record_line}")?;
        }
        output.csf_count += 1;
    }
    ensure!(
        pending_record.is_empty(),
        "final CSF record has {} of 3 lines",
        pending_record.len()
    );
    ensure!(
        !blocks.is_empty(),
        "CSF file contains no CSF records"
    );

    // --- Stage the `.w` copies aligned with the block outputs. ---
    let mut w_copies: Vec<Option<PendingCopy>> = Vec::with_capacity(blocks.len());
    if let Some(source) = &w_source {
        for block in &blocks {
            let final_path = block.final_path.with_extension("w");
            ensure_output_does_not_alias_input(&final_path, input_csf, "CSF")?;
            ensure_output_does_not_alias_input(&final_path, source, "orbital")?;
            ensure!(
                overwrite || !final_path.exists(),
                "output already exists: {} (set overwrite to replace it)",
                final_path.display()
            );
            let (temporary, mut file) = create_temporary_output(&final_path)?;
            let mut origin = File::open(source)
                .with_context(|| format!("failed to open orbital file {}", source.display()))?;
            std::io::copy(&mut origin, &mut file).with_context(|| {
                format!("failed to copy orbital file {}", source.display())
            })?;
            file.sync_all()?;
            w_copies.push(Some(PendingCopy {
                final_path,
                temporary,
            }));
        }
    } else {
        w_copies.resize_with(blocks.len(), || None);
    }

    // --- Flush everything, then publish; report partial publication. ---
    for block in &mut blocks {
        block.writer.flush()?;
        block.writer.get_ref().sync_all()?;
    }
    let mut published: Vec<PathBuf> = Vec::new();
    let mut publish = |temporary: &Path, final_path: &Path| -> Result<()> {
        if let Err(error) = publish_temporary_output(temporary, final_path, overwrite) {
            bail!(
                "failed to publish {}; already published: {:?}; {error:#}",
                final_path.display(),
                published
            );
        }
        published.push(final_path.to_path_buf());
        Ok(())
    };
    for block in &blocks {
        publish(block.temporary.path(), &block.final_path)?;
    }
    for copy in w_copies.iter().flatten() {
        publish(copy.temporary.path(), &copy.final_path)?;
    }

    let input_csf_count = blocks.iter().map(|block| block.csf_count).sum();
    Ok(JBlockSplitStats {
        input_csf_count,
        block_count: blocks.len(),
        outputs: blocks
            .into_iter()
            .enumerate()
            .map(|(block_index, block)| JBlockOutputStats {
                output_file: block.final_path.display().to_string(),
                w_file: match w_copies.get(block_index) {
                    Some(Some(copy)) => Some(copy.final_path.display().to_string()),
                    _ => None,
                },
                block_index,
                total_two_j: block.total_two_j,
                parity: parity_label(block.parity).to_owned(),
                csf_count: block.csf_count,
            })
            .collect(),
    })
}
