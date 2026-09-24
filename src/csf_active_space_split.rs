//! Split one CSF Parquet stream into overlapping GRASP-style active spaces.
//!
//! The input is read once in bounded Arrow batches. Each target has one
//! buffered writer; no target materializes the complete input or its output.

use crate::atomic_output::{
    TemporaryOutput, create_temporary_output, ensure_distinct_inputs,
    ensure_output_does_not_alias_input, publish_temporary_output,
};
use crate::csfs_conversion::HeaderData;
use anyhow::{Context, Result, bail, ensure};
use arrow::array::{Array, StringArray, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const BATCH_ROWS: usize = 65_536;

/// One independent output. The same source CSF can belong to several targets.
#[derive(Debug, Clone)]
pub struct ActiveSpaceTarget {
    pub maximum_orbitals: String,
    pub output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveSpaceOutputStats {
    pub output_file: String,
    pub maximum_orbitals: String,
    pub csf_count: usize,
    pub block_lengths: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveSpaceSplitStats {
    pub input_csf_count: usize,
    pub block_count: usize,
    pub outputs: Vec<ActiveSpaceOutputStats>,
}

struct PendingTarget {
    spec: ActiveSpaceTarget,
    allowed: HashSet<String>,
    writer: BufWriter<File>,
    temporary: TemporaryOutput,
    counts: Vec<usize>,
}

fn parse_orbital(token: &str) -> Result<(usize, char)> {
    let nonrel = token.strip_suffix('-').unwrap_or(token);
    let letter_pos = nonrel
        .find(|c: char| !c.is_ascii_digit())
        .context("orbital must start with a positive principal quantum number")?;
    let (number, letter) = nonrel.split_at(letter_pos);
    ensure!(
        !number.is_empty() && letter.len() == 1 && letter.chars().all(|c| c.is_ascii_lowercase()),
        "invalid orbital {token:?}; expected n followed by one lowercase letter and optional '-'"
    );
    let n = number
        .parse::<usize>()
        .context("invalid principal quantum number")?;
    ensure!(n > 0, "orbital principal quantum number must be positive");
    Ok((n, letter.chars().next().expect("one letter")))
}

fn parse_maxima(spec: &str) -> Result<HashMap<char, usize>> {
    ensure!(
        !spec.trim().is_empty(),
        "active orbital set cannot be empty"
    );
    let mut maxima = HashMap::new();
    for entry in spec.split(',') {
        let token = entry.trim();
        ensure!(
            !token.ends_with('-'),
            "active orbital limit must use nonrelativistic notation: {token:?}"
        );
        let (n, l) = parse_orbital(token)?;
        ensure!(
            maxima.insert(l, n).is_none(),
            "duplicate active orbital symmetry {l}"
        );
    }
    Ok(maxima)
}

fn select_row(
    line: &str,
    peel: &HashSet<String>,
    targets: &[PendingTarget],
    selected: &mut [bool],
) -> Result<()> {
    ensure!(line.is_ascii(), "CSF occupation line is not ASCII");
    let mut rest = line;
    selected.fill(true);
    let mut saw_orbital = false;
    loop {
        let Some(open) = rest.find('(') else {
            ensure!(
                rest.trim().is_empty(),
                "invalid CSF occupation line: {line:?}"
            );
            break;
        };
        let orbital = rest[..open].trim();
        parse_orbital(orbital).with_context(|| format!("invalid CSF occupation line: {line:?}"))?;
        ensure!(
            peel.contains(orbital),
            "CSF orbital {orbital:?} is absent from peel header"
        );
        let suffix = &rest[open + 1..];
        let close = suffix
            .find(')')
            .with_context(|| format!("unclosed occupation in {line:?}"))?;
        let _count = suffix[..close]
            .trim()
            .parse::<usize>()
            .with_context(|| format!("invalid occupation in {line:?}"))?;
        // A listed zero-occupation orbital still requires its header entry.
        saw_orbital = true;
        for (target, keep) in targets.iter().zip(selected.iter_mut()) {
            *keep &= target.allowed.contains(orbital);
        }
        rest = &suffix[close + 1..];
    }
    ensure!(saw_orbital, "CSF occupation line has no orbitals: {line:?}");
    Ok(())
}

fn header_peel(header: &HeaderData) -> Result<Vec<String>> {
    ensure!(
        header.header_info.header_lines.len() == 5,
        "CSF header must contain five lines"
    );
    let mut peel = Vec::new();
    let mut seen = HashSet::new();
    for token in header.header_info.header_lines[3].split_whitespace() {
        parse_orbital(token)?;
        ensure!(seen.insert(token), "duplicate peel orbital {token:?}");
        peel.push(token.to_owned());
    }
    ensure!(!peel.is_empty(), "CSF peel header is empty");
    Ok(peel)
}

fn format_peel(peel: &[String]) -> String {
    peel.iter()
        .map(|token| {
            let display = if token.ends_with('-') {
                token.clone()
            } else {
                format!("{token} ")
            };
            format!("{display:>5}")
        })
        .collect::<String>()
        .trim_end()
        .to_owned()
}

fn advance_blocks(
    targets: &mut [PendingTarget],
    lengths: &[usize],
    block: &mut usize,
    consumed: &mut usize,
) -> Result<()> {
    while *block < lengths.len() && *consumed == lengths[*block] {
        *block += 1;
        *consumed = 0;
        if *block < lengths.len() {
            for target in targets.iter_mut() {
                writeln!(target.writer, " *")?;
            }
        }
    }
    Ok(())
}

/// Split one CSF Parquet file into GRASP-compatible CSF text files.
///
/// All output paths must be absent and have existing parent directories. Each
/// completed file is published atomically without overwriting an existing path.
/// Publication of *several* files is not a transaction: if a later publication
/// fails, earlier completed outputs remain in place and the error names them.
pub fn split_csfs_by_active_spaces(
    input_parquet: &Path,
    header_path: &Path,
    targets: &[ActiveSpaceTarget],
) -> Result<ActiveSpaceSplitStats> {
    ensure!(
        !targets.is_empty(),
        "at least one active orbital set is required"
    );
    ensure_distinct_inputs(
        input_parquet,
        header_path,
        "Parquet and header inputs must differ",
    )?;
    let header: HeaderData = toml::from_str(&fs::read_to_string(header_path)?)?;
    let peel = header_peel(&header)?;
    let peel_set: HashSet<String> = peel.iter().cloned().collect();
    let lengths = &header.block_info.block_lengths;
    ensure!(
        header.block_info.block_count == lengths.len(),
        "header block count does not match block_lengths"
    );
    ensure!(!lengths.is_empty(), "header has no symmetry blocks");
    let expected_rows = lengths
        .iter()
        .try_fold(0usize, |sum, n| sum.checked_add(*n))
        .context("header CSF count overflow")?;
    ensure!(
        expected_rows == header.conversion_stats.csf_count,
        "header CSF count does not match block lengths"
    );

    // Validate every target before creating any temporary files.
    let mut validated = Vec::with_capacity(targets.len());
    let mut output_names = HashSet::new();
    for spec in targets {
        ensure_output_does_not_alias_input(&spec.output, input_parquet, "Parquet")?;
        ensure_output_does_not_alias_input(&spec.output, header_path, "header")?;
        ensure!(
            !spec.output.exists(),
            "output already exists: {}",
            spec.output.display()
        );
        let parent = spec
            .output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let canonical = parent
            .canonicalize()
            .with_context(|| format!("output parent does not exist: {}", parent.display()))?;
        let name = spec.output.file_name().context("output needs a filename")?;
        ensure!(
            output_names.insert(canonical.join(name)),
            "duplicate output path: {}",
            spec.output.display()
        );
        let maxima = parse_maxima(&spec.maximum_orbitals)?;
        let allowed: HashSet<String> = peel
            .iter()
            .filter_map(|orbital| {
                let (n, l) = parse_orbital(orbital).expect("validated peel orbital");
                maxima
                    .get(&l)
                    .filter(|maximum| n <= **maximum)
                    .map(|_| orbital.clone())
            })
            .collect();
        validated.push((spec.clone(), allowed));
    }

    let reader_file = File::open(input_parquet)?;
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(reader_file)?
        .with_batch_size(BATCH_ROWS)
        .build()?;
    let mut pending = Vec::with_capacity(targets.len());
    for (spec, allowed) in validated {
        let (temporary, file) = create_temporary_output(&spec.output)?;
        let mut writer = BufWriter::new(file);
        for (index, line) in header.header_info.header_lines.iter().enumerate() {
            if index == 3 {
                let filtered: Vec<String> = peel
                    .iter()
                    .filter(|orbital| allowed.contains(*orbital))
                    .cloned()
                    .collect();
                writeln!(writer, "{}", format_peel(&filtered))?;
            } else {
                writeln!(writer, "{}", line.trim_end())?;
            }
        }
        pending.push(PendingTarget {
            spec,
            allowed,
            writer,
            temporary,
            counts: vec![0; lengths.len()],
        });
    }

    let mut block = 0usize;
    let mut consumed = 0usize;
    let mut position = 0usize;
    let mut selected = vec![false; pending.len()];
    while let Some(batch) = reader.next() {
        let batch = batch?;
        ensure!(
            batch.num_columns() == 4,
            "CSF Parquet must have exactly four columns"
        );
        let idx = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .context("CSF Parquet idx column is not UInt64")?;
        let line1 = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("CSF Parquet line1 column is not Utf8")?;
        let line2 = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("CSF Parquet line2 column is not Utf8")?;
        let line3 = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("CSF Parquet line3 column is not Utf8")?;
        for row in 0..batch.num_rows() {
            advance_blocks(&mut pending, lengths, &mut block, &mut consumed)?;
            ensure!(
                block < lengths.len(),
                "Parquet has more rows than header block lengths"
            );
            ensure!(
                !idx.is_null(row)
                    && !line1.is_null(row)
                    && !line2.is_null(row)
                    && !line3.is_null(row),
                "CSF Parquet contains null values at row {position}"
            );
            ensure!(
                idx.value(row) == position as u64,
                "CSF Parquet idx is not contiguous at row {position}"
            );
            select_row(line1.value(row), &peel_set, &pending, &mut selected)
                .with_context(|| format!("invalid CSF at Parquet row {position}"))?;
            for (target, keep) in pending.iter_mut().zip(&selected) {
                if *keep {
                    writeln!(target.writer, "{}", line1.value(row).trim_end())?;
                    writeln!(target.writer, "{}", line2.value(row).trim_end())?;
                    writeln!(target.writer, "{}", line3.value(row).trim_end())?;
                    target.counts[block] += 1;
                }
            }
            position += 1;
            consumed += 1;
        }
    }
    ensure!(
        position == expected_rows,
        "Parquet ended after {position} rows; header expects {expected_rows}"
    );
    advance_blocks(&mut pending, lengths, &mut block, &mut consumed)?;
    ensure!(
        block == lengths.len(),
        "Parquet ended before the final symmetry block"
    );

    for target in &mut pending {
        target.writer.flush()?;
        target.writer.get_ref().sync_all()?;
    }
    let stats = ActiveSpaceSplitStats {
        input_csf_count: position,
        block_count: lengths.len(),
        outputs: pending
            .iter()
            .map(|target| ActiveSpaceOutputStats {
                output_file: target.spec.output.display().to_string(),
                maximum_orbitals: target.spec.maximum_orbitals.clone(),
                csf_count: target.counts.iter().sum(),
                block_lengths: target.counts.clone(),
            })
            .collect(),
    };
    let mut published = Vec::new();
    for target in pending {
        drop(target.writer);
        if let Err(error) =
            publish_temporary_output(target.temporary.path(), &target.spec.output, false)
        {
            bail!(
                "failed to publish {}; already published: {:?}; {error:#}",
                target.spec.output.display(),
                published
            );
        }
        published.push(target.spec.output.clone());
    }
    Ok(stats)
}
