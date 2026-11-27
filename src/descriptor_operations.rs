//! Ordered numerical CSF transformations over descriptor/header datasets.
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::configuration_generation::Subshell;
use crate::descriptor_dataset::{
    DatasetHeader, DatasetStats, DatasetWriter, DescriptorRow, DescriptorRows, header_path,
    output_identity,
};
use crate::descriptor_schema::{empty_slot, row_symmetry};

/// One requested active-space filter and its destination.
#[derive(Debug, Clone)]
pub struct ActiveSpaceTarget {
    /// Maximum nonrelativistic orbital for each included `l`, such as `5s,5p,4d`.
    pub maximum_orbitals: String,
    /// Path of the output descriptor Parquet file.
    pub output: PathBuf,
}

/// Merge orbital sequences without changing any input's relative coupling order.
pub(crate) fn common_header<'header>(
    mut headers: impl Iterator<Item = &'header DatasetHeader> + Clone,
) -> Result<DatasetHeader> {
    let mut sources = headers.clone();
    let first = sources.next().context("no input datasets")?;
    let mut header = first.clone();
    for other in sources {
        header.add_truncations_from(other)?;
    }
    let mut labels = Vec::<String>::new();
    let mut edges = HashSet::<(String, String)>::new();
    for source in headers.clone() {
        ensure!(
            source.core == header.core,
            "closed-core orbital tables differ"
        );
        for label in &source.peel {
            if !labels.contains(label) {
                labels.push(label.clone());
            }
        }
        for pair in source.peel.windows(2) {
            edges.insert((pair[0].clone(), pair[1].clone()));
        }
    }
    let mut ordered = Vec::new();
    while !labels.is_empty() {
        let index = labels
            .iter()
            .position(|label| !edges.iter().any(|(_, after)| after == label))
            .context("incompatible orbital ordering; recoupling is not supported")?;
        let label = labels.remove(index);
        edges.retain(|(before, _)| before != &label);
        ordered.push(label);
    }
    if let Some(source) = headers.find(|source| source.peel == ordered) {
        header.lines[3].clone_from(&source.lines[3]);
    } else {
        header.lines[3] = ordered.join("  ");
    }
    header.peel = ordered;
    Ok(header)
}

pub(crate) fn mapping(source: &[String], target: &[String]) -> Result<Vec<usize>> {
    let indices = source
        .iter()
        .map(|label| {
            target
                .iter()
                .position(|value| value == label)
                .context("output orbital table omits an input orbital")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        indices.windows(2).all(|pair| pair[0] < pair[1]),
        "incompatible orbital ordering; recoupling is not supported"
    );
    Ok(indices)
}

pub(crate) fn remap(row: &[i8], indices: &[usize], width: usize) -> Vec<i8> {
    let mut mapped = Vec::with_capacity(width);
    let mut source = row[..row.len() - 2]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(indices)
        .peekable();
    let mut cumulative = 1;
    for index in 0..(width - 2) / 4 {
        if let Some((slot, _)) = source.next_if(|(_, target)| **target == index) {
            mapped.extend_from_slice(slot);
            cumulative = slot[3];
        } else {
            mapped.extend_from_slice(&empty_slot(cumulative));
        }
    }
    debug_assert!(source.next().is_none());
    mapped.extend_from_slice(&row[row.len() - 2..]);
    mapped
}

/// Append descriptor datasets in input order, preserving explicit block boundaries.
pub fn merge(
    inputs: &[(PathBuf, PathBuf)],
    output: &Path,
    output_header: &Path,
) -> Result<DatasetStats> {
    ensure!(!inputs.is_empty(), "no input datasets");
    let headers = inputs
        .iter()
        .map(|(data, header)| Ok(DescriptorRows::open(data, header)?.into_header()))
        .collect::<Result<Vec<_>>>()?;
    let common = common_header(headers.iter())?;
    let width = common.layout().row_len();
    let input_paths = inputs
        .iter()
        .flat_map(|(a, b)| [a.as_path(), b.as_path()])
        .collect::<Vec<_>>();
    let mut writer = DatasetWriter::try_new(common, output, output_header, &input_paths)?;
    let counts = headers
        .iter()
        .map(DatasetHeader::zero_order_counts)
        .collect::<Result<Option<Vec<_>>>>()?
        .map(|blocks| blocks.into_iter().flatten().collect::<Vec<_>>());
    let mut block_offset = 0;
    for ((input, header_path), header) in inputs.iter().zip(&headers) {
        let indices = mapping(&header.peel, writer.peel_subshells())?;
        let mut rows = DescriptorRows::open(input, header_path)?;
        while let Some(row) = rows.next_descriptor_row()? {
            writer.push(
                &remap(&row.values, &indices, width),
                block_offset + row.block_index,
            )?;
        }
        block_offset += header.block_lengths.len();
    }
    writer
        .prepare_with_zero_order_counts(counts.as_deref())?
        .publish()
}

/// Zero-order and first-order counts for one explicit symmetry block.
#[derive(Debug, Serialize)]
pub struct PartitionBlock {
    /// Zero-based block index shared by both inputs and the output.
    pub block_index: usize,
    /// Physical total `2J`, without the V3 storage offset of one.
    pub total_two_j: i8,
    /// Signed parity code, `1` for even and `-1` for odd.
    pub parity: i8,
    /// Number of reference rows written at the start of the output block.
    pub zero_count: usize,
    /// Number of full-basis rows appended after omitting exact reference matches.
    pub first_order_count: usize,
}

/// Input counts and output statistics for a [`zero_first`] run.
#[derive(Debug, Serialize)]
pub struct PartitionStats {
    /// Statistics for the published zero-first dataset.
    #[serde(flatten)]
    pub dataset: DatasetStats,
    /// Total CSF rows in the reference input.
    pub zero_csf_count: usize,
    /// Total CSF rows in the full-basis input before excluding reference matches.
    pub full_csf_count: usize,
    /// Total full-basis rows appended after excluding reference matches in each block.
    pub first_order_count: usize,
    /// Counts for each explicit symmetry block in output order.
    pub blocks: Vec<PartitionBlock>,
}

/// Put exact reference rows first within each explicitly aligned symmetry block.
/// Memory is bounded by one reference block plus the readers' batch buffers.
pub fn zero_first(
    reference: &Path,
    reference_header: &Path,
    full: &Path,
    full_header: &Path,
    output: &Path,
    output_header: &Path,
) -> Result<PartitionStats> {
    let mut zero = DescriptorRows::open(reference, reference_header)?;
    let mut all = DescriptorRows::open(full, full_header)?;
    ensure!(
        zero.header().block_lengths.len() == all.header().block_lengths.len(),
        "reference/full block counts differ"
    );
    ensure!(
        zero.header().electron_count == all.header().electron_count,
        "reference/full electron counts differ"
    );
    let common = common_header([zero.header(), all.header()].into_iter())?;
    let zero_map = mapping(&zero.header().peel, &common.peel)?;
    let full_map = mapping(&all.header().peel, &common.peel)?;
    let width = common.layout().row_len();
    let mut writer = DatasetWriter::try_new(
        common,
        output,
        output_header,
        &[reference, reference_header, full, full_header],
    )?;
    let mut blocks = Vec::with_capacity(zero.header().block_lengths.len());
    for block in 0..zero.header().block_lengths.len() {
        let zero_count = zero.header().block_lengths[block];
        let full_count = all.header().block_lengths[block];
        let mut reference_rows = HashSet::new();
        for row in zero.read_rows(zero_count) {
            let row = remap(&row?, &zero_map, width);
            writer.push(&row, block)?;
            reference_rows.insert(row);
        }
        let (two_j, parity) = zero.block_symmetry().context("empty reference block")?;
        let mut first_order_count = 0;
        for row in all.read_rows(full_count) {
            let row = row?;
            ensure!(
                row_symmetry(&row) == (two_j, parity),
                "reference/full block symmetries differ"
            );
            let row = remap(&row, &full_map, width);
            if !reference_rows.contains(&row) {
                writer.push(&row, block)?;
                first_order_count += 1;
            }
        }
        blocks.push(PartitionBlock {
            block_index: block,
            total_two_j: two_j,
            parity,
            zero_count,
            first_order_count,
        });
    }
    ensure!(
        zero.next_descriptor_row()?.is_none() && all.next_descriptor_row()?.is_none(),
        "unconsumed CSF blocks"
    );
    let zero_csf_count = zero.header().row_count();
    let full_csf_count = all.header().row_count();
    let first_order_count = blocks.iter().map(|b| b.first_order_count).sum();
    let counts = blocks
        .iter()
        .map(|block| block.zero_count)
        .collect::<Vec<_>>();
    Ok(PartitionStats {
        dataset: writer.finish_zero_first(&counts)?,
        zero_csf_count,
        full_csf_count,
        first_order_count,
        blocks,
    })
}

/// Resolve a selected basis against a full basis without decoding CSF strings.
/// Exact occupation, state, cumulative coupling and symmetry identify each CSF.
/// Each result is the corresponding full-basis ordinal, or `None` for an
/// unmatched row when `require_all` is false.
pub fn match_indices(
    full: &Path,
    full_header: &Path,
    selected: &Path,
    selected_header: &Path,
    require_all: bool,
) -> Result<Vec<Option<u64>>> {
    let mut pool = DescriptorRows::open(full, full_header)?;
    let mut seed = DescriptorRows::open(selected, selected_header)?;
    let common = common_header([pool.header(), seed.header()].into_iter())?;
    let full_map = mapping(&pool.header().peel, &common.peel)?;
    let seed_map = mapping(&seed.header().peel, &common.peel)?;
    let width = common.layout().row_len();
    let mut requested = HashMap::<Vec<i8>, usize>::new();
    let mut matched = Vec::new();
    while let Some(row) = seed.next_descriptor_row()? {
        ensure!(
            requested
                .insert(remap(&row.values, &seed_map, width), row.ordinal)
                .is_none(),
            "duplicate selected CSF makes alignment ambiguous"
        );
        matched.push(None);
    }
    while let Some(row) = pool.next_descriptor_row()? {
        if let Some(index) = requested.get(&remap(&row.values, &full_map, width)) {
            ensure!(
                matched[*index].is_none(),
                "duplicate full-pool CSF makes alignment ambiguous"
            );
            matched[*index] = Some(u64::try_from(row.ordinal)?);
        }
    }
    if require_all {
        ensure!(
            matched.iter().all(Option::is_some),
            "selected CSFs are not contained in the full basis"
        );
    }
    Ok(matched)
}

/// Keep rows whose occupied peel subshells all fit the supplied active space.
///
/// Limits are comma- or space-separated nonrelativistic labels, with at most
/// one maximum `n` per `l`. A limit applies to both relativistic partners.
/// The closed core and the relative order of surviving rows are preserved.
pub fn filter_active_space(
    input: &Path,
    input_header: &Path,
    output: &Path,
    output_header: &Path,
    active_space: &str,
) -> Result<DatasetStats> {
    let mut results = split_active_spaces(
        input,
        input_header,
        &[(output.into(), output_header.into(), active_space.into())],
    )?;
    Ok(results.remove(0))
}

/// Validate all targets, stream the source once, then prepare every pair before publication.
pub fn split_active_spaces(
    input: &Path,
    input_header: &Path,
    targets: &[(PathBuf, PathBuf, String)],
) -> Result<Vec<DatasetStats>> {
    ensure!(!targets.is_empty(), "no active-space targets");
    let mut source = DescriptorRows::open(input, input_header)?;
    let prefix_ends = source.header().zero_order_ends()?;
    let mut outputs = HashSet::new();
    let mut prepared = Vec::new();
    for (output, output_header, active_space) in targets {
        for path in [output, output_header] {
            ensure!(
                outputs.insert(output_identity(path)?),
                "duplicate output path"
            );
        }
        let mut maxima = HashMap::<u8, u8>::new();
        for label in active_space.split([',', ' ']).filter(|s| !s.is_empty()) {
            let orbital: Subshell = label.parse()?;
            ensure!(
                !label.ends_with('-'),
                "active-space limits must use nonrelativistic orbital names"
            );
            ensure!(
                maxima.insert(orbital.l(), orbital.n()).is_none(),
                "duplicate active orbital symmetry: {label}"
            );
        }
        ensure!(!maxima.is_empty(), "active space is empty");
        let retained = source
            .header()
            .peel
            .iter()
            .map(|label| {
                let shell: Subshell = label.parse()?;
                Ok(maxima.get(&shell.l()).is_some_and(|n| shell.n() <= *n))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut header = source.header().clone();
        header.peel = header
            .peel
            .into_iter()
            .zip(&retained)
            .filter_map(|(s, keep)| keep.then_some(s))
            .collect();
        header.lines[3] = header.peel.join("  ");
        let writer = DatasetWriter::try_new(header, output, output_header, &[input, input_header])?
            .with_batch_rows(8192 / targets.len());
        prepared.push((retained, writer, BTreeMap::<usize, usize>::new()));
    }
    while let Some(DescriptorRow {
        ordinal,
        block_index: block,
        values: row,
    }) = source.next_descriptor_row()?
    {
        for (retained, writer, counts) in &mut prepared {
            if retained
                .iter()
                .enumerate()
                .any(|(i, keep)| !keep && row[4 * i] != 0)
            {
                continue;
            }
            let mut filtered = Vec::new();
            for (i, keep) in retained.iter().enumerate() {
                if *keep {
                    filtered.extend_from_slice(&row[4 * i..4 * i + 4]);
                }
            }
            filtered.extend_from_slice(&row[row.len() - 2..]);
            writer.push(&filtered, block)?;
            if let Some(ends) = &prefix_ends {
                *counts.entry(block).or_default() += usize::from(ordinal < ends[block]);
            }
        }
    }
    let ready = prepared
        .into_iter()
        .map(|(_, writer, counts)| {
            let counts = prefix_ends
                .as_ref()
                .map(|_| counts.into_values().collect::<Vec<_>>());
            writer.prepare_with_zero_order_counts(counts.as_deref())
        })
        .collect::<Result<Vec<_>>>()?;
    crate::descriptor_dataset::PreparedDataset::publish_all(ready)
}

/// Split explicit source blocks; duplicate J/parity blocks get distinct names.
pub fn split_blocks(
    input: &Path,
    input_header: &Path,
    directory: &Path,
    prefix: &str,
) -> Result<Vec<DatasetStats>> {
    ensure!(
        !prefix.is_empty() && !prefix.contains(['/', '\\']),
        "invalid output prefix"
    );
    let mut source = DescriptorRows::open(input, input_header)?;
    let header = source.header().clone();
    let zero_order_counts = header.zero_order_counts()?;
    let mut results = Vec::with_capacity(header.block_lengths.len());
    for (block, &length) in header.block_lengths.iter().enumerate() {
        let mut rows = source.read_rows(length);
        let first = rows.next().context("empty source block")??;
        let (j, parity) = row_symmetry(&first);
        let p = if parity == 1 { "even" } else { "odd" };
        let path = directory.join(format!("{prefix}_b{block:03}_{j}_{p}_desc.parquet"));
        let mut writer = DatasetWriter::try_new(
            header.clone(),
            &path,
            &header_path(&path)?,
            &[input, input_header],
        )?;
        writer.push(&first, 0)?;
        for row in rows {
            writer.push(&row?, 0)?;
        }
        // A whole block keeps its source row order and therefore its partition.
        let counts = zero_order_counts
            .as_ref()
            .map(|counts| std::slice::from_ref(&counts[block]));
        results.push(writer.prepare_with_zero_order_counts(counts)?);
    }
    ensure!(
        source.next_descriptor_row()?.is_none(),
        "unconsumed CSF blocks"
    );
    crate::descriptor_dataset::PreparedDataset::publish_all(results)
}
