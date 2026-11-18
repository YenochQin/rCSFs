//! Ordered numerical CSF transformations over descriptor/header datasets.
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::configuration_generation::Subshell;
use crate::descriptor_dataset::{
    DatasetHeader, DatasetStats, DatasetWriter, DescriptorRows, header_path, output_identity,
};
use crate::descriptor_schema::{UNOCCUPIED_V2, row_symmetry};

#[derive(Debug, Clone)]
pub struct ActiveSpaceTarget {
    pub maximum_orbitals: String,
    pub output: PathBuf,
}

/// Merge orbital sequences without changing any input's relative coupling order.
fn common_header(headers: &[DatasetHeader]) -> Result<DatasetHeader> {
    let mut header = headers.first().context("no input datasets")?.clone();
    for other in &headers[1..] {
        header.add_truncations_from(other)?;
    }
    let mut labels = Vec::<String>::new();
    let mut edges = HashSet::<(String, String)>::new();
    for source in headers {
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
    if let Some(source) = headers.iter().find(|source| source.peel == ordered) {
        header.lines[3].clone_from(&source.lines[3]);
    } else {
        header.lines[3] = ordered.join("  ");
    }
    header.peel = ordered;
    Ok(header)
}

fn mapping(source: &DatasetHeader, target: &DatasetHeader) -> Result<Vec<usize>> {
    source
        .peel
        .iter()
        .map(|label| {
            target
                .peel
                .iter()
                .position(|value| value == label)
                .context("output orbital table omits an input orbital")
        })
        .collect()
}

fn remap(row: &[i8], indices: &[usize], width: usize) -> Vec<i8> {
    let mut mapped = UNOCCUPIED_V2.repeat((width - 2) / 4);
    mapped.extend_from_slice(&row[row.len() - 2..]);
    for (source, target) in indices.iter().enumerate() {
        mapped[4 * target..4 * target + 4].copy_from_slice(&row[4 * source..4 * source + 4]);
    }
    mapped
}

pub fn merge(
    inputs: &[(PathBuf, PathBuf)],
    output: &Path,
    output_header: &Path,
) -> Result<DatasetStats> {
    ensure!(!inputs.is_empty(), "no input datasets");
    let headers = inputs
        .iter()
        .map(|(data, header)| Ok(DescriptorRows::open(data, header)?.header))
        .collect::<Result<Vec<_>>>()?;
    let common = common_header(&headers)?;
    let width = common.layout().row_len();
    let input_paths = inputs
        .iter()
        .flat_map(|(a, b)| [a.as_path(), b.as_path()])
        .collect::<Vec<_>>();
    let mut writer = DatasetWriter::new(common.clone(), output, output_header, &input_paths)?;
    let mut block_offset = 0;
    for ((input, header_path), header) in inputs.iter().zip(&headers) {
        let indices = mapping(header, &common)?;
        let mut rows = DescriptorRows::open(input, header_path)?;
        while let Some((_, block, row)) = rows.next_row()? {
            writer.push(&remap(&row, &indices, width), block_offset + block)?;
        }
        block_offset += header.block_lengths.len();
    }
    writer.finish()
}

#[derive(Debug, Serialize)]
pub struct PartitionBlock {
    pub block_index: usize,
    pub total_two_j: i8,
    pub parity: i8,
    pub zero_count: usize,
    pub first_order_count: usize,
}

#[derive(Debug, Serialize)]
pub struct PartitionStats {
    #[serde(flatten)]
    pub dataset: DatasetStats,
    pub zero_csf_count: usize,
    pub full_csf_count: usize,
    pub first_order_count: usize,
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
        zero.header.block_lengths.len() == all.header.block_lengths.len(),
        "reference/full block counts differ"
    );
    let common = common_header(&[zero.header.clone(), all.header.clone()])?;
    let zero_map = mapping(&zero.header, &common)?;
    let full_map = mapping(&all.header, &common)?;
    let width = common.layout().row_len();
    let mut writer = DatasetWriter::new(
        common,
        output,
        output_header,
        &[reference, reference_header, full, full_header],
    )?;
    let mut blocks = Vec::with_capacity(zero.header.block_lengths.len());
    for block in 0..zero.header.block_lengths.len() {
        let zero_count = zero.header.block_lengths[block];
        let full_count = all.header.block_lengths[block];
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
        zero.next_row()?.is_none() && all.next_row()?.is_none(),
        "unconsumed CSF blocks"
    );
    let zero_csf_count = zero.header.row_count();
    let full_csf_count = all.header.row_count();
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
/// Exact descriptor channels (including omitted versus printed zero) are retained.
pub fn match_indices(
    full: &Path,
    full_header: &Path,
    selected: &Path,
    selected_header: &Path,
    require_all: bool,
) -> Result<Vec<Option<u64>>> {
    let mut pool = DescriptorRows::open(full, full_header)?;
    let mut seed = DescriptorRows::open(selected, selected_header)?;
    let common = common_header(&[pool.header.clone(), seed.header.clone()])?;
    let full_map = mapping(&pool.header, &common)?;
    let seed_map = mapping(&seed.header, &common)?;
    let width = common.layout().row_len();
    let mut requested = HashMap::<Vec<i8>, usize>::new();
    let mut matched = Vec::new();
    while let Some((ordinal, _, row)) = seed.next_row()? {
        ensure!(
            requested
                .insert(remap(&row, &seed_map, width), ordinal)
                .is_none(),
            "duplicate selected CSF makes alignment ambiguous"
        );
        matched.push(None);
    }
    while let Some((ordinal, _, row)) = pool.next_row()? {
        if let Some(index) = requested.get(&remap(&row, &full_map, width)) {
            ensure!(
                matched[*index].is_none(),
                "duplicate full-pool CSF makes alignment ambiguous"
            );
            matched[*index] = Some(u64::try_from(ordinal)?);
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
            .header
            .peel
            .iter()
            .map(|label| {
                let shell: Subshell = label.parse()?;
                Ok(maxima.get(&shell.l()).is_some_and(|n| shell.n() <= *n))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut header = source.header.clone();
        header.peel = header
            .peel
            .into_iter()
            .zip(&retained)
            .filter_map(|(s, keep)| keep.then_some(s))
            .collect();
        header.lines[3] = header.peel.join("  ");
        let writer = DatasetWriter::new(header, output, output_header, &[input, input_header])?
            .with_batch_rows(8192 / targets.len());
        prepared.push((retained, writer));
    }
    while let Some((_, block, row)) = source.next_row()? {
        for (retained, writer) in &mut prepared {
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
        }
    }
    let ready = prepared
        .into_iter()
        .map(|(_, w)| w.prepare())
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
    let header = source.header.clone();
    let zero_order_counts = header.zero_order_counts()?;
    let mut results = Vec::with_capacity(header.block_lengths.len());
    for (block, &length) in header.block_lengths.iter().enumerate() {
        let mut rows = source.read_rows(length);
        let first = rows.next().context("empty source block")??;
        let (j, parity) = row_symmetry(&first);
        let p = if parity == 1 { "even" } else { "odd" };
        let path = directory.join(format!("{prefix}_b{block:03}_{j}_{p}_descriptors.parquet"));
        let mut writer = DatasetWriter::new(
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
    ensure!(source.next_row()?.is_none(), "unconsumed CSF blocks");
    crate::descriptor_dataset::PreparedDataset::publish_all(results)
}
