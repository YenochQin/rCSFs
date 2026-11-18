//! Ordered numerical CSF transformations over descriptor/header datasets.
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::configuration_generation::Subshell;
use crate::descriptor_dataset::{
    DatasetHeader, DatasetStats, DatasetWriter, DescriptorRows, header_path, output_identity,
};

#[derive(Debug, Clone)]
pub struct ActiveSpaceTarget {
    pub maximum_orbitals: String,
    pub output: PathBuf,
}

/// Merge orbital sequences without changing any input's relative coupling order.
fn common_header(headers: &[DatasetHeader]) -> Result<DatasetHeader> {
    let mut header = headers.first().context("no input datasets")?.clone();
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
    let mut mapped = [0, -1, 0, -1].repeat((width - 2) / 4);
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
    let mut z = zero.next_row()?;
    let mut f = all.next_row()?;
    let mut blocks = Vec::new();
    for block in 0..zero.header.block_lengths.len() {
        let mut reference_rows = HashSet::new();
        let mut zero_count = 0;
        let mut first_count = 0;
        let mut symmetry = None;
        while z.as_ref().is_some_and(|r| r.1 == block) {
            let (_, _, row) = z.take().expect("reference row");
            let row = remap(&row, &zero_map, width);
            symmetry = Some((row[width - 2], row[width - 1]));
            writer.push(&row, block)?;
            reference_rows.insert(row);
            zero_count += 1;
            z = zero.next_row()?;
        }
        let (two_j, parity) = symmetry.context("empty reference block")?;
        while f.as_ref().is_some_and(|r| r.1 == block) {
            let (_, _, row) = f.take().expect("full row");
            let row = remap(&row, &full_map, width);
            ensure!(
                (row[width - 2], row[width - 1]) == (two_j, parity),
                "reference/full block symmetries differ"
            );
            if !reference_rows.contains(&row) {
                writer.push(&row, block)?;
                first_count += 1;
            }
            f = all.next_row()?;
        }
        blocks.push(PartitionBlock {
            block_index: block,
            total_two_j: two_j,
            parity,
            zero_count,
            first_order_count: first_count,
        });
    }
    ensure!(z.is_none() && f.is_none(), "unconsumed CSF blocks");
    let zero_csf_count = zero.header.row_count();
    let full_csf_count = all.header.row_count();
    let first_order_count = blocks.iter().map(|b| b.first_order_count).sum();
    Ok(PartitionStats {
        dataset: writer.finish()?,
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
    ready.into_iter().map(|r| r.publish()).collect()
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
    let mut current = None;
    let mut current_block = None;
    let mut results = Vec::new();
    while let Some((_, block, row)) = source.next_row()? {
        if current_block != Some(block) {
            if let Some(writer) = current.take() {
                results.push(DatasetWriter::prepare(writer)?);
            }
            let j = row[row.len() - 2];
            let p = if row[row.len() - 1] == 1 {
                "even"
            } else {
                "odd"
            };
            let path = directory.join(format!("{prefix}_b{block:03}_{j}_{p}_descriptors.parquet"));
            current = Some(DatasetWriter::new(
                header.clone(),
                &path,
                &header_path(&path)?,
                &[input, input_header],
            )?);
            current_block = Some(block);
        }
        current.as_mut().expect("block writer").push(&row, 0)?;
    }
    if let Some(writer) = current {
        results.push(writer.prepare()?);
    }
    results.into_iter().map(|r| r.publish()).collect()
}
