//! Descriptor CSF datasets: strict paired input, bounded row I/O and publication.
//!
//! The numeric Parquet and its byte-bound TOML header are one logical dataset.
//! Transformations spool numeric batches until the output block table is known;
//! they never restore temporary CSF text or three-string Parquet files.

use anyhow::{Context, Result, ensure};
use arrow::array::{Array, ArrayRef, Int8Array};
use arrow::record_batch::RecordBatch;
use arrow_ipc::{reader::FileReader, writer::FileWriter};
use parquet::arrow::{
    ArrowWriter, arrow_reader::ParquetRecordBatchReader,
    arrow_reader::ParquetRecordBatchReaderBuilder,
};
use parquet::file::properties::WriterProperties;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::atomic_output::{
    TemporaryOutput, create_temporary_output, ensure_distinct_inputs,
    ensure_output_does_not_alias_input, publish_temporary_output,
};
use crate::complete_csf::CompleteCsfFile;
use crate::configuration_generation::Subshell;
use crate::csfs_descriptor::CSFDescriptorGenerator;
use crate::descriptor_schema::{
    DescriptorLayout, DescriptorVersion, output_kv_metadata, output_schema,
    validate_descriptor_storage, validate_output_schema, validate_record,
    validate_seniority_encoding,
};
use crate::descriptor_v2::decode_v2_into;

const BATCH_ROWS: usize = 8192;

pub fn read_header(path: &Path) -> Result<DatasetHeader> {
    DatasetHeader::parse(toml::from_str(&fs::read_to_string(path)?)?, true)
}

pub fn header_path(descriptors: &Path) -> Result<PathBuf> {
    ensure!(
        descriptors.extension().is_some_and(|s| s == "parquet"),
        "expected descriptor Parquet: {}",
        descriptors.display()
    );
    let stem = descriptors
        .file_stem()
        .and_then(|s| s.to_str())
        .context("expected UTF-8 descriptor filename")?;
    let stem = stem.strip_suffix("_descriptors").unwrap_or(stem);
    Ok(descriptors.with_file_name(format!("{stem}_header.toml")))
}

#[derive(Clone, Debug)]
pub struct DatasetHeader {
    pub document: toml::Value,
    pub lines: [String; 5],
    pub peel: Vec<String>,
    pub core: Vec<Subshell>,
    pub block_lengths: Vec<usize>,
    pub electron_count: Option<usize>,
}

/// Legacy bound headers may omit the conversion statistics.
fn truncated_count(document: &toml::Value) -> Result<i64> {
    let Some(stats) = document.get("conversion_stats") else {
        return Ok(0);
    };
    let stats = stats.as_table().context("invalid conversion_stats table")?;
    stats
        .get("truncated_count")
        .map(|v| {
            v.as_integer()
                .filter(|n| *n >= 0)
                .context("invalid truncated_count")
        })
        .transpose()
        .map(|count| count.unwrap_or(0))
}

impl DatasetHeader {
    pub(crate) fn add_truncations_from(&mut self, source: &Self) -> Result<()> {
        let added = truncated_count(&source.document)?;
        if added != 0 {
            let total = truncated_count(&self.document)?
                .checked_add(added)
                .context("source truncated_count overflow")?;
            self.document
                .as_table_mut()
                .context("invalid header document")?
                .entry("conversion_stats")
                .or_insert_with(|| toml::Table::new().into())
                .as_table_mut()
                .context("invalid conversion_stats table")?
                .insert("truncated_count".into(), total.into());
        }
        Ok(())
    }

    pub fn from_lines(lines: [String; 5]) -> Result<Self> {
        let mut document = toml::Table::new();
        let mut info = toml::Table::new();
        info.insert(
            "header_lines".into(),
            toml::Value::Array(lines.iter().cloned().map(toml::Value::String).collect()),
        );
        document.insert("header_info".into(), info.into());
        Self::parse(document.into(), false)
    }

    fn parse(document: toml::Value, require_blocks: bool) -> Result<Self> {
        let raw = document
            .get("header_info")
            .and_then(|v| v.get("header_lines"))
            .and_then(toml::Value::as_array)
            .context("missing header_info.header_lines")?;
        let lines: [String; 5] = raw
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .context("header line must be a string")
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected five header lines"))?;
        for (index, label) in [
            (0, "Core subshells:"),
            (2, "Peel subshells:"),
            (4, "CSF(s):"),
        ] {
            ensure!(lines[index].trim() == label, "invalid header label");
        }
        ensure!(
            lines.iter().all(|s| !s.contains(['\r', '\n', '\0'])),
            "embedded newline or NUL in header"
        );
        let core = lines[1]
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<Vec<Subshell>>>()?;
        let peel = lines[3]
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let orbitals = core
            .iter()
            .copied()
            .chain(
                peel.iter()
                    .map(|s| s.parse())
                    .collect::<Result<Vec<Subshell>>>()?,
            )
            .collect::<Vec<_>>();
        ensure!(!orbitals.is_empty(), "empty orbital table");
        ensure!(
            orbitals.iter().collect::<HashSet<_>>().len() == orbitals.len(),
            "duplicate core/peel orbital"
        );
        ensure!(
            peel.len() <= usize::from(u16::MAX),
            "too many peel orbitals"
        );
        let block_lengths = document
            .get("block_info")
            .and_then(|v| v.get("block_lengths"))
            .map(|v| {
                v.as_array()
                    .context("block_lengths must be an array")?
                    .iter()
                    .map(|v| {
                        let n = v
                            .as_integer()
                            .and_then(|n| usize::try_from(n).ok())
                            .context("invalid block length")?;
                        ensure!(n > 0, "empty symmetry block");
                        Ok(n)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        if require_blocks {
            ensure!(!block_lengths.is_empty(), "missing symmetry blocks");
        }
        let total = block_lengths
            .iter()
            .try_fold(0usize, |sum, n| sum.checked_add(*n))
            .context("CSF count overflow")?;
        for (section, key, expected) in [
            ("block_info", "block_count", block_lengths.len()),
            ("conversion_stats", "csf_count", total),
        ] {
            if let Some(value) = document.get(section).and_then(|v| v.get(key)) {
                ensure!(
                    value.as_integer().and_then(|n| usize::try_from(n).ok()) == Some(expected),
                    "inconsistent {key}"
                );
            }
        }
        let electron_count = document
            .get("header_info")
            .and_then(|v| v.get("electron_count"))
            .map(|v| {
                v.as_integer()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|n| *n > 0)
                    .context("invalid electron_count")
            })
            .transpose()?;
        truncated_count(&document)?;
        Ok(Self {
            document,
            lines,
            peel,
            core,
            block_lengths,
            electron_count,
        })
    }

    pub fn layout(&self) -> DescriptorLayout {
        DescriptorLayout::new(DescriptorVersion::V2, self.peel.len())
    }

    pub fn row_count(&self) -> usize {
        self.block_lengths.iter().sum()
    }

    fn validate_row(&mut self, row: &[i8]) -> Result<()> {
        ensure!(
            row.len() == self.layout().row_len(),
            "descriptor width mismatch"
        );
        if self.peel.is_empty() {
            ensure!(
                !self.core.is_empty() && row == [0, 1],
                "invalid closed-core descriptor"
            );
        } else {
            let (mut occupied, mut couplings) = (Vec::new(), Vec::new());
            let (total_j, _) = decode_v2_into(
                row,
                self.layout(),
                &self.peel,
                &mut occupied,
                &mut couplings,
            )?;
            if occupied.is_empty() {
                ensure!(
                    !self.core.is_empty() && total_j == 0,
                    "invalid empty peel record"
                );
            } else {
                validate_record(&self.peel, &occupied, &couplings)?;
            }
            let shell_j = occupied
                .iter()
                .map(|shell| shell.state.map_or(0, |s| s.two_j))
                .collect::<Vec<_>>();
            let last_nonzero = shell_j.iter().rposition(|j| *j != 0);
            let mut previous = 0u16;
            for (index, j) in shell_j.iter().copied().enumerate() {
                let printed = couplings
                    .iter()
                    .find(|c| usize::from(c.boundary) == index + 1)
                    .map(|c| c.two_j);
                let coupled = if let Some(value) = printed {
                    value
                } else if last_nonzero.is_none_or(|last| index >= last) {
                    total_j
                } else if j == 0 {
                    previous
                } else if previous == 0 {
                    j
                } else {
                    anyhow::bail!("missing nontrivial intermediate coupling")
                };
                ensure!(
                    coupled >= previous.abs_diff(j)
                        && coupled <= previous + j
                        && (previous + j - coupled).is_multiple_of(2),
                    "invalid angular momentum coupling"
                );
                previous = coupled;
            }
            ensure!(previous == total_j, "incorrect final angular momentum");
        }
        let mut electrons: usize = self.core.iter().map(|s| usize::from(s.capacity())).sum();
        let mut parity = 0usize;
        for (index, label) in self.peel.iter().enumerate() {
            let orbital: Subshell = label.parse()?;
            let population = usize::try_from(row[index * 4]).context("negative occupation")?;
            ensure!(
                population <= usize::from(orbital.capacity()),
                "occupation exceeds capacity"
            );
            electrons += population;
            parity += population * usize::from(orbital.l());
        }
        ensure!(electrons > 0, "empty electron basis");
        ensure!(
            row[row.len() - 1] == if parity.is_multiple_of(2) { 1 } else { -1 },
            "incorrect parity"
        );
        if let Some(expected) = self.electron_count {
            ensure!(expected == electrons, "electron count mismatch");
        }
        self.electron_count = Some(electrons);
        Ok(())
    }

    pub(crate) fn finalized(mut self, lengths: &[usize]) -> Result<Self> {
        ensure!(!lengths.is_empty(), "cannot publish an empty CSF dataset");
        if self.peel.is_empty() {
            ensure!(lengths == [1], "closed-core dataset must contain one CSF");
        }
        self.block_lengths = lengths.to_vec();
        let count = self.row_count();
        let truncated = truncated_count(&self.document)?;
        let root = self
            .document
            .as_table_mut()
            .context("invalid header document")?;
        let info = root
            .get_mut("header_info")
            .and_then(toml::Value::as_table_mut)
            .context("invalid header_info")?;
        info.insert(
            "header_lines".into(),
            self.lines
                .iter()
                .cloned()
                .map(toml::Value::String)
                .collect::<Vec<_>>()
                .into(),
        );
        info.insert(
            "electron_count".into(),
            i64::try_from(self.electron_count.context("missing electron count")?)?.into(),
        );
        let mut blocks = toml::Table::new();
        blocks.insert("block_count".into(), i64::try_from(lengths.len())?.into());
        blocks.insert(
            "block_lengths".into(),
            lengths
                .iter()
                .map(|n| i64::try_from(*n).map(toml::Value::Integer))
                .collect::<std::result::Result<Vec<_>, _>>()?
                .into(),
        );
        root.insert("block_info".into(), blocks.into());
        let mut stats = toml::Table::new();
        stats.insert("csf_count".into(), i64::try_from(count)?.into());
        stats.insert(
            "total_lines".into(),
            i64::try_from(count.checked_mul(3).context("line count overflow")?)?.into(),
        );
        stats.insert("truncated_count".into(), truncated.into());
        root.insert("conversion_stats".into(), stats.into());
        Ok(self)
    }

    #[cfg(feature = "python")]
    pub(crate) fn read_metadata(self) -> Result<toml::Value> {
        // Older bound headers can omit inferred values. Fill those fields while
        // preserving the source's conversion history, including its line count.
        let mut document = self.document.clone();
        let lengths = self.block_lengths.clone();
        let defaults = self.finalized(&lengths)?.document;
        let root = document.as_table_mut().context("invalid header document")?;
        for name in ["header_info", "block_info", "conversion_stats"] {
            let section = root
                .entry(name)
                .or_insert_with(|| toml::Table::new().into())
                .as_table_mut()
                .with_context(|| format!("invalid {name} table"))?;
            for (key, value) in defaults[name].as_table().expect("finalized table") {
                section.entry(key).or_insert_with(|| value.clone());
            }
        }
        Ok(document)
    }
}

pub struct DescriptorRows {
    pub header: DatasetHeader,
    reader: ParquetRecordBatchReader,
    batch: Option<RecordBatch>,
    batch_row: usize,
    ordinal: usize,
    block: usize,
    block_rows: usize,
    symmetry: Option<(i8, i8)>,
}

impl DescriptorRows {
    pub fn open(path: &Path, header_path: &Path) -> Result<Self> {
        ensure_distinct_inputs(path, header_path, "descriptor and header paths must differ")?;
        let bytes = fs::read(header_path)?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let header = DatasetHeader::parse(toml::from_str(std::str::from_utf8(&bytes)?)?, true)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
        let metadata = builder.metadata().file_metadata();
        ensure!(
            usize::try_from(metadata.num_rows()).ok() == Some(header.row_count()),
            "descriptor/header row count mismatch"
        );
        let kv = metadata
            .key_value_metadata()
            .context("missing descriptor metadata")?;
        let get = |key: &str| -> Result<&str> {
            let entries = kv.iter().filter(|v| v.key == key).collect::<Vec<_>>();
            ensure!(entries.len() == 1, "missing or duplicate {key}");
            entries[0]
                .value
                .as_deref()
                .with_context(|| format!("missing {key} value"))
        };
        for (key, expected) in [
            ("descriptor_version", "2"),
            ("channels_per_subshell", "4"),
            ("missing_sentinel", "-1"),
            ("normalized", "false"),
        ] {
            ensure!(get(key)? == expected, "invalid {key}");
        }
        ensure!(
            get("source_header_sha256")? == hash,
            "descriptor source_header_sha256 does not match the hash of the supplied header"
        );
        ensure!(
            get("subshell_count")?.parse::<usize>()? == header.peel.len(),
            "peel count mismatch"
        );
        validate_seniority_encoding(kv)?;
        validate_descriptor_storage(kv)?;
        validate_output_schema(builder.schema(), header.layout())?;
        for (key, expected) in [
            ("peel_subshells", header.peel.clone()),
            ("feature_columns", header.layout().feature_column_names()),
            ("global_columns", header.layout().global_column_names()),
        ] {
            ensure!(
                serde_json::from_str::<Vec<String>>(get(key)?)? == expected,
                "invalid {key}"
            );
        }
        if header.peel.is_empty() {
            ensure!(
                header.block_lengths == [1],
                "closed-core dataset must contain one CSF"
            );
        }
        Ok(Self {
            header,
            reader: builder.with_batch_size(BATCH_ROWS).build()?,
            batch: None,
            batch_row: 0,
            ordinal: 0,
            block: 0,
            block_rows: 0,
            symmetry: None,
        })
    }

    pub fn next_row(&mut self) -> Result<Option<(usize, usize, Vec<i8>)>> {
        while self
            .batch
            .as_ref()
            .is_none_or(|b| self.batch_row == b.num_rows())
        {
            self.batch = self.reader.next().transpose()?;
            self.batch_row = 0;
            if self.batch.is_none() {
                ensure!(
                    self.ordinal == self.header.row_count(),
                    "truncated descriptor payload"
                );
                return Ok(None);
            }
        }
        ensure!(
            self.ordinal < self.header.row_count(),
            "too many descriptor rows"
        );
        let batch = self.batch.as_ref().expect("batch loaded");
        let row = batch
            .columns()
            .iter()
            .map(|column| {
                ensure!(column.null_count() == 0, "null descriptor channel");
                Ok(column
                    .as_any()
                    .downcast_ref::<Int8Array>()
                    .context("non-Int8 descriptor channel")?
                    .value(self.batch_row))
            })
            .collect::<Result<Vec<_>>>()?;
        if self.block_rows == self.header.block_lengths[self.block] {
            self.block += 1;
            self.block_rows = 0;
            self.symmetry = None;
        }
        self.header.validate_row(&row)?;
        let symmetry = (row[row.len() - 2], row[row.len() - 1]);
        ensure!(
            self.symmetry.is_none_or(|s| s == symmetry),
            "mixed symmetry within explicit block"
        );
        self.symmetry = Some(symmetry);
        let result = (self.ordinal, self.block, row);
        self.ordinal += 1;
        self.block_rows += 1;
        self.batch_row += 1;
        Ok(Some(result))
    }
}

#[derive(Debug, Serialize)]
pub struct DatasetStats {
    pub success: bool,
    pub output_file: String,
    pub header_file: String,
    pub record_count: usize,
    pub block_count: usize,
    pub block_lengths: Vec<usize>,
    pub electron_count: usize,
}

pub struct DatasetWriter {
    header: DatasetHeader,
    output: PathBuf,
    header_output: PathBuf,
    spool: TemporaryOutput,
    writer: FileWriter<File>,
    columns: Vec<Vec<i8>>,
    lengths: Vec<usize>,
    current_block: Option<usize>,
    symmetry: Option<(i8, i8)>,
    batch_rows: usize,
}

pub(crate) struct PreparedDataset {
    header_temp: TemporaryOutput,
    output_temp: TemporaryOutput,
    header_path: PathBuf,
    output_path: PathBuf,
    stats: DatasetStats,
}

impl PreparedDataset {
    pub(crate) fn publish(self) -> Result<DatasetStats> {
        crate::atomic_output::publish_staged_outputs(&self.files(), false)?;
        Ok(self.stats)
    }

    fn files(&self) -> [(&Path, &Path); 2] {
        [
            (self.header_temp.path(), &self.header_path),
            (self.output_temp.path(), &self.output_path),
        ]
    }

    pub(crate) fn publish_all(datasets: Vec<Self>) -> Result<Vec<DatasetStats>> {
        let files = datasets.iter().flat_map(Self::files).collect::<Vec<_>>();
        crate::atomic_output::publish_staged_outputs(&files, false)?;
        Ok(datasets.into_iter().map(|d| d.stats).collect())
    }
}

pub(crate) fn output_identity(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok(parent
        .canonicalize()?
        .join(path.file_name().context("output needs a filename")?))
}

impl DatasetWriter {
    pub fn new(
        header: DatasetHeader,
        output: &Path,
        header_output: &Path,
        inputs: &[&Path],
    ) -> Result<Self> {
        ensure!(
            output.extension().is_some_and(|e| e == "parquet"),
            "output must be descriptor Parquet"
        );
        ensure!(
            output_identity(output)? != output_identity(header_output)?,
            "output descriptor and header paths alias"
        );
        crate::atomic_output::ensure_new_output(output)?;
        crate::atomic_output::ensure_new_output(header_output)?;
        ensure_distinct_inputs(
            output,
            header_output,
            "output descriptor and header paths must differ",
        )?;
        for input in inputs {
            ensure_output_does_not_alias_input(output, input, "CSF")?;
            ensure_output_does_not_alias_input(header_output, input, "CSF")?;
        }
        let (spool, file) = create_temporary_output(output)?;
        let schema = output_schema(header.layout())?;
        let writer = FileWriter::try_new(file, &schema)?;
        let columns = (0..header.layout().row_len())
            .map(|_| Vec::with_capacity(BATCH_ROWS))
            .collect();
        Ok(Self {
            header,
            output: output.into(),
            header_output: header_output.into(),
            spool,
            writer,
            columns,
            lengths: Vec::new(),
            current_block: None,
            symmetry: None,
            batch_rows: BATCH_ROWS,
        })
    }

    pub(crate) fn with_batch_rows(mut self, rows: usize) -> Self {
        self.batch_rows = rows.clamp(1, BATCH_ROWS);
        self.columns = (0..self.header.layout().row_len())
            .map(|_| Vec::with_capacity(self.batch_rows))
            .collect();
        self
    }

    pub fn push(&mut self, row: &[i8], block: usize) -> Result<()> {
        self.header.validate_row(row)?;
        if self.current_block != Some(block) {
            self.lengths.push(0);
            self.current_block = Some(block);
            self.symmetry = None;
        }
        let symmetry = (row[row.len() - 2], row[row.len() - 1]);
        ensure!(
            self.symmetry.is_none_or(|s| s == symmetry),
            "mixed output block symmetry"
        );
        self.symmetry = Some(symmetry);
        *self.lengths.last_mut().expect("block initialized") += 1;
        for (column, value) in self.columns.iter_mut().zip(row) {
            column.push(*value);
        }
        if self.columns[0].len() == self.batch_rows {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.columns[0].is_empty() {
            return Ok(());
        }
        let arrays = self
            .columns
            .iter_mut()
            .map(|column| {
                Arc::new(Int8Array::from(std::mem::replace(
                    column,
                    Vec::with_capacity(self.batch_rows),
                ))) as ArrayRef
            })
            .collect();
        self.writer.write(&RecordBatch::try_new(
            output_schema(self.header.layout())?,
            arrays,
        )?)?;
        Ok(())
    }

    pub fn finish(self) -> Result<DatasetStats> {
        self.prepare()?.publish()
    }

    pub(crate) fn prepare(mut self) -> Result<PreparedDataset> {
        self.flush()?;
        self.writer.finish()?;
        drop(self.writer);
        let header = self.header.finalized(&self.lengths)?;
        let bytes = toml::to_string(&header.document)?.into_bytes();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let (header_temp, mut header_file) = create_temporary_output(&self.header_output)?;
        header_file.write_all(&bytes)?;
        header_file.sync_all()?;
        drop(header_file);
        let metadata = output_kv_metadata(
            header.layout(),
            &header.peel,
            Some(&hash),
            self.header_output.file_name().and_then(|s| s.to_str()),
        );
        let properties = WriterProperties::builder()
            .set_key_value_metadata(Some(metadata))
            .set_max_row_group_row_count(Some(BATCH_ROWS))
            .set_compression(crate::csfs_descriptor::parquet_batch::parse_compression(
                None,
            )?)
            .build();
        let (output_temp, output_file) = create_temporary_output(&self.output)?;
        let mut output = ArrowWriter::try_new(
            output_file,
            output_schema(header.layout())?,
            Some(properties),
        )?;
        for batch in FileReader::try_new(File::open(self.spool.path())?, None)? {
            output.write(&batch?)?;
        }
        output.close()?;
        let stats = DatasetStats {
            success: true,
            output_file: self.output.display().to_string(),
            header_file: self.header_output.display().to_string(),
            record_count: header.row_count(),
            block_count: self.lengths.len(),
            block_lengths: self.lengths,
            electron_count: header.electron_count.expect("validated electron count"),
        };
        Ok(PreparedDataset {
            header_temp,
            output_temp,
            header_path: self.header_output,
            output_path: self.output,
            stats,
        })
    }
}

/// Import GRASP text directly to numeric descriptors, without string Parquet.
pub fn import_text(input: &Path, output: &Path, header_output: &Path) -> Result<DatasetStats> {
    let mut lines = BufReader::new(File::open(input)?).lines();
    let mut header_lines = Vec::new();
    for _ in 0..5 {
        header_lines.push(lines.next().context("truncated CSF header")??);
    }
    let header = DatasetHeader::from_lines(header_lines.try_into().expect("five header lines"))?;
    let closed_core = header.peel.is_empty();
    let has_core = !header.core.is_empty();
    let width = header.layout().row_len();
    let parser = CSFDescriptorGenerator::new(header.peel.clone());
    let mut writer = DatasetWriter::new(header, output, header_output, &[input])?;
    let mut block = 0;
    let mut count_in_block = 0;
    while let Some(line) = lines.next() {
        let first = line?;
        if first.trim() == "*" {
            ensure!(count_in_block > 0, "empty CSF block");
            block += 1;
            count_in_block = 0;
            continue;
        }
        let second = lines.next().context("truncated CSF state line")??;
        let third = lines.next().context("truncated CSF coupling line")??;
        let row = if closed_core
            || (has_core
                && first.trim().is_empty()
                && second.trim().is_empty()
                && third.trim() == "0+")
        {
            ensure!(
                first.trim().is_empty() && second.trim().is_empty() && third.trim() == "0+",
                "invalid closed-core text CSF"
            );
            let mut row = [0, -1, 0, -1].repeat((width - 2) / 4);
            row.extend([0, 1]);
            row
        } else {
            parser.parse_csf(&first, &second, &third)?
        };
        writer.push(&row, block)?;
        count_in_block += 1;
    }
    if count_in_block == 0 && block == 0 && closed_core {
        writer.push(&[0, 1], 0)?;
    } else {
        ensure!(count_in_block > 0, "empty or trailing CSF block");
    }
    writer.finish()
}

/// Select unique increasing input ordinals, retaining source block boundaries.
pub fn select(
    input: &Path,
    header: &Path,
    output: &Path,
    output_header: &Path,
    indices: &[u64],
) -> Result<DatasetStats> {
    ensure!(!indices.is_empty(), "empty CSF selection");
    ensure!(
        indices.windows(2).all(|w| w[0] < w[1]),
        "CSF indices must be unique and increasing"
    );
    let mut source = DescriptorRows::open(input, header)?;
    ensure!(
        indices.last().copied().unwrap() < u64::try_from(source.header.row_count())?,
        "CSF index out of range"
    );
    let mut output = DatasetWriter::new(
        source.header.clone(),
        output,
        output_header,
        &[input, header],
    )?;
    let mut selected = indices.iter().copied().peekable();
    while let Some((ordinal, block, row)) = source.next_row()? {
        if selected.peek() == Some(&(ordinal as u64)) {
            output.push(&row, block)?;
            selected.next();
        }
    }
    output.finish()
}

pub fn restore_text(
    input: &Path,
    header: &Path,
    output: &Path,
    indices: Option<&[u64]>,
) -> Result<(usize, u64)> {
    ensure!(!output.exists(), "output already exists");
    ensure_output_does_not_alias_input(output, input, "descriptor")?;
    ensure_output_does_not_alias_input(output, header, "header")?;
    let mut source = DescriptorRows::open(input, header)?;
    if let Some(indices) = indices {
        ensure!(
            !indices.is_empty() && indices.windows(2).all(|w| w[0] < w[1]),
            "CSF indices must be nonempty, unique and increasing"
        );
        ensure!(
            indices.last().copied().unwrap() < source.header.row_count() as u64,
            "CSF index out of range"
        );
    }
    let (temporary, file) = create_temporary_output(output)?;
    let mut writer = BufWriter::new(file);
    for line in &source.header.lines {
        writeln!(writer, "{line}")?;
    }
    let mut selected = indices.map(|v| v.iter().copied().peekable());
    let mut prior_block = None;
    let mut count = 0;
    while let Some((ordinal, block, row)) = source.next_row()? {
        if let Some(selection) = &mut selected {
            if selection.peek() != Some(&(ordinal as u64)) {
                continue;
            }
            selection.next();
        }
        if prior_block.is_some_and(|b| b != block) {
            writeln!(writer, " *")?;
        }
        prior_block = Some(block);
        if row[..row.len() - 2]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|slot| slot[0] == 0)
        {
            writeln!(writer, "\n\n0+")?;
        } else {
            let (mut occupied, mut couplings) = (Vec::new(), Vec::new());
            let (j, parity) = decode_v2_into(
                &row,
                source.header.layout(),
                &source.header.peel,
                &mut occupied,
                &mut couplings,
            )?;
            let (a, b, c) = CompleteCsfFile::format_record_parts(
                &source.header.peel,
                &occupied,
                &couplings,
                j,
                parity,
            )?;
            writeln!(writer, "{a}\n{b}\n{c}")?;
        }
        count += 1;
    }
    writer.flush()?;
    drop(writer);
    let bytes = fs::metadata(temporary.path())?.len();
    publish_temporary_output(temporary.path(), output, false)?;
    Ok((count, bytes))
}

/// In-memory integer records for algorithms that require the complete basis.
pub(crate) fn read_complete(input: &Path) -> Result<(CompleteCsfFile, DatasetHeader)> {
    let mut rows = DescriptorRows::open(input, &header_path(input)?)?;
    let mut result =
        CompleteCsfFile::new_for_restore(rows.header.lines.clone(), rows.header.peel.clone());
    let mut previous = None;
    while let Some((_, block, row)) = rows.next_row()? {
        let (mut occupied, mut couplings) = (Vec::new(), Vec::new());
        let (j, parity) = decode_v2_into(
            &row,
            rows.header.layout(),
            &rows.header.peel,
            &mut occupied,
            &mut couplings,
        )?;
        result.append_restored_record(&occupied, &couplings, j, parity, previous != Some(block))?;
        previous = Some(block);
    }
    Ok((result, rows.header))
}

pub(crate) fn write_complete(
    file: &CompleteCsfFile,
    header: DatasetHeader,
    output: &Path,
) -> Result<DatasetStats> {
    let layout = header.layout();
    let mut writer = DatasetWriter::new(header, output, &header_path(output)?, &[])?;
    for (index, block) in file.blocks.iter().enumerate() {
        let first = usize::try_from(block.record_start)?;
        let end = first + usize::try_from(block.record_len)?;
        for record in &file.records[first..end] {
            let mut row = vec![0; layout.row_len()];
            crate::descriptor_v2::encode_v2(file, record, &mut row)?;
            writer.push(&row, index)?;
        }
    }
    writer.finish()
}
