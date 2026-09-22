//! CSF Descriptor Generation Module
//!
//! This module converts Configuration State Function (CSF) data into descriptor arrays
//! for machine learning applications. Each CSF is parsed into a fixed-length array
//! containing electron counts and angular momentum coupling values.

use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::complete_csf::{IntermediateCoupling, OccupiedSubshell, Parity, SubshellState};
#[cfg(test)]
use crate::descriptor_schema::MISSING;
use crate::descriptor_schema::{
    DescriptorLayout, DescriptorVersion, output_schema, validate_record,
};
use crate::descriptor_v2::{decode_v2_into, write_feature_row};

pub(crate) fn parse_peel_subshells_from_header_lines(
    header_lines: &[String],
) -> Result<Vec<String>> {
    let line = header_lines
        .get(3)
        .ok_or_else(|| anyhow::anyhow!("header_lines[3] is missing"))?;
    let subshells = line
        .split_whitespace()
        .filter(|subshell| {
            subshell.chars().any(|character| character.is_alphabetic())
                && subshell.chars().all(|character| {
                    character.is_alphanumeric()
                        || character == '+'
                        || character == '-'
                        || character == '_'
                })
        })
        .map(str::to_owned)
        .collect::<Vec<_>>();

    if subshells.is_empty() {
        return Err(anyhow::anyhow!(
            "Could not find peel subshells in header lines"
        ));
    }
    Ok(subshells)
}

/// Restore a complete V2 descriptor Parquet file without materializing its
/// rows or a `CompleteCsfFile` in memory.
///
/// The descriptor's row order is already the final CSF order.  Header block
/// lengths therefore let this function emit each separator while scanning
/// record batches, with only one row's integer and decoded-state buffers.
pub(crate) fn restore_v2_descriptor_parquet_stream(
    descriptor_path: &Path,
    header_path: &Path,
    output_path: &Path,
) -> Result<(usize, u64)> {
    restore_v2_descriptor_parquet_to_outputs(descriptor_path, header_path, output_path, None)
}

/// Stream V2 descriptors once and fan out canonical records to CSF text and,
/// when requested, the existing three-line CSF Parquet representation.
pub(crate) fn restore_v2_descriptor_parquet_to_outputs(
    descriptor_path: &Path,
    header_path: &Path,
    output_path: &Path,
    csf_parquet_path: Option<&Path>,
) -> Result<(usize, u64)> {
    use crate::atomic_output::{
        create_temporary_output, ensure_output_does_not_alias_input, publish_temporary_output,
    };
    use crate::complete_csf::CompleteCsfFile;
    use arrow::array::{Array, Int32Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;
    use std::io::{BufWriter, Write};

    ensure_output_does_not_alias_input(output_path, descriptor_path, "descriptor")?;
    ensure_output_does_not_alias_input(output_path, header_path, "header")?;
    let header_toml = std::fs::read_to_string(header_path)
        .with_context(|| format!("failed to read {}", header_path.display()))?;
    let (header_lines, block_lengths) = parse_restore_header_from_toml(&header_toml)?;
    ensure!(
        !block_lengths.is_empty() && block_lengths.iter().all(|&length| length > 0),
        "header block_lengths must contain only non-empty blocks"
    );
    let peel_subshells = parse_peel_subshells_from_header_lines(&header_lines)?;
    let expected_hash = crate::descriptor_schema::hash_header_file(header_path)?;

    let file = File::open(descriptor_path)
        .with_context(|| format!("failed to open {}", descriptor_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("failed to create descriptor Parquet reader")?;
    let metadata = builder.metadata().file_metadata().clone();
    let kv = metadata
        .key_value_metadata()
        .context("descriptor file has no key-value metadata; cannot verify version")?;
    let get = |key: &str| {
        kv.iter()
            .find(|entry| entry.key == key)
            .and_then(|entry| entry.value.clone())
    };
    let version_tag: u8 = get("descriptor_version")
        .context("descriptor file is missing descriptor_version metadata")?
        .parse()
        .context("descriptor_version metadata is not a valid integer")?;
    ensure!(
        DescriptorVersion::from_tag(version_tag)? == DescriptorVersion::V2,
        "streaming restoration only supports V2 descriptor files"
    );
    let subshell_count: usize = get("subshell_count")
        .context("descriptor file is missing subshell_count metadata")?
        .parse()
        .context("subshell_count metadata is not a valid integer")?;
    ensure!(
        subshell_count == peel_subshells.len(),
        "descriptor subshell_count {subshell_count} differs from header Peel count {}",
        peel_subshells.len()
    );
    if let Some(stored_hash) = get("source_header_sha256") {
        ensure!(
            stored_hash == expected_hash,
            "header file does not match the hash recorded in the descriptor file"
        );
    }
    let expected_rows = block_lengths.iter().try_fold(0usize, |total, &length| {
        total
            .checked_add(length)
            .context("header block count overflow")
    })?;
    ensure!(
        usize::try_from(metadata.num_rows()).ok() == Some(expected_rows),
        "descriptor has {} rows but header block_lengths total {expected_rows}",
        metadata.num_rows()
    );
    let layout = DescriptorLayout::new(DescriptorVersion::V2, subshell_count);
    let expected_schema = output_schema(layout, false)?;
    let schema = builder.schema();
    ensure!(
        schema.fields().len() == expected_schema.fields().len(),
        "descriptor schema has {} columns, expected {}",
        schema.fields().len(),
        expected_schema.fields().len()
    );
    for (actual, expected) in schema.fields().iter().zip(expected_schema.fields()) {
        ensure!(
            actual.name() == expected.name()
                && actual.data_type() == expected.data_type()
                && !actual.is_nullable(),
            "descriptor schema field {:?} does not match V2 contract",
            actual.name()
        );
    }

    let mut reader = builder
        .build()
        .context("failed to build descriptor Parquet reader")?;
    let (temporary, file) = create_temporary_output(output_path)?;
    let mut writer = BufWriter::new(file);
    let mut csf_parquet = if let Some(path) = csf_parquet_path {
        ensure!(
            !path.exists(),
            "CSF Parquet output already exists: {}",
            path.display()
        );
        ensure_output_does_not_alias_input(path, descriptor_path, "descriptor")?;
        ensure_output_does_not_alias_input(path, header_path, "header")?;
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("failed to create CSF Parquet {}", path.display()))?;
        Some((
            path,
            crate::csf_output::csf_parquet::schema(),
            crate::csf_output::csf_parquet::writer(file)?,
        ))
    } else {
        None
    };
    for line in &header_lines {
        writeln!(writer, "{line}")?;
    }
    let mut row_values = vec![0; layout.row_len()];
    let mut occupied = Vec::new();
    let mut couplings = Vec::new();
    let mut record_count = 0usize;
    let mut block_index = 0usize;
    let mut records_in_block = 0usize;
    for batch in &mut reader {
        let batch = batch.context("failed to decode descriptor record batch")?;
        ensure!(
            batch.num_columns() == layout.row_len(),
            "descriptor batch has {} columns, expected {}",
            batch.num_columns(),
            layout.row_len()
        );
        let columns = batch
            .columns()
            .iter()
            .map(|column| {
                ensure!(
                    column.null_count() == 0,
                    "descriptor columns must not contain nulls"
                );
                column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .context("descriptor column is not Int32")
            })
            .collect::<Result<Vec<_>>>()?;
        let mut batches = csf_parquet.as_ref().map(|(_, schema, _)| {
            crate::csf_output::csf_parquet::BatchBuilder::new(schema.clone(), batch.num_rows())
        });
        for row in 0..batch.num_rows() {
            if records_in_block == block_lengths[block_index] {
                block_index += 1;
                records_in_block = 0;
                ensure!(
                    block_index < block_lengths.len(),
                    "descriptor exceeds header blocks"
                );
                writeln!(writer, " *")?;
            }
            for (value, column) in row_values.iter_mut().zip(&columns) {
                *value = column.value(row);
            }
            let (total_two_j, parity) =
                decode_v2_into(&row_values, layout, &mut occupied, &mut couplings)?;
            validate_record(&peel_subshells, &occupied, &couplings)?;
            let formatted = CompleteCsfFile::format_record_parts(
                &peel_subshells,
                &occupied,
                &couplings,
                total_two_j,
                parity,
            )?;
            let (formatted1, formatted2, formatted3) = formatted;
            writeln!(writer, "{formatted1}")?;
            writeln!(writer, "{formatted2}")?;
            writeln!(writer, "{formatted3}")?;
            if let Some(builder) = &mut batches {
                builder.push(
                    u64::try_from(record_count)?,
                    &[formatted1, formatted2, formatted3],
                );
            }
            record_count = record_count
                .checked_add(1)
                .context("restored CSF record count overflow")?;
            records_in_block += 1;
        }
        if let (Some((_, _, parquet_writer)), Some(builder)) = (&mut csf_parquet, batches) {
            parquet_writer
                .write(&builder.finish()?)
                .context("failed to write CSF Parquet batch")?;
        }
    }
    ensure!(
        record_count == expected_rows
            && block_index + 1 == block_lengths.len()
            && records_in_block == block_lengths[block_index],
        "descriptor row count or block boundaries disagree with header"
    );
    writer.flush()?;
    drop(writer);
    if let Some((_, _, parquet_writer)) = csf_parquet {
        parquet_writer
            .close()
            .context("failed to close CSF Parquet writer")?;
    }
    let output_bytes = std::fs::metadata(temporary.path())?.len();
    publish_temporary_output(temporary.path(), output_path, false)?;
    Ok((record_count, output_bytes))
}

/// Parquet reading/writing support
pub mod parquet_batch {
    use super::*;
    use arrow::array::{StringArray, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::arrow::arrow_writer::ArrowWriter;
    use std::fs::File;
    use std::path::PathBuf;
    use std::sync::Arc;

    struct ParquetFileGuard {
        writer: Option<ArrowWriter<File>>,
        path: PathBuf,
        cleanup_on_drop: bool,
    }

    impl ParquetFileGuard {
        fn new(writer: ArrowWriter<File>, path: PathBuf) -> Self {
            Self {
                writer: Some(writer),
                path,
                cleanup_on_drop: true,
            }
        }

        fn finish(mut self) -> Result<()> {
            if let Some(writer) = self.writer.take() {
                writer
                    .close()
                    .with_context(|| "Failed to close Parquet writer")?;
            }
            self.cleanup_on_drop = false;
            Ok(())
        }
    }

    impl Drop for ParquetFileGuard {
        fn drop(&mut self) {
            let _ = self.writer.take().map(|w| w.close());
            if self.cleanup_on_drop {
                let _ = std::fs::remove_file(&self.path);
            }
        }
    }

    /// Parse a user-provided compression specifier into a parquet `Compression`.
    ///
    /// Accepted values (case-insensitive):
    /// - `None`                       → default `ZSTD(3)` (backward compatible)
    /// - `"none"` / `"uncompressed"`  → `UNCOMPRESSED`
    /// - `"snappy"`                   → `SNAPPY`
    /// - `"zstd"`                     → `ZSTD(3)` (default level)
    /// - `"zstd-N"` (N in 1..=22)     → `ZSTD(N)`
    ///
    /// Any other value returns an error.
    pub(crate) fn parse_compression(
        compression: Option<&str>,
    ) -> Result<parquet::basic::Compression> {
        use parquet::basic::Compression;

        let Some(spec) = compression else {
            return Ok(Compression::ZSTD(
                parquet::basic::ZstdLevel::try_new(3).expect("zstd level 3 is always valid"),
            ));
        };

        let lower = spec.trim().to_ascii_lowercase();
        match lower.as_str() {
            "none" | "uncompressed" => Ok(Compression::UNCOMPRESSED),
            "snappy" => Ok(Compression::SNAPPY),
            "zstd" => Ok(Compression::ZSTD(
                parquet::basic::ZstdLevel::try_new(3).expect("zstd level 3 is always valid"),
            )),
            other if other.starts_with("zstd-") => {
                let level_str = &other["zstd-".len()..];
                let level: i32 = level_str.parse().map_err(|_| {
                    anyhow::anyhow!(
                        "invalid zstd level '{}': expected integer 1..=22",
                        level_str
                    )
                })?;
                let zstd_level = parquet::basic::ZstdLevel::try_new(level).map_err(|_| {
                    anyhow::anyhow!("zstd level {} out of range (expected 1..=22)", level)
                })?;
                Ok(Compression::ZSTD(zstd_level))
            }
            other => Err(anyhow::anyhow!(
                "unknown compression '{}': expected one of none/uncompressed/snappy/zstd/zstd-N",
                other
            )),
        }
    }

    /// Read peel subshells from a header TOML file
    ///
    /// # Arguments
    /// * `header_path` - Path to the header TOML file
    ///
    /// # Returns
    /// * `Ok(Vec<String>)` - List of peel subshell names
    /// * `Err(anyhow::Error)` - Error if parsing fails
    pub fn read_peel_subshells_from_header(header_path: &Path) -> Result<Vec<String>> {
        use toml::Value;

        let mut toml_content = read_to_string(header_path)
            .with_context(|| format!("Failed to read header file: {}", header_path.display()))?;

        // Normalize line endings and trim whitespace
        toml_content = toml_content.replace("\r\n", "\n");
        let toml_content = toml_content.trim();

        // Parse the TOML content using from_str instead of parse()
        let toml_value: Value = toml::from_str(toml_content).with_context(|| {
            format!("Failed to parse TOML from file: {}", header_path.display())
        })?;

        // Get header_lines from [header_info] section
        let header_lines = toml_value
            .get("header_info")
            .and_then(|v| v.get("header_lines"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("header_info.header_lines not found in TOML"))?;

        let peel_line = header_lines
            .get(3)
            .and_then(|line| line.as_str())
            .ok_or_else(|| anyhow::anyhow!("header_lines[3] is not a string"))?;
        let header_lines = vec![
            String::new(),
            String::new(),
            String::new(),
            peel_line.to_owned(),
        ];
        parse_peel_subshells_from_header_lines(&header_lines)
    }

    /// Find the header file for a given parquet file
    ///
    /// # Arguments
    /// * `parquet_path` - Path to the parquet file
    ///
    /// # Returns
    /// * `Some(PathBuf)` - Path to the header file if found
    /// * `None` - No header file found
    pub fn find_header_file(parquet_path: &Path) -> Option<PathBuf> {
        let parquet_stem = parquet_path.file_stem()?.to_str()?;
        let parent_dir = parquet_path.parent()?;

        // Common header file patterns
        let patterns = std::vec![
            format!("{}_header.toml", parquet_stem),
            format!("{}.toml", parquet_stem),
            format!("{}_header", parquet_stem),
        ];

        for pattern in patterns {
            let header_path = parent_dir.join(&pattern);
            if header_path.exists() {
                return Some(header_path);
            }
        }

        None
    }

    /// Result statistics for batch descriptor generation
    #[derive(Debug)]
    pub struct BatchDescriptorStats {
        pub input_file: String,
        pub output_file: String,
        pub csf_count: usize,
        pub descriptor_count: usize,
        pub orbital_count: usize,
        pub descriptor_size: usize,
        pub descriptor_version: u8,
        pub channels_per_subshell: usize,
    }

    /// Generate descriptors from a parquet file and write to Parquet file
    ///
    /// # Arguments
    /// * `input_parquet` - Path to input parquet file (must have line1, line2, line3 columns)
    /// * `output_file` - Path to output Parquet file for descriptors
    /// * `peel_subshells` - Optional list of subshell names (auto-detected if None)
    /// * `header_path` - Optional path to header TOML file
    /// * `normalize` - Whether to normalize descriptors (default: false). Rejected for V2
    ///   (design doc §4.4/§5.1): the V1 per-subshell normalization denominators
    ///   are not meaningful for a 4-wide row and are P1 scope.
    /// * `version` - Descriptor format version (default V1).
    /// * `compression` - Optional parquet compression specifier (default: `zstd-3`).
    ///   See [`parse_compression`] for accepted values.
    ///
    /// # Returns
    /// * `Ok(BatchDescriptorStats)` - Statistics about the batch operation
    /// * `Err(String)` - Error message if operation fails
    ///
    /// # Output Format
    /// Parquet with configurable compression (default ZSTD level 3) - columnar format, Polars compatible
    /// Read with: `polars.read_parquet()` or `pyarrow.parquet.read_table()`
    pub fn generate_descriptors_from_parquet(
        input_parquet: &Path,
        output_file: &Path,
        peel_subshells: Option<Vec<String>>,
        header_path: Option<PathBuf>,
        normalize: bool,
        version: DescriptorVersion,
        compression: Option<&str>,
    ) -> Result<BatchDescriptorStats> {
        // Step 1: Determine peel_subshells and, when available, the header
        // path whose bytes get hashed into the output KV metadata (design
        // doc §6, plan D5 layer 2: "that hash is the enforcement mechanism
        // for header/block/descriptor matching").
        let resolved_header_path = match &header_path {
            Some(path) => Some(path.clone()),
            None => find_header_file(input_parquet),
        };
        let peel_subshells = match peel_subshells {
            Some(s) => s,
            None => {
                let header = resolved_header_path.clone().ok_or_else(|| {
                    anyhow::anyhow!(
                        "Could not auto-detect header file. Please provide peel_subshells or header_path."
                    )
                })?;
                read_peel_subshells_from_header(&header)?
            }
        };

        let layout = DescriptorLayout::new(version, peel_subshells.len());
        let orbital_count = layout.subshell_count();
        let descriptor_size = layout.row_len();

        // Step 2: Create descriptor generator
        let generator =
            super::CSFDescriptorGenerator::new_with_version(peel_subshells.clone(), version);

        // Step 3: Open input parquet file
        let file = std::fs::File::open(input_parquet).with_context(|| {
            format!("Failed to open input parquet: {}", input_parquet.display())
        })?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file).with_context(|| {
            format!(
                "Failed to create parquet reader: {}",
                input_parquet.display()
            )
        })?;

        let _schema = builder.schema();

        let mut reader = builder
            .build()
            .with_context(|| "Failed to build parquet reader")?;

        // Step 4: Create output Parquet writer
        use parquet::file::properties::WriterProperties;

        let output_schema = crate::descriptor_schema::output_schema(layout, normalize)?;
        let source_header_sha256 = resolved_header_path
            .as_ref()
            .map(|path| crate::descriptor_schema::hash_header_file(path))
            .transpose()?;
        let source_header_filename = resolved_header_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .map(str::to_owned);
        let kv_metadata = crate::descriptor_schema::output_kv_metadata(
            layout,
            &peel_subshells,
            normalize,
            source_header_sha256.as_deref(),
            source_header_filename.as_deref(),
        );

        let output_file_handle = std::fs::File::create(output_file)
            .with_context(|| format!("Failed to create output file: {}", output_file.display()))?;

        // Normalized Float32 descriptors have near-unique values where dictionary
        // encoding is pure overhead (confirmed by benchmark: ~11.5s saved at 48 workers).
        // Raw Int32 descriptors may have enough repetition for dictionary to help,
        // but this has not been A/B benchmarked yet — keep parquet default (enabled).
        let props = WriterProperties::builder()
            .set_compression(parse_compression(compression)?)
            .set_dictionary_enabled(!normalize)
            .set_key_value_metadata(Some(kv_metadata))
            .build();

        let writer = ArrowWriter::try_new(output_file_handle, output_schema.clone(), Some(props))
            .with_context(|| "Failed to create Parquet writer")?;
        let mut writer_guard = ParquetFileGuard::new(writer, output_file.to_path_buf());

        // Step 5: Process each batch
        let mut total_csfs = 0;
        let mut descriptor_count = 0;

        loop {
            match reader.next() {
                Some(Ok(batch)) => {
                    let batch_size = batch.num_rows();

                    // Get columns by index (parquet schema: idx, line1, line2, line3)
                    let idx_col = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or_else(|| anyhow::anyhow!("idx column is not uint64 type"))?;

                    let line1_col = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| anyhow::anyhow!("line1 column is not string type"))?;

                    let line2_col = batch
                        .column(2)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| anyhow::anyhow!("line2 column is not string type"))?;

                    let line3_col = batch
                        .column(3)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| anyhow::anyhow!("line3 column is not string type"))?;

                    // Process each row
                    use arrow::array::{Array, Float32Builder, Int32Builder};
                    use std::sync::Arc;

                    // Initialize builders for each column (avoids transpose overhead)
                    // Use Float32Builder for normalized output, Int32Builder for raw descriptors
                    if normalize {
                        use crate::descriptor_normalization::{
                            infer_two_j_target, normalize_descriptor_per_csf,
                        };

                        let mut builders: Vec<Float32Builder> = (0..descriptor_size)
                            .map(|_| arrow::array::Float32Builder::with_capacity(batch_size))
                            .collect();

                        for i in 0..batch_size {
                            let line1 = line1_col.value(i);
                            let line2 = line2_col.value(i);
                            let line3 = line3_col.value(i);
                            let idx = idx_col.value(i);

                            match generator.parse_csf(line1, line2, line3) {
                                Ok(descriptor) => {
                                    let two_j_target = infer_two_j_target(&descriptor);
                                    let normalized = match normalize_descriptor_per_csf(
                                        &descriptor,
                                        &peel_subshells,
                                        two_j_target,
                                    ) {
                                        Ok(normalized) => normalized,
                                        Err(e) => {
                                            eprintln!(
                                                "Warning: Failed to normalize CSF at index {}: {}",
                                                idx, e
                                            );
                                            vec![0.0f32; descriptor_size]
                                        }
                                    };
                                    for (col_idx, &val) in normalized.iter().enumerate() {
                                        builders[col_idx].append_value(val);
                                    }
                                }
                                Err(e) => {
                                    if version == DescriptorVersion::V2 {
                                        return Err(e).with_context(|| {
                                            format!("Failed to parse V2 CSF at index {idx}")
                                        });
                                    }
                                    eprintln!(
                                        "Warning: Failed to parse CSF at index {}: {}",
                                        idx, e
                                    );
                                    for builder in &mut builders {
                                        builder.append_value(0.0f32);
                                    }
                                }
                            }
                            descriptor_count += 1;
                        }

                        // Convert builders to Arrow arrays
                        let column_arrays: Vec<Arc<dyn Array>> = builders
                            .into_iter()
                            .map(|mut b| Arc::new(b.finish()) as Arc<dyn Array>)
                            .collect();

                        // Create output record batch
                        use arrow::record_batch::RecordBatch;
                        let output_batch =
                            RecordBatch::try_new(output_schema.clone(), column_arrays)
                                .with_context(|| "Failed to create output batch")?;

                        writer_guard
                            .writer
                            .as_mut()
                            .expect("writer exists until finish")
                            .write(&output_batch)
                            .with_context(|| "Failed to write batch")?;
                    } else {
                        let mut row = vec![0i32; descriptor_size];
                        let fallback_value = match version {
                            DescriptorVersion::V1 => 0,
                            DescriptorVersion::V2 => crate::descriptor_schema::MISSING,
                        };
                        let mut builders: Vec<Int32Builder> = (0..descriptor_size)
                            .map(|_| arrow::array::Int32Builder::with_capacity(batch_size))
                            .collect();

                        for i in 0..batch_size {
                            let line1 = line1_col.value(i);
                            let line2 = line2_col.value(i);
                            let line3 = line3_col.value(i);
                            let idx = idx_col.value(i);

                            match generator.parse_row_into(line1, line2, line3, &mut row) {
                                Ok(()) => {
                                    // Append directly to column builders
                                    for (col_idx, &val) in row.iter().enumerate() {
                                        builders[col_idx].append_value(val);
                                    }
                                }
                                Err(e) => {
                                    if version == DescriptorVersion::V2 {
                                        return Err(e).with_context(|| {
                                            format!("Failed to parse V2 CSF at index {idx}")
                                        });
                                    }
                                    eprintln!(
                                        "Warning: Failed to parse CSF at index {}: {}",
                                        idx, e
                                    );
                                    for builder in &mut builders {
                                        builder.append_value(fallback_value);
                                    }
                                }
                            }
                            descriptor_count += 1;
                        }

                        // Convert builders to Arrow arrays
                        let column_arrays: Vec<Arc<dyn Array>> = builders
                            .into_iter()
                            .map(|mut b| Arc::new(b.finish()) as Arc<dyn Array>)
                            .collect();

                        // Create output record batch
                        use arrow::record_batch::RecordBatch;
                        let output_batch =
                            RecordBatch::try_new(output_schema.clone(), column_arrays)
                                .with_context(|| "Failed to create output batch")?;

                        writer_guard
                            .writer
                            .as_mut()
                            .expect("writer exists until finish")
                            .write(&output_batch)
                            .with_context(|| "Failed to write batch")?;
                    }

                    total_csfs += batch_size;
                }
                Some(Err(e)) => {
                    return Err(anyhow::anyhow!("Error reading parquet batch: {}", e));
                }
                None => break,
            }
        }

        // Step 6: Finalize writer
        writer_guard.finish()?;

        Ok(BatchDescriptorStats {
            input_file: input_parquet.to_string_lossy().to_string(),
            output_file: output_file.to_string_lossy().to_string(),
            csf_count: total_csfs,
            descriptor_count,
            orbital_count,
            descriptor_size,
            descriptor_version: version.tag(),
            channels_per_subshell: layout.channels_per_subshell(),
        })
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Pipeline Parallel Descriptor Generation
    ////////////////////////////////////////////////////////////////////////////////

    type DescriptorRow = (u64, Arc<str>, Arc<str>, Arc<str>);

    /// Work item sent from reader to workers
    struct WorkItem {
        batch_idx: usize,
        rows: Vec<DescriptorRow>,
    }

    /// Descriptor columns produced by workers.
    enum DescriptorColumns {
        Raw(Vec<Vec<i32>>),
        Normalized(Vec<Vec<f32>>),
    }

    /// Result item sent from workers to writer
    struct ResultItem {
        batch_idx: usize,
        batch_size: usize,
        columns: DescriptorColumns,
    }

    #[cfg(test)]
    fn transpose_i32_rows(rows: Vec<Vec<i32>>, descriptor_size: usize) -> Vec<Vec<i32>> {
        let batch_size = rows.len();
        let mut columns: Vec<Vec<i32>> = (0..descriptor_size)
            .map(|_| Vec::with_capacity(batch_size))
            .collect();

        for row in rows {
            for (col_idx, column) in columns.iter_mut().enumerate() {
                column.push(row.get(col_idx).copied().unwrap_or(0));
            }
        }

        columns
    }

    #[cfg(test)]
    fn transpose_f32_rows(rows: Vec<Vec<f32>>, descriptor_size: usize) -> Vec<Vec<f32>> {
        let batch_size = rows.len();
        let mut columns: Vec<Vec<f32>> = (0..descriptor_size)
            .map(|_| Vec::with_capacity(batch_size))
            .collect();

        for row in rows {
            for (col_idx, column) in columns.iter_mut().enumerate() {
                column.push(row.get(col_idx).copied().unwrap_or(0.0));
            }
        }

        columns
    }

    fn descriptor_chunk_size(batch_size: usize, rayon_thread_count: usize) -> usize {
        let target_chunks = rayon_thread_count.saturating_mul(4).max(1);
        batch_size.div_ceil(target_chunks).clamp(256, 8192)
    }

    fn build_raw_descriptor_columns_parallel(
        pool: &rayon::ThreadPool,
        generator: Arc<super::CSFDescriptorGenerator>,
        rows: Vec<DescriptorRow>,
    ) -> Result<Vec<Vec<i32>>> {
        use rayon::prelude::*;

        let descriptor_size = generator.layout().row_len();
        let fallback_value = match generator.layout().version() {
            DescriptorVersion::V1 => 0,
            DescriptorVersion::V2 => crate::descriptor_schema::MISSING,
        };
        let batch_size = rows.len();
        let chunk_size = descriptor_chunk_size(batch_size, pool.current_num_threads());

        let chunk_columns: Vec<Result<(usize, Vec<Vec<i32>>)>> = pool.install(|| {
            rows.par_chunks(chunk_size)
                .enumerate()
                .map(|(chunk_idx, chunk)| {
                    let mut columns: Vec<Vec<i32>> = (0..descriptor_size)
                        .map(|_| Vec::with_capacity(chunk.len()))
                        .collect();
                    let mut descriptor = vec![0i32; descriptor_size];

                    for (idx, line1, line2, line3) in chunk {
                        if let Err(e) =
                            generator.parse_row_into(line1, line2, line3, &mut descriptor)
                        {
                            if generator.layout().version() == DescriptorVersion::V2 {
                                return Err(e).with_context(|| {
                                    format!("Failed to parse V2 CSF at index {idx}")
                                });
                            }
                            eprintln!("Warning: Failed to parse CSF at index {}: {}", idx, e);
                            descriptor.fill(fallback_value);
                        }
                        for (col_idx, column) in columns.iter_mut().enumerate() {
                            column.push(descriptor[col_idx]);
                        }
                    }

                    Ok((chunk_idx, columns))
                })
                .collect()
        });
        let chunk_columns = chunk_columns.into_iter().collect::<Result<Vec<_>>>()?;

        Ok(pool.install(|| {
            (0..descriptor_size)
                .into_par_iter()
                .map(|col_idx| {
                    let mut col = Vec::with_capacity(batch_size);
                    for (_, chunk) in &chunk_columns {
                        col.extend(chunk[col_idx].iter().copied());
                    }
                    col
                })
                .collect()
        }))
    }

    fn build_normalized_descriptor_columns_parallel(
        pool: &rayon::ThreadPool,
        generator: Arc<super::CSFDescriptorGenerator>,
        peel_subshells: Arc<Vec<String>>,
        rows: Vec<DescriptorRow>,
    ) -> Vec<Vec<f32>> {
        use crate::descriptor_normalization::{infer_two_j_target, normalize_descriptor_per_csf};
        use rayon::prelude::*;

        let descriptor_size = 3 * generator.orbital_count();
        let batch_size = rows.len();
        let chunk_size = descriptor_chunk_size(batch_size, pool.current_num_threads());

        let chunk_columns: Vec<(usize, Vec<Vec<f32>>)> = pool.install(|| {
            rows.par_chunks(chunk_size)
                .enumerate()
                .map(|(chunk_idx, chunk)| {
                    let mut columns: Vec<Vec<f32>> = (0..descriptor_size)
                        .map(|_| Vec::with_capacity(chunk.len()))
                        .collect();
                    let mut descriptor = vec![0i32; descriptor_size];
                    let mut normalized = vec![0.0f32; descriptor_size];

                    for (idx, line1, line2, line3) in chunk {
                        match generator.parse_csf_into(line1, line2, line3, &mut descriptor) {
                            Ok(()) => {
                                let two_j_target = infer_two_j_target(&descriptor);
                                match normalize_descriptor_per_csf(
                                    &descriptor,
                                    &peel_subshells,
                                    two_j_target,
                                ) {
                                    Ok(values) if values.len() == descriptor_size => {
                                        normalized.copy_from_slice(&values);
                                    }
                                    Ok(values) => {
                                        eprintln!(
                                            "Warning: Normalized descriptor at index {} has length {}, expected {}",
                                            idx,
                                            values.len(),
                                            descriptor_size
                                        );
                                        normalized.fill(0.0);
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "Warning: Failed to normalize CSF at index {}: {}",
                                            idx, e
                                        );
                                        normalized.fill(0.0);
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Warning: Failed to parse CSF at index {}: {}", idx, e);
                                normalized.fill(0.0);
                            }
                        }

                        for (col_idx, column) in columns.iter_mut().enumerate() {
                            column.push(normalized[col_idx]);
                        }
                    }

                    (chunk_idx, columns)
                })
                .collect()
        });

        pool.install(|| {
            (0..descriptor_size)
                .into_par_iter()
                .map(|col_idx| {
                    let mut col = Vec::with_capacity(batch_size);
                    for (_, chunk) in &chunk_columns {
                        col.extend(chunk[col_idx].iter().copied());
                    }
                    col
                })
                .collect()
        })
    }

    pub(crate) fn descriptor_pipeline_channel_capacity(num_workers: usize) -> usize {
        num_workers.clamp(1, 8)
    }

    /// Generate descriptors from parquet with full pipeline parallelization
    ///
    /// This implementation uses a producer-consumer pipeline with three stages:
    /// 1. **Reader thread**: Continuously reads parquet batches and sends to work channel
    /// 2. **Worker threads (Rayon)**: Parse CSFs in parallel, send results to result channel
    /// 3. **Writer thread**: Receives results in order and writes to parquet file
    ///
    /// All three stages run concurrently, maximizing CPU utilization and I/O overlap.
    ///
    /// Output format: Parquet with configurable compression (default ZSTD level 3).
    /// Normalized Float32 output uses PLAIN encoding (dictionary disabled, confirmed
    /// optimal for near-unique values). Raw Int32 output keeps dictionary encoding
    /// enabled (parquet default, pending A/B benchmark).
    ///
    /// # Arguments
    /// * `input_parquet` - Path to input parquet file
    /// * `output_file` - Path to output Parquet file
    /// * `peel_subshells` - List of subshell names
    /// * `num_workers` - Number of worker threads (default: CPU core count)
    /// * `normalize` - Whether to normalize descriptors (default: false). Rejected for
    ///   V2 (design doc §4.4/§5.1).
    /// * `version` - Descriptor format version (default V1).
    /// * `header_path` - Optional header TOML whose SHA-256 is bound into the
    ///   output's KV metadata (design doc §6, plan D5).
    /// * `compression` - Optional parquet compression specifier (default: `zstd-3`).
    ///   See [`parse_compression`] for accepted values.
    pub fn generate_descriptors_from_parquet_parallel(
        input_parquet: &Path,
        output_file: &Path,
        peel_subshells: Vec<String>,
        num_workers: Option<usize>,
        normalize: bool,
        version: DescriptorVersion,
        header_path: Option<&Path>,
        compression: Option<&str>,
    ) -> Result<BatchDescriptorStats> {
        use arrow::array::{Array, StringArray, UInt64Array};
        use arrow::record_batch::RecordBatch;
        use crossbeam_channel::{Receiver, Sender, bounded};
        use parquet::file::properties::WriterProperties;
        use std::collections::BTreeMap;
        use std::sync::Arc;
        // Determine worker count
        let num_workers = num_workers.unwrap_or_else(num_cpus::get);
        if num_workers == 0 {
            return Err(anyhow::anyhow!("num_workers must be greater than 0"));
        }

        let layout = DescriptorLayout::new(version, peel_subshells.len());
        let orbital_count = layout.subshell_count();
        let descriptor_size = layout.row_len();

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 1: Setup channels with bounded capacity
        ////////////////////////////////////////////////////////////////////////////////
        let channel_capacity = descriptor_pipeline_channel_capacity(num_workers);
        let (work_tx, work_rx): (Sender<WorkItem>, Receiver<WorkItem>) = bounded(channel_capacity);
        let (result_tx, result_rx): (Sender<ResultItem>, Receiver<ResultItem>) =
            bounded(channel_capacity);

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 2: Setup output schema and writer (multi-column format for better performance)
        ////////////////////////////////////////////////////////////////////////////////
        let schema = crate::descriptor_schema::output_schema(layout, normalize)?;
        let source_header_sha256 = header_path
            .map(crate::descriptor_schema::hash_header_file)
            .transpose()?;
        let source_header_filename = header_path
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .map(str::to_owned);
        let kv_metadata = crate::descriptor_schema::output_kv_metadata(
            layout,
            &peel_subshells,
            normalize,
            source_header_sha256.as_deref(),
            source_header_filename.as_deref(),
        );

        let output_file_handle = std::fs::File::create(output_file)
            .with_context(|| format!("Failed to create output file: {}", output_file.display()))?;

        // See sequential path for dictionary encoding rationale.
        let props = WriterProperties::builder()
            .set_compression(parse_compression(compression)?)
            .set_dictionary_enabled(!normalize)
            .set_key_value_metadata(Some(kv_metadata))
            .build();

        let writer = ArrowWriter::try_new(output_file_handle, schema.clone(), Some(props))
            .with_context(|| "Failed to create Parquet writer")?;
        let mut writer_guard = ParquetFileGuard::new(writer, output_file.to_path_buf());

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 3: Spawn reader thread
        ////////////////////////////////////////////////////////////////////////////////
        let input_path = input_parquet.to_path_buf();
        let reader_handle = std::thread::spawn(move || {
            use std::fs::File;
            let file = match File::open(&input_path) {
                Ok(f) => f,
                Err(e) => {
                    let _ = work_tx.send(WorkItem {
                        batch_idx: usize::MAX, // Error sentinel
                        rows: vec![],
                    });
                    return Err(anyhow::anyhow!("Failed to open input parquet: {}", e));
                }
            };

            let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
                Ok(b) => b,
                Err(e) => {
                    let _ = work_tx.send(WorkItem {
                        batch_idx: usize::MAX,
                        rows: vec![],
                    });
                    return Err(anyhow::anyhow!("Failed to create parquet reader: {}", e));
                }
            };

            let mut reader = match builder.with_batch_size(65536).build() {
                Ok(r) => r,
                Err(e) => {
                    let _ = work_tx.send(WorkItem {
                        batch_idx: usize::MAX,
                        rows: vec![],
                    });
                    return Err(anyhow::anyhow!("Failed to build parquet reader: {}", e));
                }
            };

            let mut batch_idx = 0usize;
            let mut total_csfs = 0usize;

            loop {
                match reader.next() {
                    Some(Ok(batch)) => {
                        let batch_size = batch.num_rows();
                        total_csfs += batch_size;

                        let idx_col = match batch.column(0).as_any().downcast_ref::<UInt64Array>() {
                            Some(col) => col,
                            None => return Err(anyhow::anyhow!("idx column is not uint64 type")),
                        };

                        let line1_col = match batch.column(1).as_any().downcast_ref::<StringArray>()
                        {
                            Some(col) => col,
                            None => return Err(anyhow::anyhow!("line1 column is not string type")),
                        };

                        let line2_col = match batch.column(2).as_any().downcast_ref::<StringArray>()
                        {
                            Some(col) => col,
                            None => return Err(anyhow::anyhow!("line2 column is not string type")),
                        };

                        let line3_col = match batch.column(3).as_any().downcast_ref::<StringArray>()
                        {
                            Some(col) => col,
                            None => return Err(anyhow::anyhow!("line3 column is not string type")),
                        };

                        // Copy row strings into Arc<str> so worker threads can own them safely.
                        let rows: Vec<DescriptorRow> = (0..batch_size)
                            .map(|i| {
                                (
                                    idx_col.value(i),
                                    line1_col.value(i).into(),
                                    line2_col.value(i).into(),
                                    line3_col.value(i).into(),
                                )
                            })
                            .collect();

                        let work_item = WorkItem { batch_idx, rows };
                        if work_tx.send(work_item).is_err() {
                            return Err(anyhow::anyhow!("Failed to send work item"));
                        }
                        batch_idx += 1;
                    }
                    Some(Err(e)) => {
                        return Err(anyhow::anyhow!("Error reading parquet batch: {}", e));
                    }
                    None => break,
                }
            }

            Ok(total_csfs)
        });

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 4: Compute thread - process each batch with a bounded Rayon pool
        ////////////////////////////////////////////////////////////////////////////////
        let peel_subshells_for_normalization = Arc::new(peel_subshells.clone());
        let generator = Arc::new(super::CSFDescriptorGenerator::new_with_version(
            peel_subshells,
            version,
        ));
        let mut worker_handles = Vec::new();

        let rayon_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build()
            .with_context(|| "Failed to create descriptor worker thread pool")?;

        {
            let generator_clone = generator.clone();
            let result_tx_clone = result_tx.clone();
            let work_rx_clone = work_rx.clone();
            let peel_subshells_for_normalization = peel_subshells_for_normalization.clone();
            let normalize_enabled = normalize;

            worker_handles.push(std::thread::spawn(move || {
                let mut first_error = None;
                while let Ok(work_item) = work_rx_clone.recv() {
                    // Check for error sentinel
                    if work_item.batch_idx == usize::MAX {
                        return Err(anyhow::anyhow!("Reader thread encountered an error"));
                    }

                    let batch_idx = work_item.batch_idx;
                    let batch_size = work_item.rows.len();
                    if first_error.is_some() {
                        continue;
                    }
                    let columns = if normalize_enabled {
                        let cols = build_normalized_descriptor_columns_parallel(
                            &rayon_pool,
                            generator_clone.clone(),
                            peel_subshells_for_normalization.clone(),
                            work_item.rows,
                        );
                        DescriptorColumns::Normalized(cols)
                    } else {
                        let cols = match build_raw_descriptor_columns_parallel(
                            &rayon_pool,
                            generator_clone.clone(),
                            work_item.rows,
                        ) {
                            Ok(cols) => cols,
                            Err(error) => {
                                first_error = Some(error);
                                continue;
                            }
                        };
                        DescriptorColumns::Raw(cols)
                    };

                    let result_item = ResultItem {
                        batch_idx,
                        batch_size,
                        columns,
                    };
                    if result_tx_clone.send(result_item).is_err() {
                        return Err(anyhow::anyhow!("Failed to send result item"));
                    }
                }

                match first_error {
                    Some(error) => Err(error),
                    None => Ok(()),
                }
            }));
        }

        // Drop our clone of the result_tx so the writer can properly detect when workers are done
        drop(result_tx);

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 5: Writer thread - maintain order and write to parquet (multi-column format)
        ////////////////////////////////////////////////////////////////////////////////
        let writer_handle: std::thread::JoinHandle<Result<usize>> = std::thread::spawn(move || {
            use arrow::array::{Float32Array, Int32Array};

            let mut pending: BTreeMap<usize, ResultItem> = BTreeMap::new();
            let mut next_write_idx = 0usize;
            let mut total_descriptors = 0usize;

            while let Ok(result_item) = result_rx.recv() {
                let batch_idx = result_item.batch_idx;
                pending.insert(batch_idx, result_item);

                // Write all consecutive batches we have
                while let Some(result_item) = pending.remove(&next_write_idx) {
                    let batch_size = result_item.batch_size;
                    if batch_size == 0 {
                        next_write_idx += 1;
                        continue;
                    }
                    total_descriptors += batch_size;

                    let column_arrays: Vec<Arc<dyn Array>> = if normalize {
                        let columns = match result_item.columns {
                            DescriptorColumns::Normalized(columns) => columns,
                            DescriptorColumns::Raw(_) => {
                                return Err(anyhow::anyhow!(
                                    "Expected normalized descriptor columns"
                                ));
                            }
                        };
                        columns
                            .into_iter()
                            .map(|column| Arc::new(Float32Array::from(column)) as Arc<dyn Array>)
                            .collect()
                    } else {
                        let columns = match result_item.columns {
                            DescriptorColumns::Raw(columns) => columns,
                            DescriptorColumns::Normalized(_) => {
                                return Err(anyhow::anyhow!("Expected raw descriptor columns"));
                            }
                        };
                        columns
                            .into_iter()
                            .map(|column| Arc::new(Int32Array::from(column)) as Arc<dyn Array>)
                            .collect()
                    };

                    let output_batch = match RecordBatch::try_new(schema.clone(), column_arrays) {
                        Ok(b) => b,
                        Err(e) => {
                            return Err(anyhow::anyhow!("Failed to create output batch: {}", e));
                        }
                    };

                    if writer_guard
                        .writer
                        .as_mut()
                        .expect("writer exists until finish")
                        .write(&output_batch)
                        .is_err()
                    {
                        return Err(anyhow::anyhow!("Failed to write batch"));
                    }

                    next_write_idx += 1;
                }
            }

            writer_guard.finish()?;
            Ok(total_descriptors)
        });

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 6: Wait for all threads and collect results
        ////////////////////////////////////////////////////////////////////////////////
        let mut errors = Vec::new();

        let reader_result = match reader_handle.join() {
            Ok(Ok(result)) => Some(result),
            Ok(Err(e)) => {
                errors.push(format!("Reader thread failed: {:#}", e));
                None
            }
            Err(e) => {
                errors.push(format!("Reader thread panicked: {:?}", e));
                None
            }
        };

        // Wait for all worker threads
        for (i, handle) in worker_handles.into_iter().enumerate() {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    errors.push(format!("Worker thread {} failed: {:#}", i, e));
                }
                Err(e) => {
                    errors.push(format!("Worker thread {} panicked: {:?}", i, e));
                }
            }
        }

        let writer_result = match writer_handle.join() {
            Ok(Ok(result)) => Some(result),
            Ok(Err(e)) => {
                errors.push(format!("Writer thread failed: {:#}", e));
                None
            }
            Err(e) => {
                errors.push(format!("Writer thread panicked: {:?}", e));
                None
            }
        };

        if !errors.is_empty() {
            let _ = std::fs::remove_file(output_file);
            return Err(anyhow::anyhow!(
                "Parallel descriptor generation failed: {}",
                errors.join("; ")
            ));
        }

        let total_csfs = reader_result.expect("reader result exists when no errors occurred");
        let total_descriptors =
            writer_result.expect("writer result exists when no errors occurred");

        Ok(BatchDescriptorStats {
            input_file: input_parquet.to_string_lossy().to_string(),
            output_file: output_file.to_string_lossy().to_string(),
            csf_count: total_csfs,
            descriptor_count: total_descriptors,
            orbital_count,
            descriptor_size,
            descriptor_version: version.tag(),
            channels_per_subshell: layout.channels_per_subshell(),
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::csfs_descriptor::CSFDescriptorGenerator;

        #[test]
        fn build_raw_descriptor_columns_parallel_matches_parse_csf_rows() {
            let generator = Arc::new(CSFDescriptorGenerator::new(vec![
                "5s".to_string(),
                "4d-".to_string(),
                "4d".to_string(),
            ]));
            let rows = vec![
                (
                    0u64,
                    Arc::<str>::from("  5s ( 2)  4d-( 4)  4d ( 6)"),
                    Arc::<str>::from("                   3/2      "),
                    Arc::<str>::from("                        4-  "),
                ),
                (
                    1u64,
                    Arc::<str>::from("  5s ( 0)  4d-( 4)  4d ( 6)"),
                    Arc::<str>::from("                   5/2      "),
                    Arc::<str>::from("                        4-  "),
                ),
            ];
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();

            let columns =
                build_raw_descriptor_columns_parallel(&pool, generator.clone(), rows.clone())
                    .unwrap();

            let expected_rows: Vec<Vec<i32>> = rows
                .iter()
                .map(|(_, line1, line2, line3)| generator.parse_csf(line1, line2, line3).unwrap())
                .collect();
            let expected_columns = transpose_i32_rows(expected_rows, generator.orbital_count() * 3);
            assert_eq!(columns, expected_columns);
        }

        #[test]
        fn build_normalized_descriptor_columns_parallel_matches_row_path() {
            use crate::descriptor_normalization::{
                infer_two_j_target, normalize_descriptor_per_csf,
            };

            let peel_subshells =
                Arc::new(vec!["5s".to_string(), "4d-".to_string(), "4d".to_string()]);
            let generator = Arc::new(CSFDescriptorGenerator::new((*peel_subshells).clone()));
            let rows = vec![
                (
                    0u64,
                    Arc::<str>::from("  5s ( 2)  4d-( 4)  4d ( 6)"),
                    Arc::<str>::from("                   3/2      "),
                    Arc::<str>::from("                        4-  "),
                ),
                (
                    1u64,
                    Arc::<str>::from("  5s ( 0)  4d-( 4)  4d ( 6)"),
                    Arc::<str>::from("                   5/2      "),
                    Arc::<str>::from("                        4-  "),
                ),
            ];
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();

            let columns = build_normalized_descriptor_columns_parallel(
                &pool,
                generator.clone(),
                peel_subshells.clone(),
                rows.clone(),
            );

            let expected_rows: Vec<Vec<f32>> = rows
                .iter()
                .map(|(_, line1, line2, line3)| {
                    let descriptor = generator.parse_csf(line1, line2, line3).unwrap();
                    let two_j_target = infer_two_j_target(&descriptor);
                    normalize_descriptor_per_csf(&descriptor, &peel_subshells, two_j_target)
                        .unwrap()
                })
                .collect();
            let expected_columns = transpose_f32_rows(expected_rows, generator.orbital_count() * 3);
            assert_eq!(columns, expected_columns);
        }

        #[test]
        fn descriptor_chunk_size_keeps_slack_for_high_worker_counts() {
            assert_eq!(descriptor_chunk_size(65_536, 8), 2048);
            assert!(descriptor_chunk_size(65_536, 46) <= 512);
            assert!(descriptor_chunk_size(65_536, 46) >= 256);
        }

        #[test]
        fn parse_compression_defaults_to_zstd3() {
            use parquet::basic::Compression;
            match parse_compression(None).unwrap() {
                Compression::ZSTD(level) => assert_eq!(level.compression_level(), 3),
                other => panic!("expected ZSTD(3), got {:?}", other),
            }
        }

        #[test]
        fn parse_compression_handles_named_codecs() {
            use parquet::basic::Compression;
            assert!(matches!(
                parse_compression(Some("none")).unwrap(),
                Compression::UNCOMPRESSED
            ));
            assert!(matches!(
                parse_compression(Some("uncompressed")).unwrap(),
                Compression::UNCOMPRESSED
            ));
            assert!(matches!(
                parse_compression(Some("snappy")).unwrap(),
                Compression::SNAPPY
            ));
        }

        #[test]
        fn parse_compression_handles_zstd_levels() {
            use parquet::basic::Compression;
            for level in [1, 3, 9, 19, 22] {
                let spec = format!("zstd-{}", level);
                match parse_compression(Some(&spec)).unwrap() {
                    Compression::ZSTD(parsed) => assert_eq!(parsed.compression_level(), level),
                    other => panic!("expected ZSTD({}), got {:?}", level, other),
                }
            }
            // Bare "zstd" is level 3
            match parse_compression(Some("zstd")).unwrap() {
                Compression::ZSTD(parsed) => assert_eq!(parsed.compression_level(), 3),
                other => panic!("expected ZSTD(3), got {:?}", other),
            }
        }

        #[test]
        fn parse_compression_is_case_insensitive_and_trims_whitespace() {
            use parquet::basic::Compression;
            assert!(matches!(
                parse_compression(Some("NONE")).unwrap(),
                Compression::UNCOMPRESSED
            ));
            assert!(matches!(
                parse_compression(Some("  Snappy ")).unwrap(),
                Compression::SNAPPY
            ));
            match parse_compression(Some("ZSTD-19")).unwrap() {
                Compression::ZSTD(parsed) => assert_eq!(parsed.compression_level(), 19),
                other => panic!("expected ZSTD(19), got {:?}", other),
            }
        }

        #[test]
        fn parse_compression_rejects_unknown_and_out_of_range() {
            assert!(parse_compression(Some("lzma")).is_err());
            assert!(parse_compression(Some("gzip")).is_err());
            assert!(parse_compression(Some("zstd-0")).is_err());
            assert!(parse_compression(Some("zstd-23")).is_err());
            assert!(parse_compression(Some("zstd--1")).is_err());
            assert!(parse_compression(Some("zstd-abc")).is_err());
        }
    }
}

/// Convert a J-value string to its doubled integer representation (2J)
///
/// # Arguments
/// * `j_str` - J value as string, e.g., "3/2", "2", "5/2"
///
/// # Returns
/// * `Ok(i32)` - The doubled J value (2J)
/// * `Err(String)` - Error message if parsing fails
///
/// # Examples
/// ```text
/// // Fractional J values:
/// j_to_double_j("3/2") => Ok(3)
/// j_to_double_j("5/2") => Ok(5)
///
/// // Integer J values:
/// j_to_double_j("2")  => Ok(4)
/// j_to_double_j("4")  => Ok(8)
/// ```
pub fn j_to_double_j(j_str: &str) -> Result<i32> {
    let trimmed = j_str.trim();

    // Handle fractional J values (e.g., "3/2" -> 3)
    if let Some(slash_pos) = trimmed.find('/') {
        let numerator: i32 = trimmed[..slash_pos]
            .parse()
            .with_context(|| format!("Invalid J value numerator: {}", trimmed))?;
        return Ok(numerator);
    }

    // Handle integer J values (e.g., "2" -> 4, "4-" -> 8)
    // Remove trailing parity indicator if present
    let cleaned = trimmed.trim_end_matches('-').trim_end_matches('+');
    cleaned
        .parse::<i32>()
        .map(|j| j * 2)
        .with_context(|| format!("Invalid J value: {}", trimmed))
}

/// Chunk a string into fixed-size pieces
///
/// # Arguments
/// * `s` - The string to chunk
/// * `chunk_size` - Size of each chunk
///
/// # Returns
/// Vector of string chunks
#[cfg(test)]
fn chunk_string(s: &str, chunk_size: usize) -> Vec<&str> {
    s.as_bytes()
        .chunks(chunk_size)
        .map(|chunk| std::str::from_utf8(chunk).expect("ASCII input keeps chunks valid UTF-8"))
        .collect()
}

fn fixed_width_field(line: &str, start: usize, width: usize) -> &str {
    if start >= line.len() {
        return "";
    }
    let end = start.saturating_add(width).min(line.len());
    line.get(start..end).unwrap_or("")
}

fn fixed_width_trimmed_field(line: &str, start: usize, width: usize) -> &str {
    fixed_width_field(line, start, width).trim()
}

/// Parse a V2 line2 state field, keeping the seniority digit `kopp1` writes
/// at field offsets 3-4 (`"s;"`) rather than discarding it as V1's
/// `parse_csf_into` does.
fn parse_v2_state_field(field: &str) -> Result<Option<SubshellState>> {
    let trimmed = field.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let (seniority, j_value) = match trimmed.split_once(';') {
        Some((seniority, j_value)) => {
            ensure!(
                field.len() == 9
                    && field.as_bytes().get(3).is_some_and(u8::is_ascii_digit)
                    && field.as_bytes().get(4) == Some(&b';'),
                "seniority must occupy field offsets 3 and 4 as a single digit followed by ';'"
            );
            let seniority: u8 = seniority
                .trim()
                .parse()
                .with_context(|| format!("invalid seniority {seniority:?}"))?;
            ensure!(seniority <= 9, "seniority exceeds one GRASP output digit");
            (Some(seniority), j_value)
        }
        None => (None, trimmed),
    };
    let two_j =
        u16::try_from(j_to_double_j(j_value.trim())?).context("state 2J value is negative")?;
    Ok(Some(SubshellState { two_j, seniority }))
}

pub(crate) fn coupling_signature_from_descriptor_into(
    descriptor: &[i32],
    signature: &mut Vec<i32>,
) -> Result<()> {
    if !descriptor.len().is_multiple_of(3) {
        return Err(anyhow::anyhow!(
            "descriptor length {} is not a multiple of 3",
            descriptor.len()
        ));
    }

    signature.clear();
    signature.extend(
        descriptor
            .as_chunks::<3>()
            .0
            .iter()
            .filter(|triplet| triplet[0] > 0)
            .map(|triplet| triplet[2]),
    );
    Ok(())
}

/// CSF Descriptor Generator
///
/// This struct maintains the state needed to convert CSF data into descriptor arrays.
pub struct CSFDescriptorGenerator {
    /// List of peel subshell names (e.g., ["5s", "4d-", "4d", ...])
    peel_subshells: Vec<String>,
    /// Map from subshell name to index for O(1) lookup
    orbital_index_map: HashMap<String, usize>,
    /// Number of orbitals (cached for performance)
    orbital_count: usize,
    /// Row shape for this generator's descriptor version (plan D8 seam 1).
    layout: DescriptorLayout,
    /// Count missing-subshell warnings so malformed input cannot flood stderr.
    missing_subshell_warning_count: AtomicUsize,
}

impl CSFDescriptorGenerator {
    /// Create a new V1 CSF descriptor generator
    ///
    /// # Arguments
    /// * `peel_subshells` - List of subshell names (e.g., ["5s", "4d-", "4d"])
    ///
    /// # Returns
    /// A new generator instance
    pub fn new(peel_subshells: Vec<String>) -> Self {
        Self::new_with_version(peel_subshells, DescriptorVersion::V1)
    }

    /// Create a new CSF descriptor generator for a specific descriptor version.
    pub fn new_with_version(peel_subshells: Vec<String>, version: DescriptorVersion) -> Self {
        let orbital_count = peel_subshells.len();
        let orbital_index_map: HashMap<_, _> = peel_subshells
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        Self {
            peel_subshells,
            orbital_index_map,
            orbital_count,
            layout: DescriptorLayout::new(version, orbital_count),
            missing_subshell_warning_count: AtomicUsize::new(0),
        }
    }

    /// Get the number of orbitals
    pub fn orbital_count(&self) -> usize {
        self.orbital_count
    }

    /// Get the peel subshells list
    pub fn peel_subshells(&self) -> &[String] {
        &self.peel_subshells
    }

    /// Get this generator's descriptor row layout.
    pub fn layout(&self) -> DescriptorLayout {
        self.layout
    }

    /// Parse one CSF's three lines into `row`, dispatching on `self.layout().version()`.
    ///
    /// `row.len()` must equal `self.layout().row_len()`. This is the single
    /// entry point through which both the V1 and V2 text parsers are called
    /// (plan D8 seam 1), so export code never branches on version itself.
    pub fn parse_row_into(
        &self,
        line1: &str,
        line2: &str,
        line3: &str,
        row: &mut [i32],
    ) -> Result<()> {
        match self.layout.version() {
            DescriptorVersion::V1 => self.parse_csf_into(line1, line2, line3, row),
            DescriptorVersion::V2 => self.parse_csf_v2_into(line1, line2, line3, row),
        }
    }

    /// Parse a single CSF into a descriptor array
    ///
    /// # Arguments
    /// * `line1` - First line: subshell configurations and electron counts
    /// * `line2` - Second line: intermediate J coupling values
    /// * `line3` - Third line: final coupling and total J value
    ///
    /// # Returns
    /// A vector of i32 descriptor values
    ///
    /// # CSF Format Example
    /// ```text
    /// line1: "  5s ( 2)  4d-( 4)  4d ( 6)"
    /// line2: "                   3/2      "
    /// line3: "                        4-  "
    /// ```
    pub fn parse_csf(&self, line1: &str, line2: &str, line3: &str) -> Result<Vec<i32>> {
        let mut descriptor = vec![0i32; 3 * self.orbital_count];
        self.parse_csf_into(line1, line2, line3, &mut descriptor)?;
        Ok(descriptor)
    }

    pub fn parse_csf_into(
        &self,
        line1: &str,
        line2: &str,
        line3: &str,
        descriptor: &mut [i32],
    ) -> Result<()> {
        let expected_len = 3 * self.orbital_count;
        if descriptor.len() != expected_len {
            return Err(anyhow::anyhow!(
                "descriptor buffer length {} does not match expected {}",
                descriptor.len(),
                expected_len
            ));
        }
        if !line1.is_ascii() || !line2.is_ascii() || !line3.is_ascii() {
            return Err(anyhow::anyhow!("CSF lines must be ASCII fixed-width text"));
        }

        descriptor.fill(0);

        // Step 1: Preprocess the three lines
        let subshells_line = line1.trim_end();
        let line_length = subshells_line.len();

        // Extract coupling line (remove first 4 and last 5 characters)
        let coupling_line_raw = line3.trim_end();
        let coupling_line = coupling_line_raw
            .get(4..coupling_line_raw.len().saturating_sub(5))
            .unwrap_or(coupling_line_raw);

        // Step 2: Extract final J value from the end of line3
        // Extract final J value from the end of line3 (last 5 chars, minus 1 trailing char)
        let final_j_str = coupling_line_raw
            .get(
                coupling_line_raw.len().saturating_sub(5)
                    ..coupling_line_raw.len().saturating_sub(1),
            )
            .unwrap_or("");
        let final_double_j = j_to_double_j(final_j_str)?;

        // Step 3: Chunk lines into 9-character blocks
        let block_count = line_length.div_ceil(9);

        // Step 4: Process each subshell block
        for i in 0..block_count {
            let start = i * 9;
            let subshell_charges = fixed_width_field(subshells_line, start, 9);

            // Extract subshell name (first 5 characters, trimmed)
            let subshell = fixed_width_trimmed_field(subshell_charges, 0, 5);
            if subshell.is_empty() {
                continue;
            }

            // Extract electron number (characters 6-8, i.e., indices 6 and 7)
            let subshell_electron_num: i32 = if subshell_charges.len() >= 8 {
                fixed_width_trimmed_field(subshell_charges, 6, 2)
                    .parse()
                    .unwrap_or(0)
            } else {
                0
            };

            // Check if this is the last subshell
            let is_last = i + 1 == block_count;

            let middle_item = fixed_width_field(line2.trim_end(), start, 9);
            let coupling_item = fixed_width_field(coupling_line, start, 9);

            // Process middle J coupling value (line 2)
            let mut temp_middle_item: i32 = 0;
            if !middle_item.trim().is_empty() {
                // If semicolon separated, take the last value
                let middle_value = if let Some(semi_pos) = middle_item.find(';') {
                    &middle_item[semi_pos + 1..]
                } else {
                    middle_item
                };
                temp_middle_item = j_to_double_j(middle_value).unwrap_or(0);
            }

            // Process coupling J value (line 3)
            let mut temp_coupling_item: i32 = 0;
            if !coupling_item.trim().is_empty() {
                temp_coupling_item = j_to_double_j(coupling_item).unwrap_or(0);
            } else if !middle_item.trim().is_empty() {
                // If line 3 is empty but line 2 has a value, use line 2's value
                temp_coupling_item = temp_middle_item;
            }

            // Special handling: last subshell uses final J value
            if is_last {
                temp_coupling_item = final_double_j;
            }

            // Step 5: Find orbital index in the peel subshells list
            if let Some(&orbs_idx) = self.orbital_index_map.get(subshell) {
                let descriptor_idx = orbs_idx * 3;

                if subshell_electron_num == 0 {
                    descriptor[descriptor_idx] = 0;
                    descriptor[descriptor_idx + 1] = 0;
                    descriptor[descriptor_idx + 2] = 0;
                } else {
                    descriptor[descriptor_idx] = subshell_electron_num;
                    descriptor[descriptor_idx + 1] = temp_middle_item;
                    descriptor[descriptor_idx + 2] = temp_coupling_item;
                }
            } else {
                let warning_idx = self
                    .missing_subshell_warning_count
                    .fetch_add(1, Ordering::Relaxed);
                if warning_idx < 5 {
                    eprintln!("Warning: {} not found in orbs list", subshell);
                } else if warning_idx == 5 {
                    eprintln!("Warning: further subshell-not-found warnings suppressed");
                }
            }
        }

        // Unoccupied orbitals remain with all zeros (default initialization)

        Ok(())
    }

    /// Parse one CSF's three lines into a V2 row (design doc §3, plan D2/D3).
    ///
    /// Unlike [`Self::parse_csf_into`] (V1), this keeps the seniority digit
    /// `kopp1` writes at line2 offset 3, never folds the total `2J` into the
    /// last occupied subshell's field (it becomes the separate
    /// `total_two_j` global column), and never back-fills line2's value into
    /// an unprinted line3 coupling slot. Every one of V1's silent-recovery
    /// paths — a too-short coupling slice, an unparsable J value, an orbital
    /// missing from the peel table, a truncated or explicit-zero electron
    /// count — is a hard error here, and out-of-Peel-order or duplicated
    /// occupied subshells are rejected by [`validate_record`] rather than
    /// silently overwritten.
    pub fn parse_csf_v2_into(
        &self,
        line1: &str,
        line2: &str,
        line3: &str,
        row: &mut [i32],
    ) -> Result<()> {
        ensure!(
            self.layout.version() == DescriptorVersion::V2,
            "parse_csf_v2_into requires a V2-configured generator"
        );
        ensure!(
            row.len() == self.layout.row_len(),
            "row buffer length {} does not match expected {}",
            row.len(),
            self.layout.row_len()
        );
        ensure!(
            line1.is_ascii() && line2.is_ascii() && line3.is_ascii(),
            "CSF lines must be ASCII fixed-width text"
        );

        let subshells_line = line1.trim_end();
        let line_length = subshells_line.len();
        ensure!(
            line_length > 0 && line_length.is_multiple_of(9),
            "occupation line length {line_length} is not a positive multiple of 9"
        );
        let field_count = line_length / 9;

        let coupling_line_raw = line3.trim_end();
        ensure!(
            coupling_line_raw.len() > 5,
            "coupling line {coupling_line_raw:?} is too short to hold a total J and parity"
        );
        let coupling_line = coupling_line_raw
            .get(4..coupling_line_raw.len() - 5)
            .with_context(|| {
                format!(
                    "coupling line {coupling_line_raw:?} is too short for {field_count} subshells"
                )
            })?;

        let final_j_str =
            &coupling_line_raw[coupling_line_raw.len() - 5..coupling_line_raw.len() - 1];
        let final_two_j =
            u16::try_from(j_to_double_j(final_j_str)?).context("total 2J is negative")?;

        let parity_byte = coupling_line_raw.as_bytes()[coupling_line_raw.len() - 1];
        let parity = match parity_byte {
            b'+' => Parity::Even,
            b'-' => Parity::Odd,
            other => bail!(
                "invalid parity byte {:?} in coupling line {coupling_line_raw:?}",
                char::from(other)
            ),
        };

        let mut occupied = Vec::with_capacity(field_count);
        let mut couplings = Vec::new();
        let trimmed_line2 = line2.trim_end();

        for i in 0..field_count {
            let start = i * 9;
            let subshell_field = fixed_width_field(subshells_line, start, 9);
            ensure!(
                subshell_field.len() == 9,
                "occupation field at position {i} is truncated"
            );

            let subshell = fixed_width_trimmed_field(subshell_field, 0, 5);
            ensure!(!subshell.is_empty(), "empty subshell field at position {i}");
            ensure!(
                subshell_field.as_bytes()[5] == b'(' && subshell_field.as_bytes()[8] == b')',
                "malformed occupation field {subshell_field:?} at position {i}"
            );

            let electron_field = fixed_width_trimmed_field(subshell_field, 6, 2);
            let electrons: u8 = electron_field.parse().with_context(|| {
                format!("invalid occupation {electron_field:?} at position {i}")
            })?;

            let &orbital_index = self.orbital_index_map.get(subshell).with_context(|| {
                format!("{subshell:?} at position {i} is absent from peel subshells")
            })?;

            let middle_field = fixed_width_field(trimmed_line2, start, 9);
            let state = parse_v2_state_field(middle_field)
                .with_context(|| format!("invalid state field {middle_field:?} at position {i}"))?;

            occupied.push(OccupiedSubshell {
                subshell_index: u16::try_from(orbital_index)?,
                occupation: electrons,
                state,
            });

            let coupling_field = fixed_width_field(coupling_line, start, 9);
            let trimmed_coupling = coupling_field.trim();
            if !trimmed_coupling.is_empty() {
                let boundary = u16::try_from(i + 1)?;
                let two_j = u16::try_from(j_to_double_j(trimmed_coupling)?)
                    .context("coupling 2J is negative")?;
                couplings.push(IntermediateCoupling { boundary, two_j });
            }
        }

        validate_record(&self.peel_subshells, &occupied, &couplings)?;
        write_feature_row(self.layout, &occupied, &couplings, final_two_j, parity, row)
    }

    /// V1-only: coupling signatures are a V1 triplet-derived concept (design
    /// doc §5.1 P0 explicitly keeps this interface unchanged rather than
    /// making it version-generic).
    pub(crate) fn parse_coupling_signature_into(
        &self,
        line1: &str,
        line2: &str,
        line3: &str,
        descriptor: &mut [i32],
        signature: &mut Vec<i32>,
    ) -> Result<()> {
        ensure!(
            self.layout.version() == DescriptorVersion::V1,
            "coupling signatures are only defined for V1 descriptors"
        );
        self.parse_csf_into(line1, line2, line3, descriptor)?;
        coupling_signature_from_descriptor_into(descriptor, signature)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Python Bindings (PyO3)
////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-exposed function to generate descriptors from parquet file (parallel version)
///
/// Output format: Parquet file with multiple `col_0, col_1, ..., col_N` Int32 columns
/// and configurable compression (default ZSTD level 3)
/// - Each column corresponds to one position in the descriptor array
/// - Much faster than List column format for large datasets
/// - Read with: `df = pl.read_parquet(); descriptors = df[["col_0", "col_1", ...]].to_numpy()`
///
/// This version uses streaming batch processing with 65536 rows/batch for low memory usage
/// and better I/CPU balance on multi-core systems. Multi-column format avoids ListArray overhead.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    input_parquet,
    output_file,
    peel_subshells,
    num_workers=None,
    normalize=false,
    compression=None,
    *,
    descriptor_version=2,
    header_path=None
))]
#[allow(clippy::too_many_arguments)] // PyO3 exposes one argument per Python parameter.
fn py_generate_descriptors_from_parquet(
    py: Python,
    input_parquet: String,
    output_file: String,
    peel_subshells: Vec<String>,
    num_workers: Option<usize>,
    normalize: bool,
    compression: Option<String>,
    descriptor_version: u8,
    header_path: Option<String>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    use pyo3::types::PyDict;
    use std::path::Path;

    let input_path = Path::new(&input_parquet).to_path_buf();
    let output_path = Path::new(&output_file).to_path_buf();
    let header_path_buf = header_path
        .map(|path| Path::new(&path).to_path_buf())
        .or_else(|| parquet_batch::find_header_file(&input_path));

    if matches!(num_workers, Some(0)) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_workers must be greater than 0",
        ));
    }
    let version = DescriptorVersion::from_tag(descriptor_version)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Release the GIL during the long-running operation
    let stats = py
        .detach(|| {
            parquet_batch::generate_descriptors_from_parquet_parallel(
                &input_path,
                &output_path,
                peel_subshells,
                num_workers,
                normalize,
                version,
                header_path_buf.as_deref(),
                compression.as_deref(),
            )
        })
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("success", true)?;
    dict.set_item("input_file", stats.input_file)?;
    dict.set_item("output_file", stats.output_file)?;
    dict.set_item("csf_count", stats.csf_count)?;
    dict.set_item("descriptor_count", stats.descriptor_count)?;
    dict.set_item("orbital_count", stats.orbital_count)?;
    dict.set_item("descriptor_size", stats.descriptor_size)?;
    dict.set_item("descriptor_version", stats.descriptor_version)?;
    dict.set_item("channels_per_subshell", stats.channels_per_subshell)?;
    Ok(dict.into())
}

/// Python-exposed function to restore CSFs from a V2 descriptor Parquet file
/// and its source `{stem}_header.toml` (design doc §3.4, plan D5/D7).
///
/// The header path must be explicit: `find_header_file`'s candidate list
/// includes `{stem}.toml`, which collides with the descriptor sidecar name
/// `cli.py` writes, so auto-detection is not offered here.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (descriptor_parquet, header_path, output, indices=None))]
fn py_restore_csfs_from_descriptors(
    py: Python,
    descriptor_parquet: String,
    header_path: String,
    output: String,
    indices: Option<Vec<u64>>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    use crate::atomic_output::{
        create_temporary_output, ensure_output_does_not_alias_input, publish_temporary_output,
    };
    use crate::descriptor_v2::restore_file;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use pyo3::types::PyDict;
    use std::path::Path;

    let descriptor_path = Path::new(&descriptor_parquet).to_path_buf();
    let header_path_buf = Path::new(&header_path).to_path_buf();
    let output_path = Path::new(&output).to_path_buf();

    let stats = py
        .detach(|| -> anyhow::Result<(usize, u64)> {
            if indices.is_none() {
                return restore_v2_descriptor_parquet_stream(
                    &descriptor_path,
                    &header_path_buf,
                    &output_path,
                );
            }
            ensure_output_does_not_alias_input(&output_path, &descriptor_path, "descriptor")?;
            ensure_output_does_not_alias_input(&output_path, &header_path_buf, "header")?;

            let expected_hash = crate::descriptor_schema::hash_header_file(&header_path_buf)?;
            let file = std::fs::File::open(&descriptor_path)
                .with_context(|| format!("failed to open {}", descriptor_path.display()))?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                .with_context(|| "failed to create parquet reader")?;
            let metadata = builder.metadata().file_metadata().clone();
            let kv = metadata
                .key_value_metadata()
                .context("descriptor file has no key-value metadata; cannot verify version")?;
            let get = |key: &str| {
                kv.iter()
                    .find(|entry| entry.key == key)
                    .and_then(|entry| entry.value.clone())
            };
            let version_tag: u8 = get("descriptor_version")
                .context("descriptor file is missing descriptor_version metadata")?
                .parse()
                .context("descriptor_version metadata is not a valid integer")?;
            let version = DescriptorVersion::from_tag(version_tag)?;
            ensure!(
                version == DescriptorVersion::V2,
                "py_restore_csfs_from_descriptors only supports V2 descriptor files"
            );
            let subshell_count: usize = get("subshell_count")
                .context("descriptor file is missing subshell_count metadata")?
                .parse()
                .context("subshell_count metadata is not a valid integer")?;
            if let Some(stored_hash) = get("source_header_sha256") {
                ensure!(
                    stored_hash == expected_hash,
                    "header file does not match the hash recorded in the descriptor file"
                );
            }

            let layout = DescriptorLayout::new(version, subshell_count);
            let mut reader = builder
                .build()
                .with_context(|| "failed to build parquet reader")?;
            let mut rows: Vec<Vec<i32>> = Vec::new();
            loop {
                match reader.next() {
                    Some(Ok(batch)) => {
                        use arrow::array::{Array, Int32Array};
                        let columns: Vec<&Int32Array> = (0..batch.num_columns())
                            .map(|i| {
                                batch
                                    .column(i)
                                    .as_any()
                                    .downcast_ref::<Int32Array>()
                                    .context("descriptor column is not Int32")
                            })
                            .collect::<anyhow::Result<Vec<_>>>()?;
                        for row_idx in 0..batch.num_rows() {
                            rows.push(columns.iter().map(|column| column.value(row_idx)).collect());
                        }
                    }
                    Some(Err(e)) => {
                        return Err(anyhow::anyhow!("error reading parquet batch: {e}"));
                    }
                    None => break,
                }
            }

            let header_toml = std::fs::read_to_string(&header_path_buf)
                .with_context(|| format!("failed to read {}", header_path_buf.display()))?;
            let (header_lines, block_lengths) = parse_restore_header_from_toml(&header_toml)?;

            let (temp, file) = create_temporary_output(&output_path)?;
            let restored = restore_file(
                &rows,
                layout,
                header_lines,
                Some(&block_lengths),
                indices.as_deref(),
            )?;
            restored.write_to(std::io::BufWriter::new(file))?;
            let record_count = restored.records.len();
            let bytes_written = std::fs::metadata(temp.path())?.len();
            publish_temporary_output(temp.path(), &output_path, false)?;
            Ok((record_count, bytes_written))
        })
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("success", true)?;
    dict.set_item("output_file", output)?;
    dict.set_item("record_count", stats.0)?;
    dict.set_item("output_bytes", stats.1)?;
    Ok(dict.into())
}

fn parse_restore_header_from_toml(toml_content: &str) -> Result<([String; 5], Vec<usize>)> {
    use toml::Value;

    let toml_content = toml_content.replace("\r\n", "\n");
    let toml_value: Value =
        toml::from_str(toml_content.trim()).with_context(|| "failed to parse header TOML")?;
    let header_lines = toml_value
        .get("header_info")
        .and_then(|v| v.get("header_lines"))
        .and_then(|v| v.as_array())
        .context("header_info.header_lines not found in TOML")?;
    let lines: Vec<String> = header_lines
        .iter()
        .map(|line| {
            line.as_str()
                .map(str::to_owned)
                .context("header_lines entry is not a string")
        })
        .collect::<Result<Vec<_>>>()?;
    let header_lines = lines.try_into().map_err(|lines: Vec<String>| {
        anyhow::anyhow!("expected 5 header lines, got {}", lines.len())
    })?;
    let block_values = toml_value
        .get("block_info")
        .and_then(|v| v.get("block_lengths"))
        .and_then(|v| v.as_array())
        .context("block_info.block_lengths not found in TOML")?;
    let block_lengths = block_values
        .iter()
        .map(|value| {
            let length = value
                .as_integer()
                .context("block_lengths entry is not an integer")?;
            usize::try_from(length).context("block_lengths entry must be non-negative")
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((header_lines, block_lengths))
}

/// Python-exposed function to read peel subshells from header file
#[cfg(feature = "python")]
#[pyfunction]
fn py_read_peel_subshells(header_path: String) -> PyResult<Vec<String>> {
    use std::path::Path;
    parquet_batch::read_peel_subshells_from_header(Path::new(&header_path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Register the Python module functions and classes
#[cfg(feature = "python")]
pub fn register_descriptor_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        py_generate_descriptors_from_parquet,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_restore_csfs_from_descriptors, module)?)?;
    module.add_function(wrap_pyfunction!(py_read_peel_subshells, module)?)?;

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////
// Rust Tests
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_j_to_double_j_fractional() {
        assert_eq!(j_to_double_j("3/2").unwrap(), 3);
        assert_eq!(j_to_double_j("5/2").unwrap(), 5);
        assert_eq!(j_to_double_j("1/2").unwrap(), 1);
    }

    #[test]
    fn test_j_to_double_j_integer() {
        assert_eq!(j_to_double_j("2").unwrap(), 4);
        assert_eq!(j_to_double_j("3").unwrap(), 6);
        assert_eq!(j_to_double_j("4").unwrap(), 8);
    }

    #[test]
    fn test_j_to_double_j_with_parity() {
        assert_eq!(j_to_double_j("4-").unwrap(), 8);
        assert_eq!(j_to_double_j("3+").unwrap(), 6);
    }

    #[test]
    fn test_j_to_double_j_invalid() {
        assert!(j_to_double_j("invalid").is_err());
        assert!(j_to_double_j("abc/def").is_err());
    }

    #[test]
    fn test_chunk_string() {
        let result = chunk_string("abcdefghi", 3);
        assert_eq!(result, vec!["abc", "def", "ghi"]);
    }

    #[test]
    fn test_descriptor_generator_creation() {
        let subshells = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string()];
        let generator = CSFDescriptorGenerator::new(subshells.clone());

        assert_eq!(generator.orbital_count(), 3);
        assert_eq!(generator.peel_subshells(), &subshells);
    }

    #[test]
    fn parse_csf_into_reuses_caller_buffer_and_matches_parse_csf() {
        let subshells = vec![
            "5s".to_string(),
            "4d-".to_string(),
            "4d".to_string(),
            "5p-".to_string(),
            "5p".to_string(),
            "6s".to_string(),
        ];
        let generator = CSFDescriptorGenerator::new(subshells);
        let line1 = "  5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)";
        let line2 = "                   3/2               2        ";
        let line3 = "                                           4-  ";

        let expected = generator.parse_csf(line1, line2, line3).unwrap();
        let mut descriptor = vec![99i32; generator.orbital_count() * 3];

        generator
            .parse_csf_into(line1, line2, line3, &mut descriptor)
            .unwrap();

        assert_eq!(descriptor, expected);
    }

    #[test]
    fn parse_csf_into_rejects_wrong_buffer_size() {
        let generator = CSFDescriptorGenerator::new(vec!["5s".to_string()]);
        let mut too_short = vec![0i32; 2];

        let result = generator.parse_csf_into("  5s ( 2)", "", "      0  ", &mut too_short);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("descriptor buffer length"),
            "error should explain buffer length mismatch"
        );
    }

    #[test]
    fn parse_csf_into_zero_electron_subshell_keeps_triplet_zero() {
        let generator = CSFDescriptorGenerator::new(vec![
            "5s".to_string(),
            "4d-".to_string(),
            "4d".to_string(),
        ]);
        let line1 = "  5s ( 0)  4d-( 4)  4d ( 6)";
        let line2 = "                   5/2      ";
        let line3 = "                        4-  ";
        let mut descriptor = vec![99i32; generator.orbital_count() * 3];

        generator
            .parse_csf_into(line1, line2, line3, &mut descriptor)
            .unwrap();

        assert_eq!(&descriptor[0..3], &[0, 0, 0]);
    }

    #[test]
    fn parse_csf_into_treats_short_coupling_lines_as_right_padded() {
        let generator = CSFDescriptorGenerator::new(vec![
            "5s".to_string(),
            "4d-".to_string(),
            "4d".to_string(),
        ]);
        let line1 = "  5s ( 2)  4d-( 4)  4d ( 6)";
        let short_line2 = "                   3/2";
        let short_line3 = "                        4-  ";
        let padded_line2 = format!("{:<width$}", short_line2, width = line1.len());
        let padded_line3 = format!("{:<width$}", short_line3, width = line1.len() + 9);

        let short_result = generator
            .parse_csf(line1, short_line2, short_line3)
            .unwrap();
        let padded_result = generator
            .parse_csf(line1, padded_line2.as_str(), padded_line3.as_str())
            .unwrap();

        assert_eq!(short_result, padded_result);
    }

    #[test]
    fn parse_csf_into_treats_truncated_electron_field_as_empty() {
        let generator = CSFDescriptorGenerator::new(vec!["5s".to_string()]);
        let mut descriptor = vec![0i32; generator.orbital_count() * 3];

        generator
            .parse_csf_into("  5s (2", "", "    0-", &mut descriptor)
            .unwrap();

        assert_eq!(descriptor[0], 0);
    }

    #[test]
    fn coupling_signature_keeps_occupied_zero_and_omits_unoccupied_positions() {
        let descriptor = [2, 1, 0, 0, 0, 0, 3, 5, 8];
        let mut signature = Vec::new();

        coupling_signature_from_descriptor_into(&descriptor, &mut signature).unwrap();

        assert_eq!(signature, [0, 8]);
    }

    #[test]
    fn parse_coupling_signature_reuses_all_descriptor_fixed_width_rules() {
        let generator = CSFDescriptorGenerator::new(vec![
            "5s".to_string(),
            "4d-".to_string(),
            "4d".to_string(),
        ]);
        let line1 = "  5s ( 2)  4d-( 4)  4d ( 6)";
        // The second 9-character field uses the value after ';'. Its aligned
        // line3 field is empty, so coupling falls back to that middle value.
        let line2 = "             1;3/2";
        let line3 = "                        4-  ";
        let mut descriptor = vec![0; generator.orbital_count() * 3];
        let mut signature = Vec::new();

        generator
            .parse_coupling_signature_into(line1, line2, line3, &mut descriptor, &mut signature)
            .unwrap();

        assert_eq!(descriptor, [2, 0, 0, 4, 3, 3, 6, 0, 8]);
        assert_eq!(signature, [0, 3, 8]);
    }

    #[test]
    fn descriptor_pipeline_channel_capacity_limits_high_worker_memory_pressure() {
        assert_eq!(parquet_batch::descriptor_pipeline_channel_capacity(1), 1);
        assert_eq!(parquet_batch::descriptor_pipeline_channel_capacity(2), 2);
        assert_eq!(parquet_batch::descriptor_pipeline_channel_capacity(8), 8);
        assert_eq!(parquet_batch::descriptor_pipeline_channel_capacity(48), 8);
    }

    ////////////////////////////////////////////////////////////////////////////
    // V2 text-path parser tests
    ////////////////////////////////////////////////////////////////////////////

    fn v2_generator(subshells: Vec<String>) -> CSFDescriptorGenerator {
        CSFDescriptorGenerator::new_with_version(subshells, DescriptorVersion::V2)
    }

    /// Generate one physically legal CSF record and return its three
    /// canonical text lines, so V2 text-parser tests never rely on
    /// hand-typed field literals that might not satisfy the real subshell
    /// state tables (as `tests/fixtures/complete.csf` predates
    /// `validate_record` and does not).
    fn generate_one_record_text(
        configuration: Vec<crate::csf_generation::SubshellOccupation>,
        two_j: u16,
    ) -> (Vec<String>, String, String, String) {
        use crate::csf_generation::{GenerationRequest, generate_csfs};

        let request = GenerationRequest {
            core_subshells: Vec::new(),
            configuration,
            min_two_j: two_j,
            max_two_j: two_j,
        };
        let generated = generate_csfs(&request).unwrap();
        let record = &generated.records[0];
        let mut buffer = Vec::new();
        generated.write_record_to(record, &mut buffer).unwrap();
        let text = String::from_utf8(buffer).unwrap();
        let mut lines = text.lines();
        (
            generated.subshells.clone(),
            lines.next().unwrap().to_owned(),
            lines.next().unwrap().to_owned(),
            lines.next().unwrap().to_owned(),
        )
    }

    #[test]
    fn parse_csf_v2_into_basic_six_orbitals() {
        use crate::csf_generation::SubshellOccupation;

        let (subshells, line1, line2, line3) = generate_one_record_text(
            vec![
                SubshellOccupation {
                    subshell: "5s".parse().unwrap(),
                    electrons: 2,
                },
                SubshellOccupation {
                    subshell: "4d-".parse().unwrap(),
                    electrons: 4,
                },
                SubshellOccupation {
                    subshell: "4d".parse().unwrap(),
                    electrons: 3,
                },
            ],
            3,
        );
        let generator = v2_generator(subshells.clone());

        let mut row = vec![0i32; generator.layout().row_len()];
        generator
            .parse_csf_v2_into(&line1, &line2, &line3, &mut row)
            .unwrap();

        // Row length is 4*3+2 = 14.
        assert_eq!(row.len(), 14);
        assert_eq!(subshells.len(), 3);

        // 4d- (index 1, base 4): closed subshell (occupation == capacity),
        // so the generator never prints its state — MISSING, not V1's
        // line2 back-fill of the neighboring coupling.
        assert_eq!(row[4], 4); // occupation
        assert_eq!(row[5], MISSING); // no printed 2J
        assert_eq!(row[6], MISSING); // no printed seniority

        // 4d (index 2, base 8, last occupied position): total 2J is a
        // separate global column, never folded into this subshell's slot.
        assert_eq!(row[8], 3); // occupation
        assert_eq!(row[10], MISSING); // no printed seniority for a single hole
        assert_eq!(row[11], MISSING); // last position: 2K boundary never printed

        let total_index = generator.layout().total_two_j_index().unwrap();
        assert_eq!(row[total_index], 3);
    }

    #[test]
    fn parse_csf_v2_into_reads_parity_byte() {
        let subshells = vec!["5s".to_string()];
        let generator = v2_generator(subshells);
        let mut row = vec![0i32; generator.layout().row_len()];

        generator
            .parse_csf_v2_into("  5s ( 1)", "", "        3+", &mut row)
            .unwrap();
        assert_eq!(row[row.len() - 1], 1);

        let error = generator
            .parse_csf_v2_into("  5s ( 1)", "", "        3x", &mut row)
            .unwrap_err();
        assert!(error.to_string().contains("invalid parity byte"));
    }

    #[test]
    fn parse_csf_v2_into_keeps_printed_seniority() {
        // 4f(4) at 2J=4 (doubled: 8) is only reachable with seniority 2 or 4
        // (the design doc §2.2 collision sample); seniority occupies the
        // fixed field offsets 3-4 (`kopp1`'s "s;"), J right-aligned in the
        // remaining 9-wide field.
        let subshells = vec!["4f".to_string()];
        let generator = v2_generator(subshells);
        let mut row = vec![0i32; generator.layout().row_len()];

        generator
            .parse_csf_v2_into("  4f ( 4)", "   2;   4", "        4+", &mut row)
            .unwrap();

        assert_eq!(row[0], 4); // occupation
        assert_eq!(row[1], 8); // 2J = 4 -> 8
        assert_eq!(row[2], 2); // seniority kept, not discarded like V1
    }

    #[test]
    fn parse_csf_v2_into_errors_where_v1_silently_recovers() {
        let subshells = vec!["5s".to_string(), "4d-".to_string(), "4d".to_string()];
        let generator = v2_generator(subshells);
        let mut row = vec![0i32; generator.layout().row_len()];

        // (a) Coupling line too short to slice with the fixed 4/5 offsets.
        assert!(
            generator
                .parse_csf_v2_into("  5s ( 2)  4d-( 4)  4d ( 6)", "", "  4-", &mut row)
                .is_err()
        );

        // (b) Unparsable line2 J value must propagate, not become printed 0.
        assert!(
            generator
                .parse_csf_v2_into(
                    "  5s ( 2)  4d-( 4)  4d ( 6)",
                    "         garbage         ",
                    "                        4-  ",
                    &mut row
                )
                .is_err()
        );

        // (c) Unparsable line3 coupling value must propagate.
        assert!(
            generator
                .parse_csf_v2_into(
                    "  5s ( 2)  4d-( 4)  4d ( 6)",
                    "                   3/2      ",
                    "                garbage4-  ",
                    &mut row
                )
                .is_err()
        );

        // (d) Orbital missing from the peel map must error, not warn-and-skip.
        let missing_generator = v2_generator(vec!["5s".to_string()]);
        let mut missing_row = vec![0i32; missing_generator.layout().row_len()];
        assert!(
            missing_generator
                .parse_csf_v2_into("  6p ( 2)", "", "        0+", &mut missing_row)
                .is_err()
        );

        // (e) Truncated/unparsable electron count must error, not become 0.
        assert!(
            generator
                .parse_csf_v2_into("  5s (2  ", "", "                        0+", &mut row)
                .is_err()
        );

        // (f) Out-of-Peel-order occupied subshells must error via validate_record.
        let reordered_generator = v2_generator(vec!["4d-".to_string(), "5s".to_string()]);
        let mut reordered_row = vec![0i32; reordered_generator.layout().row_len()];
        assert!(
            reordered_generator
                .parse_csf_v2_into(
                    "  5s ( 2)  4d-( 4)",
                    "",
                    "           0+",
                    &mut reordered_row
                )
                .is_err()
        );
    }

    #[test]
    fn parse_csf_v2_into_rejects_zero_electron_count() {
        // V1 accepts an explicit "( 0)" occupation and zero-fills the triplet;
        // V2 requires validate_record's occupation >= 1 instead.
        let generator = v2_generator(vec!["5s".to_string()]);
        let mut row = vec![0i32; generator.layout().row_len()];
        let error = generator
            .parse_csf_v2_into("  5s ( 0)", "", "        0+", &mut row)
            .unwrap_err();
        assert!(error.to_string().contains("outside 1"));
    }

    #[test]
    fn v2_rejects_normalize() {
        use crate::descriptor_schema::{DescriptorLayout, output_schema};
        let layout = DescriptorLayout::new(DescriptorVersion::V2, 2);
        assert!(output_schema(layout, true).is_err());
    }

    #[test]
    fn text_and_complete_producers_agree_on_v2() {
        use crate::complete_csf::CompleteCsfFile;
        use crate::descriptor_v2::encode_v2;
        use std::io::Cursor;

        // The shared fixture (`tests/fixtures/complete.csf`) prints a state
        // outside GRASP's real state tables (see descriptor_v2.rs test
        // comments), so build a physically legal multi-record file with the
        // generator instead, then compare the two V2 producers row for row.
        use crate::csf_generation::{GenerationRequest, SubshellOccupation, generate_csfs};
        let request = GenerationRequest {
            core_subshells: Vec::new(),
            configuration: vec![
                SubshellOccupation {
                    subshell: "4f-".parse().unwrap(),
                    electrons: 3,
                },
                SubshellOccupation {
                    subshell: "4f".parse().unwrap(),
                    electrons: 4,
                },
            ],
            min_two_j: 1,
            max_two_j: 5,
        };
        let generated = generate_csfs(&request).unwrap();
        let mut text = Vec::new();
        generated.write_to(&mut text).unwrap();
        let parsed = CompleteCsfFile::parse_reader(Cursor::new(text)).unwrap();

        let generator = v2_generator(parsed.subshells.clone());
        for record in &parsed.records {
            let (line1, line2, line3) = format_record_for_test(&parsed, record);
            let mut text_row = vec![0i32; generator.layout().row_len()];
            generator
                .parse_csf_v2_into(&line1, &line2, &line3, &mut text_row)
                .unwrap();

            let mut complete_row = vec![0i32; generator.layout().row_len()];
            encode_v2(&parsed, record, &mut complete_row).unwrap();

            assert_eq!(text_row, complete_row);
        }
    }

    fn format_record_for_test(
        file: &crate::complete_csf::CompleteCsfFile,
        record: &crate::complete_csf::CsfRecord,
    ) -> (String, String, String) {
        let mut buffer = Vec::new();
        file.write_record_to(record, &mut buffer).unwrap();
        let text = String::from_utf8(buffer).unwrap();
        let mut lines = text.lines();
        (
            lines.next().unwrap().to_owned(),
            lines.next().unwrap().to_owned(),
            lines.next().unwrap().to_owned(),
        )
    }
}
