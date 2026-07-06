//! CSF Descriptor Generation Module
//!
//! This module converts Configuration State Function (CSF) data into descriptor arrays
//! for machine learning applications. Each CSF is parsed into a fixed-length array
//! containing electron counts and angular momentum coupling values.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

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
                parquet::basic::ZstdLevel::try_new(3)
                    .expect("zstd level 3 is always valid"),
            ));
        };

        let lower = spec.trim().to_ascii_lowercase();
        match lower.as_str() {
            "none" | "uncompressed" => Ok(Compression::UNCOMPRESSED),
            "snappy" => Ok(Compression::SNAPPY),
            "zstd" => Ok(Compression::ZSTD(
                parquet::basic::ZstdLevel::try_new(3)
                    .expect("zstd level 3 is always valid"),
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

        // Peel subshells are on line 4 (index 3): "  2s   2p-  2p   3s..."
        if let Some(line_value) = header_lines.get(3) {
            let line = line_value
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("header_lines[3] is not a string"))?;

            // Parse space-separated subshell names
            let parts: Vec<&str> = line.split_whitespace().collect();

            // Filter valid subshell names (must contain at least one letter)
            let subshells: Vec<String> = parts
                .into_iter()
                .filter(|s| {
                    s.chars().any(|c| c.is_alphabetic())
                        && s.chars()
                            .all(|c| c.is_alphanumeric() || c == '+' || c == '-' || c == '_')
                })
                .map(|s| s.to_string())
                .collect();

            if !subshells.is_empty() {
                return Ok(subshells);
            }
        }

        Err(anyhow::anyhow!(
            "Could not find peel subshells in header file"
        ))
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
    }

    /// Generate descriptors from a parquet file and write to Parquet file
    ///
    /// # Arguments
    /// * `input_parquet` - Path to input parquet file (must have line1, line2, line3 columns)
    /// * `output_file` - Path to output Parquet file for descriptors
    /// * `peel_subshells` - Optional list of subshell names (auto-detected if None)
    /// * `header_path` - Optional path to header TOML file
    /// * `normalize` - Whether to normalize descriptors (default: false)
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
        compression: Option<&str>,
    ) -> Result<BatchDescriptorStats> {
        // Step 1: Determine peel_subshells
        let peel_subshells = match peel_subshells {
            Some(s) => s,
            None => {
                // Try to find and read header file
                let header = match header_path {
                    Some(h) => h,
                    None => find_header_file(input_parquet)
                        .ok_or_else(|| anyhow::anyhow!("Could not auto-detect header file. Please provide peel_subshells or header_path."))?,
                };
                read_peel_subshells_from_header(&header)?
            }
        };

        let orbital_count = peel_subshells.len();
        let descriptor_size = 3 * orbital_count;

        // Step 2: Create descriptor generator
        let generator = super::CSFDescriptorGenerator::new(peel_subshells.clone());

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
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        // Output schema: descriptor columns (one column per descriptor element)
        // Use Float32 for normalized output, Int32 for raw descriptors
        let output_type = if normalize {
            DataType::Float32
        } else {
            DataType::Int32
        };
        let mut fields = Vec::with_capacity(descriptor_size);
        for i in 0..descriptor_size {
            fields.push(Field::new(format!("col_{}", i), output_type.clone(), false));
        }
        let output_schema = Arc::new(Schema::new(fields));

        let output_file_handle = std::fs::File::create(output_file)
            .with_context(|| format!("Failed to create output file: {}", output_file.display()))?;

        // Normalized Float32 descriptors have near-unique values where dictionary
        // encoding is pure overhead (confirmed by benchmark: ~11.5s saved at 48 workers).
        // Raw Int32 descriptors may have enough repetition for dictionary to help,
        // but this has not been A/B benchmarked yet — keep parquet default (enabled).
        let props = WriterProperties::builder()
            .set_compression(parse_compression(compression)?)
            .set_dictionary_enabled(!normalize)
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
                        let mut builders: Vec<Int32Builder> = (0..descriptor_size)
                            .map(|_| arrow::array::Int32Builder::with_capacity(batch_size))
                            .collect();

                        for i in 0..batch_size {
                            let line1 = line1_col.value(i);
                            let line2 = line2_col.value(i);
                            let line3 = line3_col.value(i);
                            let idx = idx_col.value(i);

                            match generator.parse_csf(line1, line2, line3) {
                                Ok(descriptor) => {
                                    // Append directly to column builders
                                    for (col_idx, &val) in descriptor.iter().enumerate() {
                                        builders[col_idx].append_value(val);
                                    }
                                }
                                Err(e) => {
                                    eprintln!(
                                        "Warning: Failed to parse CSF at index {}: {}",
                                        idx, e
                                    );
                                    for builder in &mut builders {
                                        builder.append_value(0i32);
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

    #[derive(Debug, Default)]
    struct StageTimings {
        parallel: std::time::Duration,
        merge: std::time::Duration,
    }

    #[derive(Debug, Default)]
    struct ComputeStats {
        batches_processed: usize,
        rows_processed: usize,
        parallel_elapsed: std::time::Duration,
        merge_elapsed: std::time::Duration,
        recv_wait_elapsed: std::time::Duration,
        send_elapsed: std::time::Duration,
    }

    #[derive(Debug, Default)]
    struct WriterStats {
        total_descriptors: usize,
        lifetime: std::time::Duration,
        recv_wait_elapsed: std::time::Duration,
        array_build_elapsed: std::time::Duration,
        batch_build_elapsed: std::time::Duration,
        write_elapsed: std::time::Duration,
        finish_elapsed: std::time::Duration,
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
    ) -> (Vec<Vec<i32>>, StageTimings) {
        use rayon::prelude::*;
        use std::time::Instant;

        let descriptor_size = 3 * generator.orbital_count();
        let batch_size = rows.len();
        let chunk_size = descriptor_chunk_size(batch_size, pool.current_num_threads());
        let mut timings = StageTimings::default();

        let parallel_start = Instant::now();
        let chunk_columns: Vec<(usize, Vec<Vec<i32>>)> = pool.install(|| {
            rows.par_chunks(chunk_size)
                .enumerate()
                .map(|(chunk_idx, chunk)| {
                    let mut columns: Vec<Vec<i32>> = (0..descriptor_size)
                        .map(|_| Vec::with_capacity(chunk.len()))
                        .collect();
                    let mut descriptor = vec![0i32; descriptor_size];

                    for (idx, line1, line2, line3) in chunk {
                        if let Err(e) =
                            generator.parse_csf_into(line1, line2, line3, &mut descriptor)
                        {
                            eprintln!("Warning: Failed to parse CSF at index {}: {}", idx, e);
                            descriptor.fill(0);
                        }
                        for (col_idx, column) in columns.iter_mut().enumerate() {
                            column.push(descriptor[col_idx]);
                        }
                    }

                    (chunk_idx, columns)
                })
                .collect()
        });
        timings.parallel = parallel_start.elapsed();

        let merge_start = Instant::now();
        let columns: Vec<Vec<i32>> = pool.install(|| {
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
        });
        timings.merge = merge_start.elapsed();

        (columns, timings)
    }

    fn build_normalized_descriptor_columns_parallel(
        pool: &rayon::ThreadPool,
        generator: Arc<super::CSFDescriptorGenerator>,
        peel_subshells: Arc<Vec<String>>,
        rows: Vec<DescriptorRow>,
    ) -> (Vec<Vec<f32>>, StageTimings) {
        use crate::descriptor_normalization::{infer_two_j_target, normalize_descriptor_per_csf};
        use rayon::prelude::*;
        use std::time::Instant;

        let descriptor_size = 3 * generator.orbital_count();
        let batch_size = rows.len();
        let chunk_size = descriptor_chunk_size(batch_size, pool.current_num_threads());
        let mut timings = StageTimings::default();

        let parallel_start = Instant::now();
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
        timings.parallel = parallel_start.elapsed();

        let merge_start = Instant::now();
        let columns: Vec<Vec<f32>> = pool.install(|| {
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
        });
        timings.merge = merge_start.elapsed();

        (columns, timings)
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
    /// Output format: Parquet with configurable compression (default ZSTD level 3),
    /// PLAIN encoding (dictionary disabled for throughput)
    ///
    /// # Arguments
    /// * `input_parquet` - Path to input parquet file
    /// * `output_file` - Path to output Parquet file
    /// * `peel_subshells` - List of subshell names
    /// * `num_workers` - Number of worker threads (default: CPU core count)
    /// * `normalize` - Whether to normalize descriptors (default: false)
    /// * `compression` - Optional parquet compression specifier (default: `zstd-3`).
    ///   See [`parse_compression`] for accepted values.
    pub fn generate_descriptors_from_parquet_parallel(
        input_parquet: &Path,
        output_file: &Path,
        peel_subshells: Vec<String>,
        num_workers: Option<usize>,
        normalize: bool,
        compression: Option<&str>,
    ) -> Result<BatchDescriptorStats> {
        use arrow::array::{Array, StringArray, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use crossbeam_channel::{Receiver, Sender, bounded};
        use parquet::file::properties::WriterProperties;
        use std::collections::BTreeMap;
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        let total_start = Instant::now();

        // Determine worker count
        let num_workers = num_workers.unwrap_or_else(num_cpus::get);
        if num_workers == 0 {
            return Err(anyhow::anyhow!("num_workers must be greater than 0"));
        }

        let orbital_count = peel_subshells.len();
        let descriptor_size = 3 * orbital_count;

        println!("开始生成描述符...");
        println!("输入: {:?} | 输出: {:?}", input_parquet, output_file);
        println!(
            "Worker: {} | 轨道: {} | 描述符大小: {}",
            num_workers, orbital_count, descriptor_size
        );
        if normalize {
            println!("归一化: 启用 (per-CSF physics-correct normalization)");
        }

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
        // Use Float32 for normalized output, Int32 for raw descriptors
        let output_type = if normalize {
            DataType::Float32
        } else {
            DataType::Int32
        };
        let mut fields = Vec::with_capacity(descriptor_size);
        for i in 0..descriptor_size {
            fields.push(Field::new(format!("col_{}", i), output_type.clone(), false));
        }
        let schema = Arc::new(Schema::new(fields));

        let output_file_handle = std::fs::File::create(output_file)
            .with_context(|| format!("Failed to create output file: {}", output_file.display()))?;

        // See sequential path for dictionary encoding rationale.
        let props = WriterProperties::builder()
            .set_compression(parse_compression(compression)?)
            .set_dictionary_enabled(!normalize)
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
            let reader_start = Instant::now();
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
            let mut read_elapsed = Duration::ZERO;
            let mut row_copy_elapsed = Duration::ZERO;
            let mut send_wait_elapsed = Duration::ZERO;

            loop {
                let read_start = Instant::now();
                let next_batch = reader.next();
                read_elapsed += read_start.elapsed();

                match next_batch {
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
                        let copy_start = Instant::now();
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
                        row_copy_elapsed += copy_start.elapsed();

                        let work_item = WorkItem { batch_idx, rows };
                        let send_start = Instant::now();
                        match work_tx.send(work_item) {
                            Ok(()) => {}
                            Err(_) => return Err(anyhow::anyhow!("Failed to send work item")),
                        }
                        send_wait_elapsed += send_start.elapsed();
                        batch_idx += 1;

                        if total_csfs.is_multiple_of(10_000_000) {
                            println!("[读取进度] {} 个 CSF", total_csfs);
                        }
                    }
                    Some(Err(e)) => {
                        return Err(anyhow::anyhow!("Error reading parquet batch: {}", e));
                    }
                    None => break,
                }
            }

            println!("[读取完成] {} 个 CSF", total_csfs);
            Ok((
                total_csfs,
                batch_idx,
                reader_start.elapsed(),
                read_elapsed,
                row_copy_elapsed,
                send_wait_elapsed,
            ))
        });

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 4: Compute thread - process each batch with a bounded Rayon pool
        ////////////////////////////////////////////////////////////////////////////////
        let peel_subshells_for_normalization = Arc::new(peel_subshells.clone());
        let generator = Arc::new(super::CSFDescriptorGenerator::new(peel_subshells));
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
                let mut batches_processed = 0usize;
                let mut rows_processed = 0usize;
                let mut parallel_elapsed = Duration::ZERO;
                let mut merge_elapsed = Duration::ZERO;
                let mut recv_wait_elapsed = Duration::ZERO;
                let mut send_elapsed = Duration::ZERO;

                loop {
                    let recv_start = Instant::now();
                    let work_item = match work_rx_clone.recv() {
                        Ok(item) => item,
                        Err(_) => break,
                    };
                    recv_wait_elapsed += recv_start.elapsed();

                    // Check for error sentinel
                    if work_item.batch_idx == usize::MAX {
                        return Err(anyhow::anyhow!("Reader thread encountered an error"));
                    }

                    let batch_idx = work_item.batch_idx;
                    let batch_size = work_item.rows.len();
                    let columns = if normalize_enabled {
                        let (cols, timings) =
                            build_normalized_descriptor_columns_parallel(
                                &rayon_pool,
                                generator_clone.clone(),
                                peel_subshells_for_normalization.clone(),
                                work_item.rows,
                            );
                        parallel_elapsed += timings.parallel;
                        merge_elapsed += timings.merge;
                        DescriptorColumns::Normalized(cols)
                    } else {
                        let (cols, timings) = build_raw_descriptor_columns_parallel(
                            &rayon_pool,
                            generator_clone.clone(),
                            work_item.rows,
                        );
                        parallel_elapsed += timings.parallel;
                        merge_elapsed += timings.merge;
                        DescriptorColumns::Raw(cols)
                    };

                    let result_item = ResultItem {
                        batch_idx,
                        batch_size,
                        columns,
                    };
                    let send_start = Instant::now();
                    if result_tx_clone.send(result_item).is_err() {
                        return Err(anyhow::anyhow!("Failed to send result item"));
                    }
                    send_elapsed += send_start.elapsed();
                    batches_processed += 1;
                    rows_processed += batch_size;
                }

                Ok(ComputeStats {
                    batches_processed,
                    rows_processed,
                    parallel_elapsed,
                    merge_elapsed,
                    recv_wait_elapsed,
                    send_elapsed,
                })
            }));
        }

        // Drop our clone of the result_tx so the writer can properly detect when workers are done
        drop(result_tx);

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 5: Writer thread - maintain order and write to parquet (multi-column format)
        ////////////////////////////////////////////////////////////////////////////////
        let writer_handle: std::thread::JoinHandle<Result<WriterStats>> =
            std::thread::spawn(move || {
                use arrow::array::{Float32Array, Int32Array};

                let writer_start = Instant::now();
                let mut pending: BTreeMap<usize, ResultItem> = BTreeMap::new();
                let mut next_write_idx = 0usize;
                let mut total_descriptors = 0usize;
                let mut total_batches_written = 0usize;
                let mut recv_wait_elapsed = Duration::ZERO;
                let mut array_build_elapsed = Duration::ZERO;
                let mut batch_build_elapsed = Duration::ZERO;
                let mut write_elapsed = Duration::ZERO;

                loop {
                    let recv_start = Instant::now();
                    let result_item = match result_rx.recv() {
                        Ok(item) => item,
                        Err(_) => break,
                    };
                    recv_wait_elapsed += recv_start.elapsed();

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

                        let array_start = Instant::now();
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
                                .map(|column| {
                                    Arc::new(Float32Array::from(column)) as Arc<dyn Array>
                                })
                                .collect()
                        } else {
                            let columns = match result_item.columns {
                                DescriptorColumns::Raw(columns) => columns,
                                DescriptorColumns::Normalized(_) => {
                                    return Err(anyhow::anyhow!(
                                        "Expected raw descriptor columns"
                                    ));
                                }
                            };
                            columns
                                .into_iter()
                                .map(|column| Arc::new(Int32Array::from(column)) as Arc<dyn Array>)
                                .collect()
                        };
                        array_build_elapsed += array_start.elapsed();

                        let batch_start = Instant::now();
                        let output_batch = match RecordBatch::try_new(schema.clone(), column_arrays)
                        {
                            Ok(b) => b,
                            Err(e) => {
                                return Err(anyhow::anyhow!(
                                    "Failed to create output batch: {}",
                                    e
                                ));
                            }
                        };
                        batch_build_elapsed += batch_start.elapsed();

                        let write_start = Instant::now();
                        if writer_guard
                            .writer
                            .as_mut()
                            .expect("writer exists until finish")
                            .write(&output_batch)
                            .is_err()
                        {
                            return Err(anyhow::anyhow!("Failed to write batch"));
                        }
                        write_elapsed += write_start.elapsed();

                        total_batches_written += 1;
                        next_write_idx += 1;

                        if total_batches_written.is_multiple_of(100) {
                            println!("[写入进度] {} 个描述符", total_descriptors);
                        }
                    }
                }

                let finish_start = Instant::now();
                writer_guard.finish()?;
                let finish_elapsed = finish_start.elapsed();
                println!("[写入完成] {} 个描述符", total_descriptors);
                Ok(WriterStats {
                    total_descriptors,
                    lifetime: writer_start.elapsed(),
                    recv_wait_elapsed,
                    array_build_elapsed,
                    batch_build_elapsed,
                    write_elapsed,
                    finish_elapsed,
                })
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
        let mut total_parallel_elapsed = Duration::ZERO;
        let mut total_merge_elapsed = Duration::ZERO;
        let mut total_recv_wait_elapsed = Duration::ZERO;
        let mut total_send_elapsed = Duration::ZERO;
        let mut total_compute_rows = 0usize;
        let mut total_compute_batches = 0usize;
        for (i, handle) in worker_handles.into_iter().enumerate() {
            match handle.join() {
                Ok(Ok(stats)) => {
                    total_parallel_elapsed += stats.parallel_elapsed;
                    total_merge_elapsed += stats.merge_elapsed;
                    total_recv_wait_elapsed += stats.recv_wait_elapsed;
                    total_send_elapsed += stats.send_elapsed;
                    total_compute_rows += stats.rows_processed;
                    total_compute_batches += stats.batches_processed;
                }
                Ok(Err(e)) => {
                    errors.push(format!("Worker thread {} failed: {:#}", i, e));
                }
                Err(e) => {
                    errors.push(format!("Worker thread {} panicked: {:?}", i, e));
                }
            }
        }
        println!(
            "[计算完成] batches: {} | rows: {} | parallel: {:.2?} | merge: {:.2?} | wait_reader: {:.2?} | wait_writer: {:.2?}",
            total_compute_batches,
            total_compute_rows,
            total_parallel_elapsed,
            total_merge_elapsed,
            total_recv_wait_elapsed,
            total_send_elapsed,
        );

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
            return Err(anyhow::anyhow!(
                "Parallel descriptor generation failed: {}",
                errors.join("; ")
            ));
        }

        let (total_csfs, _, reader_elapsed, read_elapsed, row_copy_elapsed, reader_send_wait) =
            reader_result.expect("reader result exists when no errors occurred");
        let writer_stats = writer_result.expect("writer result exists when no errors occurred");

        println!("====================================");
        println!("处理完成！");
        println!(
            "输入 CSF: {} | 生成描述符: {}",
            total_csfs, writer_stats.total_descriptors
        );
        println!(
            "轨道数: {} | 描述符大小: {}",
            orbital_count, descriptor_size
        );
        println!(
            "耗时: total {:.2?}",
            total_start.elapsed(),
        );
        println!(
            "  reader:   lifetime {:.2?} | read_decode {:.2?} | row_copy {:.2?} | send_wait {:.2?}",
            reader_elapsed, read_elapsed, row_copy_elapsed, reader_send_wait,
        );
        println!(
            "  compute:  parallel {:.2?} | merge {:.2?} | wait_reader {:.2?} | wait_writer {:.2?}",
            total_parallel_elapsed,
            total_merge_elapsed,
            total_recv_wait_elapsed,
            total_send_elapsed,
        );
        println!(
            "  writer:   lifetime {:.2?} | recv_wait {:.2?} | array_build {:.2?} | batch_build {:.2?} | write {:.2?} | finish {:.2?}",
            writer_stats.lifetime,
            writer_stats.recv_wait_elapsed,
            writer_stats.array_build_elapsed,
            writer_stats.batch_build_elapsed,
            writer_stats.write_elapsed,
            writer_stats.finish_elapsed,
        );
        println!("====================================");

        Ok(BatchDescriptorStats {
            input_file: input_parquet.to_string_lossy().to_string(),
            output_file: output_file.to_string_lossy().to_string(),
            csf_count: total_csfs,
            descriptor_count: writer_stats.total_descriptors,
            orbital_count,
            descriptor_size,
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

            let (columns, _timings) =
                build_raw_descriptor_columns_parallel(&pool, generator.clone(), rows.clone());

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

            let (columns, _timings) = build_normalized_descriptor_columns_parallel(
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
            assert!(matches!(parse_compression(Some("none")).unwrap(), Compression::UNCOMPRESSED));
            assert!(matches!(
                parse_compression(Some("uncompressed")).unwrap(),
                Compression::UNCOMPRESSED
            ));
            assert!(matches!(parse_compression(Some("snappy")).unwrap(), Compression::SNAPPY));
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
            assert!(matches!(parse_compression(Some("NONE")).unwrap(), Compression::UNCOMPRESSED));
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
    /// Count missing-subshell warnings so malformed input cannot flood stderr.
    missing_subshell_warning_count: AtomicUsize,
}

impl CSFDescriptorGenerator {
    /// Create a new CSF descriptor generator
    ///
    /// # Arguments
    /// * `peel_subshells` - List of subshell names (e.g., ["5s", "4d-", "4d"])
    ///
    /// # Returns
    /// A new generator instance
    pub fn new(peel_subshells: Vec<String>) -> Self {
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
    compression=None
))]
fn py_generate_descriptors_from_parquet(
    py: Python,
    input_parquet: String,
    output_file: String,
    peel_subshells: Vec<String>,
    num_workers: Option<usize>,
    normalize: bool,
    compression: Option<String>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    use pyo3::types::PyDict;
    use std::path::Path;

    let input_path = Path::new(&input_parquet).to_path_buf();
    let output_path = Path::new(&output_file).to_path_buf();

    if matches!(num_workers, Some(0)) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_workers must be greater than 0",
        ));
    }

    // Release the GIL during the long-running operation
    let stats = py
        .detach(|| {
            parquet_batch::generate_descriptors_from_parquet_parallel(
                &input_path,
                &output_path,
                peel_subshells,
                num_workers,
                normalize,
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
    Ok(dict.into())
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
    fn descriptor_pipeline_channel_capacity_limits_high_worker_memory_pressure() {
        assert_eq!(parquet_batch::descriptor_pipeline_channel_capacity(1), 1);
        assert_eq!(parquet_batch::descriptor_pipeline_channel_capacity(2), 2);
        assert_eq!(parquet_batch::descriptor_pipeline_channel_capacity(8), 8);
        assert_eq!(parquet_batch::descriptor_pipeline_channel_capacity(48), 8);
    }
}
