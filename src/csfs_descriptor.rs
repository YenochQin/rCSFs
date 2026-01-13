//! CSF Descriptor Generation Module
//!
//! This module converts Configuration State Function (CSF) data into descriptor arrays
//! for machine learning applications. Each CSF is parsed into a fixed-length array
//! containing electron counts and angular momentum coupling values.

use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::Path;

/// Parquet reading/writing support
pub mod parquet_batch {
    use super::*;
    use arrow::array::{StringArray, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::path::PathBuf;
    use std::sync::Arc;

    /// Read peel subshells from a header TOML file
    ///
    /// # Arguments
    /// * `header_path` - Path to the header TOML file
    ///
    /// # Returns
    /// * `Ok(Vec<String>)` - List of peel subshell names
    /// * `Err(String)` - Error message if parsing fails
    pub fn read_peel_subshells_from_header(header_path: &Path) -> Result<Vec<String>, String> {
        use toml::Value;

        let mut toml_content = read_to_string(header_path)
            .map_err(|e| format!("Failed to read header file: {}", e))?;

        // Normalize line endings and trim whitespace
        toml_content = toml_content.replace("\r\n", "\n");
        let toml_content = toml_content.trim();

        // Parse the TOML content using from_str instead of parse()
        let toml_value: Value =
            toml::from_str(toml_content).map_err(|e| format!("Failed to parse TOML: {}", e))?;

        // Get header_lines from [header_info] section
        let header_lines = toml_value
            .get("header_info")
            .and_then(|v| v.get("header_lines"))
            .and_then(|v| v.as_array())
            .ok_or("header_info.header_lines not found in TOML")?;

        // Peel subshells are on line 4 (index 3): "  2s   2p-  2p   3s..."
        if let Some(line_value) = header_lines.get(3) {
            let line = line_value
                .as_str()
                .ok_or("header_lines[3] is not a string")?;

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

        Err("Could not find peel subshells in header file".to_string())
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
    /// * `max_cumulative_doubled_j` - Maximum cumulative 2J value for normalization (required if normalize=true)
    ///
    /// # Returns
    /// * `Ok(BatchDescriptorStats)` - Statistics about the batch operation
    /// * `Err(String)` - Error message if operation fails
    ///
    /// # Output Format
    /// Parquet with ZSTD compression (level 3) - columnar format, Polars compatible
    /// Read with: `polars.read_parquet()` or `pyarrow.parquet.read_table()`
    pub fn generate_descriptors_from_parquet(
        input_parquet: &Path,
        output_file: &Path,
        peel_subshells: Option<Vec<String>>,
        header_path: Option<PathBuf>,
        normalize: bool,
        max_cumulative_doubled_j: Option<i32>,
    ) -> Result<BatchDescriptorStats, String> {
        // Step 1: Determine peel_subshells
        let peel_subshells = match peel_subshells {
            Some(s) => s,
            None => {
                // Try to find and read header file
                let header = match header_path {
                    Some(h) => h,
                    None => find_header_file(input_parquet)
                        .ok_or("Could not auto-detect header file. Please provide peel_subshells or header_path.")?,
                };
                read_peel_subshells_from_header(&header)?
            }
        };

        let orbital_count = peel_subshells.len();
        let descriptor_size = 3 * orbital_count;

        // Validate normalization parameters
        let max_cumulative_doubled_j = if normalize {
            match max_cumulative_doubled_j {
                Some(v) => v,
                None => return Err("max_cumulative_doubled_j is required when normalize=true".to_string()),
            }
        } else {
            0  // Unused when normalize=false
        };

        // Pre-compute normalization reciprocals if needed
        let normalization_reciprocals: Option<Vec<f32>> = if normalize {
            use crate::descriptor_normalization::{get_subshells_properties, compute_properties_reciprocals};
            let denominators = get_subshells_properties(&peel_subshells, max_cumulative_doubled_j)?;
            Some(compute_properties_reciprocals(&denominators)?)
        } else {
            None
        };

        // Step 2: Create descriptor generator
        let generator = super::CSFDescriptorGenerator::new(peel_subshells.clone());

        // Step 3: Open input parquet file
        let file = std::fs::File::open(input_parquet)
            .map_err(|e| format!("Failed to open input parquet: {}", e))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| format!("Failed to create parquet reader: {}", e))?;

        let _schema = builder.schema();

        let mut reader = builder
            .build()
            .map_err(|e| format!("Failed to build parquet reader: {}", e))?;

        // Step 4: Create output Parquet writer (ZSTD compression)
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::arrow_writer::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        // Output schema: descriptor columns (one column per descriptor element)
        // Use Float32 for normalized output, Int32 for raw descriptors
        let output_type = if normalize { DataType::Float32 } else { DataType::Int32 };
        let mut fields = Vec::with_capacity(descriptor_size);
        for i in 0..descriptor_size {
            fields.push(Field::new(format!("col_{}", i), output_type.clone(), false));
        }
        let output_schema = Arc::new(Schema::new(fields));

        let output_file_handle = std::fs::File::create(output_file)
            .map_err(|e| format!("Failed to create output file: {}", e))?;

        // Use ZSTD compression for better I/O performance and smaller file size
        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(
                parquet::basic::ZstdLevel::try_new(3).unwrap(),
            ))
            .build();

        let mut writer = ArrowWriter::try_new(
            output_file_handle,
            output_schema.clone(),
            Some(props),
        )
            .map_err(|e| format!("Failed to create Parquet writer: {}", e))?;

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
                        .ok_or("idx column is not uint64 type")?;

                    let line1_col = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or("line1 column is not string type")?;

                    let line2_col = batch
                        .column(2)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or("line2 column is not string type")?;

                    let line3_col = batch
                        .column(3)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or("line3 column is not string type")?;

                    // Process each row
                    use arrow::array::{Array, Int32Builder, Float32Builder};
                    use std::sync::Arc;

                    // Initialize builders for each column (avoids transpose overhead)
                    // Use Float32Builder for normalized output, Int32Builder for raw descriptors
                    if normalize {
                        let mut builders: Vec<Float32Builder> =
                            (0..descriptor_size)
                                .map(|_| arrow::array::Float32Builder::with_capacity(batch_size))
                                .collect();

                        let reciprocals = normalization_reciprocals.as_ref().unwrap();

                        for i in 0..batch_size {
                            let line1 = line1_col.value(i);
                            let line2 = line2_col.value(i);
                            let line3 = line3_col.value(i);
                            let idx = idx_col.value(i);

                            match generator.parse_csf(line1, line2, line3) {
                                Ok(descriptor) => {
                                    // Normalize and append directly to column builders
                                    for (col_idx, &val) in descriptor.iter().enumerate() {
                                        let normalized_val = val as f32 * reciprocals[col_idx];
                                        builders[col_idx].append_value(normalized_val);
                                    }
                                    descriptor_count += 1;
                                }
                                Err(e) => {
                                    eprintln!("Warning: Failed to parse CSF at index {}: {}", idx, e);
                                }
                            }
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
                                .map_err(|e| format!("Failed to create output batch: {}", e))?;

                        writer
                            .write(&output_batch)
                            .map_err(|e| format!("Failed to write batch: {}", e))?;
                    } else {
                        let mut builders: Vec<Int32Builder> =
                            (0..descriptor_size)
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
                                    descriptor_count += 1;
                                }
                                Err(e) => {
                                    eprintln!("Warning: Failed to parse CSF at index {}: {}", idx, e);
                                }
                            }
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
                                .map_err(|e| format!("Failed to create output batch: {}", e))?;

                        writer
                            .write(&output_batch)
                            .map_err(|e| format!("Failed to write batch: {}", e))?;
                    }

                    total_csfs += batch_size;
                }
                Some(Err(e)) => {
                    return Err(format!("Error reading parquet batch: {}", e));
                }
                None => break,
            }
        }

        // Step 6: Finalize writer
        writer
            .finish()
            .map_err(|e| format!("Failed to finish writer: {}", e))?;

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

    /// Work item sent from reader to workers
    struct WorkItem {
        batch_idx: usize,
        rows: Vec<(u64, Arc<str>, Arc<str>, Arc<str>)>,
    }

    /// Result item sent from workers to writer
    struct ResultItem {
        batch_idx: usize,
        descriptors: Vec<Vec<i32>>,
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
    /// Output format: Parquet with ZSTD compression (level 3)
    ///
    /// # Arguments
    /// * `input_parquet` - Path to input parquet file
    /// * `output_file` - Path to output Parquet file
    /// * `peel_subshells` - List of subshell names
    /// * `num_workers` - Number of worker threads (default: CPU core count)
    /// * `normalize` - Whether to normalize descriptors (default: false)
    /// * `max_cumulative_doubled_j` - Maximum cumulative 2J value for normalization (required if normalize=true)
    pub fn generate_descriptors_from_parquet_parallel(
        input_parquet: &Path,
        output_file: &Path,
        peel_subshells: Vec<String>,
        num_workers: Option<usize>,
        normalize: bool,
        max_cumulative_doubled_j: Option<i32>,
    ) -> Result<BatchDescriptorStats, String> {
        use arrow::array::{Array, StringArray, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::arrow_writer::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;
        use crossbeam_channel::{bounded, Sender, Receiver};
        use std::collections::BTreeMap;

        // Determine worker count
        let num_workers = num_workers.unwrap_or_else(|| {
            let default = num_cpus::get();
            default
        });

        let orbital_count = peel_subshells.len();
        let descriptor_size = 3 * orbital_count;

        // Validate normalization parameters
        let max_cumulative_doubled_j = if normalize {
            match max_cumulative_doubled_j {
                Some(v) => v,
                None => return Err("max_cumulative_doubled_j is required when normalize=true".to_string()),
            }
        } else {
            0  // Unused when normalize=false
        };

        // Pre-compute normalization reciprocals if needed
        let normalization_reciprocals: Option<Vec<f32>> = if normalize {
            use crate::descriptor_normalization::{get_subshells_properties, compute_properties_reciprocals};
            let denominators = get_subshells_properties(&peel_subshells, max_cumulative_doubled_j)?;
            Some(compute_properties_reciprocals(&denominators)?)
        } else {
            None
        };

        println!("开始生成描述符...");
        println!("输入: {:?} | 输出: {:?}", input_parquet, output_file);
        println!("Worker: {} | 轨道: {} | 描述符大小: {}", num_workers, orbital_count, descriptor_size);
        if normalize {
            println!("归一化: 启用 (max_cumulative_doubled_j = {})", max_cumulative_doubled_j);
        }

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 1: Setup channels with bounded capacity
        ////////////////////////////////////////////////////////////////////////////////
        let channel_capacity = num_workers * 2;
        let (work_tx, work_rx): (Sender<WorkItem>, Receiver<WorkItem>) = bounded(channel_capacity);
        let (result_tx, result_rx): (Sender<ResultItem>, Receiver<ResultItem>) = bounded(channel_capacity);

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 2: Setup output schema and writer (multi-column format for better performance)
        ////////////////////////////////////////////////////////////////////////////////
        // Use Float32 for normalized output, Int32 for raw descriptors
        let output_type = if normalize { DataType::Float32 } else { DataType::Int32 };
        let mut fields = Vec::with_capacity(descriptor_size);
        for i in 0..descriptor_size {
            fields.push(Field::new(format!("col_{}", i), output_type.clone(), false));
        }
        let schema = Arc::new(Schema::new(fields));

        let output_file_handle = std::fs::File::create(output_file)
            .map_err(|e| format!("Failed to create output file: {}", e))?;

        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(
                parquet::basic::ZstdLevel::try_new(3).unwrap(),
            ))
            .build();

        let mut writer = ArrowWriter::try_new(
            output_file_handle,
            schema.clone(),
            Some(props),
        )
            .map_err(|e| format!("Failed to create Parquet writer: {}", e))?;

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
                        batch_idx: usize::MAX,  // Error sentinel
                        rows: vec![],
                    });
                    return Err(format!("Failed to open input parquet: {}", e));
                }
            };

            let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
                Ok(b) => b,
                Err(e) => {
                    let _ = work_tx.send(WorkItem {
                        batch_idx: usize::MAX,
                        rows: vec![],
                    });
                    return Err(format!("Failed to create parquet reader: {}", e));
                }
            };

            let mut reader = match builder.with_batch_size(65536).build() {
                Ok(r) => r,
                Err(e) => {
                    let _ = work_tx.send(WorkItem {
                        batch_idx: usize::MAX,
                        rows: vec![],
                    });
                    return Err(format!("Failed to build parquet reader: {}", e));
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
                            None => return Err("idx column is not uint64 type".to_string()),
                        };

                        let line1_col = match batch.column(1).as_any().downcast_ref::<StringArray>() {
                            Some(col) => col,
                            None => return Err("line1 column is not string type".to_string()),
                        };

                        let line2_col = match batch.column(2).as_any().downcast_ref::<StringArray>() {
                            Some(col) => col,
                            None => return Err("line2 column is not string type".to_string()),
                        };

                        let line3_col = match batch.column(3).as_any().downcast_ref::<StringArray>() {
                            Some(col) => col,
                            None => return Err("line3 column is not string type".to_string()),
                        };

                        // Extract rows as Arc<str> for zero-copy sharing across threads
                        let rows: Vec<(u64, Arc<str>, Arc<str>, Arc<str>)> = (0..batch_size)
                            .map(|i| (
                                idx_col.value(i),
                                line1_col.value(i).into(),
                                line2_col.value(i).into(),
                                line3_col.value(i).into(),
                            ))
                            .collect();

                        let work_item = WorkItem { batch_idx, rows };
                        match work_tx.send(work_item) {
                            Ok(()) => {},
                            Err(_) => return Err("Failed to send work item".to_string()),
                        }
                        batch_idx += 1;

                        if total_csfs % 10_000_000 == 0 {
                            println!("[读取进度] {} 个 CSF", total_csfs);
                        }
                    }
                    Some(Err(e)) => {
                        return Err(format!("Error reading parquet batch: {}", e));
                    }
                    None => break,
                }
            }

            println!("[读取完成] {} 个 CSF", total_csfs);
            Ok((total_csfs, batch_idx))
        });

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 4: Multiple Worker threads - compete to process items from channel
        ////////////////////////////////////////////////////////////////////////////////
        let generator = Arc::new(super::CSFDescriptorGenerator::new(peel_subshells));
        let mut worker_handles = Vec::new();

        // Spawn multiple worker threads, all competing on the same channel
        for _worker_id in 0..num_workers {
            let generator_clone = generator.clone();
            let result_tx_clone = result_tx.clone();
            let work_rx_clone = work_rx.clone();

            worker_handles.push(std::thread::spawn(move || {
                use rayon::prelude::*;

                let descriptor_size = 3 * generator_clone.orbital_count();

                // Each worker competes to receive work items
                while let Ok(work_item) = work_rx_clone.recv() {
                    // Check for error sentinel
                    if work_item.batch_idx == usize::MAX {
                        return Err("Reader thread encountered an error".to_string());
                    }

                    let batch_idx = work_item.batch_idx;
                    let descriptors: Vec<Vec<i32>> = work_item.rows
                        .into_par_iter()
                        .map(|(idx, ref line1, ref line2, ref line3)| {
                            match generator_clone.parse_csf(line1, line2, line3) {
                                Ok(desc) => desc,
                                Err(e) => {
                                    eprintln!("Warning: Failed to parse CSF at index {}: {}", idx, e);
                                    vec![0i32; descriptor_size]
                                }
                            }
                        })
                        .collect();

                    let result_item = ResultItem { batch_idx, descriptors };
                    if result_tx_clone.send(result_item).is_err() {
                        return Err("Failed to send result item".to_string());
                    }
                }

                Ok(())
            }));
        }

        // Drop our clone of the result_tx so the writer can properly detect when workers are done
        drop(result_tx);

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 5: Writer thread - maintain order and write to parquet (multi-column format)
        ////////////////////////////////////////////////////////////////////////////////
        let writer_handle = std::thread::spawn(move || {
            use arrow::array::{Int32Builder, Float32Builder};

            let mut pending: BTreeMap<usize, Vec<Vec<i32>>> = BTreeMap::new();
            let mut next_write_idx = 0usize;
            let mut total_descriptors = 0usize;
            let mut total_batches_written = 0usize;

            // Clone normalization reciprocals for writer thread
            let normalization_reciprocals_clone = normalization_reciprocals.clone();

            while let Ok(result_item) = result_rx.recv() {
                let batch_idx = result_item.batch_idx;
                let descriptors = result_item.descriptors;

                // Insert into pending map
                pending.insert(batch_idx, descriptors);

                // Write all consecutive batches we have
                while let Some(descriptors) = pending.remove(&next_write_idx) {
                    let batch_size = descriptors.len();
                    if batch_size == 0 {
                        next_write_idx += 1;
                        continue;
                    }
                    total_descriptors += batch_size;

                    // Build and write batch based on normalization setting
                    if normalize {
                        let reciprocals = normalization_reciprocals_clone.as_ref().unwrap();
                        let mut builders: Vec<Float32Builder> =
                            (0..descriptor_size)
                                .map(|_| Float32Builder::with_capacity(batch_size))
                                .collect();

                        // Build normalized columns
                        for desc in &descriptors {
                            for (col_idx, &val) in desc.iter().enumerate() {
                                let normalized_val = val as f32 * reciprocals[col_idx];
                                builders[col_idx].append_value(normalized_val);
                            }
                        }

                        // Convert builders to Arrow arrays
                        let column_arrays: Vec<Arc<dyn Array>> = builders
                            .into_iter()
                            .map(|mut b| Arc::new(b.finish()) as Arc<dyn Array>)
                            .collect();

                        let output_batch = match RecordBatch::try_new(schema.clone(), column_arrays) {
                            Ok(b) => b,
                            Err(e) => return Err(format!("Failed to create output batch: {}", e)),
                        };

                        if writer.write(&output_batch).is_err() {
                            return Err("Failed to write batch".to_string());
                        }
                    } else {
                        let mut builders: Vec<Int32Builder> =
                            (0..descriptor_size)
                                .map(|_| Int32Builder::with_capacity(batch_size))
                                .collect();

                        // Build columns directly (no ListArray overhead)
                        for desc in &descriptors {
                            for (col_idx, &val) in desc.iter().enumerate() {
                                builders[col_idx].append_value(val);
                            }
                        }

                        // Convert builders to Arrow arrays
                        let column_arrays: Vec<Arc<dyn Array>> = builders
                            .into_iter()
                            .map(|mut b| Arc::new(b.finish()) as Arc<dyn Array>)
                            .collect();

                        let output_batch = match RecordBatch::try_new(schema.clone(), column_arrays) {
                            Ok(b) => b,
                            Err(e) => return Err(format!("Failed to create output batch: {}", e)),
                        };

                        if writer.write(&output_batch).is_err() {
                            return Err("Failed to write batch".to_string());
                        }
                    }

                    total_batches_written += 1;
                    next_write_idx += 1;

                    if total_batches_written % 100 == 0 {
                        println!("[写入进度] {} 个描述符", total_descriptors);
                    }
                }
            }

            // Finalize writer
            match writer.close() {
                Ok(_) => {
                    println!("[写入完成] {} 个描述符", total_descriptors);
                    Ok((total_descriptors, total_batches_written))
                }
                Err(e) => Err(format!("Failed to close writer: {}", e)),
            }
        });

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 6: Wait for all threads and collect results
        ////////////////////////////////////////////////////////////////////////////////
        let reader_result = reader_handle.join()
            .map_err(|e| format!("Reader thread panicked: {:?}", e))?
            .map_err(|e| format!("Reader thread failed: {}", e))?;

        let (total_csfs, _) = reader_result;

        // Wait for all worker threads
        for (i, handle) in worker_handles.into_iter().enumerate() {
            handle.join()
                .map_err(|e| format!("Worker thread {} panicked: {:?}", i, e))?
                .map_err(|e| format!("Worker thread {} failed: {}", i, e))?;
        }

        let writer_result = writer_handle.join()
            .map_err(|e| format!("Writer thread panicked: {:?}", e))?
            .map_err(|e| format!("Writer thread failed: {}", e))?;

        let (total_descriptors, _) = writer_result;

        println!("====================================");
        println!("处理完成！");
        println!("输入 CSF: {} | 生成描述符: {}", total_csfs, total_descriptors);
        println!("轨道数: {} | 描述符大小: {}", orbital_count, descriptor_size);
        println!("====================================");

        Ok(BatchDescriptorStats {
            input_file: input_parquet.to_string_lossy().to_string(),
            output_file: output_file.to_string_lossy().to_string(),
            csf_count: total_csfs,
            descriptor_count: total_descriptors,
            orbital_count,
            descriptor_size,
        })
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
pub fn j_to_double_j(j_str: &str) -> Result<i32, String> {
    let trimmed = j_str.trim();

    // Handle fractional J values (e.g., "3/2" -> 3)
    if let Some(slash_pos) = trimmed.find('/') {
        let numerator: i32 = trimmed[..slash_pos]
            .parse()
            .map_err(|_| format!("Invalid J value numerator: {}", trimmed))?;
        return Ok(numerator);
    }

    // Handle integer J values (e.g., "2" -> 4, "4-" -> 8)
    // Remove trailing parity indicator if present
    let cleaned = trimmed.trim_end_matches('-').trim_end_matches('+');
    cleaned
        .parse::<i32>()
        .map(|j| j * 2)
        .map_err(|_| format!("Invalid J value: {}", trimmed))
}

/// Chunk a string into fixed-size pieces
///
/// # Arguments
/// * `s` - The string to chunk
/// * `chunk_size` - Size of each chunk
///
/// # Returns
/// Vector of string chunks
fn chunk_string(s: &str, chunk_size: usize) -> Vec<&str> {
    s.as_bytes()
        .chunks(chunk_size)
        .map(|chunk| std::str::from_utf8(chunk).unwrap_or(""))
        .collect()
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
    pub fn parse_csf(&self, line1: &str, line2: &str, line3: &str) -> Result<Vec<i32>, String> {
        // Initialize descriptor array with zeros
        let mut descriptor = vec![0i32; 3 * self.orbital_count];
        let mut occupied_orbitals = Vec::new();

        // Step 1: Preprocess the three lines
        let subshells_line = line1.trim_end();
        let line_length = subshells_line.len();

        // Pad middle line to match line1 length
        let middle_line = format!("{:<width$}", line2.trim_end(), width = line_length);

        // Extract coupling line (remove first 4 and last 5 characters)
        let coupling_line_raw = line3.trim_end();
        let coupling_trimmed = coupling_line_raw
            .get(4..coupling_line_raw.len().saturating_sub(5))
            .unwrap_or(coupling_line_raw);
        let coupling_line = format!("{:<width$}", coupling_trimmed, width = line_length);

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
        let subshell_list = chunk_string(subshells_line, 9);
        let middle_list = chunk_string(&middle_line, 9);
        let coupling_list = chunk_string(&coupling_line, 9);

        // Step 4: Process each subshell block
        for (i, ((subshell_charges, middle_item), coupling_item)) in subshell_list
            .iter()
            .zip(middle_list.iter())
            .zip(coupling_list.iter())
            .enumerate()
        {
            // Extract subshell name (first 5 characters, trimmed)
            let subshell = subshell_charges
                .get(0..5)
                .map(|s: &str| s.trim())
                .unwrap_or("")
                .to_string();

            // Extract electron number (characters 6-8, i.e., indices 6 and 7)
            let subshell_electron_num: i32 = if subshell_charges.len() >= 8 {
                subshell_charges[6..8].trim().parse().unwrap_or(0)
            } else {
                0
            };

            // Check if this is the last subshell
            let is_last = i == subshell_list.len() - 1;

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
            if let Some(&orbs_idx) = self.orbital_index_map.get(&subshell) {
                let descriptor_idx = orbs_idx * 3;

                // Ensure we don't go out of bounds
                if descriptor_idx + 3 <= descriptor.len() {
                    // If electron count is 0, set all three values to 0
                    if subshell_electron_num == 0 {
                        descriptor[descriptor_idx] = 0;
                        descriptor[descriptor_idx + 1] = 0;
                        descriptor[descriptor_idx + 2] = 0;
                    } else {
                        descriptor[descriptor_idx] = subshell_electron_num;
                        descriptor[descriptor_idx + 1] = temp_middle_item;
                        descriptor[descriptor_idx + 2] = temp_coupling_item;
                    }

                    occupied_orbitals.push(orbs_idx);
                }
            } else {
                // Warning: subshell not in list (skip as Python does)
                eprintln!("Warning: {} not found in orbs list", subshell);
            }
        }

        // Unoccupied orbitals remain with all zeros (default initialization)

        Ok(descriptor)
    }
}

//////////////////////////////////////////////////////////////////////////////
/// Python Bindings (PyO3)
//////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-exposed function to generate descriptors from parquet file (parallel version)
///
/// Output format: Parquet file with multiple `col_0, col_1, ..., col_N` Int32 columns and ZSTD compression (level 3)
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
    max_cumulative_doubled_j=None
))]
fn py_generate_descriptors_from_parquet(
    py: Python,
    input_parquet: String,
    output_file: String,
    peel_subshells: Vec<String>,
    num_workers: Option<usize>,
    normalize: bool,
    max_cumulative_doubled_j: Option<i32>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    use pyo3::types::PyDict;
    use std::path::Path;

    let input_path = Path::new(&input_parquet).to_path_buf();
    let output_path = Path::new(&output_file).to_path_buf();

    // Release the GIL during the long-running operation
    let stats = py
        .detach(|| {
            parquet_batch::generate_descriptors_from_parquet_parallel(
                &input_path,
                &output_path,
                peel_subshells,
                num_workers,
                normalize,
                max_cumulative_doubled_j,
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
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
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

//////////////////////////////////////////////////////////////////////////////
/// Rust Tests
//////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_j_to_double_j_fractional() {
        assert_eq!(j_to_double_j("3/2"), Ok(3));
        assert_eq!(j_to_double_j("5/2"), Ok(5));
        assert_eq!(j_to_double_j("1/2"), Ok(1));
    }

    #[test]
    fn test_j_to_double_j_integer() {
        assert_eq!(j_to_double_j("2"), Ok(4));
        assert_eq!(j_to_double_j("3"), Ok(6));
        assert_eq!(j_to_double_j("4"), Ok(8));
    }

    #[test]
    fn test_j_to_double_j_with_parity() {
        assert_eq!(j_to_double_j("4-"), Ok(8));
        assert_eq!(j_to_double_j("3+"), Ok(6));
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
}
