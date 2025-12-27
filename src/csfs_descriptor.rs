//! CSF Descriptor Generation Module
//!
//! This module converts Configuration State Function (CSF) data into descriptor arrays
//! for machine learning applications. Each CSF is parsed into a fixed-length array
//! containing electron counts and angular momentum coupling values.

use std::collections::HashMap;
use std::path::Path;
use std::fs::read_to_string;
use std::thread;
use std::sync::{Arc, Mutex};
use crossbeam_channel::{bounded, Sender, Receiver};

/// Parquet reading support
pub mod parquet_batch {
    use super::*;
    use std::path::PathBuf;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use arrow::array::{StringArray, UInt64Array};

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
        let toml_value: Value = toml::from_str(toml_content)
            .map_err(|e| format!("Failed to parse TOML: {}", e))?;

        // Get header_lines from [header_info] section
        let header_lines = toml_value.get("header_info")
            .and_then(|v| v.get("header_lines"))
            .and_then(|v| v.as_array())
            .ok_or("header_info.header_lines not found in TOML")?;

        // Peel subshells are on line 4 (index 3): "  2s   2p-  2p   3s..."
        if let Some(line_value) = header_lines.get(3) {
            let line = line_value.as_str()
                .ok_or("header_lines[3] is not a string")?;

            // Parse space-separated subshell names
            let parts: Vec<&str> = line.split_whitespace().collect();

            // Filter valid subshell names (must contain at least one letter)
            let subshells: Vec<String> = parts
                .into_iter()
                .filter(|s| {
                    s.chars().any(|c| c.is_alphabetic()) &&
                    s.chars().all(|c| c.is_alphanumeric() || c == '+' || c == '-' || c == '_')
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
        let patterns = std::vec! [
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

    /// Generate descriptors from a parquet file and write to a new parquet file
    ///
    /// # Arguments
    /// * `input_parquet` - Path to input parquet file (must have line1, line2, line3 columns)
    /// * `output_parquet` - Path to output parquet file for descriptors
    /// * `peel_subshells` - Optional list of subshell names (auto-detected if None)
    /// * `header_path` - Optional path to header TOML file
    ///
    /// # Returns
    /// * `Ok(BatchDescriptorStats)` - Statistics about the batch operation
    /// * `Err(String)` - Error message if operation fails
    pub fn generate_descriptors_from_parquet(
        input_parquet: &Path,
        output_parquet: &Path,
        peel_subshells: Option<Vec<String>>,
        header_path: Option<PathBuf>,
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

        // Step 2: Create descriptor generator
        let generator = super::CSFDescriptorGenerator::new(peel_subshells.clone());

        // Step 3: Open input parquet file
        let file = std::fs::File::open(input_parquet)
            .map_err(|e| format!("Failed to open input parquet: {}", e))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| format!("Failed to create parquet reader: {}", e))?;

        let schema = builder.schema();
        eprintln!("Input parquet schema: {}", schema);

        let mut reader = builder.build()
            .map_err(|e| format!("Failed to build parquet reader: {}", e))?;

        // Step 4: Create output parquet writer
        use parquet::arrow::ArrowWriter;
        use arrow::datatypes::{Schema, Field, DataType};
        use std::sync::Arc;

        // Output schema: descriptor columns (one column per descriptor element)
        let mut fields = Vec::with_capacity(descriptor_size);
        for i in 0..descriptor_size {
            fields.push(Field::new(format!("col_{}", i), DataType::Float32, false));
        }
        let output_schema = Arc::new(Schema::new(fields));

        let output_file = std::fs::File::create(output_parquet)
            .map_err(|e| format!("Failed to create output parquet: {}", e))?;

        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(parquet::basic::ZstdLevel::default()))
            .build();

        let mut writer = ArrowWriter::try_new(output_file, output_schema.clone(), Some(props))
            .map_err(|e| format!("Failed to create arrow writer: {}", e))?;

        // Step 5: Process each batch
        let mut total_csfs = 0;
        let mut descriptor_count = 0;

        eprintln!("Starting descriptor generation from parquet...");

        loop {
            match reader.next() {
                Some(Ok(batch)) => {
                    let batch_size = batch.num_rows();
                    eprintln!("Processing batch of {} CSFs (total: {})", batch_size, total_csfs);

                    // Get columns by index (parquet schema: idx, line1, line2, line3)
                    let idx_col = batch.column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or("idx column is not uint64 type")?;

                    let line1_col = batch.column(1)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or("line1 column is not string type")?;

                    let line2_col = batch.column(2)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or("line2 column is not string type")?;

                    let line3_col = batch.column(3)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or("line3 column is not string type")?;

                    // Process each row
                    use arrow::array::{Float32Array, Array};
                    use std::sync::Arc;

                    let mut descriptors_vec = Vec::with_capacity(batch_size);

                    for i in 0..batch_size {
                        if i > 0 && i % 100 == 0 {
                            eprintln!("  Progress: {}/{} in batch", i, batch_size);
                        }
                        let line1 = line1_col.value(i);
                        let line2 = line2_col.value(i);
                        let line3 = line3_col.value(i);
                        let idx = idx_col.value(i);

                        match generator.parse_csf(line1, line2, line3) {
                            Ok(descriptor) => {
                                descriptors_vec.push(descriptor);
                                descriptor_count += 1;
                            }
                            Err(e) => {
                                eprintln!("Warning: Failed to parse CSF at index {}: {}", idx, e);
                            }
                        }
                    }

                    total_csfs += batch_size;

                    // Convert descriptors to separate Arrow arrays (one per column)
                    // Each descriptor becomes a row, with each element in its own column
                    let column_arrays: Vec<Arc<dyn Array>> = (0..descriptor_size)
                        .map(|col_idx| {
                            let values: Vec<f32> = descriptors_vec
                                .iter()
                                .map(|desc| desc[col_idx])
                                .collect();
                            Arc::new(Float32Array::from(values)) as Arc<dyn Array>
                        })
                        .collect();

                    // Create output record batch
                    use arrow::record_batch::RecordBatch;
                    let output_batch = RecordBatch::try_new(
                        output_schema.clone(),
                        column_arrays,
                    ).map_err(|e| format!("Failed to create output batch: {}", e))?;

                    writer.write(&output_batch)
                        .map_err(|e| format!("Failed to write batch: {}", e))?;

                    eprintln!("  Processed {} descriptors", descriptor_count);
                }
                Some(Err(e)) => {
                    return Err(format!("Error reading parquet batch: {}", e));
                }
                None => break,
            }
        }

        // Step 6: Finalize writer
        writer.close()
            .map_err(|e| format!("Failed to close writer: {}", e))?;

        eprintln!("Descriptor generation complete!");
        eprintln!("  Total CSFs processed: {}", total_csfs);
        eprintln!("  Descriptors generated: {}", descriptor_count);
        eprintln!("  Orbital count: {}", orbital_count);
        eprintln!("  Descriptor size: {}", descriptor_size);

        Ok(BatchDescriptorStats {
            input_file: input_parquet.to_string_lossy().to_string(),
            output_file: output_parquet.to_string_lossy().to_string(),
            csf_count: total_csfs,
            descriptor_count,
            orbital_count,
            descriptor_size,
        })
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Parallel Descriptor Generation (Streaming + Multi-threaded per batch)
    ////////////////////////////////////////////////////////////////////////////////

    /// Work item for descriptor generation
    #[derive(Debug)]
    struct DescriptorWorkItem {
        batch_idx: usize,
        start_row_in_batch: usize,
        rows: Vec<(u64, String, String, String)>, // (idx, line1, line2, line3)
    }

    /// Processed descriptor result
    #[derive(Debug)]
    struct DescriptorResult {
        batch_idx: usize,
        start_row_in_batch: usize,
        descriptors: Vec<Vec<f32>>,
    }

    /// Generate descriptors from parquet with parallel processing
    ///
    /// This function combines:
    /// 1. Streaming batch reading from parquet (memory efficient)
    /// 2. Multi-threaded parallel processing within each batch (CPU efficient)
    ///
    /// # Arguments
    /// * `input_parquet` - Path to input parquet file
    /// * `output_parquet` - Path to output parquet file
    /// * `peel_subshells` - List of subshell names
    /// * `num_workers` - Number of worker threads (default: CPU core count)
    /// * `rows_per_task` - Number of rows per parallel task (default: 1000)
    pub fn generate_descriptors_from_parquet_parallel(
        input_parquet: &Path,
        output_parquet: &Path,
        peel_subshells: Vec<String>,
        num_workers: Option<usize>,
        rows_per_task: Option<usize>,
    ) -> Result<BatchDescriptorStats, String> {
        let num_workers = num_workers.unwrap_or_else(|| num_cpus::get());
        let rows_per_task = rows_per_task.unwrap_or(1000);
        let work_queue_size = num_workers * 2;
        // Use unbounded channel for results to prevent deadlock
        let (result_sender, result_receiver) = crossbeam_channel::unbounded::<DescriptorResult>();

        let orbital_count = peel_subshells.len();
        let descriptor_size = 3 * orbital_count;

        eprintln!("启动并行描述符生成:");
        eprintln!("  工作线程数: {}", num_workers);
        eprintln!("  每任务行数: {}", rows_per_task);
        eprintln!("  轨道数: {}", orbital_count);
        eprintln!("  描述符大小: {}", descriptor_size);

        // Create descriptor generator (shared by all workers)
        let generator = Arc::new(super::CSFDescriptorGenerator::new(peel_subshells.clone()));

        // Create channel for work distribution (bounded)
        let (work_sender, work_receiver) = bounded::<DescriptorWorkItem>(work_queue_size);

        // Statistics
        let total_csfs = Arc::new(Mutex::new(0usize));
        let descriptor_count = Arc::new(Mutex::new(0usize));

        ////////////////////////////////////////////////////////////////////////////////
        // Step 1: Start worker threads
        ////////////////////////////////////////////////////////////////////////////////
        let mut worker_handles = Vec::new();
        for worker_id in 0..num_workers {
            let work_receiver = work_receiver.clone();
            let result_sender = result_sender.clone();
            let generator = generator.clone();

            let handle = thread::spawn(move || {
                descriptor_worker(worker_id, work_receiver, result_sender, generator);
            });

            worker_handles.push(handle);
        }

        ////////////////////////////////////////////////////////////////////////////////
        // Step 2: Start separate reader and writer threads (run concurrently)
        ////////////////////////////////////////////////////////////////////////////////

        // Reader thread: reads parquet and sends work items
        let reader_thread = thread::spawn({
            let input_parquet = input_parquet.to_path_buf();
            let work_sender = work_sender.clone();
            let total_csfs = total_csfs.clone();

            move || {
                reader_thread(input_parquet, work_sender, total_csfs, rows_per_task)
            }
        });

        // Writer thread: collects results and writes to parquet
        let writer_thread = thread::spawn({
            let output_parquet = output_parquet.to_path_buf();
            let result_receiver = result_receiver;
            let descriptor_count = descriptor_count.clone();
            let descriptor_size = 3 * orbital_count;

            move || {
                writer_thread(output_parquet, result_receiver, descriptor_count, descriptor_size, rows_per_task)
            }
        });

        // Drop sender copies for worker threads
        drop(work_sender);
        drop(result_sender);

        ////////////////////////////////////////////////////////////////////////////////
        // Step 3: Wait for completion
        ////////////////////////////////////////////////////////////////////////////////
        let csf_count = reader_thread.join().unwrap()?;
        let result = writer_thread.join().unwrap()?;

        for handle in worker_handles {
            handle.join().unwrap();
        }

        eprintln!("\n并行描述符生成完成！");
        eprintln!("  总 CSF 数: {}", csf_count);
        eprintln!("  生成描述符数: {}", result.descriptor_count);

        Ok(BatchDescriptorStats {
            input_file: input_parquet.to_string_lossy().to_string(),
            output_file: output_parquet.to_string_lossy().to_string(),
            csf_count,
            descriptor_count: result.descriptor_count,
            orbital_count,
            descriptor_size,
        })
    }

    /// Worker thread for descriptor generation
    fn descriptor_worker(
        worker_id: usize,
        work_receiver: Receiver<DescriptorWorkItem>,
        result_sender: Sender<DescriptorResult>,
        generator: Arc<super::CSFDescriptorGenerator>,
    ) {
        eprintln!("Worker {} started", worker_id);
        let mut work_count = 0usize;
        for work_item in work_receiver {
            work_count += 1;
            let mut descriptors = Vec::with_capacity(work_item.rows.len());

            for (idx, line1, line2, line3) in work_item.rows {
                match generator.parse_csf(&line1, &line2, &line3) {
                    Ok(descriptor) => {
                        descriptors.push(descriptor);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to parse CSF at index {}: {}", idx, e);
                        // Push zero descriptor on error
                        descriptors.push(vec![0.0f32; descriptor_size_from_generator(&generator)]);
                    }
                }
            }

            let result = DescriptorResult {
                batch_idx: work_item.batch_idx,
                start_row_in_batch: work_item.start_row_in_batch,
                descriptors,
            };

            eprintln!("Worker {} sending result for batch_idx={}, start_row={}, size={}", worker_id, result.batch_idx, result.start_row_in_batch, result.descriptors.len());
            result_sender.send(result).unwrap();
        }
        eprintln!("Worker {} finished, processed {} work items", worker_id, work_count);
    }

    /// Helper to get descriptor size from generator
    fn descriptor_size_from_generator(generator: &super::CSFDescriptorGenerator) -> usize {
        3 * generator.orbital_count()
    }

    /// Writer result with descriptor count
    #[derive(Debug)]
    struct WriterResult {
        pub descriptor_count: usize,
    }

    /// Reader thread: streams batches from input parquet and sends work items
    fn reader_thread(
        input_parquet: PathBuf,
        work_sender: Sender<DescriptorWorkItem>,
        total_csfs: Arc<Mutex<usize>>,
        rows_per_task: usize,
    ) -> Result<usize, String> {
        let file = std::fs::File::open(&input_parquet)
            .map_err(|e| format!("Failed to open input parquet: {}", e))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| format!("Failed to create parquet reader: {}", e))?;

        let schema = builder.schema();
        eprintln!("Input parquet schema: {}", schema);

        let mut reader = builder.build()
            .map_err(|e| format!("Failed to build parquet reader: {}", e))?;

        let mut global_batch_idx = 0usize;
        let mut total_processed = 0usize;

        eprintln!("开始流式处理 parquet 文件...");

        loop {
            match reader.next() {
                Some(Ok(batch)) => {
                    let batch_size = batch.num_rows();
                    eprintln!("读取批次 {} ({} 行)", global_batch_idx, batch_size);

                    // Get columns by index (parquet schema: idx, line1, line2, line3)
                    let idx_col = batch.column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or("idx column is not uint64 type")?;

                    let line1_col = batch.column(1)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or("line1 column is not string type")?;

                    let line2_col = batch.column(2)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or("line2 column is not string type")?;

                    let line3_col = batch.column(3)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or("line3 column is not string type")?;

                    // Extract rows and send work items
                    let mut start_row = 0;
                    while start_row < batch_size {
                        let end_row = (start_row + rows_per_task).min(batch_size);

                        let mut rows = Vec::with_capacity(end_row - start_row);
                        for i in start_row..end_row {
                            rows.push((
                                idx_col.value(i),
                                line1_col.value(i).to_string(),
                                line2_col.value(i).to_string(),
                                line3_col.value(i).to_string(),
                            ));
                        }

                        let work_item = DescriptorWorkItem {
                            batch_idx: global_batch_idx,
                            start_row_in_batch: start_row,
                            rows,
                        };

                        work_sender.send(work_item)
                            .map_err(|e| format!("Failed to send work item: {}", e))?;

                        start_row = end_row;
                    }

                    // Update totals
                    {
                        let mut count = total_csfs.lock().unwrap();
                        *count += batch_size;
                    }
                    total_processed += batch_size;

                    global_batch_idx += 1;
                }
                Some(Err(e)) => {
                    return Err(format!("Error reading parquet batch: {}", e));
                }
                None => break,
            }
        }

        eprintln!("读取完成，共 {} 个批次，{} 行", global_batch_idx, total_processed);
        Ok(total_processed)
    }

    /// Writer thread: collects results and writes to output parquet in order
    fn writer_thread(
        output_parquet: PathBuf,
        result_receiver: Receiver<DescriptorResult>,
        descriptor_count: Arc<Mutex<usize>>,
        descriptor_size: usize,
        rows_per_task: usize,
    ) -> Result<WriterResult, String> {
        use parquet::arrow::ArrowWriter;
        use arrow::datatypes::{Schema, Field, DataType};
        use arrow::array::{Float32Array, Array};
        use std::collections::BTreeMap;
        use std::sync::Arc;

        ////////////////////////////////////////////////////////////////////////////////
        // Create output parquet writer
        ////////////////////////////////////////////////////////////////////////////////
        let mut fields = Vec::with_capacity(descriptor_size);
        for i in 0..descriptor_size {
            fields.push(Field::new(format!("col_{}", i), DataType::Float32, false));
        }
        let output_schema = Arc::new(Schema::new(fields));

        let output_file = std::fs::File::create(&output_parquet)
            .map_err(|e| format!("Failed to create output parquet: {}", e))?;

        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(parquet::basic::ZstdLevel::default()))
            .build();

        let mut writer = ArrowWriter::try_new(output_file, output_schema.clone(), Some(props))
            .map_err(|e| format!("Failed to create arrow writer: {}", e))?;

        ////////////////////////////////////////////////////////////////////////////////
        // Collect and write results in order
        ////////////////////////////////////////////////////////////////////////////////
        eprintln!("开始收集并写入结果...");

        let mut pending_results: BTreeMap<(usize, usize), Vec<Vec<f32>>> = BTreeMap::new();
        let mut next_write_idx = (0usize, 0usize); // (batch_idx, start_row_in_batch)
        let mut total_processed = 0usize;
        let mut total_written = 0usize;

        for result in result_receiver {
            let key = (result.batch_idx, result.start_row_in_batch);
            let result_size = result.descriptors.len();
            pending_results.insert(key, result.descriptors);
            total_processed += result_size;
            eprintln!("Writer received result for batch_idx={}, start_row={}, size={}", result.batch_idx, result.start_row_in_batch, result_size);

            // Write consecutive results
            while let Some(descs) = pending_results.remove(&next_write_idx) {
                // Convert to Arrow arrays (column-oriented)
                let column_arrays: Vec<Arc<dyn Array>> = (0..descriptor_size)
                    .map(|col_idx| {
                        let values: Vec<f32> = descs
                            .iter()
                            .map(|desc| desc[col_idx])
                            .collect();
                        Arc::new(Float32Array::from(values)) as Arc<dyn Array>
                    })
                    .collect();

                // Create output record batch
                use arrow::record_batch::RecordBatch;
                let output_batch = RecordBatch::try_new(
                    output_schema.clone(),
                    column_arrays,
                ).map_err(|e| format!("Failed to create output batch: {}", e))?;

                writer.write(&output_batch)
                    .map_err(|e| format!("Failed to write batch: {}", e))?;

                // Update descriptor count
                {
                    let mut count = descriptor_count.lock().unwrap();
                    *count += descs.len();
                }
                total_written += descs.len();

                // Move to next expected result
                // Key insight: batch splitting always creates (batch_idx, 0) and (batch_idx, rows_per_task)
                // We can determine what to expect next by checking current start_row
                let (curr_batch_idx, curr_start_row) = next_write_idx;

                if curr_start_row == 0 {
                    // We just wrote (batch_idx, 0)
                    // If we also have (batch_idx, rows_per_task) pending, we need it next
                    // Otherwise, move to next batch
                    let split_key = (curr_batch_idx, rows_per_task);
                    if pending_results.contains_key(&split_key) {
                        // Split batch exists, expect it next
                        next_write_idx = (curr_batch_idx, rows_per_task);
                    } else {
                        // No split batch, move to next batch
                        next_write_idx = (curr_batch_idx + 1, 0);
                    }
                } else {
                    // We just wrote the second part of a split batch (start_row > 0)
                    // Move to next batch
                    next_write_idx = (curr_batch_idx + 1, 0);
                }
            }

            if total_processed > 0 && total_written % 10000 == 0 {
                eprintln!("进度: {} 描述符已生成并写入", total_written);
            }
        }

        eprintln!("Writer loop ended. total_processed={}, total_written={}", total_processed, total_written);

        ////////////////////////////////////////////////////////////////////////////////
        // Finalize writer
        ////////////////////////////////////////////////////////////////////////////////
        writer.close()
            .map_err(|e| format!("Failed to close writer: {}", e))?;

        let desc_count = *descriptor_count.lock().unwrap();

        eprintln!("描述符生成完成！");
        eprintln!("  生成描述符数: {}", desc_count);

        Ok(WriterResult {
            descriptor_count: desc_count,
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

    if trimmed.is_empty() {
        return Ok(0); // Empty string represents 0
    }

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
    /// A vector of f32 descriptor values
    ///
    /// # CSF Format Example
    /// ```text
    /// line1: "  5s ( 2)  4d-( 4)  4d ( 6)"
    /// line2: "                   3/2      "
    /// line3: "                        4-  "
    /// ```
    pub fn parse_csf(
        &self,
        line1: &str,
        line2: &str,
        line3: &str,
    ) -> Result<Vec<f32>, String> {
        // Initialize descriptor array with zeros
        let mut descriptor = vec![0.0f32; 3 * self.orbital_count];
        let mut occupied_orbitals = Vec::new();

        // Step 1: Preprocess the three lines
        let subshells_line = line1.trim_end();
        let line_length = subshells_line.len();

        // Pad middle line to match line1 length
        let middle_line = format!("{:<width$}", line2.trim_end(), width = line_length);

        // Extract coupling line (remove first 4 and last 5 characters)
        let coupling_line_raw = line3.trim_end();
        let coupling_start = if coupling_line_raw.len() > 5 { 4 } else { 0 };
        let coupling_end = if coupling_line_raw.len() > 5 { coupling_line_raw.len() - 5 } else { coupling_line_raw.len() };
        let coupling_line = format!("{:<width$}",
            &coupling_line_raw[coupling_start..coupling_end],
            width = line_length
        );

        // Step 2: Extract final J value from the end of line3
        let final_j_start = if coupling_line_raw.len() > 5 { coupling_line_raw.len() - 5 } else { 0 };
        let final_j_end = if coupling_line_raw.len() > 1 { coupling_line_raw.len() - 1 } else { coupling_line_raw.len() };
        let final_j_str = &coupling_line_raw[final_j_start..final_j_end];
        let final_double_j = j_to_double_j(final_j_str)?;

        // Step 3: Chunk lines into 9-character blocks
        let subshell_list = chunk_string(subshells_line, 9);
        let middle_list = chunk_string(&middle_line, 9);
        let coupling_list = chunk_string(&coupling_line, 9);

        // Step 4: Process each subshell block
        for (i, ((subshell_charges, middle_item), coupling_item)) in
            subshell_list.iter().zip(middle_list.iter()).zip(coupling_list.iter()).enumerate()
        {
            // Extract subshell name (first 5 characters, trimmed)
            let subshell = subshell_charges.get(0..5)
                .map(|s: &str| s.trim())
                .unwrap_or("")
                .to_string();

            // Extract electron number (characters 6-8, i.e., indices 6 and 7)
            let subshell_electron_num: f32 = if subshell_charges.len() >= 8 {
                subshell_charges[6..8].trim().parse()
                    .unwrap_or(0.0)
            } else {
                0.0
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
                    descriptor[descriptor_idx] = subshell_electron_num;
                    descriptor[descriptor_idx + 1] = temp_middle_item as f32;
                    descriptor[descriptor_idx + 2] = temp_coupling_item as f32;

                    occupied_orbitals.push(orbs_idx);
                }
            } else {
                // Warning: subshell not in list (skip as Python does)
                eprintln!("Warning: {} not found in orbs list", subshell);
            }
        }

        // Step 6: Fill unoccupied orbitals with final J value (position 2)
        let all_orbitals: std::collections::HashSet<_> =
            (0..self.orbital_count).collect();
        let occupied: std::collections::HashSet<_> =
            occupied_orbitals.iter().cloned().collect();
        let remaining: Vec<_> =
            all_orbitals.difference(&occupied).cloned().collect();

        for idx in remaining {
            let descriptor_idx = idx * 3 + 2;
            if descriptor_idx < descriptor.len() {
                descriptor[descriptor_idx] = final_double_j as f32;
            }
        }

        Ok(descriptor)
    }
}

//////////////////////////////////////////////////////////////////////////////
/// Python Bindings (PyO3)
//////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-exposed CSF Descriptor Generator class
#[cfg(feature = "python")]
#[pyclass(name = "CSFDescriptorGenerator")]
pub struct PyCSFDescriptorGenerator {
    inner: CSFDescriptorGenerator,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCSFDescriptorGenerator {
    /// Create a new descriptor generator
    #[new]
    fn new(peel_subshells: Vec<String>) -> Self {
        Self {
            inner: CSFDescriptorGenerator::new(peel_subshells),
        }
    }

    /// Get the number of orbitals
    fn orbital_count(&self) -> usize {
        self.inner.orbital_count()
    }

    /// Get the peel subshells list
    fn peel_subshells(&self) -> Vec<String> {
        self.inner.peel_subshells().to_vec()
    }

    /// Parse a single CSF into a descriptor array
    ///
    /// Args:
    ///     line1: First line of CSF (subshell configurations)
    ///     line2: Second line of CSF (intermediate J coupling)
    ///     line3: Third line of CSF (final coupling and total J)
    ///
    /// Returns:
    ///     List of float32 descriptor values
    fn parse_csf(&self, line1: &str, line2: &str, line3: &str) -> PyResult<Vec<f32>> {
        self.inner
            .parse_csf(line1, line2, line3)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Parse CSF from a list of 3 strings (Python list format)
    fn parse_csf_from_list(&self, csf_lines: Vec<String>) -> PyResult<Vec<f32>> {
        if csf_lines.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "csf_lines must contain exactly 3 elements",
            ));
        }
        self.inner
            .parse_csf(&csf_lines[0], &csf_lines[1], &csf_lines[2])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Batch parse multiple CSFs
    ///
    /// Args:
    ///     csf_list: List of CSF data, each being a list of 3 strings
    ///
    /// Returns:
    ///     List of descriptor arrays
    fn batch_parse_csfs(&self, csf_list: Vec<Vec<String>>) -> PyResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(csf_list.len());

        for (idx, csf_lines) in csf_list.into_iter().enumerate() {
            match self.inner.parse_csf(
                &csf_lines.get(0).map(|s| s.as_str()).unwrap_or(""),
                &csf_lines.get(1).map(|s| s.as_str()).unwrap_or(""),
                &csf_lines.get(2).map(|s| s.as_str()).unwrap_or(""),
            ) {
                Ok(descriptor) => results.push(descriptor),
                Err(e) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Error parsing CSF at index {}: {}",
                        idx, e
                    )))
                }
            }
        }

        Ok(results)
    }

    /// Get the configuration as a dictionary
    fn get_config(&self, py: Python) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("orbital_count", self.inner.orbital_count())?;
        dict.set_item("peel_subshells", self.inner.peel_subshells())?;
        Ok(dict.into())
    }
}

/// Python-exposed function to generate descriptors from parquet file
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    input_parquet,
    output_parquet,
    peel_subshells=None,
    header_path=None
))]
fn py_generate_descriptors_from_parquet(
    py: Python,
    input_parquet: String,
    output_parquet: String,
    peel_subshells: Option<Vec<String>>,
    header_path: Option<String>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    use std::path::Path;
    use pyo3::types::PyDict;

    let header_path_buf = header_path.as_ref().map(|s| Path::new(s).to_path_buf());
    let input_path = Path::new(&input_parquet).to_path_buf();
    let output_path = Path::new(&output_parquet).to_path_buf();

    // Release the GIL during the long-running operation
    let stats = py.detach(
        || parquet_batch::generate_descriptors_from_parquet(
            &input_path,
            &output_path,
            peel_subshells,
            header_path_buf,
        )
    ).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

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

/// Python-exposed function to generate descriptors from parquet file (parallel version)
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    input_parquet,
    output_parquet,
    peel_subshells,
    num_workers=None,
    rows_per_task=None
))]
fn py_generate_descriptors_from_parquet_parallel(
    py: Python,
    input_parquet: String,
    output_parquet: String,
    peel_subshells: Vec<String>,
    num_workers: Option<usize>,
    rows_per_task: Option<usize>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    use std::path::Path;
    use pyo3::types::PyDict;

    let input_path = Path::new(&input_parquet).to_path_buf();
    let output_path = Path::new(&output_parquet).to_path_buf();

    // Release the GIL during the long-running operation
    let stats = py.detach(
        || parquet_batch::generate_descriptors_from_parquet_parallel(
            &input_path,
            &output_path,
            peel_subshells,
            num_workers,
            rows_per_task,
        )
    ).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

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

/// Register the Python module functions and classes
#[cfg(feature = "python")]
pub fn register_descriptor_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCSFDescriptorGenerator>()?;
    module.add_function(wrap_pyfunction!(py_generate_descriptors_from_parquet, module)?)?;
    module.add_function(wrap_pyfunction!(py_generate_descriptors_from_parquet_parallel, module)?)?;
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
    fn test_j_to_double_j_empty() {
        assert_eq!(j_to_double_j(""), Ok(0));
        assert_eq!(j_to_double_j("   "), Ok(0));
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
