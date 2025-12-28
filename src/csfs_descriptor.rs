//! CSF Descriptor Generation Module
//!
//! This module converts Configuration State Function (CSF) data into descriptor arrays
//! for machine learning applications. Each CSF is parsed into a fixed-length array
//! containing electron counts and angular momentum coupling values.

use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::Path;

/// Parquet reading support
pub mod parquet_batch {
    use super::*;
    use arrow::array::{StringArray, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::path::PathBuf;

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

    /// Generate descriptors from a parquet file and write to Arrow IPC file
    ///
    /// # Arguments
    /// * `input_parquet` - Path to input parquet file (must have line1, line2, line3 columns)
    /// * `output_file` - Path to output Arrow IPC/Feather file for descriptors
    /// * `peel_subshells` - Optional list of subshell names (auto-detected if None)
    /// * `header_path` - Optional path to header TOML file
    ///
    /// # Returns
    /// * `Ok(BatchDescriptorStats)` - Statistics about the batch operation
    /// * `Err(String)` - Error message if operation fails
    ///
    /// # Output Format
    /// Arrow IPC (Feather) - columnar format, Polars compatible
    /// Read with: `polars.read_ipc()` or `pyarrow.feather.read_table()`
    pub fn generate_descriptors_from_parquet(
        input_parquet: &Path,
        output_file: &Path,
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

        let _schema = builder.schema();

        let mut reader = builder
            .build()
            .map_err(|e| format!("Failed to build parquet reader: {}", e))?;

        // Step 4: Create output Arrow IPC writer
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::FileWriter;
        use std::sync::Arc;

        // Output schema: descriptor columns (one column per descriptor element)
        let mut fields = Vec::with_capacity(descriptor_size);
        for i in 0..descriptor_size {
            fields.push(Field::new(format!("col_{}", i), DataType::Int32, false));
        }
        let output_schema = Arc::new(Schema::new(fields));

        let output_file_handle = std::fs::File::create(output_file)
            .map_err(|e| format!("Failed to create output file: {}", e))?;

        let mut writer = FileWriter::try_new(output_file_handle, output_schema.as_ref())
            .map_err(|e| format!("Failed to create IPC writer: {}", e))?;

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
                    use arrow::array::{Array, Int32Array};
                    use std::sync::Arc;

                    let mut descriptors_vec = Vec::with_capacity(batch_size);

                    for i in 0..batch_size {
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
                            let values: Vec<i32> =
                                descriptors_vec.iter().map(|desc| desc[col_idx]).collect();
                            Arc::new(Int32Array::from(values)) as Arc<dyn Array>
                        })
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
    // Parallel Descriptor Generation (Rayon-based, streaming for low memory)
    ////////////////////////////////////////////////////////////////////////////////

    /// Generate descriptors from parquet with parallel processing using rayon
    ///
    /// This implementation uses streaming batch processing to minimize memory usage:
    /// 1. Read parquet in batches
    /// 2. Parse CSFs to descriptors in parallel
    /// 3. Convert descriptors to columnar format in parallel
    /// 4. Write batch to Arrow IPC/Feather file
    /// 5. Repeat until all data processed
    ///
    /// Output format: Arrow IPC (Feather), which is:
    /// - Polars compatible (pl.read_ipc())
    /// - Faster to write than Parquet (no compression overhead)
    /// - Still columnar and efficient
    ///
    /// # Arguments
    /// * `input_parquet` - Path to input parquet file
    /// * `output_file` - Path to output Arrow IPC/Feather file
    /// * `peel_subshells` - List of subshell names
    /// * `num_workers` - Number of worker threads (default: CPU core count)
    pub fn generate_descriptors_from_parquet_parallel(
        input_parquet: &Path,
        output_file: &Path,
        peel_subshells: Vec<String>,
        num_workers: Option<usize>,
    ) -> Result<BatchDescriptorStats, String> {
        use rayon::prelude::*;
        use arrow::ipc::writer::FileWriter;
        use arrow::array::{Array, Int32Array, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        // Configure rayon thread pool if specified
        if let Some(n) = num_workers {
            println!("配置 Rayon 线程池，使用 {} 个 worker", n);
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global()
                .map_err(|e| format!("Failed to configure rayon thread pool: {}", e))?;
        }

        let orbital_count = peel_subshells.len();
        let descriptor_size = 3 * orbital_count;
        println!("轨道数量: {}, 描述符大小: {}", orbital_count, descriptor_size);
        let generator = std::sync::Arc::new(super::CSFDescriptorGenerator::new(peel_subshells));

        println!("开始生成描述符（Arrow IPC 流式并行版本）");
        println!("输入文件: {:?}", input_parquet);
        println!("输出文件: {:?}", output_file);

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 1: Setup output schema and writer (Arrow IPC/Feather)
        ////////////////////////////////////////////////////////////////////////////////
        let mut fields = Vec::with_capacity(descriptor_size);
        for i in 0..descriptor_size {
            fields.push(Field::new(format!("col_{}", i), DataType::Int32, false));
        }
        let schema = Arc::new(Schema::new(fields));

        let output_file_handle = std::fs::File::create(output_file)
            .map_err(|e| format!("Failed to create output file: {}", e))?;

        let mut writer = FileWriter::try_new(output_file_handle, schema.as_ref())
            .map_err(|e| format!("Failed to create IPC writer: {}", e))?;

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 2: Stream processing - read, process, write in batches
        ////////////////////////////////////////////////////////////////////////////////
        let file = std::fs::File::open(input_parquet)
            .map_err(|e| format!("Failed to open input parquet: {}", e))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| format!("Failed to create parquet reader: {}", e))?;

        let mut reader = builder
            .build()
            .map_err(|e| format!("Failed to build parquet reader: {}", e))?;

        let mut total_csfs = 0;
        let mut total_descriptors = 0;

        // Process each batch from input parquet
        loop {
            match reader.next() {
                Some(Ok(batch)) => {
                    let batch_size = batch.num_rows();

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

                    // Extract rows (borrowed, no allocation)
                    let batch_rows: Vec<(u64, &str, &str, &str)> = (0..batch_size)
                        .map(|i| (
                            idx_col.value(i),
                            line1_col.value(i),
                            line2_col.value(i),
                            line3_col.value(i),
                        ))
                        .collect();

                    // Parse CSFs to descriptors (parallel)
                    let generator_ref = generator.as_ref();
                    let descriptors: Vec<Vec<i32>> = batch_rows
                        .into_par_iter()
                        .map(|(idx, line1, line2, line3)| {
                            match generator_ref.parse_csf(line1, line2, line3) {
                                Ok(desc) => desc,
                                Err(e) => {
                                    eprintln!("Warning: Failed to parse CSF at index {}: {}", idx, e);
                                    vec![0i32; descriptor_size]
                                }
                            }
                        })
                        .collect();

                    // Convert descriptors to columnar format (parallel)
                    let column_arrays: Vec<Arc<dyn Array>> = (0..descriptor_size)
                        .into_par_iter()
                        .map(|col_idx| {
                            let values: Vec<i32> = descriptors.iter().map(|desc| desc[col_idx]).collect();
                            Arc::new(Int32Array::from(values)) as Arc<dyn Array>
                        })
                        .collect();

                    // Write batch to Arrow IPC file
                    let output_batch = RecordBatch::try_new(schema.clone(), column_arrays)
                        .map_err(|e| format!("Failed to create output batch: {}", e))?;

                    writer.write(&output_batch)
                        .map_err(|e| format!("Failed to write batch: {}", e))?;

                    total_csfs += batch_size;
                    total_descriptors += descriptors.len();

                    // Memory is automatically released after each iteration

                    // Print progress every 10 million CSFs
                    if total_csfs % 10_000_000 == 0 {
                        println!("已处理 {} 个 CSF", total_csfs);
                    }
                }
                Some(Err(e)) => {
                    return Err(format!("Error reading parquet batch: {}", e));
                }
                None => break,
            }
        }

        ////////////////////////////////////////////////////////////////////////////////
        // Phase 3: Finalize writer
        ////////////////////////////////////////////////////////////////////////////////
        println!("完成写入 Arrow IPC 文件...");
        writer.finish()
            .map_err(|e| format!("Failed to finish writer: {}", e))?;

        println!("描述符导出完成！");
        println!("输入 CSF 数量: {}", total_csfs);
        println!("生成描述符数量: {}", total_descriptors);
        println!("轨道数量: {}", orbital_count);
        println!("描述符大小: {}", descriptor_size);

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
                    descriptor[descriptor_idx] = subshell_electron_num;
                    descriptor[descriptor_idx + 1] = temp_middle_item;
                    descriptor[descriptor_idx + 2] = temp_coupling_item;

                    occupied_orbitals.push(orbs_idx);
                }
            } else {
                // Warning: subshell not in list (skip as Python does)
                eprintln!("Warning: {} not found in orbs list", subshell);
            }
        }

        // Step 6: Fill unoccupied orbitals with final J value (position 2)
        let all_orbitals: std::collections::HashSet<_> = (0..self.orbital_count).collect();
        let occupied: std::collections::HashSet<_> = occupied_orbitals.iter().cloned().collect();
        let remaining: Vec<_> = all_orbitals.difference(&occupied).cloned().collect();

        for idx in remaining {
            let descriptor_idx = idx * 3 + 2;
            if descriptor_idx < descriptor.len() {
                descriptor[descriptor_idx] = final_double_j;
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
    ///     List of int32 descriptor values
    fn parse_csf(&self, line1: &str, line2: &str, line3: &str) -> PyResult<Vec<i32>> {
        self.inner
            .parse_csf(line1, line2, line3)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Parse CSF from a list of 3 strings (Python list format)
    fn parse_csf_from_list(&self, csf_lines: Vec<String>) -> PyResult<Vec<i32>> {
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
    fn batch_parse_csfs(&self, csf_list: Vec<Vec<String>>) -> PyResult<Vec<Vec<i32>>> {
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
                    )));
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
///
/// Output format: Arrow IPC (Feather) file
/// Read with: polars.read_ipc() or pyarrow.feather.read_table()
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    input_parquet,
    output_file,
    peel_subshells=None,
    header_path=None
))]
fn py_generate_descriptors_from_parquet(
    py: Python,
    input_parquet: String,
    output_file: String,
    peel_subshells: Option<Vec<String>>,
    header_path: Option<String>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    use pyo3::types::PyDict;
    use std::path::Path;

    let header_path_buf = header_path.as_ref().map(|s| Path::new(s).to_path_buf());
    let input_path = Path::new(&input_parquet).to_path_buf();
    let output_path = Path::new(&output_file).to_path_buf();

    // Release the GIL during the long-running operation
    let stats = py
        .detach(|| {
            parquet_batch::generate_descriptors_from_parquet(
                &input_path,
                &output_path,
                peel_subshells,
                header_path_buf,
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

/// Python-exposed function to generate descriptors from parquet file (parallel version)
///
/// Output format: Arrow IPC (Feather) file
/// Read with: polars.read_ipc() or pyarrow.feather.read_table()
///
/// This version uses streaming batch processing for low memory usage.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    input_parquet,
    output_file,
    peel_subshells,
    num_workers=None
))]
fn py_generate_descriptors_from_parquet_parallel(
    py: Python,
    input_parquet: String,
    output_file: String,
    peel_subshells: Vec<String>,
    num_workers: Option<usize>,
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

/// Register the Python module functions and classes
#[cfg(feature = "python")]
pub fn register_descriptor_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCSFDescriptorGenerator>()?;
    module.add_function(wrap_pyfunction!(
        py_generate_descriptors_from_parquet,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_generate_descriptors_from_parquet_parallel,
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
