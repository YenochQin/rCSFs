use arrow::array::{StringBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use toml;

/// Number of header lines at the beginning of a CSF file
const CSF_HEADER_LINE_COUNT: usize = 5;

/// Maximum line length (in bytes) before emitting a strong warning about memory usage.
/// The BufRead::lines() iterator allocates the full line before truncation.
/// For practical purposes this is acceptable since:
/// 1. OS/system limits typically bound single line length
/// 2. The max_line_len parameter limits what we actually store
/// 3. Temporary allocations are freed immediately
/// Lines exceeding this threshold will trigger a warning but still be processed.
const MAX_LINE_WARNING_THRESHOLD: usize = 1024 * 1024; // 1 MB

/// Validate and get the parent directory of a path, preventing directory traversal.
/// Returns the parent directory or the current directory if the path has no parent.
/// This function ensures that the returned path is canonicalized to prevent
/// path traversal attacks via `..` components.
fn safe_parent_dir(path: &Path) -> PathBuf {
    path.parent()
        .and_then(|p| p.canonicalize().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Extracts the first CSF_HEADER_LINE_COUNT header lines from a CSF file.
/// Returns a vector of exactly CSF_HEADER_LINE_COUNT strings (empty strings if file has fewer lines).
fn extract_header_lines(csfs_path: &Path) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let input_file = File::open(csfs_path)?;
    let reader = BufReader::new(input_file);
    let mut lines_iter = reader.lines();

    let mut headers = Vec::with_capacity(CSF_HEADER_LINE_COUNT);

    for _i in 0..CSF_HEADER_LINE_COUNT {
        match lines_iter.next() {
            Some(Ok(line)) => {
                headers.push(line);
            }
            Some(Err(e)) => return Err(e.into()),
            None => {
                println!("警告: 文件少于 {} 行 Header", CSF_HEADER_LINE_COUNT);
                break;
            }
        }
    }

    // Ensure headers has exactly CSF_HEADER_LINE_COUNT lines, fill with empty strings if needed
    while headers.len() < CSF_HEADER_LINE_COUNT {
        headers.push(String::new());
    }

    Ok(headers)
}

#[derive(Serialize, Deserialize, Debug)]
struct HeaderInfo {
    header_lines: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConversionStats {
    pub csf_count: usize,
    pub total_lines: usize,
    pub truncated_count: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct HeaderData {
    header_info: HeaderInfo,
    conversion_stats: ConversionStats,
}

/// Convert CSF text file to Parquet format using parallel processing.
///
/// This function is optimized for large-scale data processing. It uses a streaming
/// approach with rayon-based parallel processing:
/// - Stream: Read file in batches to avoid loading 34GB+ into memory
/// - Parallel: Process each batch with rayon's work-stealing
/// - Order: Maintain CSF order in output
///
/// # Arguments
///
/// * `csfs_path` - Path to input CSF file
/// * `output_path` - Path to output Parquet file
/// * `max_line_len` - Maximum line length (lines longer than this are truncated)
/// * `chunk_size` - Number of lines per read batch (larger = fewer I/O ops but more memory)
///
/// # Returns
///
/// Returns `ConversionStats` containing:
/// * `csf_count` - Number of CSFs processed
/// * `total_lines` - Total number of lines processed
/// * `truncated_count` - Number of lines that were truncated
///
/// # Architecture
///
/// ```
/// File → [Read batch] → [Rayon parallel process] → [Write ordered] → repeat
/// ```
///
/// - **Streaming**: Read file in batches (don't load entire 34GB into memory)
/// - **Parallel**: Each batch processed with rayon's par_iter (all cores used automatically)
/// - **Ordered**: Results written in CSF order (par_iter + collect preserves order)
///
/// # Header File
///
/// Automatically generates `[input_file_stem]_header.toml` in the output directory
/// containing the 5-line header and conversion statistics.
pub fn convert_csfs_to_parquet_parallel(
    csfs_path: &Path,
    output_path: &Path,
    max_line_len: usize,
    chunk_size: usize,
    num_workers: Option<usize>,
) -> Result<ConversionStats, Box<dyn std::error::Error + Send + Sync>> {
    // Configure rayon thread pool if num_workers is specified
    if let Some(n) = num_workers {
        println!("配置 Rayon 线程池，使用 {} 个 worker", n);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
    }

    println!("开始并行转换 CSF 文件");
    println!("输入文件: {:?}", csfs_path);
    println!("输出文件: {:?}", output_path);
    println!("最大行长度: {}", max_line_len);
    println!("批处理大小: {}", chunk_size);

    // --- 1. 读取 Header (5行) ---
    let headers = extract_header_lines(csfs_path)?;

    // --- 2. 创建 Parquet 写入器 ---
    let schema = Arc::new(Schema::new(vec![
        Field::new("idx", DataType::UInt64, false),
        Field::new("line1", DataType::Utf8, false),
        Field::new("line2", DataType::Utf8, false),
        Field::new("line3", DataType::Utf8, false),
    ]));

    let output_file = File::create(&output_path)?;
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::UNCOMPRESSED)
        .build();
    let mut writer = ArrowWriter::try_new(output_file, schema.clone(), Some(props))?;
    println!("Parquet 写入器已创建，使用无压缩");

    // --- 3. 流式读取 + 批量并行处理 ---
    let input_file = File::open(csfs_path)?;
    let reader = BufReader::new(input_file);
    let mut lines_iter = reader.lines();

    // Skip header lines
    for _ in 0..CSF_HEADER_LINE_COUNT {
        if lines_iter.next().is_none() {
            break;
        }
    }

    let mut batch_lines = Vec::with_capacity(chunk_size);
    let mut csf_count = 0;
    let mut total_lines = 0;
    let mut truncated_count = 0;

    println!("开始并行处理 CSF 数据...");

    loop {
        // Read a batch of lines
        let mut lines_read = 0;
        for _ in 0..chunk_size {
            match lines_iter.next() {
                Some(Ok(line)) => {
                    total_lines += 1;
                    lines_read += 1;
                    batch_lines.push(line);
                }
                Some(Err(e)) => return Err(e.into()),
                None => break,
            }
        }

        if batch_lines.is_empty() {
            break;
        }

        // Ensure we have complete CSFs (3 lines each)
        let num_full_csfs = batch_lines.len() / 3;
        if num_full_csfs == 0 {
            if lines_read == 0 {
                break;
            }
            continue;
        }

        let lines_to_process: Vec<String> = batch_lines.drain(..num_full_csfs * 3).collect();

        // Create chunks for parallel processing
        let chunks: Vec<&[String]> = lines_to_process.chunks(3).collect();

        // Process batch in parallel using rayon
        let batch_results: Vec<(u64, String, String, String, bool)> = chunks
            .into_par_iter()
            .enumerate()
            .map(|(i, chunk)| {
                if chunk.len() != 3 {
                    return ((csf_count + i) as u64, String::new(), String::new(), String::new(), false);
                }

                let process_line = |line: &str, max_len: usize| -> (String, bool) {
                    if line.len() > MAX_LINE_WARNING_THRESHOLD {
                        eprintln!("警告: 行过长 ({} bytes)", line.len());
                    }
                    if line.len() > max_len {
                        (line.chars().take(max_len).collect::<String>(), true)
                    } else {
                        (line.to_string(), false)
                    }
                };

                let (line1, t1) = process_line(&chunk[0], max_line_len);
                let (line2, t2) = process_line(&chunk[1], max_line_len);
                let (line3, t3) = process_line(&chunk[2], max_line_len);

                ((csf_count + i) as u64, line1, line2, line3, t1 || t2 || t3)
            })
            .collect();

        // Write results in order (par_iter + collect preserves order)
        let mut idx_builder = UInt64Builder::with_capacity(num_full_csfs);
        let mut line1_builder = StringBuilder::with_capacity(num_full_csfs, num_full_csfs * max_line_len);
        let mut line2_builder = StringBuilder::with_capacity(num_full_csfs, num_full_csfs * max_line_len);
        let mut line3_builder = StringBuilder::with_capacity(num_full_csfs, num_full_csfs * max_line_len);

        for (idx, line1, line2, line3, truncated) in batch_results {
            idx_builder.append_value(idx);
            line1_builder.append_value(&line1);
            line2_builder.append_value(&line2);
            line3_builder.append_value(&line3);
            if truncated {
                truncated_count += 1;
            }
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(idx_builder.finish()),
                Arc::new(line1_builder.finish()),
                Arc::new(line2_builder.finish()),
                Arc::new(line3_builder.finish()),
            ],
        )?;

        writer.write(&batch)?;

        csf_count += num_full_csfs;
    }

    // --- 4. 完成写入 ---
    writer.close()?;

    let final_stats = ConversionStats {
        csf_count,
        total_lines,
        truncated_count,
    };

    println!("\n并行转换完成！");
    println!("总行数: {}", total_lines);
    println!("CSF 数量: {}", csf_count);
    println!("截断行数: {}", truncated_count);
    if truncated_count > 0 {
        println!("警告: {} 行被截断，考虑增加 max_line_len 参数", truncated_count);
    }

    // --- 5. 创建 TOML 头部文件 ---
    let header_data = HeaderData {
        header_info: HeaderInfo {
            header_lines: headers,
        },
        conversion_stats: final_stats.clone(),
    };

    let header_dir = safe_parent_dir(output_path);
    let input_file_stem = csfs_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("csfs");
    let header_filename = format!("{}_header.toml", input_file_stem);
    let header_path = header_dir.join(header_filename);
    let toml_string = toml::to_string_pretty(&header_data)?;
    std::fs::write(&header_path, toml_string)?;

    println!("Header 文件: {:?}", header_path);

    Ok(final_stats)
}

/// Convert CSF text file to Parquet format using sequential processing.
///
/// This is the simpler, single-threaded conversion function suitable for
/// small to medium-sized files or when parallel processing overhead is not justified.
///
/// # Arguments
///
/// * `csfs_path` - Path to input CSF file
/// * `output_path` - Path to output Parquet file
/// * `max_line_len` - Maximum line length (lines longer than this are truncated)
/// * `chunk_size` - Number of lines per batch processing (larger = better efficiency)
///
/// # Returns
///
/// Returns `ConversionStats` containing:
/// * `csf_count` - Number of CSFs processed
/// * `total_lines` - Total number of lines processed
/// * `truncated_count` - Number of lines that were truncated
///
/// # CSF File Format
///
/// Expected format:
/// * Lines 1-5: Header metadata (extracted to separate TOML file)
/// * Lines 6+: CSF data in groups of exactly 3 lines per CSF
///   * Line 1: CSF identifier/configuration
///   * Line 2: Additional parameters
///   * Line 3: More parameters or coefficients
///
/// # Header File
///
/// Automatically generates `[input_file_stem]_header.toml` in the output directory
/// containing the 5-line header and conversion statistics.
pub fn convert_csfs_to_parquet(
    csfs_path: &Path,
    output_path: &Path,
    max_line_len: usize,
    chunk_size: usize,
) -> Result<ConversionStats, Box<dyn std::error::Error + Send + Sync>> {
    println!("开始转换，最大行长度: {}", max_line_len);
    println!("输入文件: {:?}", csfs_path);
    println!("输出文件: {:?}", output_path);

    // 打开输入文件
    let input_file = File::open(csfs_path)?;
    let reader = BufReader::new(input_file);
    let mut lines_iter = reader.lines();

    // --- 1. 处理 Header (5行) ---
    let headers = extract_header_lines(csfs_path)?;

    // Skip header lines in the main iterator since extract_header_lines uses its own file handle
    for _ in 0..CSF_HEADER_LINE_COUNT {
        if lines_iter.next().is_none() {
            break;
        }
    }

    // --- 2. 创建 Arrow Schema ---
    let schema = Arc::new(Schema::new(vec![
        Field::new("idx", DataType::UInt64, false),
        Field::new("line1", DataType::Utf8, false),
        Field::new("line2", DataType::Utf8, false),
        Field::new("line3", DataType::Utf8, false),
    ]));

    // --- 3. 创建 Parquet 写入器 ---
    let output_file = File::create(output_path)?;
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::UNCOMPRESSED)
        .set_write_batch_size(chunk_size as usize)
        .build();

    let mut writer = ArrowWriter::try_new(output_file, schema.clone(), Some(props))?;
    println!("Parquet 写入器已创建，使用无压缩");

    // --- 4. 批量处理 ---
    let mut batch_lines = Vec::with_capacity(chunk_size);
    let mut csf_count = 0;
    let mut total_lines = 0;
    let mut truncated_count = 0;

    println!("开始处理 CSF 数据...");

    loop {
        // 读取 chunk_size 行
        let mut lines_read_this_iteration = 0;
        for _ in 0..chunk_size {
            match lines_iter.next() {
                Some(Ok(line)) => {
                    total_lines += 1;
                    lines_read_this_iteration += 1;

                    // Warn about excessively long lines (potential memory issue)
                    if line.len() > MAX_LINE_WARNING_THRESHOLD {
                        eprintln!(
                            "警告: 第 {} 行过长 ({} bytes)，可能消耗大量内存",
                            total_lines, line.len()
                        );
                    }

                    // 检查是否需要截断
                    if line.len() > max_line_len {
                        truncated_count += 1;
                        if truncated_count <= 5 {
                            // 只打印前5个截断警告
                            println!(
                                "警告: 第 {} 行被截断 ({} > {})",
                                total_lines,
                                line.len(),
                                max_line_len
                            );
                        }
                    }

                    // 截断到 max_line_len
                    let processed_line = if line.len() > max_line_len {
                        line.chars().take(max_line_len).collect::<String>()
                    } else {
                        line
                    };
                    batch_lines.push(processed_line);
                }
                Some(Err(e)) => return Err(e.into()),
                None => break,
            }
        }

        if batch_lines.is_empty() {
            break;
        }

        // 确保是 3 的倍数
        let num_full_csfs = batch_lines.len() / 3;
        if num_full_csfs == 0 {
            // No more lines to read and incomplete CSF remaining - discard and exit
            if lines_read_this_iteration == 0 {
                break;
            }
            // 保留剩余行给下一轮
            continue;
        }

        // 处理完整的 CSF 块
        let lines_to_process = batch_lines.drain(..num_full_csfs * 3).collect::<Vec<_>>();

        // 创建 Arrow 数组
        let mut idx_builder = UInt64Builder::with_capacity(num_full_csfs);
        let mut line1_builder =
            StringBuilder::with_capacity(num_full_csfs, num_full_csfs * max_line_len);
        let mut line2_builder =
            StringBuilder::with_capacity(num_full_csfs, num_full_csfs * max_line_len);
        let mut line3_builder =
            StringBuilder::with_capacity(num_full_csfs, num_full_csfs * max_line_len);

        for (i, chunk) in lines_to_process.chunks(3).enumerate() {
            if chunk.len() == 3 {
                idx_builder.append_value((csf_count + i) as u64);
                line1_builder.append_value(&chunk[0]);
                line2_builder.append_value(&chunk[1]);
                line3_builder.append_value(&chunk[2]);
            }
        }

        // 构建 RecordBatch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(idx_builder.finish()),
                Arc::new(line1_builder.finish()),
                Arc::new(line2_builder.finish()),
                Arc::new(line3_builder.finish()),
            ],
        )?;

        // 写入 Parquet
        writer.write(&batch)?;

        csf_count += num_full_csfs;
        if csf_count % 100000 == 0 {
            println!("已处理 {} 个 CSF", csf_count);
        }
    }

    // 完成写入
    writer.close()?;

    // 创建 TOML 头部数据
    let header_data = HeaderData {
        header_info: HeaderInfo {
            header_lines: headers.clone(),
        },
        conversion_stats: ConversionStats {
            csf_count,
            total_lines,
            truncated_count,
        },
    };

    // 保存头部数据为 [输入文件名前缀]_header.toml 文件
    let header_dir = safe_parent_dir(output_path);
    let input_file_stem = csfs_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("csfs");
    let header_filename = format!("{}_header.toml", input_file_stem);
    let header_path = header_dir.join(header_filename);
    let toml_string = toml::to_string_pretty(&header_data)?;
    std::fs::write(&header_path, toml_string)?;

    // 统计信息
    println!("\n转换完成！");
    println!("================ 统计信息 ================");
    println!("总行数: {}", total_lines);
    println!("CSF 数量: {}", csf_count);
    println!("截断行数: {}", truncated_count);
    if truncated_count > 0 {
        println!(
            "警告: {} 行被截断，考虑增加 max_line_len 参数",
            truncated_count
        );
    }
    println!("输出文件: {:?}", output_path);
    println!("Header 文件: {:?}", header_path);
    println!("==========================================");

    Ok(header_data.conversion_stats)
}

/// Get Parquet file metadata without reading the actual data.
///
/// This function efficiently retrieves file information and metadata
/// without loading the entire file into memory.
///
/// # Arguments
///
/// * `parquet_path` - Path to the Parquet file
///
/// # Returns
///
/// Returns a Python dictionary containing:
/// * `file_path` - Path to the Parquet file
/// * `file_size` - File size in bytes
/// * `num_rows` - Number of rows in the file
/// * `num_columns` - Number of columns
/// * `compression` - Compression method used (e.g., "GZIP")
///
/// # Errors
///
/// Returns an error if the file cannot be opened or is not a valid Parquet file.
pub fn get_parquet_metadata(
    parquet_path: &Path,
) -> Result<pyo3::Py<pyo3::PyAny>, Box<dyn std::error::Error + Send + Sync>> {
    use pyo3::{
        Python,
        types::{PyDict, PyDictMethods},
    };

    // 获取文件大小
    let file_metadata = std::fs::metadata(parquet_path)?;
    let file_size = file_metadata.len();

    // 打开 Parquet 文件
    let file = File::open(parquet_path)?;
    let reader = SerializedFileReader::new(file)?;
    let metadata = reader.metadata();

    // 获取行数和列数
    let num_rows = metadata.file_metadata().num_rows();
    let schema = metadata.file_metadata().schema();
    let num_columns = schema.get_fields().len();

    // 创建 Python 字典并返回
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("file_path", parquet_path.to_string_lossy().as_ref())?;
        dict.set_item("file_size", file_size)?;
        dict.set_item("num_rows", num_rows)?;
        dict.set_item("num_columns", num_columns)?;
        dict.set_item(
            "compression",
            metadata.file_metadata().created_by().unwrap_or("Unknown"),
        )?;

        Ok(dict.into())
    })
}
