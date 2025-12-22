use arrow::array::{StringBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use toml;
use std::thread;
use std::sync::Mutex;
use crossbeam_channel::{bounded, Sender, Receiver};

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

#[derive(Debug, Clone)]
struct ProcessedChunk {
    start_csf_index: usize,
    idx_values: Vec<u64>,
    line1_values: Vec<String>,
    line2_values: Vec<String>,
    line3_values: Vec<String>,
    csf_count: usize,
    line_count: usize,
    truncated_count: usize,
}

#[derive(Debug)]
struct WorkChunk {
    #[allow(dead_code)]
    chunk_id: usize,
    start_csf_index: usize,
    lines: Vec<String>,
}


pub fn convert_csfs_to_parquet_parallel(
    csfs_path: &Path,
    output_path: &Path,
    max_line_len: usize,
    chunk_size: usize,
    num_workers: Option<usize>,
) -> Result<ConversionStats, Box<dyn std::error::Error + Send + Sync>> {
    let num_workers = num_workers.unwrap_or_else(|| num_cpus::get());
    let queue_size = num_workers * 2; // Bounded queue to prevent memory explosion

    println!("启动并行处理: {} 个工作线程, 队列大小: {}", num_workers, queue_size);
    println!("输入文件: {:?}", csfs_path);
    println!("输出文件: {:?}", output_path);

    // 创建通信通道
    let (work_sender, work_receiver) = bounded(queue_size);
    let (result_sender, result_receiver) = bounded(queue_size);

    // 统计信息 (使用 Mutex 保证线程安全)
    let total_stats = Arc::new(Mutex::new(ConversionStats {
        csf_count: 0,
        total_lines: 0,
        truncated_count: 0,
    }));

    // --- 1. 读取 Header (5行) ---
    let input_file = File::open(csfs_path)?;
    let reader = BufReader::new(input_file);
    let mut lines_iter = reader.lines();

    let mut headers = Vec::with_capacity(5);
    println!("读取 Header...");
    for _i in 0..5 {
        match lines_iter.next() {
            Some(Ok(line)) => {
                headers.push(line);
                // println!("  Header {}: {} 字符", i + 1, headers.last().unwrap().len());
            }
            Some(Err(e)) => return Err(e.into()),
            None => {
                println!("警告: 文件少于 5 行 Header");
                break;
            }
        }
    }

    // 确保 headers 有 5 行，不足的用空字符串填充
    while headers.len() < 5 {
        headers.push(String::new());
    }

    // --- 2. 启动工作线程 ---
    let mut worker_handles = Vec::new();
    for worker_id in 0..num_workers {
        let work_receiver = work_receiver.clone();
        let result_sender = result_sender.clone();
        let max_line_len = max_line_len;

        let handle = thread::spawn(move || {
            worker_process(worker_id, work_receiver, result_sender, max_line_len);
        });

        worker_handles.push(handle);
    }

    // --- 3. 启动文件读取线程 ---
    let reader_thread = thread::spawn({
        let work_sender = work_sender.clone();
        let csfs_path = csfs_path.to_owned();
        move || {
            file_reader_thread(csfs_path, chunk_size, work_sender)
        }
    });

    // Drop sender copies for worker threads after starting reader thread
    drop(work_sender);
    drop(result_sender);

    // --- 4. 启动写入线程 ---
    let writer_thread = thread::spawn({
        let result_receiver = result_receiver;
        let output_path = output_path.to_owned();
        let total_stats = total_stats.clone();
        move || {
            writer_thread(result_receiver, output_path, total_stats)
        }
    });

    // --- 5. 等待所有线程完成 ---
    reader_thread.join().unwrap()?;

    for handle in worker_handles {
        handle.join().unwrap();
    }

    let final_stats = writer_thread.join().unwrap()?;

    // --- 6. 创建 TOML 头部文件 ---
    let header_data = HeaderData {
        header_info: HeaderInfo {
            header_lines: headers,
        },
        conversion_stats: final_stats.clone(),
    };

    // 使用输入文件名的前缀 + _header.toml
    let header_dir = output_path.parent().unwrap_or_else(|| Path::new("."));
    let input_file_stem = csfs_path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("csfs");
    let header_filename = format!("{}_header.toml", input_file_stem);
    let header_path = header_dir.join(header_filename);
    let toml_string = toml::to_string_pretty(&header_data)?;
    std::fs::write(&header_path, toml_string)?;

    println!("\n并行转换完成！");
    println!("================ 最终统计 ================");
    println!("总行数: {}", final_stats.total_lines);
    println!("CSF 数量: {}", final_stats.csf_count);
    println!("截断行数: {}", final_stats.truncated_count);
    println!("输出文件: {:?}", output_path);
    println!("Header 文件: {:?}", header_path);
    println!("==========================================");

    Ok(final_stats)
}

// 工作线程函数
fn worker_process(
    _worker_id: usize,
    work_receiver: Receiver<WorkChunk>,
    result_sender: Sender<ProcessedChunk>,
    max_line_len: usize,
) {
    for work_chunk in work_receiver {
        let mut idx_values = Vec::with_capacity(work_chunk.lines.len() / 3);
        let mut line1_values = Vec::with_capacity(work_chunk.lines.len() / 3);
        let mut line2_values = Vec::with_capacity(work_chunk.lines.len() / 3);
        let mut line3_values = Vec::with_capacity(work_chunk.lines.len() / 3);
        let mut csf_count = 0;
        let mut truncated_count = 0;

        // 处理行，确保是 3 的倍数
        let _full_csfs = work_chunk.lines.len() / 3;
        for chunk in work_chunk.lines.chunks(3) {
            if chunk.len() == 3 {
                // 处理每一行
                let process_line = |line: &str| -> (String, bool) {
                    if line.len() > max_line_len {
                        let truncated = line.chars().take(max_line_len).collect::<String>();
                        (truncated, true)
                    } else {
                        (line.to_string(), false)
                    }
                };

                let (line1, trunc1) = process_line(&chunk[0]);
                let (line2, trunc2) = process_line(&chunk[1]);
                let (line3, trunc3) = process_line(&chunk[2]);

                idx_values.push((work_chunk.start_csf_index + csf_count) as u64);
                line1_values.push(line1);
                line2_values.push(line2);
                line3_values.push(line3);

                if trunc1 || trunc2 || trunc3 {
                    truncated_count += 1;
                }

                csf_count += 1;
            }
        }

        let processed_chunk = ProcessedChunk {
            start_csf_index: work_chunk.start_csf_index,
            idx_values,
            line1_values,
            line2_values,
            line3_values,
            csf_count,
            line_count: work_chunk.lines.len(),
            truncated_count,
        };

        result_sender.send(processed_chunk).unwrap();
    }
}

// 文件读取线程函数
fn file_reader_thread(
    csfs_path: PathBuf,
    chunk_size: usize,
    work_sender: Sender<WorkChunk>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let input_file = File::open(&csfs_path)?;
    let reader = BufReader::new(input_file);
    let mut lines_iter = reader.lines();

    // 跳过前 5 行 header
    for _ in 0..5 {
        if lines_iter.next().is_none() {
            break;
        }
    }

    let mut chunk_id = 0;
    let mut chunk_lines = Vec::with_capacity(chunk_size);
    let mut global_csf_count = 0;

    println!("开始读取 CSF 数据...");

    for line_result in lines_iter {
        match line_result {
            Ok(line) => {
                chunk_lines.push(line);

                if chunk_lines.len() >= chunk_size {
                    // 确保发送的块包含完整的 CSF（3 的倍数行）
                    let num_full_csfs = chunk_lines.len() / 3;
                    let lines_to_send = num_full_csfs * 3;

                    if lines_to_send > 0 {
                        let start_csf_index = global_csf_count;

                        // 发送完整 CSF 的行
                        let work_chunk = WorkChunk {
                            chunk_id,
                            start_csf_index,
                            lines: chunk_lines[..lines_to_send].to_vec(),
                        };

                        work_sender.send(work_chunk)?;

                        // 更新全局 CSF 计数
                        global_csf_count += num_full_csfs;

                        // 保留剩余的行到下一块
                        let remaining_lines = chunk_lines[lines_to_send..].to_vec();
                        chunk_lines = remaining_lines;
                        chunk_id += 1;

                        if chunk_id % 100 == 0 {
                            // println!("已发送 {} 个数据块", chunk_id);
                        }
                    }
                }
            }
            Err(e) => return Err(e.into()),
        }
    }

    // 发送剩余的行（确保是 3 的倍数）
    if !chunk_lines.is_empty() {
        let num_full_csfs = chunk_lines.len() / 3;
        if num_full_csfs > 0 {
            let lines_to_send = num_full_csfs * 3;
            let work_chunk = WorkChunk {
                chunk_id,
                start_csf_index: global_csf_count,
                lines: chunk_lines[..lines_to_send].to_vec(),
            };
            work_sender.send(work_chunk)?;
        }
    }

    // println!("文件读取完成，共发送 {} 个数据块", chunk_id + 1);
    Ok(())
}

// 写入线程函数
fn writer_thread(
    result_receiver: Receiver<ProcessedChunk>,
    output_path: PathBuf,
    total_stats: Arc<Mutex<ConversionStats>>,
) -> Result<ConversionStats, Box<dyn std::error::Error + Send + Sync>> {
    // 创建 Arrow Schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("idx", DataType::UInt64, false),
        Field::new("line1", DataType::Utf8, false),
        Field::new("line2", DataType::Utf8, false),
        Field::new("line3", DataType::Utf8, false),
    ]));

    // 创建 Parquet 写入器
    let output_file = File::create(&output_path)?;
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::GZIP(
            parquet::basic::GzipLevel::try_new(4)?
        ))
        .build();

    let mut writer = ArrowWriter::try_new(output_file, schema.clone(), Some(props))?;

    // 使用基于 CSF 索引的有序写入缓冲区
    let mut write_buffer = std::collections::BTreeMap::new();
    let mut next_csf_index = 0;

    println!("开始有序写入 Parquet 文件...");

    for processed_chunk in result_receiver {
        write_buffer.insert(processed_chunk.start_csf_index, processed_chunk);

        // 按 CSF 索引顺序写入连续的块
        while let Some((start_index, chunk)) = write_buffer.iter().next().map(|(k, v)| (*k, v.clone())) {
            if start_index == next_csf_index {
                // 这是下一个期望的块
                write_buffer.remove(&start_index);

                // 创建 Arrow 数组
                let mut idx_builder = UInt64Builder::with_capacity(chunk.csf_count);
                let mut line1_builder = StringBuilder::with_capacity(chunk.csf_count, chunk.csf_count * 256);
                let mut line2_builder = StringBuilder::with_capacity(chunk.csf_count, chunk.csf_count * 256);
                let mut line3_builder = StringBuilder::with_capacity(chunk.csf_count, chunk.csf_count * 256);

                for i in 0..chunk.csf_count {
                    idx_builder.append_value(chunk.idx_values[i]);
                    line1_builder.append_value(&chunk.line1_values[i]);
                    line2_builder.append_value(&chunk.line2_values[i]);
                    line3_builder.append_value(&chunk.line3_values[i]);
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

                // 更新统计信息
                {
                    let mut stats = total_stats.lock().unwrap();
                    stats.csf_count += chunk.csf_count;
                    stats.total_lines += chunk.line_count;
                    stats.truncated_count += chunk.truncated_count;
                }

                next_csf_index += chunk.csf_count;

                if next_csf_index % 50000 == 0 {
                    let stats = total_stats.lock().unwrap();
                    println!("已写入 {} 个 CSF", stats.csf_count);
                }
            } else {
                // 下一个块还没到达，等待
                break;
            }
        }
    }

    // 完成写入
    writer.close()?;

    // 返回最终统计信息
    let stats = total_stats.lock().unwrap().clone();
    println!("Parquet 写入完成，共处理 {} 个 CSF", next_csf_index);

    Ok(stats)
}

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
    let mut headers = Vec::with_capacity(5);
    println!("读取 Header...");
    for _i in 0..5 {
        match lines_iter.next() {
            Some(Ok(line)) => {
                headers.push(line);
                // println!("  Header {}: {} 字符", i + 1, headers.last().unwrap().len());
            }
            Some(Err(e)) => return Err(e.into()),
            None => {
                println!("警告: 文件少于 5 行 Header");
                break;
            }
        }
    }

    // 确保 headers 有 5 行，不足的用空字符串填充
    while headers.len() < 5 {
        headers.push(String::new());
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
        .set_compression(parquet::basic::Compression::GZIP(
            parquet::basic::GzipLevel::try_new(4)?
        ))
        .set_write_batch_size(chunk_size as usize)
        .build();
    
    let mut writer = ArrowWriter::try_new(output_file, schema.clone(), Some(props))?;
    println!("Parquet 写入器已创建，使用 GZIP 压缩");

    // --- 4. 批量处理 ---
    let mut batch_lines = Vec::with_capacity(chunk_size);
    let mut csf_count = 0;
    let mut total_lines = 0;
    let mut truncated_count = 0;

    println!("开始处理 CSF 数据...");

    loop {
        // 读取 chunk_size 行
        for _ in 0..chunk_size {
            match lines_iter.next() {
                Some(Ok(line)) => {
                    total_lines += 1;
                    
                    // 检查是否需要截断
                    if line.len() > max_line_len {
                        truncated_count += 1;
                        if truncated_count <= 5 { // 只打印前5个截断警告
                            println!("警告: 第 {} 行被截断 ({} > {})", 
                                total_lines, line.len(), max_line_len);
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
            // 保留剩余行给下一轮
            continue;
        }

        // 处理完整的 CSF 块
        let lines_to_process = batch_lines.drain(..num_full_csfs * 3).collect::<Vec<_>>();
        
        // 创建 Arrow 数组
        let mut idx_builder = UInt64Builder::with_capacity(num_full_csfs);
        let mut line1_builder = StringBuilder::with_capacity(num_full_csfs, num_full_csfs * max_line_len);
        let mut line2_builder = StringBuilder::with_capacity(num_full_csfs, num_full_csfs * max_line_len);
        let mut line3_builder = StringBuilder::with_capacity(num_full_csfs, num_full_csfs * max_line_len);

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
    let header_dir = output_path.parent().unwrap_or_else(|| Path::new("."));
    let input_file_stem = csfs_path.file_stem()
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
        println!("警告: {} 行被截断，考虑增加 max_line_len 参数", truncated_count);
    }
    println!("输出文件: {:?}", output_path);
    println!("Header 文件: {:?}", header_path);
    println!("==========================================");

    Ok(header_data.conversion_stats)
}

/// 获取 Parquet 文件基本信息（仅元数据）
pub fn get_parquet_metadata(parquet_path: &Path) -> Result<pyo3::Py<pyo3::PyAny>, Box<dyn std::error::Error + Send + Sync>> {
    use pyo3::{Python, types::{PyDict, PyDictMethods}};

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
        dict.set_item("compression", metadata.file_metadata().created_by().unwrap_or("Unknown"))?;

        Ok(dict.into())
    })
}
