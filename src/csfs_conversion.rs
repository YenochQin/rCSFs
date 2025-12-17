use arrow::array::StringBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;


pub fn convert_csfs_to_parquet(
    csfs_path: &Path,
    output_path: &Path,
    max_line_len: usize,
    chunk_size: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
    for i in 0..5 {
        match lines_iter.next() {
            Some(Ok(line)) => {
                headers.push(line);
                println!("  Header {}: {} 字符", i + 1, headers.last().unwrap().len());
            }
            Some(Err(e)) => return Err(e.into()),
            None => {
                println!("警告: 文件少于 5 行 Header");
                break;
            }
        }
    }
    
    // 保存 headers 到单独的文件
    let header_path = output_path.with_extension("headers.txt");
    std::fs::write(&header_path, headers.join("\n"))?;
    println!("Header 已保存到: {:?}", header_path);

    // --- 2. 创建 Arrow Schema ---
    let schema = Arc::new(Schema::new(vec![
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
        let mut line1_builder = StringBuilder::with_capacity(num_full_csfs, num_full_csfs * max_line_len);
        let mut line2_builder = StringBuilder::with_capacity(num_full_csfs, num_full_csfs * max_line_len);
        let mut line3_builder = StringBuilder::with_capacity(num_full_csfs, num_full_csfs * max_line_len);

        for chunk in lines_to_process.chunks(3) {
            if chunk.len() == 3 {
                line1_builder.append_value(&chunk[0]);
                line2_builder.append_value(&chunk[1]);
                line3_builder.append_value(&chunk[2]);
            }
        }

        // 构建 RecordBatch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
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

    Ok(())
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
