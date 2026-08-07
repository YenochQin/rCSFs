use arrow::array::{ArrayRef, StringBuilder, UInt32Builder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Error as IoError, ErrorKind};
use std::path::Path;
use std::sync::Arc;

use crate::csfs_conversion::{BlockInfo, ConversionStats, HeaderData, HeaderInfo};

const CSF_HEADER_LINE_COUNT: usize = 5;

#[derive(Debug)]
struct DataLine {
    value: String,
    number: usize,
    block_id: u32,
    truncated: bool,
}

fn process_data_line_owned(mut line: DataLine, max_line_len: usize) -> Result<DataLine, IoError> {
    if !line.value.is_ascii() {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!(
                "CSF data line {} contains non-ASCII text; fixed-width CSF parsing requires ASCII input",
                line.number
            ),
        ));
    }

    if line.value.len() > max_line_len {
        line.value.truncate(max_line_len);
        line.truncated = true;
    }
    Ok(line)
}

/// Read a complete CSF file into one Arrow record batch.
///
/// Block separator lines (`*`, ignoring surrounding whitespace) are omitted.
/// When requested, `block_id` identifies the zero-based block for every row.
pub fn read_csfs_to_record_batch(
    csfs_path: &Path,
    max_line_len: usize,
    num_workers: Option<usize>,
    include_block_id: bool,
) -> Result<(HeaderData, RecordBatch), Box<dyn std::error::Error + Send + Sync>> {
    if max_line_len == 0 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "max_line_len must be greater than 0",
        )
        .into());
    }
    if matches!(num_workers, Some(0)) {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "num_workers must be greater than 0",
        )
        .into());
    }

    let file = File::open(csfs_path)?;
    let reader = BufReader::new(file);
    let mut header_lines = Vec::with_capacity(CSF_HEADER_LINE_COUNT);
    let mut data_lines = Vec::new();
    let mut current_block_line_count = 0usize;
    let mut block_lengths = Vec::new();
    let mut saw_separator = false;
    let mut block_id = 0u32;
    let mut total_lines = 0usize;

    for (zero_based_line_number, line) in reader.lines().enumerate() {
        let line = line?;
        if zero_based_line_number < CSF_HEADER_LINE_COUNT {
            header_lines.push(line);
            continue;
        }

        total_lines += 1;
        let data_line_number = zero_based_line_number + 1 - CSF_HEADER_LINE_COUNT;
        if line.trim() == "*" {
            if !current_block_line_count.is_multiple_of(3) {
                return Err(IoError::new(
                    ErrorKind::InvalidData,
                    format!(
                        "CSF block ending at data line {data_line_number} has \
                         {current_block_line_count} data lines, not a multiple of 3"
                    ),
                )
                .into());
            }
            block_lengths.push(current_block_line_count / 3);
            current_block_line_count = 0;
            saw_separator = true;
            block_id = block_id.checked_add(1).ok_or_else(|| {
                IoError::new(ErrorKind::InvalidData, "CSF file contains too many blocks")
            })?;
            continue;
        }

        current_block_line_count += 1;
        data_lines.push(DataLine {
            value: line,
            number: data_line_number,
            block_id,
            truncated: false,
        });
    }

    header_lines.resize(CSF_HEADER_LINE_COUNT, String::new());

    // Match the established conversion behavior: ignore an incomplete final CSF.
    let incomplete_line_count = current_block_line_count % 3;
    if incomplete_line_count != 0 {
        data_lines.truncate(data_lines.len() - incomplete_line_count);
    }

    let process = || {
        data_lines
            .into_par_iter()
            .map(|line| process_data_line_owned(line, max_line_len))
            .collect::<Result<Vec<_>, IoError>>()
    };

    let rows = match num_workers {
        Some(worker_count) => rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()?
            .install(process)?,
        None => process()?,
    };

    let row_count = rows.len() / 3;
    let truncated_count = rows.iter().filter(|row| row.truncated).count();
    if saw_separator || current_block_line_count > 0 {
        block_lengths.push(current_block_line_count / 3);
    }
    let header_data = HeaderData {
        header_info: HeaderInfo { header_lines },
        block_info: BlockInfo {
            block_count: block_lengths.len(),
            block_lengths,
        },
        conversion_stats: ConversionStats {
            csf_count: row_count,
            total_lines,
            truncated_count,
        },
    };
    let line1_bytes = rows.iter().step_by(3).map(|row| row.value.len()).sum();
    let line2_bytes = rows
        .iter()
        .skip(1)
        .step_by(3)
        .map(|row| row.value.len())
        .sum();
    let line3_bytes = rows
        .iter()
        .skip(2)
        .step_by(3)
        .map(|row| row.value.len())
        .sum();
    let mut idx_builder = UInt64Builder::with_capacity(row_count);
    let mut block_id_builder = include_block_id.then(|| UInt32Builder::with_capacity(row_count));
    let mut line1_builder = StringBuilder::with_capacity(row_count, line1_bytes);
    let mut line2_builder = StringBuilder::with_capacity(row_count, line2_bytes);
    let mut line3_builder = StringBuilder::with_capacity(row_count, line3_bytes);

    for (idx, lines) in rows.chunks_exact(3).enumerate() {
        debug_assert_eq!(lines[0].block_id, lines[1].block_id);
        debug_assert_eq!(lines[1].block_id, lines[2].block_id);
        idx_builder.append_value(idx as u64);
        if let Some(builder) = &mut block_id_builder {
            builder.append_value(lines[0].block_id);
        }
        line1_builder.append_value(&lines[0].value);
        line2_builder.append_value(&lines[1].value);
        line3_builder.append_value(&lines[2].value);
    }

    let mut fields = vec![Field::new("idx", DataType::UInt64, false)];
    let mut columns: Vec<ArrayRef> = vec![Arc::new(idx_builder.finish())];
    if let Some(mut builder) = block_id_builder {
        fields.push(Field::new("block_id", DataType::UInt32, false));
        columns.push(Arc::new(builder.finish()));
    }
    fields.extend([
        Field::new("line1", DataType::Utf8, false),
        Field::new("line2", DataType::Utf8, false),
        Field::new("line3", DataType::Utf8, false),
    ]);
    columns.extend([
        Arc::new(line1_builder.finish()) as ArrayRef,
        Arc::new(line2_builder.finish()) as ArrayRef,
        Arc::new(line3_builder.finish()) as ArrayRef,
    ]);

    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
    Ok((header_data, batch))
}
