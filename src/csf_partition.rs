//! CSF zero-first partition core.
//!
//! Reorders a CSF list so that, within each symmetry block (J^P), the zero-order
//! reference CSFs are locked to the head of the block and the first-order
//! complement (the full block's CSFs that are not in the zero-order block) is
//! appended after them. This mirrors the behavior of GRASP2018's `rcsfzerofirst`
//! Fortran utility, but operates on the Parquet representation produced by
//! [`crate::csfs_conversion::convert_csfs_to_parquet`] instead of raw CSF text.
//!
//! Match semantics: a CSF is identified by the exact string equality of its
//! three-line record `(line1, line2, line3)` — identical to
//! `lodcsl_Part.f90:52-54` in the Fortran reference. Trailing-whitespace
//! differences cause a miss-match; this is preserved as grasp-compatible
//! behavior.
//!
//! Block alignment is by index: the i-th block of the zero-order file is paired
//! with the i-th block of the full file. [`partition_csfs`] validates that both
//! files report the same `block_count`.
//!
//! Memory profile: per block, the zero-order block is fully loaded (it is the
//! reference and typically small) to build the anti-match `HashSet`; the full
//! block is read in batch-sized chunks. Peak memory ≈ zero-block size + one
//! Parquet batch. This matches the grasp original, which also `allocate`s the
//! full `Found`/`C_shell`/... arrays per block.

use crate::csfs_conversion::HeaderData;
use arrow::array::{RecordBatch, StringArray};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Error as IoError, ErrorKind, Write};
use std::path::Path;

/// Statistics returned from a zero-first partition operation.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PartitionStats {
    /// Number of symmetry blocks processed.
    pub block_count: usize,
    /// Total CSFs in the zero-order reference file (`sum(zero block_lengths)`).
    pub zero_csf_count: usize,
    /// Total CSFs in the full file (`sum(full block_lengths)`).
    pub full_csf_count: usize,
    /// Total CSFs written to the output (`zero + first-order complement`).
    pub output_csf_count: usize,
    /// Complement count: full CSFs that were not found in the zero-order reference.
    pub first_order_count: usize,
}

/// GRASP-style block separator written between symmetry blocks.
///
/// The Fortran original emits `' *'` (leading space + asterisk); rCSFs' own
/// `is_block_separator` accepts any trim-equal-to-`*` line on input, but we
/// emit the canonical grasp form on output for downstream compatibility.
const BLOCK_SEPARATOR: &str = " *";

/// Number of Parquet rows to pull per batch when streaming the full block.
///
/// Balances Parquet I/O granularity against peak memory. The reader already
/// yields `RecordBatch`es of its own chosen size; this only caps how many
/// decoded triples we hold before flushing them through the anti-match check.
const FULL_BLOCK_BATCH_ROWS: usize = 65_536;

/// A single CSF row read from Parquet: the `(line1, line2, line3)` triple.
/// The `idx` column is intentionally ignored — file order already matches the
/// global `idx` order written by `convert_csfs`, so `block_lengths` is
/// sufficient to slice blocks.
type CsfRow = (String, String, String);

/// Streaming reader over the CSF Parquet schema `(idx, line1, line2, line3)`.
struct CsfRowStream {
    reader: ParquetRecordBatchReader,
    pending: Vec<CsfRow>,
}

impl CsfRowStream {
    fn open(parquet_path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = File::open(parquet_path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        Ok(Self {
            reader,
            pending: Vec::new(),
        })
    }

    /// Decode the next `RecordBatch` into `pending`. Returns `false` at EOF.
    fn refill(&mut self) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        match self.reader.next() {
            None => Ok(false),
            Some(Ok(batch)) => {
                self.ingest_batch(&batch)?;
                Ok(true)
            }
            Some(Err(e)) => Err(Box::new(e)),
        }
    }

    fn ingest_batch(
        &mut self,
        batch: &RecordBatch,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let line1 = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                IoError::new(ErrorKind::InvalidData, "Parquet line1 column is not Utf8")
            })?;
        let line2 = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                IoError::new(ErrorKind::InvalidData, "Parquet line2 column is not Utf8")
            })?;
        let line3 = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                IoError::new(ErrorKind::InvalidData, "Parquet line3 column is not Utf8")
            })?;
        for i in 0..batch.num_rows() {
            self.pending.push((
                line1.value(i).to_string(),
                line2.value(i).to_string(),
                line3.value(i).to_string(),
            ));
        }
        Ok(())
    }

    /// Pull exactly `n` rows in file order into the caller's `Vec`.
    ///
    /// Returns an error if the stream ends before `n` rows are available —
    /// that indicates a header TOML / Parquet mismatch and must not silently
    /// truncate output.
    fn take_rows(
        &mut self,
        n: usize,
    ) -> Result<Vec<CsfRow>, Box<dyn std::error::Error + Send + Sync>> {
        let mut out: Vec<CsfRow> = Vec::with_capacity(n);
        while out.len() < n {
            if self.pending.is_empty() && !self.refill()? {
                return Err(Box::new(IoError::new(
                    ErrorKind::UnexpectedEof,
                    format!(
                        "unexpected end of Parquet stream: wanted {} rows, got {}",
                        n,
                        out.len()
                    ),
                )));
            }
            let need = n - out.len();
            let take = need.min(self.pending.len());
            out.extend(self.pending.drain(..take));
        }
        Ok(out)
    }

    /// Read up to `max` rows (fewer if the stream ends). Used for the full
    /// block, where the total is bounded by `full_lengths[b]` but we do not
    /// want to force the whole block into memory at once.
    fn take_up_to(
        &mut self,
        max: usize,
    ) -> Result<Vec<CsfRow>, Box<dyn std::error::Error + Send + Sync>> {
        let mut out: Vec<CsfRow> = Vec::with_capacity(max.min(FULL_BLOCK_BATCH_ROWS));
        while out.len() < max {
            if self.pending.is_empty() && !self.refill()? {
                break;
            }
            let need = max - out.len();
            let take = need.min(self.pending.len());
            out.extend(self.pending.drain(..take));
        }
        Ok(out)
    }
}

/// Partition CSFs into a zero-order + first-order space per symmetry block.
///
/// # Algorithm (per block `b`)
///
/// 1. Load the entire zero-order block `b` and build a `HashSet` of its
///    `(line1, line2, line3)` triples.
/// 2. Write the zero-order block verbatim to the output — this **locks** the
///    reference to the head of the block.
/// 3. Stream the full block `b` in batches; append any CSF whose triple is
///    **not** in the zero set — this is the **first-order complement**.
/// 4. Emit a ` *` block separator between blocks (never after the last).
///
/// The 5-line header is taken from the **full** file (it is the complete
/// space); the zero-order file's header is only used for `block_lengths`.
///
/// # Arguments
///
/// * `zero_parquet`, `zero_header` — the zero-order reference (its Parquet +
///   the `{stem}_header.toml` produced alongside it by `convert_csfs`).
/// * `full_parquet`, `full_header` — the complete CSF list (Parquet + header TOML).
/// * `output_csf` — destination CSF text file (created or overwritten).
///
/// # Errors
///
/// Returns an error if any file cannot be opened, the header TOML is invalid,
/// the two files disagree on `block_count`, or a Parquet stream ends before
/// the declared `block_lengths` rows have been read.
///
/// # Example (Rust integration test path)
///
/// ```no_run
/// use _rcsfs::csf_partition::partition_csfs;
///
/// let stats = partition_csfs(
///     std::path::Path::new("zero.parquet"),
///     std::path::Path::new("zero_header.toml"),
///     std::path::Path::new("full.parquet"),
///     std::path::Path::new("full_header.toml"),
///     std::path::Path::new("reordered.csf"),
/// ).expect("partition should succeed");
/// assert!(stats.first_order_count > 0);
/// ```
pub fn partition_csfs(
    zero_parquet: &Path,
    zero_header: &Path,
    full_parquet: &Path,
    full_header: &Path,
    output_csf: &Path,
) -> Result<PartitionStats, Box<dyn std::error::Error + Send + Sync>> {
    println!("开始 zero-first 划分");
    println!("零阶 Parquet: {:?}", zero_parquet);
    println!("完整 Parquet: {:?}", full_parquet);
    println!("输出 CSF: {:?}", output_csf);

    // --- 1. 读取两份 header TOML（单一真相源：csfs_conversion 的 HeaderData）---
    let zero_hdr: HeaderData = {
        let text = std::fs::read_to_string(zero_header)?;
        toml::from_str(&text)?
    };
    let full_hdr: HeaderData = {
        let text = std::fs::read_to_string(full_header)?;
        toml::from_str(&text)?
    };

    // --- 2. 校验块对齐（按块序号一一配对，块数必须相等）---
    let block_count = zero_hdr.block_info.block_count;
    if block_count != full_hdr.block_info.block_count {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidInput,
            format!(
                "block count mismatch: zero={} full={} (blocks are paired by index; \
                 counts must be equal)",
                block_count, full_hdr.block_info.block_count
            ),
        )));
    }
    let zero_lengths = &zero_hdr.block_info.block_lengths;
    let full_lengths = &full_hdr.block_info.block_lengths;
    if zero_lengths.len() != block_count || full_lengths.len() != block_count {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidData,
            format!(
                "block_lengths length does not match block_count: zero.len={} full.len={} expected={}",
                zero_lengths.len(),
                full_lengths.len(),
                block_count
            ),
        )));
    }

    let zero_csf_count: usize = zero_lengths.iter().sum();
    let full_csf_count: usize = full_lengths.iter().sum();

    // --- 3. 打开输出 + 写 5 行 header（取自完整文件，它代表完整空间）---
    let output_file = File::create(output_csf)?;
    let mut out = BufWriter::new(output_file);
    for line in &full_hdr.header_info.header_lines {
        writeln!(out, "{}", line)?;
    }

    // --- 4. 流式处理每个块 ---
    let mut zero_stream = CsfRowStream::open(zero_parquet)?;
    let mut full_stream = CsfRowStream::open(full_parquet)?;
    let mut output_csf_count: usize = 0;
    let mut first_order_count: usize = 0;

    // ============================================================
    // 输出格式契约 —— 不可随意修改（DO NOT MODIFY FORMAT）
    // ------------------------------------------------------------
    // 下方表头 println! 和 for 循环内的每块数据行 println!（格式
    // "{:>5}{:>20}{:>20}"）刻意按 GRASP Fortran 程序 RCSFzerofirst
    // 的 stdout 格式书写，是 graspkit-tools/scripts/rzf_arg.py 的隐式
    // 输入契约：
    //
    //   zf_block_input_parameter() 用正则 r"\s+\d+\s+(\d+)\s+\d+$"
    //   逐行匹配，捕获第二列（Zero-order Space，每块零阶 CSF 数）。
    //
    // 必须保持：
    //   - 表头文本与列顺序：Block | Zero-order Space | Complete Space
    //   - 数据行格式："{:>5}{:>20}{:>20}"（首列块号，第二列零阶 CSF 数，
    //     第三列完整 CSF 数；仅用空白分隔，纯数字）
    //   - 不在数据行中混入任何单位、文字、注释
    //
    // 如需调整，必须同步修改 rzf_arg.py 的解析逻辑并补回归测试。
    // ============================================================
    println!("==========================================");
    println!("   Block    Zero-order Space   Complete Space");
    for b in 0..block_count {
        let zlen = zero_lengths[b];
        let flen = full_lengths[b];

        // 4a. 读零阶块全部，建 HashSet（零阶通常是小子集）
        let zero_rows = zero_stream.take_rows(zlen)?;
        let zero_set: HashSet<(&str, &str, &str)> = zero_rows
            .iter()
            .map(|(l1, l2, l3)| (l1.as_str(), l2.as_str(), l3.as_str()))
            .collect();

        // 4b. 写零阶块（锁定到块首）
        for (l1, l2, l3) in &zero_rows {
            writeln!(out, "{}", l1)?;
            writeln!(out, "{}", l2)?;
            writeln!(out, "{}", l3)?;
        }
        output_csf_count += zlen;

        // 4c. 流式读 full 块（分批），追加补集（不在零阶集合中的 CSF）
        let mut full_taken: usize = 0;
        while full_taken < flen {
            let want = (flen - full_taken).min(FULL_BLOCK_BATCH_ROWS);
            let chunk = full_stream.take_up_to(want)?;
            if chunk.len() != want {
                return Err(Box::new(IoError::new(
                    ErrorKind::UnexpectedEof,
                    format!(
                        "full Parquet ended early in block {}: wanted {} more rows, got {}",
                        b,
                        want,
                        chunk.len()
                    ),
                )));
            }
            for (l1, l2, l3) in &chunk {
                if !zero_set.contains(&(l1.as_str(), l2.as_str(), l3.as_str())) {
                    writeln!(out, "{}", l1)?;
                    writeln!(out, "{}", l2)?;
                    writeln!(out, "{}", l3)?;
                    output_csf_count += 1;
                    first_order_count += 1;
                }
            }
            full_taken += chunk.len();
        }

        // 4d. 块分隔符（最后一块之后不写）
        if b + 1 < block_count {
            writeln!(out, "{}", BLOCK_SEPARATOR)?;
        }

        // 见上方"输出格式契约"：此行格式是 rzf_arg.py 的输入契约，不可修改。
        println!("{:>5}{:>20}{:>20}", b + 1, zlen, flen);
    }

    out.flush()?;

    println!("\n划分完成！");
    println!("================ 统计信息 ================");
    println!("块数: {}", block_count);
    println!("零阶 CSF 数: {}", zero_csf_count);
    println!("完整 CSF 数: {}", full_csf_count);
    println!("输出 CSF 数: {}", output_csf_count);
    println!("一阶补集 CSF 数: {}", first_order_count);
    println!("输出文件: {:?}", output_csf);
    println!("==========================================");

    Ok(PartitionStats {
        block_count,
        zero_csf_count,
        full_csf_count,
        output_csf_count,
        first_order_count,
    })
}
