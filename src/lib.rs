use arrow::record_batch::RecordBatchIterator;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use pyo3_arrow::PyRecordBatchReader;
use std::path::Path;

// Public modules for integration testing
pub mod csf_partition;
pub mod csfs_conversion;
pub mod csfs_descriptor;
pub mod csfs_memory;
pub mod descriptor_normalization;

#[pymodule]
fn _rcsfs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(convert_csfs, m)?)?;
    m.add_function(wrap_pyfunction!(read_csfs_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(get_parquet_info, m)?)?;
    m.add_function(wrap_pyfunction!(partition_csfs, m)?)?;

    // Register CSF descriptor module
    csfs_descriptor::register_descriptor_module(m)?;

    Ok(())
}

/// Read CSF header metadata and data rows for zero-copy import by Polars.
#[pyfunction]
#[pyo3(signature = (
    input_path,
    max_line_len=None,
    num_workers=None,
    include_block_id=false
))]
fn read_csfs_arrow(
    py: Python,
    input_path: String,
    max_line_len: Option<usize>,
    num_workers: Option<usize>,
    include_block_id: bool,
) -> PyResult<(Py<PyAny>, PyRecordBatchReader)> {
    let max_line_len = max_line_len.unwrap_or(256);
    if max_line_len == 0 {
        return Err(PyValueError::new_err("max_line_len must be greater than 0"));
    }
    if matches!(num_workers, Some(0)) {
        return Err(PyValueError::new_err("num_workers must be greater than 0"));
    }

    let (header_data, batch) = py
        .detach(|| {
            csfs_memory::read_csfs_to_record_batch(
                Path::new(&input_path),
                max_line_len,
                num_workers,
                include_block_id,
            )
        })
        .map_err(|error| PyIOError::new_err(error.to_string()))?;

    let header_info = PyDict::new(py);
    header_info.set_item("header_lines", &header_data.header_info.header_lines)?;
    let block_info = PyDict::new(py);
    block_info.set_item("block_lengths", &header_data.block_info.block_lengths)?;
    block_info.set_item("block_count", header_data.block_info.block_count)?;
    let conversion_stats = PyDict::new(py);
    conversion_stats.set_item("csf_count", header_data.conversion_stats.csf_count)?;
    conversion_stats.set_item("total_lines", header_data.conversion_stats.total_lines)?;
    conversion_stats.set_item(
        "truncated_count",
        header_data.conversion_stats.truncated_count,
    )?;
    let header = PyDict::new(py);
    header.set_item("header_info", header_info)?;
    header.set_item("block_info", block_info)?;
    header.set_item("conversion_stats", conversion_stats)?;

    let schema = batch.schema();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    Ok((header.into(), PyRecordBatchReader::new(Box::new(reader))))
}

/// Get Parquet file basic information and metadata
///
/// Args:
/// - input_path: Path to Parquet file
///
/// Returns:
/// Dictionary containing file information and metadata
#[pyfunction]
fn get_parquet_info(py: Python, input_path: String) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    py.detach(|| {
        csfs_conversion::get_parquet_metadata(Path::new(&input_path))
            .map_err(|e| PyIOError::new_err(format!("Failed to get Parquet file info: {}", e)))
    })
}

/// Convert CSF text file to Parquet format (parallel processing)
///
/// Args:
/// - input_path: Path to input CSF file
/// - output_path: Path to output Parquet file
/// - max_line_len: Maximum line length (default: 256)
/// - chunk_size: Batch processing size (default: 3000000, optimized for parallel efficiency)
/// - num_workers: Number of worker threads (default: CPU core count)
///
/// Returns:
/// Dictionary containing conversion statistics:
/// - success: Whether conversion succeeded
/// - csf_count: Number of CSFs
/// - total_lines: Total line count
/// - truncated_count: Number of truncated lines
/// - input_file: Input file path
/// - output_file: Output file path
/// - header_file: TOML header file path
/// - max_line_len: Maximum line length used
/// - chunk_size: Batch processing size used
/// - error: Error message (only present on failure)
///
/// Features:
/// - Multi-threaded parallel processing using rayon (automatically uses all CPU cores)
/// - Maintains original CSF order for consistent output file ordering
/// - Memory efficient streaming to handle large files
#[pyfunction]
#[pyo3(signature = (
    input_path,
    output_path,
    max_line_len=None,
    chunk_size=None,
    num_workers=None
))]
fn convert_csfs(
    py: Python,
    input_path: String,
    output_path: String,
    max_line_len: Option<usize>,
    chunk_size: Option<usize>,
    num_workers: Option<usize>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    // Set default parameters (optimized for parallel processing)
    let max_line_len = max_line_len.unwrap_or(256);
    let chunk_size = chunk_size.unwrap_or(3000000); // 1M CSFs = 3M lines per batch

    // Parameter validation
    if max_line_len == 0 {
        return Err(PyValueError::new_err("max_line_len must be greater than 0"));
    }
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size must be greater than 0"));
    }
    if matches!(num_workers, Some(0)) {
        return Err(PyValueError::new_err("num_workers must be greater than 0"));
    }

    // Execute parallel conversion
    let result = py.detach(|| {
        csfs_conversion::convert_csfs_to_parquet_parallel(
            Path::new(&input_path),
            Path::new(&output_path),
            max_line_len,
            chunk_size,
            num_workers,
        )
    });

    match result {
        Ok(conversion_stats) => {
            // Create result dictionary
            let stats = PyDict::new(py);
            stats.set_item("success", true)?;
            stats.set_item("input_file", &input_path)?;
            stats.set_item("output_file", &output_path)?;
            stats.set_item("max_line_len", max_line_len)?;
            stats.set_item("chunk_size", chunk_size)?;
            stats.set_item("csf_count", conversion_stats.csf_count)?;
            stats.set_item("total_lines", conversion_stats.total_lines)?;
            stats.set_item("truncated_count", conversion_stats.truncated_count)?;

            // Try to read [input_file_stem]_header.toml file path
            let output_dir = Path::new(&output_path)
                .parent()
                .unwrap_or_else(|| Path::new("."));
            let input_file_stem = Path::new(&input_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("csfs");
            let header_filename = format!("{}_header.toml", input_file_stem);
            let header_path = output_dir.join(header_filename);
            if header_path.exists() {
                stats.set_item("header_file", header_path.to_string_lossy())?;
            }

            Ok(stats.into())
        }
        Err(e) => {
            // Create error result dictionary
            let stats = PyDict::new(py);
            stats.set_item("success", false)?;
            stats.set_item("error", e.to_string())?;
            Ok(stats.into())
        }
    }
}

/// Partition CSFs into zero-order + first-order space per symmetry block.
///
/// Reads two CSF Parquet files (a zero-order reference and the complete list)
/// together with their `{stem}_header.toml` sidecars, reorders each symmetry
/// block so the zero-order CSFs are locked to the head followed by the
/// first-order complement, and writes the result as a CSF text file.
///
/// Args:
/// - zero_parquet: Path to the zero-order reference Parquet file.
/// - zero_header: Path to the zero-order `{stem}_header.toml`.
/// - full_parquet: Path to the complete-list Parquet file.
/// - full_header: Path to the complete-list `{stem}_header.toml`.
/// - output_csf: Path to the destination CSF text file.
///
/// Returns:
/// Dictionary with partition statistics:
/// - success: Whether partition succeeded
/// - zero_parquet / full_parquet / output_file: input/output paths
/// - block_count: number of symmetry blocks
/// - zero_csf_count: total CSFs in the zero-order reference
/// - full_csf_count: total CSFs in the full list
/// - output_csf_count: total CSFs written (zero + complement)
/// - first_order_count: complement count (full CSFs not in zero-order)
/// - error: error message (only present on failure)
#[pyfunction]
fn partition_csfs(
    py: Python,
    zero_parquet: String,
    zero_header: String,
    full_parquet: String,
    full_header: String,
    output_csf: String,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    let result = py.detach(|| {
        csf_partition::partition_csfs(
            Path::new(&zero_parquet),
            Path::new(&zero_header),
            Path::new(&full_parquet),
            Path::new(&full_header),
            Path::new(&output_csf),
        )
    });

    match result {
        Ok(stats) => {
            let d = PyDict::new(py);
            d.set_item("success", true)?;
            d.set_item("zero_parquet", &zero_parquet)?;
            d.set_item("full_parquet", &full_parquet)?;
            d.set_item("output_file", &output_csf)?;
            d.set_item("block_count", stats.block_count)?;
            d.set_item("zero_csf_count", stats.zero_csf_count)?;
            d.set_item("full_csf_count", stats.full_csf_count)?;
            d.set_item("output_csf_count", stats.output_csf_count)?;
            d.set_item("first_order_count", stats.first_order_count)?;
            Ok(d.into())
        }
        Err(e) => {
            let d = PyDict::new(py);
            d.set_item("success", false)?;
            d.set_item("error", e.to_string())?;
            Ok(d.into())
        }
    }
}
