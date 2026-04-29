use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use std::path::Path;

// Public modules for integration testing
pub mod csfs_conversion;
pub mod csfs_descriptor;
pub mod descriptor_normalization;

#[pymodule]
fn _rcsfs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(convert_csfs, m)?)?;
    m.add_function(wrap_pyfunction!(get_parquet_info, m)?)?;

    // Register CSF descriptor module
    csfs_descriptor::register_descriptor_module(m)?;

    Ok(())
}

/// Return basic metadata for a Parquet file.
///
/// Args:
///     input_path: Path to the Parquet file.
///
/// Returns:
///     A dictionary with file path, file size, row count, column count,
///     compression, and writer metadata.
#[pyfunction]
fn get_parquet_info(py: Python, input_path: String) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    py.detach(|| {
        csfs_conversion::get_parquet_metadata(Path::new(&input_path))
            .map_err(|e| PyIOError::new_err(format!("Failed to get Parquet file info: {}", e)))
    })
}

/// Convert a GRASP CSF text file to Parquet.
///
/// The converter streams input lines, groups each CSF's three data lines into
/// one row, and writes an ordered Parquet table suitable for later descriptor
/// generation.
///
/// Args:
///     input_path: Path to the input CSF file.
///     output_path: Path where the CSF Parquet file should be written.
///     max_line_len: Maximum input line length. Longer lines are truncated.
///     chunk_size: Number of input lines per streaming batch.
///     num_workers: Worker thread count. Use None for Rayon defaults.
///
/// Returns:
///     A dictionary containing success status, generated paths, line counts,
///     CSF count, truncation count, and error text when conversion fails.
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
