use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError};
use pyo3::types::{PyDict, PyDictMethods};
use std::path::Path;

// Public modules for integration testing
pub mod csfs_conversion;
pub mod csfs_descriptor;

#[pymodule]
fn _rcsfs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(convert_csfs, m)?)?;
    m.add_function(wrap_pyfunction!(get_parquet_info, m)?)?;
    m.add_class::<CSFProcessor>()?;

    // Register CSF descriptor module
    csfs_descriptor::register_descriptor_module(m)?;

    Ok(())
}


/// Get Parquet file basic information and metadata
///
/// Args:
/// - input_path: Path to Parquet file
///
/// Returns:
/// Dictionary containing file information and metadata
#[pyfunction]
fn get_parquet_info(
    py: Python,
    input_path: String,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    py.detach(|| {
        csfs_conversion::get_parquet_metadata(Path::new(&input_path))
            .map_err(|e| PyIOError::new_err(format!("Failed to get Parquet file info: {}", e)))
    })
}

/// CSF file processor class providing an object-oriented interface
#[pyclass]
struct CSFProcessor {
    max_line_len: usize,
    chunk_size: usize,
}

#[pymethods]
impl CSFProcessor {
    /// Create a new CSF processor instance
    #[new]
    #[pyo3(signature = (max_line_len=256, chunk_size=3000000))]
    fn new(max_line_len: Option<usize>, chunk_size: Option<usize>) -> PyResult<Self> {
        let max_line_len = max_line_len.unwrap_or(256);
        let chunk_size = chunk_size.unwrap_or(3000000);

        if max_line_len == 0 {
            return Err(PyValueError::new_err("max_line_len must be greater than 0"));
        }
        if chunk_size == 0 {
            return Err(PyValueError::new_err("chunk_size must be greater than 0"));
        }

        Ok(CSFProcessor {
            max_line_len,
            chunk_size,
        })
    }

    /// Set maximum line length
    #[setter]
    fn set_max_line_len(&mut self, value: usize) -> PyResult<()> {
        if value == 0 {
            return Err(PyValueError::new_err("max_line_len must be greater than 0"));
        }
        self.max_line_len = value;
        Ok(())
    }

    /// Set batch processing size
    #[setter]
    fn set_chunk_size(&mut self, value: usize) -> PyResult<()> {
        if value == 0 {
            return Err(PyValueError::new_err("chunk_size must be greater than 0"));
        }
        self.chunk_size = value;
        Ok(())
    }

    /// Get current configuration
    fn get_config(&self, py: Python) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        let config = PyDict::new(py);
        config.set_item("max_line_len", self.max_line_len)?;
        config.set_item("chunk_size", self.chunk_size)?;
        Ok(config.into())
    }

    /// Convert CSF file using parallel processing
    fn convert(
        &self,
        py: Python,
        input_path: String,
        output_path: String,
        num_workers: Option<usize>,
    ) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        convert_csfs(
            py,
            input_path,
            output_path,
            Some(self.max_line_len),
            Some(self.chunk_size),
            num_workers,
        )
    }

    /// Get Parquet file information
    fn get_metadata(&self, py: Python, input_path: String) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        get_parquet_info(py, input_path)
    }
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
            let output_dir = Path::new(&output_path).parent().unwrap_or_else(|| Path::new("."));
            let input_file_stem = Path::new(&input_path).file_stem()
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
