use arrow::record_batch::RecordBatchIterator;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList, PyListMethods};
use pyo3_arrow::PyRecordBatchReader;
use std::path::{Path, PathBuf};

// Public modules for integration testing
pub mod atomic_output;
pub mod complete_csf;
pub mod csf_generation;
pub mod csf_partition;
pub mod csfs_conversion;
pub mod csfs_descriptor;
pub mod csfs_memory;
pub mod descriptor_normalization;
pub mod descriptor_schema;
pub mod descriptor_v2;
pub mod interaction;
mod interaction_py;

#[pymodule]
fn _rcsfs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(convert_csfs, m)?)?;
    m.add_function(wrap_pyfunction!(read_csfs_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(get_parquet_info, m)?)?;
    m.add_function(wrap_pyfunction!(partition_csfs, m)?)?;
    m.add_function(wrap_pyfunction!(generate_csfs_from_transcript, m)?)?;
    m.add_function(wrap_pyfunction!(generate_disk_outputs_from_transcript, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_disk_generation, m)?)?;
    m.add_function(wrap_pyfunction!(select_interacting_csfs, m)?)?;

    // Register CSF descriptor module
    csfs_descriptor::register_descriptor_module(m)?;

    Ok(())
}

/// Select candidate CSFs that may interact with a reference space.
///
/// The structural-upper-bound method is conservative and intentionally does
/// not reproduce GRASP's complete angular and recoupling algebra. Its result
/// is therefore not an exact non-zero Hamiltonian test.
#[pyfunction]
#[pyo3(signature = (
    reference_csf,
    candidate_csf,
    output_csf,
    *,
    hamiltonian="dirac_coulomb",
    method="structural_upper_bound",
    num_workers=None,
    overwrite=false
))]
#[allow(clippy::too_many_arguments)] // PyO3 exposes one argument per Python parameter.
fn select_interacting_csfs(
    py: Python<'_>,
    reference_csf: &str,
    candidate_csf: &str,
    output_csf: &str,
    hamiltonian: &str,
    method: &str,
    num_workers: Option<&Bound<'_, PyAny>>,
    overwrite: bool,
) -> PyResult<Py<PyAny>> {
    interaction_py::select_interacting_csfs(
        py,
        reference_csf,
        candidate_csf,
        output_csf,
        hamiltonian,
        method,
        num_workers,
        overwrite,
    )
}

/// Read CSF header metadata and data rows for zero-copy import by Polars.
#[pyfunction]
#[pyo3(signature = (
    input_path,
    max_line_len=None,
    num_workers=None,
    include_block_id=false,
    include_coupling_signature=false,
    strict=true
))]
fn read_csfs_arrow(
    py: Python,
    input_path: String,
    max_line_len: Option<usize>,
    num_workers: Option<usize>,
    include_block_id: bool,
    include_coupling_signature: bool,
    strict: bool,
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
                include_coupling_signature,
                strict,
            )
        })
        .map_err(|error| PyIOError::new_err(format!("{error:#}")))?;

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
            d.set_item("error", format!("{e:#}"))?;
            Ok(d.into())
        }
    }
}

/// Generate CSFs from an in-memory `rcsfgenerate.log`-format transcript.
///
/// The transcript is parsed, enumerated and generated entirely in Rust and
/// written directly to `output_path`; it is never written to disk itself.
/// This backs the interactive `rcsfs csfsgenerate` CLI, which assembles the
/// transcript from the user's answers before calling this function.
///
/// Args:
/// - transcript: `rcsfgenerate.log`-format text (see `ExcitationRequest::from_transcript`).
/// - output_path: Destination CSF text file. Must not already exist.
/// - normalize: Retained for Python API compatibility; descriptors use the CLI pipeline.
/// - threads: Optional Rayon thread count; defaults to all cores.
///
/// Returns:
/// Dictionary with `success`, and on success `output_file`, `record_count`,
/// `block_count`, `unique_occupations`, optionally `descriptor_file`/
/// `descriptor_count`; on failure `error`.
#[pyfunction]
#[pyo3(signature = (
    transcript,
    output_path,

    normalize=false,
    threads=None
))]
fn generate_csfs_from_transcript(
    py: Python,
    transcript: String,
    output_path: String,

    normalize: bool,
    threads: Option<usize>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    if matches!(threads, Some(0)) {
        return Err(PyValueError::new_err("threads must be greater than 0"));
    }

    let _ = normalize;
    let result = py.detach(|| {
        crate::csf_generation::generate_csfs_from_transcript(
            &transcript,
            Path::new(&output_path),
            threads,
        )
    });

    match result {
        Ok(stats) => {
            let d = PyDict::new(py);
            d.set_item("success", true)?;
            d.set_item("output_file", &output_path)?;
            d.set_item("record_count", stats.record_count)?;
            d.set_item("block_count", stats.block_count)?;
            d.set_item("unique_occupations", stats.unique_occupations)?;
            if let Some(descriptor_count) = stats.descriptor_count {
                d.set_item("descriptor_count", descriptor_count)?;
            }
            Ok(d.into())
        }
        Err(e) => {
            let d = PyDict::new(py);
            d.set_item("success", false)?;
            d.set_item("error", format!("{e:#}"))?;
            Ok(d.into())
        }
    }
}

/// Generate private staged CSF and V2 descriptor files through the bounded
/// disk pipeline.  Python owns the final multi-file transaction and optional
/// CSF-three-line Parquet conversion.
#[pyfunction]
#[pyo3(signature = (transcript, csf_output, csf_parquet_output, descriptor_output, header_output, scratch_dir, threads=None, memory_budget_mib=None, allow_unchecked_space=false))]
#[allow(clippy::too_many_arguments)]
fn generate_disk_outputs_from_transcript(
    py: Python,
    transcript: String,
    csf_output: String,
    csf_parquet_output: String,
    descriptor_output: String,
    header_output: String,
    scratch_dir: String,
    threads: Option<usize>,
    memory_budget_mib: Option<usize>,
    allow_unchecked_space: bool,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    if matches!(threads, Some(0)) {
        return Err(PyValueError::new_err("threads must be greater than 0"));
    }
    if matches!(memory_budget_mib, Some(0)) {
        return Err(PyValueError::new_err(
            "memory_budget_mib must be greater than 0",
        ));
    }
    let options = crate::csf_generation::GenerationOptions::from_api_with_space_policy(
        threads,
        memory_budget_mib,
        Some(Path::new(&scratch_dir).to_path_buf()),
        allow_unchecked_space,
    )
    .map_err(|error| PyValueError::new_err(format!("invalid generation options: {error:#}")))?;
    let stats = py
        .detach(|| {
            crate::csf_generation::streaming::generate_disk_outputs_from_transcript_with_options(
                &transcript,
                Path::new(&csf_output),
                Path::new(&csf_parquet_output),
                Path::new(&descriptor_output),
                Path::new(&header_output),
                &options,
            )
        })
        .map_err(|error| PyIOError::new_err(format!("{error:#}")))?;
    let output = PyDict::new(py);
    output.set_item("success", true)?;
    output.set_item("output_file", csf_output)?;
    output.set_item("parquet_file", csf_parquet_output)?;
    output.set_item("descriptor_file", descriptor_output)?;
    output.set_item("header_file", header_output)?;
    output.set_item("unique_occupations", stats.unique_occupations)?;
    output.set_item("generated_count", stats.generated_count)?;
    output.set_item("record_count", stats.unique_count)?;
    output.set_item("descriptor_count", stats.unique_count)?;
    output.set_item("duplicate_count", stats.duplicate_count)?;
    output.set_item("block_count", stats.block_count)?;
    output.set_item("csf_bytes", stats.csf_bytes)?;
    output.set_item("descriptor_bytes", stats.descriptor_bytes)?;
    output.set_item("segment_codec", stats.segment_codec)?;
    output.set_item("deduplication", stats.deduplication)?;
    let resource_stats = PyDict::new(py);
    resource_stats.set_item("memory_budget_mib", stats.resource_stats.memory_budget_mib)?;
    resource_stats.set_item("budget_bytes", stats.resource_stats.budget_bytes)?;
    resource_stats.set_item(
        "peak_managed_bytes",
        stats.resource_stats.peak_managed_bytes,
    )?;
    resource_stats.set_item(
        "current_managed_bytes",
        stats.resource_stats.current_managed_bytes,
    )?;
    resource_stats.set_item("occupation_bytes", stats.resource_stats.occupation_bytes)?;
    output.set_item("resource_stats", resource_stats)?;
    let stage_stats = PyList::empty(py);
    for stage in stats.stage_stats {
        let item = PyDict::new(py);
        item.set_item("name", stage.name)?;
        item.set_item("elapsed_millis", stage.elapsed_millis)?;
        item.set_item("cpu_millis", stage.cpu_millis)?;
        item.set_item("input_records", stage.input_records)?;
        item.set_item("output_records", stage.output_records)?;
        item.set_item("input_bytes", stage.input_bytes)?;
        item.set_item("output_bytes", stage.output_bytes)?;
        stage_stats.append(item)?;
    }
    output.set_item("stage_stats", stage_stats)?;
    output.set_item("plan_stats", plan_stats_dict(py, &stats.plan_stats)?)?;
    Ok(output.into())
}

/// The counted workload and schedule, shared by the generation result and the
/// estimate-only report so both describe the plan the same way.
fn plan_stats_dict(
    py: Python<'_>,
    stats: &crate::csf_generation::PlanStats,
) -> PyResult<pyo3::Py<PyDict>> {
    let plan_stats = PyDict::new(py);
    plan_stats.set_item("task_count", stats.task_count)?;
    plan_stats.set_item("target_records_per_task", stats.target_records_per_task)?;
    plan_stats.set_item("estimated_total_records", stats.estimated_total_records)?;
    plan_stats.set_item("unique_occupations", stats.unique_occupations)?;
    plan_stats.set_item(
        "zero_record_configurations",
        stats.zero_record_configurations,
    )?;
    plan_stats.set_item("unsplittable_tasks", stats.unsplittable_tasks)?;
    plan_stats.set_item("unsplittable_records", stats.unsplittable_records)?;
    let per_task = PyDict::new(py);
    let distribution = stats.estimated_records_per_task;
    per_task.set_item("count", distribution.count)?;
    per_task.set_item("total", distribution.total)?;
    per_task.set_item("minimum", distribution.minimum)?;
    per_task.set_item("p50", distribution.p50)?;
    per_task.set_item("p95", distribution.p95)?;
    per_task.set_item("maximum", distribution.maximum)?;
    plan_stats.set_item("estimated_records_per_task", per_task)?;
    Ok(plan_stats.unbind())
}

/// Count a transcript's workload and estimate its capacity without writing.
///
/// This runs the same enumeration, counting and planning as a real disk
/// generation, so its schedule and record counts describe exactly what a run
/// would do. It creates no scratch directory and publishes no file.
#[pyfunction]
#[pyo3(signature = (
    transcript,
    threads=None,
    memory_budget_mib=None,
    scratch_dir=None,
    staging_dir=None,
    destinations=None
))]
#[allow(clippy::too_many_arguments)] // PyO3 exposes one argument per Python parameter.
fn estimate_disk_generation(
    py: Python,
    transcript: String,
    threads: Option<usize>,
    memory_budget_mib: Option<usize>,
    scratch_dir: Option<String>,
    staging_dir: Option<String>,
    destinations: Option<Vec<(String, String)>>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    if matches!(threads, Some(0)) {
        return Err(PyValueError::new_err("threads must be greater than 0"));
    }
    if matches!(memory_budget_mib, Some(0)) {
        return Err(PyValueError::new_err(
            "memory_budget_mib must be greater than 0",
        ));
    }
    let options =
        crate::csf_generation::GenerationOptions::from_api(threads, memory_budget_mib, None)
            .map_err(|error| {
                PyValueError::new_err(format!("invalid generation options: {error:#}"))
            })?;
    let layout = crate::csf_generation::streaming::SpaceLayout {
        scratch_dir: scratch_dir.map(PathBuf::from),
        staging_dir: staging_dir.map(PathBuf::from),
        destinations: destinations
            .unwrap_or_default()
            .into_iter()
            .map(|(kind, path)| {
                Ok((
                    PathBuf::from(path),
                    crate::csf_generation::ArtifactKind::from_name(&kind)?,
                ))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()
            .map_err(|error| PyValueError::new_err(format!("{error:#}")))?,
    };
    let (estimate, checks) = py
        .detach(|| {
            crate::csf_generation::streaming::estimate_disk_generation_with_layout(
                &transcript,
                &options,
                &layout,
            )
        })
        .map_err(|error| PyIOError::new_err(format!("{error:#}")))?;
    let output = PyDict::new(py);
    output.set_item("success", true)?;
    output.set_item("unique_occupations", estimate.plan_stats.unique_occupations)?;
    output.set_item(
        "pre_deduplication_records",
        estimate.capacity.pre_deduplication_records,
    )?;
    output.set_item("peel_subshells", estimate.capacity.peel_subshells)?;
    output.set_item("v2_columns", estimate.capacity.v2_columns)?;
    output.set_item("segment_codec", estimate.segment_codec)?;
    output.set_item("deduplication", estimate.deduplication)?;
    output.set_item("enumeration_millis", estimate.enumeration_millis)?;
    output.set_item("planning_millis", estimate.planning_millis)?;
    output.set_item("plan_stats", plan_stats_dict(py, &estimate.plan_stats)?)?;
    let capacity = &estimate.capacity;
    let bytes = PyDict::new(py);
    bytes.set_item("segments", capacity.segment_bytes)?;
    bytes.set_item("root_buckets", capacity.root_bucket_bytes)?;
    bytes.set_item("recursive_buckets", capacity.recursive_bucket_bytes)?;
    bytes.set_item("survivor_bitsets", capacity.survivor_bitset_bytes)?;
    bytes.set_item("scratch_peak", capacity.scratch_peak_bytes)?;
    bytes.set_item("descriptor", capacity.descriptor_bytes)?;
    bytes.set_item("csf_text", capacity.csf_text_bytes)?;
    bytes.set_item("csf_parquet", capacity.csf_parquet_bytes)?;
    bytes.set_item("staged_outputs", capacity.staged_output_bytes)?;
    bytes.set_item("required_scratch", capacity.required_scratch_bytes)?;
    bytes.set_item("required_output", capacity.required_output_bytes)?;
    output.set_item("bytes", bytes)?;
    output.set_item("assumptions", capacity.assumption_lines().to_vec())?;
    // No manifest binds the scratch of a failed run to its input and format
    // version, so scratch is never reused: a failed run restarts.
    output.set_item("failure_recovery", "restart")?;
    let space_checks = PyList::empty(py);
    for check in checks {
        let item = PyDict::new(py);
        item.set_item("path", check.path)?;
        item.set_item("required_bytes", check.required_bytes)?;
        item.set_item("free_bytes", check.free_bytes)?;
        item.set_item("sufficient", check.sufficient)?;
        space_checks.append(item)?;
    }
    output.set_item("space_checks", space_checks)?;
    Ok(output.into())
}
