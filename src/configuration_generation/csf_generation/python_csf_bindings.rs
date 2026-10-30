//! Python adapters for full-CSF generation and its capacity estimate.
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList, PyListMethods};
use std::path::{Path, PathBuf};

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
    threads=None
))]
pub(crate) fn generate_csfs_from_transcript(
    py: Python,
    transcript: String,
    output_path: String,
    threads: Option<usize>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    if matches!(threads, Some(0)) {
        return Err(PyValueError::new_err("threads must be greater than 0"));
    }

    let result = py.detach(|| {
        crate::configuration_generation::generate_csfs_from_transcript(
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
/// disk pipeline.  Python owns the final multi-file publication and optional
/// CSF-three-line Parquet conversion.
#[pyfunction]
#[pyo3(signature = (transcript, csf_output, csf_parquet_output, descriptor_output, header_output, scratch_dir, threads=None, memory_budget_mib=None, allow_unchecked_space=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_disk_outputs_from_transcript(
    py: Python,
    transcript: &Bound<'_, PyAny>,
    csf_output: String,
    csf_parquet_output: String,
    descriptor_output: String,
    header_output: String,
    scratch_dir: String,
    threads: Option<usize>,
    memory_budget_mib: Option<usize>,
    allow_unchecked_space: bool,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    let single = transcript.extract::<String>().ok();
    let multiple = if single.is_none() {
        Some(transcript.extract::<Vec<String>>().map_err(|_| {
            PyValueError::new_err("transcript must be a string or an array of strings")
        })?)
    } else {
        None
    };
    if matches!(threads, Some(0)) {
        return Err(PyValueError::new_err("threads must be greater than 0"));
    }
    if matches!(memory_budget_mib, Some(0)) {
        return Err(PyValueError::new_err(
            "memory_budget_mib must be greater than 0",
        ));
    }
    let options = crate::configuration_generation::shared_generation::GenerationOptions::from_api_with_space_policy(
        threads,
        memory_budget_mib,
        Some(Path::new(&scratch_dir).to_path_buf()),
        allow_unchecked_space,
    )
    .map_err(|error| PyValueError::new_err(format!("invalid generation options: {error:#}")))?;
    let stats = py
        .detach(|| match (single.as_deref(), multiple.as_deref()) {
            (Some(text), _) =>
                crate::configuration_generation::csf_generation::disk_csf_generation_pipeline::generate_disk_outputs_from_transcript_with_options(
                    text,
                    Path::new(&csf_output),
                    Path::new(&csf_parquet_output),
                    Path::new(&descriptor_output),
                    Path::new(&header_output),
                    &options,
                ),
            (_, Some(texts)) =>
                crate::configuration_generation::csf_generation::disk_csf_generation_pipeline::generate_disk_outputs_from_transcripts_with_options(
                    texts,
                    Path::new(&csf_output),
                    Path::new(&csf_parquet_output),
                    Path::new(&descriptor_output),
                    Path::new(&header_output),
                    &options,
                ),
            _ => unreachable!(),
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
    stats: &crate::configuration_generation::shared_generation::PlanStats,
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
pub(crate) fn estimate_disk_generation(
    py: Python,
    transcript: &Bound<'_, PyAny>,
    threads: Option<usize>,
    memory_budget_mib: Option<usize>,
    scratch_dir: Option<String>,
    staging_dir: Option<String>,
    destinations: Option<Vec<(String, String)>>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    let single = transcript.extract::<String>().ok();
    let multiple = if single.is_none() {
        Some(transcript.extract::<Vec<String>>().map_err(|_| {
            PyValueError::new_err("transcript must be a string or an array of strings")
        })?)
    } else {
        None
    };
    if matches!(threads, Some(0)) {
        return Err(PyValueError::new_err("threads must be greater than 0"));
    }
    if matches!(memory_budget_mib, Some(0)) {
        return Err(PyValueError::new_err(
            "memory_budget_mib must be greater than 0",
        ));
    }
    let options = crate::configuration_generation::shared_generation::GenerationOptions::from_api(
        threads,
        memory_budget_mib,
        None,
    )
    .map_err(|error| PyValueError::new_err(format!("invalid generation options: {error:#}")))?;
    let layout = crate::configuration_generation::csf_generation::disk_csf_generation_pipeline::SpaceLayout {
        scratch_dir: scratch_dir.map(PathBuf::from),
        staging_dir: staging_dir.map(PathBuf::from),
        destinations: destinations
            .unwrap_or_default()
            .into_iter()
            .map(|(kind, path)| {
                Ok((
                    PathBuf::from(path),
                    super::ArtifactKind::from_name(&kind)?,
                ))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()
            .map_err(|error| PyValueError::new_err(format!("{error:#}")))?,
    };
    let (estimate, checks) = py
        .detach(|| match (single.as_deref(), multiple.as_deref()) {
            (Some(text), _) =>
                crate::configuration_generation::csf_generation::disk_csf_generation_pipeline::estimate_disk_generation_with_layout(
                    text, &options, &layout,
                ),
            (_, Some(texts)) =>
                crate::configuration_generation::csf_generation::disk_csf_generation_pipeline::estimate_disk_generation_from_transcripts_with_layout(
                    texts, &options, &layout,
                ),
            _ => unreachable!(),
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
