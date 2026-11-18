//! Python adapters for full-CSF generation and its capacity estimate.
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList, PyListMethods};
use std::path::PathBuf;

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
