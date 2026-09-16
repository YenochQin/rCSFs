//! Python boundary adapter for [`crate::interaction`].
//!
//! Option decoding, error classification, and statistics serialization live
//! here so `lib.rs` keeps only registration, argument receipt, and a single
//! call into the interaction API.

use crate::complete_csf::Parity;
use crate::interaction::{
    self, HamiltonianMode, InteractionBlockStats, InteractionMethod, InteractionStats,
};
use pyo3::exceptions::{PyFileExistsError, PyIOError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList, PyListMethods};
use std::path::Path;

/// Run structural interaction selection on behalf of the Python wrapper.
#[allow(clippy::too_many_arguments)] // Mirrors the Python signature one-to-one.
pub fn select_interacting_csfs(
    py: Python<'_>,
    reference_csf: &str,
    candidate_csf: &str,
    output_csf: &str,
    hamiltonian: &str,
    method: &str,
    num_workers: Option<&Bound<'_, PyAny>>,
    overwrite: bool,
) -> PyResult<Py<PyAny>> {
    let workers = worker_count(num_workers)?;
    let mode = HamiltonianMode::parse(hamiltonian)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let interaction_method = InteractionMethod::parse(method)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    // Reported before the run so an unusable destination fails fast with the
    // dedicated exception class rather than a generic selection error.
    if !overwrite && Path::new(output_csf).exists() {
        return Err(PyFileExistsError::new_err(format!(
            "output file {output_csf} already exists (set overwrite=True to replace it)"
        )));
    }

    let stats = py
        .detach(|| {
            interaction::select_interacting_csfs(
                Path::new(reference_csf),
                Path::new(candidate_csf),
                Path::new(output_csf),
                mode,
                interaction_method,
                workers,
                overwrite,
            )
        })
        .map_err(interaction_error)?;

    let result = stats_to_dict(py, &stats, reference_csf, candidate_csf, output_csf)?;
    Ok(result.into_any().unbind())
}

/// Validate the Python worker count before any unsigned conversion.
///
/// PyO3 would reject a negative or oversized integer with `OverflowError`
/// while extracting a `usize` argument, before the function body runs. The
/// public API promises `ValueError` for every non-positive worker count, so
/// the value is classified here while it is still a Python object.
fn worker_count(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<usize>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }

    // `__index__` accepts `int` and integer-like objects while rejecting
    // `float` and `str`, matching the documented `int | None` parameter.
    let index = value
        .call_method0("__index__")
        .map_err(|_| PyTypeError::new_err("num_workers must be an integer or None"))?;
    // Comparing in Python classifies arbitrarily large integers without any
    // conversion that could raise `OverflowError` first.
    if index.lt(0)? {
        return Err(PyValueError::new_err("num_workers must be greater than 0"));
    }
    match index.extract::<usize>() {
        Ok(0) => Err(PyValueError::new_err("num_workers must be greater than 0")),
        Ok(count) => Ok(Some(count)),
        Err(_) => Err(PyValueError::new_err(
            "num_workers exceeds the maximum worker count supported on this platform",
        )),
    }
}

/// Classify a selection failure into the documented Python exception classes.
fn interaction_error(error: anyhow::Error) -> PyErr {
    let message = error.to_string();
    let io_error = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<std::io::Error>());
    match io_error.map(std::io::Error::kind) {
        Some(std::io::ErrorKind::AlreadyExists) => PyFileExistsError::new_err(message),
        Some(_) => PyIOError::new_err(message),
        None => PyValueError::new_err(message),
    }
}

const fn parity_str(parity: Parity) -> &'static str {
    match parity {
        Parity::Even => "+",
        Parity::Odd => "-",
    }
}

fn stats_to_dict<'py>(
    py: Python<'py>,
    stats: &InteractionStats,
    reference_csf: &str,
    candidate_csf: &str,
    output_csf: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("exact", stats.exact)?;
    result.set_item("hamiltonian", stats.mode.as_str())?;
    result.set_item("method", stats.method.as_str())?;
    result.set_item("reference_file", reference_csf)?;
    result.set_item("candidate_file", candidate_csf)?;
    result.set_item("output_file", output_csf)?;
    result.set_item("block_count", stats.block_count)?;
    result.set_item("reference_count", stats.reference_count)?;
    result.set_item("candidate_count", stats.candidate_count)?;
    result.set_item("exact_reference_skipped", stats.exact_reference_skipped)?;
    result.set_item("selected_count", stats.selected_count)?;
    result.set_item("rejected_count", stats.rejected_count)?;
    result.set_item("output_count", stats.output_count)?;
    result.set_item("output_bytes", stats.output_bytes)?;

    let blocks = PyList::empty(py);
    for block in &stats.blocks {
        blocks.append(block_to_dict(py, block)?)?;
    }
    result.set_item("blocks", blocks)?;
    Ok(result)
}

fn block_to_dict<'py>(
    py: Python<'py>,
    block: &InteractionBlockStats,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("block_index", block.block_index)?;
    result.set_item("total_two_j", block.total_two_j)?;
    result.set_item("parity", parity_str(block.parity))?;
    result.set_item("reference_count", block.reference_count)?;
    result.set_item("candidate_count", block.candidate_count)?;
    result.set_item("exact_reference_skipped", block.exact_reference_skipped)?;
    result.set_item("selected_count", block.selected_count)?;
    result.set_item("rejected_count", block.rejected_count)?;
    result.set_item("output_count", block.output_count)?;
    Ok(result)
}
