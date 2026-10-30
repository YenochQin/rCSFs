//! Thin native adapter; CSFG rules live in configuration_generation::csfg_generation.
use super::{CsfgRequest, generate_csfgs as generate};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use std::path::Path;

#[pyfunction]
#[pyo3(signature = (inactive_core, reference_configuration, labeling_space, active_space, j_min, j_max, excitations, csfg_output, label_output, threads=None))]
#[allow(clippy::too_many_arguments)] // Mirrors the small structured Python request.
pub(crate) fn generate_csfgs(
    py: Python<'_>,
    inactive_core: u8,
    reference_configuration: Vec<String>,
    labeling_space: String,
    active_space: String,
    j_min: u16,
    j_max: u16,
    excitations: u8,
    csfg_output: String,
    label_output: String,
    threads: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let request = CsfgRequest {
        inactive_core,
        reference_configuration,
        labeling_space,
        active_space,
        j_min,
        j_max,
        excitations,
    };
    let stats = py
        .detach(|| {
            generate(
                &request,
                Path::new(&csfg_output),
                Path::new(&label_output),
                threads,
            )
        })
        .map_err(|error| PyValueError::new_err(format!("{error:#}")))?;
    let out = PyDict::new(py);
    out.set_item("success", true)?;
    out.set_item("rcsfg_out", csfg_output)?;
    out.set_item("rlabel_out", label_output)?;
    out.set_item("record_count", stats.record_count)?;
    out.set_item("label_csf_count", stats.label_csf_count)?;
    out.set_item(
        "generator_count",
        stats.record_count - stats.label_csf_count,
    )?;
    out.set_item("expanded_csf_count", stats.expanded_csf_count)?;
    out.set_item("type_counts", stats.type_counts.to_vec())?;
    out.set_item("unique_occupations", stats.unique_occupations)?;
    out.set_item(
        "representative_occupations",
        stats.representative_occupations,
    )?;
    out.set_item("block_count", stats.blocks.len())?;
    let mut blocks = Vec::new();
    for block in stats.blocks {
        let b = PyDict::new(py);
        b.set_item("two_j", block.two_j)?;
        b.set_item("parity", if block.odd { "-" } else { "+" })?;
        b.set_item("record_count", block.record_count)?;
        b.set_item("expanded_csf_count", block.expanded_csf_count)?;
        b.set_item("label_csf_count", block.label_csf_count)?;
        blocks.push(b);
    }
    out.set_item("blocks", blocks)?;
    out.set_item("csfg_bytes", stats.csfg_bytes)?;
    out.set_item("label_bytes", stats.label_bytes)?;
    out.set_item("planning_seconds", stats.planning_seconds)?;
    out.set_item("generation_seconds", stats.generation_seconds)?;
    out.set_item("writing_seconds", stats.writing_seconds)?;
    Ok(out.into())
}
