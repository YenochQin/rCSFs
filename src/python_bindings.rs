//! Python interfaces over the shared Rust implementation.
use crate::*;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModuleMethods;
use std::path::Path;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "DESCRIPTOR_VERSION",
        descriptor_transport::DescriptorVersion::V3.tag(),
    )?;
    descriptor_python::register(m)?;
    m.add_function(wrap_pyfunction!(get_parquet_info, m)?)?;
    m.add_function(wrap_pyfunction!(select_interacting_csfs, m)?)?;
    m.add_function(wrap_pyfunction!(seniority_to_printed_v, m)?)?;
    m.add_function(wrap_pyfunction!(
        configuration_generation::csf_generation::python_csf_bindings::estimate_disk_generation,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(read_peel_subshells, m)?)?;
    Ok(())
}

#[pyfunction]
fn read_peel_subshells(header_path: String) -> PyResult<Vec<String>> {
    descriptor_dataset::read_header(Path::new(&header_path))
        .map(|h| h.peel)
        .map_err(|e| PyIOError::new_err(format!("{e:#}")))
}

#[pyfunction]
fn seniority_to_printed_v(
    two_j: u16,
    occupation: u8,
    subshell_two_j: u16,
    seniority: u8,
) -> PyResult<i32> {
    seniority::seniority_to_printed_v(two_j, occupation, subshell_two_j, seniority)
        .map_err(|error| PyValueError::new_err(format!("{error:#}")))
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

#[pyfunction]
fn get_parquet_info(py: Python, input_path: String) -> PyResult<Py<PyAny>> {
    let info = py
        .detach(|| parquet_info::read_parquet_info(Path::new(&input_path)))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    descriptor_python::python_value(py, &info)
}
