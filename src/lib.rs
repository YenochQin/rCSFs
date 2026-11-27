//! Generate GRASP configuration state functions and transform V3 descriptor datasets.

pub mod atomic_output;
pub mod complete_csf;
pub mod configuration_generation;
mod csf_output;
pub mod csfs_descriptor;
pub mod descriptor_dataset;
pub mod descriptor_operations;
#[cfg(feature = "python")]
mod descriptor_python;
pub mod descriptor_schema;
pub mod descriptor_transport;
pub mod descriptor_v3;
pub mod interaction;
#[cfg(feature = "python")]
mod interaction_py;
mod orbital_label;
mod parquet_compression;
pub mod parquet_info;
mod reporting;
pub mod seniority;
#[cfg(windows)]
mod windows_filesystem;

/// Compatibility alias for the established Rust generator interface.
pub use configuration_generation as csf_generation;

#[cfg(feature = "python")]
mod python_bindings;

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "python")]
#[pyo3::pymodule]
fn _rcsfs(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    python_bindings::register(m)
}
