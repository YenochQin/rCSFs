// Public modules for integration testing
pub mod atomic_output;
pub mod complete_csf;
pub mod configuration_generation;
pub mod csf_active_space_split;
pub mod csf_block_split;
mod csf_output;
pub mod csf_partition;
pub mod csfs_conversion;
pub mod csfs_descriptor;
pub mod csfs_memory;
pub mod descriptor_schema;
pub mod descriptor_v2;
pub mod interaction;
#[cfg(feature = "python")]
mod interaction_py;
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
