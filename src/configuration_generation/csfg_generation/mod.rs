//! Direct CSFG generation and its GRASPG-compatible paired output.
mod direct_csfg_generation_pipeline;
mod graspg_csfg_output_writer;
pub(crate) mod python_csfg_bindings;
mod representative_occupation_planning;

pub use direct_csfg_generation_pipeline::{CsfgRequest, generate_csfgs};
pub use graspg_csfg_output_writer::{CsfgBlockStats, CsfgStats};

#[cfg(test)]
mod csfg_compatibility_tests;
