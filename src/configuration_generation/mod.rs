//! Related full-CSF and CSFG generation workflows over one shared kernel.
pub mod csf_generation;
pub mod csfg_generation;
pub mod shared_generation;

// Stable public Rust interface. New callers may use the explicit submodules.
pub use csf_generation::{
    TranscriptGenerationStats, WriteStats, generate_csfs_from_transcript, write_generated_csfs,
};
pub use shared_generation::{
    EnumeratedConfiguration, EnumeratedOccupations, ExcitationRequest, GenerationRequest,
    OccupationMode, Orbital, ReferenceConfiguration, ReferenceSubshell, Subshell,
    SubshellOccupation, enumerate_occupations, generate_csfs, generate_csfs_parallel,
};
// Existing csf_generation::csfg callers remain source compatible.
pub use csfg_generation as csfg;
