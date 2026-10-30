//! Full-CSF generation, disk processing, artifact encoding and Python adapters.
mod complete_csf_generation_pipeline;
pub(crate) mod csf_artifact_encoding;
pub(crate) mod csf_output_capacity_estimation;
#[allow(dead_code)]
pub(crate) mod disk_csf_generation_pipeline;
mod disk_space_preflight;
pub(crate) mod python_csf_bindings;

pub use complete_csf_generation_pipeline::{
    TranscriptGenerationStats, WriteStats, generate_csfs_from_transcript, write_generated_csfs,
};
pub(crate) use csf_output_capacity_estimation::{
    ArtifactKind, CapacityEstimate, estimate_capacity,
};
pub(crate) use disk_space_preflight::{
    SpaceCheck, SpacePolicy, SpaceRole, check_space, preflight_run,
};
// The full-CSF pipeline consumes the same low-level interfaces as CSFG.
pub(crate) use super::shared_generation::{
    DeduplicationStrategy, EnumeratedConfiguration, EnumeratedOccupations, ExcitationRequest,
    GeneratedRecordRef, GeneratedRecordSink, GenerationOptions, GenerationPlan, GenerationRequest,
    PlanStats, PlannedTask, RecordSelection, ResourceBudget, ResourcePermit, ResourceStats,
    SegmentCodec, Subshell, SubshellOccupation, TaskSpan, enumerate_occupations,
    enumerate_occupations_with_budget, estimate_workload, generate_configuration_records,
    generate_csfs_parallel, plan_generation, report_plan, request_targets,
};
