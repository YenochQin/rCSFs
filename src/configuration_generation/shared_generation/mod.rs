//! Shared occupation, state-table and angular-coupling implementation.
mod angular_coupling_generator;
pub(crate) mod csf_record_counting;
mod electron_occupation_enumeration;
mod generation_execution_options;
#[allow(dead_code)]
pub(crate) mod generation_task_planning;
mod subshell_state_catalog;

pub(crate) use angular_coupling_generator::{
    GeneratedRecordRef, GeneratedRecordSink, PreparedGeneration, RecordSelection,
    generate_configuration_records, generate_prepared_records, prepare_configuration,
    state_prefixes,
};
pub use angular_coupling_generator::{
    GenerationRequest, Subshell, SubshellOccupation, generate_csfs, generate_csfs_parallel,
};
pub(crate) use csf_record_counting::{PreparedCounter, count_configuration_records};
pub(crate) use electron_occupation_enumeration::enumerate_occupations_with_budget;
pub use electron_occupation_enumeration::{
    EnumeratedConfiguration, EnumeratedOccupations, ExcitationRequest, OccupationMode, Orbital,
    ReferenceConfiguration, ReferenceSubshell, enumerate_occupations,
};
pub(crate) use generation_execution_options::{
    DeduplicationStrategy, GenerationOptions, ResourceBudget, ResourcePermit, ResourceStats,
    SegmentCodec,
};
pub(crate) use generation_task_planning::{
    GenerationPlan, PlanStats, PlannedTask, TaskSpan, estimate_workload, plan_generation,
    report_plan, request_targets,
};

pub(crate) use subshell_state_catalog::subshell_states;
